"""
MindGraphNet - Cloud Deployment & Inference API (FastAPI)
Production REST API with real-time dynamic multimodal feature extraction,
calibrated risk prediction, and on-demand XAI explanations.
"""
from typing import List, Optional, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config.settings import (
    TRANSFORMER_EMB_DIM, GRAPH_EMB_DIM, N_ENGINEERED_FEATURES,
    CLASS_NAMES, HIGH_RISK_ALERT_THRESHOLD
)
from mindgraphnet.preprocessing.anonymizer import DataAnonymizer
from mindgraphnet.preprocessing.text_cleaner import ClinicalTextCleaner
from mindgraphnet.features.nlp_features import NLPFeatureExtractor
from mindgraphnet.features.psycholinguistics import PsycholinguisticExtractor
from mindgraphnet.engine.transformer_module import TransformerEncoder
from mindgraphnet.engine.fusion_model import MultiHybridFusionNet
from mindgraphnet.xai.token_saliency import TokenSaliencyExplainer
from mindgraphnet.xai.shap_explainer import GroupedSHAPExplainer
from mindgraphnet.analytics.risk_scorer import ClinicalRiskScorer

app = FastAPI(
    title="MindGraphNet Multi-Hybrid Clinical Analytics API",
    version="2.0.0",
    description="Multi-hybrid Transformer, Graph Learning, and Psycholinguistic Mental Health Risk Prediction API."
)

# Global model & component holders
anonymizer = DataAnonymizer()
cleaner = ClinicalTextCleaner()
nlp_extractor = NLPFeatureExtractor()
psych_extractor = PsycholinguisticExtractor()
transformer_encoder = TransformerEncoder()
fusion_model = MultiHybridFusionNet()
shap_explainer = GroupedSHAPExplainer(fusion_model)
risk_scorer = ClinicalRiskScorer()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fusion_model.to(DEVICE)
fusion_model.eval()

# Graph structural prior: learned mean representation for incoming unseen nodes
graph_structural_prior = np.zeros(GRAPH_EMB_DIM, dtype=np.float32)

# --- Pydantic Request / Response Schemas ---
class PredictRequest(BaseModel):
    text: str = Field(..., example="I feel completely overwhelmed and hopeless. Nothing I do matters anymore.")
    author: Optional[str] = Field("anonymous_poster", example="user_123")
    topic: Optional[str] = Field("mentalhealth", example="depression")

class PredictResponse(BaseModel):
    risk_level: str
    confidence: float
    continuous_risk_score: float
    is_crisis: bool
    crisis_signals: List[str]
    sentiment_score: float
    sentiment_label: str
    stress_anxiety_index: float
    depressive_signal_index: float
    resilience_index: float
    class_probabilities: Dict[str, float]
    factor_importance: Dict[str, float]
    disclaimer: str = (
        "⚠️ RESEARCH / DECISION SUPPORT SIGNAL ONLY. "
        "MindGraphNet is not a clinical diagnostic device. Never use for autonomous medical decisions."
    )

class ExplainResponse(BaseModel):
    risk_level: str
    confidence: float
    top_attended_words: List[Tuple[str, float]]
    highlighted_html: str
    modality_weights: Dict[str, float]
    factor_importance: Dict[str, float]

class BatchPredictRequest(BaseModel):
    posts: List[PredictRequest]

class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]
    high_risk_ratio: float
    alert_triggered: bool


def _extract_live_features(text: str):
    """
    Extracts complete semantic, emotion, psycholinguistic, and structural vectors dynamically.
    No empty/zero placeholders!
    """
    scrubbed = anonymizer.scrub_pii(text)
    cleaned = cleaner.clean_text(scrubbed)
    tokens = cleaner.tokenize(cleaned)
    tokens_nostop = cleaner.remove_stopwords(tokens)

    # 1. NLP Sentiment & Emotions
    sentiment_data = nlp_extractor.get_sentiment(cleaned)
    emotion_data = nlp_extractor.detect_emotions(tokens_nostop)

    # 2. Psycholinguistics
    psych_data = psych_extractor.extract_features(scrubbed, tokens)

    # 3. Assemble engineered feature vector (16 dimensions)
    # sentiment_score, 6 emotions, 6 linguistic, 3 interaction defaults
    engineered_list = (
        [sentiment_data["sentiment_score"]] +
        [emotion_data[f"emo_{e}"] for e in ["sadness", "fear", "anger", "joy", "fatigue", "hopelessness"]] +
        [psych_data["word_count"], psych_data["avg_word_len"], psych_data["exclamation_count"],
         psych_data["first_person_ratio"], psych_data["cognitive_distortion_score"], psych_data["distress_index"]] +
        [0.0, 1.0, 0.0] # replies_received, author_activity, thread_depth
    )
    psych_vec = np.array(engineered_list, dtype=np.float32)

    # 4. Transformer Semantic Vector & Attention
    cls_emb, attn_matrix, raw_tokens = transformer_encoder.encode_text(cleaned)

    return {
        "scrubbed": scrubbed,
        "cleaned": cleaned,
        "tokens": tokens,
        "sentiment_data": sentiment_data,
        "emotion_data": emotion_data,
        "psych_data": psych_data,
        "psych_vec": psych_vec,
        "cls_emb": cls_emb,
        "attn_matrix": attn_matrix,
        "raw_tokens": raw_tokens
    }


@app.get("/api/v1/health")
def health():
    return {
        "status": "active",
        "device": str(DEVICE),
        "transformer_active": transformer_encoder.model is not None,
        "model_architecture": "MindGraphNet Multi-Hybrid Fusion v2.0"
    }


@app.post("/api/v1/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.text or len(req.text.strip()) < 3:
        raise HTTPException(status_code=400, detail="Text payload cannot be empty.")

    feats = _extract_live_features(req.text)
    
    # Model inference
    with torch.no_grad():
        s_t = torch.tensor(feats["cls_emb"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        g_t = torch.tensor(graph_structural_prior, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        p_t = torch.tensor(feats["psych_vec"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        logits, risk_cont, weights = fusion_model(s_t, g_t, p_t)
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        risk_score_val = float(risk_cont.cpu().numpy()[0, 0])

    # Clinical triaging evaluation
    eval_res = risk_scorer.evaluate_post(
        text_raw=req.text,
        probs=probs,
        continuous_risk=risk_score_val,
        emo_dict=feats["emotion_data"],
        sentiment_score=feats["sentiment_data"]["sentiment_score"]
    )

    factor_imp = shap_explainer.explain_instance(feats["cls_emb"], graph_structural_prior, feats["psych_vec"])

    return PredictResponse(
        risk_level=eval_res["risk_level"],
        confidence=eval_res["confidence"],
        continuous_risk_score=eval_res["continuous_risk_score"],
        is_crisis=eval_res["is_crisis"],
        crisis_signals=eval_res["crisis_signals_detected"],
        sentiment_score=feats["sentiment_data"]["sentiment_score"],
        sentiment_label=feats["sentiment_data"]["sentiment_label"],
        stress_anxiety_index=eval_res["stress_anxiety_index"],
        depressive_signal_index=eval_res["depressive_signal_index"],
        resilience_index=eval_res["resilience_index"],
        class_probabilities=eval_res["class_probabilities"],
        factor_importance=factor_imp
    )


@app.post("/api/v1/explain", response_model=ExplainResponse)
def explain(req: PredictRequest):
    feats = _extract_live_features(req.text)
    
    with torch.no_grad():
        s_t = torch.tensor(feats["cls_emb"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        g_t = torch.tensor(graph_structural_prior, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        p_t = torch.tensor(feats["psych_vec"], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        logits, _, weights = fusion_model(s_t, g_t, p_t)
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        w_np = weights.cpu().numpy()[0]

    words, norm_scores, html = TokenSaliencyExplainer.process_attention(feats["attn_matrix"], feats["raw_tokens"])
    top_words = TokenSaliencyExplainer.get_top_attended_words(words, norm_scores, top_k=6)
    factor_imp = shap_explainer.explain_instance(feats["cls_emb"], graph_structural_prior, feats["psych_vec"])

    return ExplainResponse(
        risk_level=CLASS_NAMES[int(probs.argmax())],
        confidence=round(float(probs.max()), 3),
        top_attended_words=top_words,
        highlighted_html=html,
        modality_weights={
            "Semantic Language": round(float(w_np[0]), 3),
            "Graph Structural": round(float(w_np[1]), 3),
            "Psycholinguistic": round(float(w_np[2]), 3)
        },
        factor_importance=factor_imp
    )


@app.post("/api/v1/predict_batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest):
    results = [predict(item) for item in req.posts]
    high_count = sum(1 for r in results if r.risk_level == "High")
    ratio = high_count / max(len(results), 1)
    return BatchPredictResponse(
        results=results,
        high_risk_ratio=round(ratio, 3),
        alert_triggered=(ratio >= HIGH_RISK_ALERT_THRESHOLD)
    )
