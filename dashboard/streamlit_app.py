"""
MindGraphNet - Interactive Clinical & Research Dashboard
A Streamlit analytics portal for mental health researchers, clinicians, and platform moderators.
"""

import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for folder in [PROJECT_ROOT, PROJECT_ROOT / "MindGraphNet"]:
    if folder.is_dir() and str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from config.settings import GRAPH_EMB_DIM
from data.loader import load_dataset
from mindgraphnet.analytics.risk_scorer import ClinicalRiskScorer
from mindgraphnet.engine.fusion_model import MultiHybridFusionNet
from mindgraphnet.engine.transformer_module import TransformerEncoder
from mindgraphnet.features.nlp_features import NLPFeatureExtractor
from mindgraphnet.features.psycholinguistics import PsycholinguisticExtractor
from mindgraphnet.preprocessing.anonymizer import DataAnonymizer
from mindgraphnet.preprocessing.text_cleaner import ClinicalTextCleaner
from mindgraphnet.xai.shap_explainer import GroupedSHAPExplainer
from mindgraphnet.xai.token_saliency import TokenSaliencyExplainer

st.set_page_config(page_title="MindGraphNet", page_icon="🧠", layout="wide")

st.markdown(
    """
    <style>
    .main { background: linear-gradient(135deg, #0f172a 0%, #111827 100%); }
    .block-container { padding-top: 2rem; }
    .stMetric>div { background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(148,163,184,0.25); border-radius: 12px; }
    .stTabs [role="tablist"] button { font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def initialize_pipeline():
    anonymizer = DataAnonymizer()
    cleaner = ClinicalTextCleaner()
    nlp_extractor = NLPFeatureExtractor()
    psych_extractor = PsycholinguisticExtractor()
    transformer_encoder = TransformerEncoder()
    fusion_model = MultiHybridFusionNet()
    shap_explainer = GroupedSHAPExplainer(fusion_model)
    risk_scorer = ClinicalRiskScorer()
    graph_structural_prior = np.zeros(GRAPH_EMB_DIM, dtype=np.float32)
    return {
        "anonymizer": anonymizer,
        "cleaner": cleaner,
        "nlp_extractor": nlp_extractor,
        "psych_extractor": psych_extractor,
        "transformer_encoder": transformer_encoder,
        "fusion_model": fusion_model,
        "shap_explainer": shap_explainer,
        "risk_scorer": risk_scorer,
        "graph_structural_prior": graph_structural_prior,
    }


def analyze_text(text: str, author: str = "anonymous_poster", topic: str = "mentalhealth") -> dict:
    pipeline = initialize_pipeline()
    if not text or len(str(text).strip()) < 3:
        return {"error": "Text input is too short."}

    features = _extract_features(pipeline, text)
    with torch.no_grad():
        s_t = torch.tensor(features["cls_emb"], dtype=torch.float32).unsqueeze(0)
        g_t = torch.tensor(pipeline["graph_structural_prior"], dtype=torch.float32).unsqueeze(0)
        p_t = torch.tensor(features["psych_vec"], dtype=torch.float32).unsqueeze(0)
        logits, risk_cont, weights = pipeline["fusion_model"](s_t, g_t, p_t)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        risk_score_val = float(risk_cont.cpu().numpy()[0, 0])

    eval_res = pipeline["risk_scorer"].evaluate_post(
        text_raw=text,
        probs=probs,
        continuous_risk=risk_score_val,
        emo_dict=features["emotion_data"],
        sentiment_score=features["sentiment_data"]["sentiment_score"],
    )
    factor_imp = pipeline["shap_explainer"].explain_instance(
        features["cls_emb"],
        pipeline["graph_structural_prior"],
        features["psych_vec"],
    )
    words, scores, html = TokenSaliencyExplainer.process_attention(
        features["attn_matrix"],
        features["raw_tokens"],
    )
    top_words = TokenSaliencyExplainer.get_top_attended_words(words, scores, top_k=6)

    response = {
        "risk_level": eval_res["risk_level"],
        "confidence": eval_res["confidence"],
        "continuous_risk_score": eval_res["continuous_risk_score"],
        "is_crisis": eval_res["is_crisis"],
        "crisis_signals": eval_res["crisis_signals_detected"],
        "sentiment_score": features["sentiment_data"]["sentiment_score"],
        "sentiment_label": features["sentiment_data"]["sentiment_label"],
        "stress_anxiety_index": eval_res["stress_anxiety_index"],
        "depressive_signal_index": eval_res["depressive_signal_index"],
        "resilience_index": eval_res["resilience_index"],
        "class_probabilities": eval_res["class_probabilities"],
        "factor_importance": factor_imp,
        "top_attended_words": top_words,
        "highlighted_html": html,
        "raw_text": text,
        "author": author,
        "topic": topic,
        "scrubbed_text": features["scrubbed"],
    }
    return response


def _extract_features(pipeline: dict, text: str) -> dict:
    scrubbed = pipeline["anonymizer"].scrub_pii(text)
    cleaned = pipeline["cleaner"].clean_text(scrubbed)
    tokens = pipeline["cleaner"].tokenize(cleaned)
    tokens_nostop = pipeline["cleaner"].remove_stopwords(tokens)

    sentiment_data = pipeline["nlp_extractor"].get_sentiment(cleaned)
    emotion_data = pipeline["nlp_extractor"].detect_emotions(tokens_nostop)
    psych_data = pipeline["psych_extractor"].extract_features(scrubbed, tokens)

    engineered_list = (
        [sentiment_data["sentiment_score"]]
        + [emotion_data[f"emo_{e}"] for e in ["sadness", "fear", "anger", "joy", "fatigue", "hopelessness"]]
        + [
            psych_data["word_count"],
            psych_data["avg_word_len"],
            psych_data["exclamation_count"],
            psych_data["first_person_ratio"],
            psych_data["cognitive_distortion_score"],
            psych_data["distress_index"],
            0.0,
            1.0,
            0.0,
        ]
    )
    psych_vec = np.array(engineered_list, dtype=np.float32)
    cls_emb, attn_matrix, raw_tokens = pipeline["transformer_encoder"].encode_text(cleaned)

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
        "raw_tokens": raw_tokens,
    }


def render_single_post_analysis():
    sample_prompts = {
        "High-risk example": "I feel like nothing matters anymore. I cannot stop thinking about disappearing and I don't want to wake up.",
        "Moderate example": "I am exhausted and overwhelmed at work. My chest is tight and I keep replaying mistakes all day.",
        "Supportive example": "I had a small win today and finally went for a walk outside. I am trying to stay consistent.",
    }

    st.subheader("Single Post Clinical Analysis")
    col1, col2 = st.columns([2, 1])
    with col1:
        prompt_choice = st.selectbox("Use example", ["Custom text", *list(sample_prompts.keys())])
        if prompt_choice == "Custom text":
            draft = st.text_area("Post text", height=180, placeholder="Paste a mental health post or comment here...")
        else:
            draft = st.text_area("Post text", value=sample_prompts[prompt_choice], height=180)
    with col2:
        author = st.text_input("Author", value="demo_user")
        topic = st.selectbox("Topic", ["mentalhealth", "depression", "anxiety", "lonely", "ptsd", "SuicideWatch", "CasualConversation"])
        run = st.button("Analyze with MindGraphNet", type="primary")

    if run and draft.strip():
        with st.spinner("Running multimodal analysis..."):
            result = analyze_text(draft, author=author, topic=topic)

        if "error" in result:
            st.error(result["error"])
            return

        st.success(f"Risk rating: {result['risk_level']} | Confidence: {result['confidence']:.3f}")

        metrics = [
            ("Risk Level", result["risk_level"]),
            ("Confidence", f"{result['confidence']:.3f}"),
            ("Continuous Score", f"{result['continuous_risk_score']:.3f}"),
            ("Crisis Trigger", "Yes" if result["is_crisis"] else "No"),
            ("Sentiment", result["sentiment_label"]),
            ("Stress/Anxiety", f"{result['stress_anxiety_index']:.3f}"),
            ("Depressive Signal", f"{result['depressive_signal_index']:.3f}"),
            ("Resilience", f"{result['resilience_index']:.3f}"),
        ]

        cols = st.columns(4)
        for idx, (label, value) in enumerate(metrics[:4]):
            cols[idx].metric(label, value)

        for idx, (label, value) in enumerate(metrics[4:], start=0):
            cols[idx % 4].metric(label, value)

        st.subheader("Class Probabilities")
        prob_df = pd.DataFrame(
            [{"Class": k, "Probability": float(v)} for k, v in result["class_probabilities"].items()]
        )
        st.plotly_chart(px.bar(prob_df, x="Class", y="Probability", color="Class", title="Risk Class Distribution"), use_container_width=True)

        st.subheader("Modality & Clinical Factor Importance")
        factor_df = pd.DataFrame(
            [{"Factor": k, "Importance": float(v)} for k, v in result["factor_importance"].items()]
        )
        st.plotly_chart(px.bar(factor_df, x="Importance", y="Factor", orientation="h", color="Factor", title="Factor Contribution"), use_container_width=True)

        st.subheader("Attention Highlights")
        if result["top_attended_words"]:
            st.write("Top attended words:", result["top_attended_words"])
        st.markdown(result["highlighted_html"], unsafe_allow_html=True)

        st.subheader("Detected Crisis Signals")
        if result["crisis_signals"]:
            st.warning(", ".join(result["crisis_signals"]))
        else:
            st.info("No explicit crisis or self-harm keywords detected in the input.")

        st.caption("Input after privacy scrub: " + result["scrubbed_text"])


def render_bulk_analysis():
    st.subheader("Bulk CSV Analysis")
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"]) 

    if uploaded is None:
        st.info("Upload a CSV with a text-like column (for example: text, post, message, content, title, body).")
        return

    df = pd.read_csv(uploaded)
    text_candidates = [c for c in df.columns if any(k in c.lower() for k in ["text", "post", "message", "content", "body", "title"]) ]
    if not text_candidates:
        st.error("No text-like column found in the CSV. Please include one of: text, post, content, message, title, or body.")
        return

    text_col = st.selectbox("Choose the text column", text_candidates)
    author_col = st.selectbox("Choose author column (optional)", [None] + list(df.columns), index=0)
    topic_col = st.selectbox("Choose topic column (optional)", [None] + list(df.columns), index=0)

    if st.button("Run bulk analysis"):
        rows = []
        for idx, row in df.iterrows():
            text_value = str(row[text_col])
            if not text_value or not text_value.strip():
                continue
            author = row[author_col] if author_col is not None and author_col in df.columns else "anonymous_poster"
            topic = row[topic_col] if topic_col is not None and topic_col in df.columns else "mentalhealth"
            result = analyze_text(text_value, author=str(author), topic=str(topic))
            if "error" in result:
                continue
            rows.append({
                "index": idx,
                "author": str(author),
                "topic": str(topic),
                "text": text_value,
                "risk_level": result["risk_level"],
                "confidence": float(result["confidence"]),
                "continuous_risk_score": float(result["continuous_risk_score"]),
                "is_crisis": bool(result["is_crisis"]),
                "sentiment_score": float(result["sentiment_score"]),
                "sentiment_label": result["sentiment_label"],
                "stress_anxiety_index": float(result["stress_anxiety_index"]),
                "depressive_signal_index": float(result["depressive_signal_index"]),
                "resilience_index": float(result["resilience_index"]),
            })

        if not rows:
            st.warning("No valid rows were analyzed from the uploaded CSV.")
            return

        result_df = pd.DataFrame(rows)
        count_by_risk = result_df["risk_level"].value_counts().reindex(["Low", "Moderate", "High"], fill_value=0)
        st.subheader("Bulk Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows analyzed", len(result_df))
        c2.metric("High-risk posts", int((result_df["risk_level"] == "High").sum()))
        c3.metric("High-risk ratio", f"{(result_df['risk_level'].eq('High').mean() * 100):.1f}%")

        st.bar_chart(count_by_risk)

        risk_table = result_df[["index", "author", "topic", "risk_level", "confidence", "is_crisis"]].sort_values("confidence", ascending=False)
        st.dataframe(risk_table, use_container_width=True)

        csv_buffer = BytesIO()
        result_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download analysis results as CSV",
            data=csv_buffer.getvalue(),
            file_name="mindgraphnet_bulk_results.csv",
            mime="text/csv",
        )


def render_dataset_overview():
    st.subheader("Benchmark Dataset Overview")
    dataset = load_dataset(use_kaggle=False, fallback_size=500)
    topic_counts = dataset["topic"].value_counts().head(10)
    st.bar_chart(topic_counts)

    st.dataframe(dataset.head(10), use_container_width=True)


def main():
    st.title("🧠 MindGraphNet Multi-Hybrid Clinical Analytics")
    st.caption("Explainable mental health risk analysis across language, graph, and psycholinguistic signals.")

    with st.sidebar:
        st.markdown("### Navigation")
        st.caption("Project root: " + str(PROJECT_ROOT))
        st.code("API: http://127.0.0.1:8000/api/v1/health\nDashboard: http://localhost:8501")
        nav = st.radio("Select workspace", ["Single Post", "Bulk CSV", "Dataset Overview"], index=0)

    if nav == "Single Post":
        render_single_post_analysis()
    elif nav == "Bulk CSV":
        render_bulk_analysis()
    else:
        render_dataset_overview()


if __name__ == "__main__":
    main()
