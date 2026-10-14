"""
MindGraphNet - Clinical Risk Scorer & Multi-Signal Triaging
Computes calibrated risk tiers, clinical sub-indices, and urgent crisis flags.
"""
from typing import Dict, Any, List
import numpy as np

CRISIS_KEYWORDS = {
    "kill myself", "end it all", "suicide", "suicidal", "want to die",
    "say goodbye", "no reason to live", "better off dead", "cutting myself",
    "hang myself", "overdose"
}

class ClinicalRiskScorer:
    @staticmethod
    def evaluate_post(text_raw: str, probs: np.ndarray, continuous_risk: float, 
                      emo_dict: Dict[str, float], sentiment_score: float) -> Dict[str, Any]:
        """
        Synthesizes multi-hybrid model outputs into structured clinical analytics.
        """
        # Determine risk tier
        class_idx = int(np.argmax(probs))
        class_names = ["Low", "Moderate", "High"]
        risk_level = class_names[class_idx]
        confidence = float(probs[class_idx])

        # Clinical sub-indices
        stress_anxiety_idx = float(np.clip(emo_dict.get("emo_fear", 0.0) * 1.5 + emo_dict.get("emo_fatigue", 0.0), 0.0, 1.0))
        depressive_idx = float(np.clip(emo_dict.get("emo_sadness", 0.0) * 1.3 + emo_dict.get("emo_hopelessness", 0.0) * 1.8, 0.0, 1.0))
        resilience_idx = float(np.clip(emo_dict.get("emo_joy", 0.0) * 2.0 + max(sentiment_score, 0.0), 0.0, 1.0))

        # Check for immediate crisis trigger
        text_lower = text_raw.lower()
        crisis_match = [kw for kw in CRISIS_KEYWORDS if kw in text_lower]
        is_crisis = (len(crisis_match) > 0) or (continuous_risk >= 0.78 and risk_level == "High")

        if is_crisis and risk_level != "High":
            risk_level = "High"
            confidence = max(confidence, 0.92)

        return {
            "risk_level": risk_level,
            "confidence": round(confidence, 3),
            "continuous_risk_score": round(float(continuous_risk), 3),
            "is_crisis": is_crisis,
            "crisis_signals_detected": crisis_match,
            "stress_anxiety_index": round(stress_anxiety_idx, 3),
            "depressive_signal_index": round(depressive_idx, 3),
            "resilience_index": round(resilience_idx, 3),
            "class_probabilities": {
                "Low": round(float(probs[0]), 3),
                "Moderate": round(float(probs[1]), 3),
                "High": round(float(probs[2]), 3)
            }
        }
