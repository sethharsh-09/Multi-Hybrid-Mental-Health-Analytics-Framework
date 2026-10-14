"""
MindGraphNet - Longitudinal User Trajectory & Behavioral Drift Analysis
Tracks temporal mental health trajectories for specific users across historical posts.
"""
from typing import Dict, List, Any
import pandas as pd
import numpy as np

class LongitudinalTracker:
    @staticmethod
    def track_user_trajectory(df_scored: pd.DataFrame, author_hash: str) -> Dict[str, Any]:
        user_posts = df_scored[df_scored["author_hash"] == author_hash].sort_values("created_at").reset_index(drop=True)
        if len(user_posts) == 0:
            return {"status": "User not found"}

        risk_history = user_posts["risk_level"].tolist()
        sentiment_history = user_posts["sentiment_score"].tolist()
        dates = user_posts["created_at"].dt.strftime("%Y-%m-%d").tolist()

        # Risk drift calculation: difference between first post and latest post
        sentiment_drift = round(float(sentiment_history[-1] - sentiment_history[0]), 3) if len(sentiment_history) > 1 else 0.0
        
        # High-risk frequency ratio
        high_risk_ratio = float((user_posts["risk_level"] == "High").mean())

        # Escalation alert: user went from Low/Moderate to consecutive High
        is_escalating = False
        if len(risk_history) >= 2:
            if risk_history[-1] == "High" and sentiment_drift < -0.25:
                is_escalating = True

        return {
            "author_hash": author_hash,
            "total_posts": len(user_posts),
            "dates": dates,
            "risk_history": risk_history,
            "sentiment_history": [round(s, 3) for s in sentiment_history],
            "sentiment_drift": sentiment_drift,
            "high_risk_ratio": round(high_risk_ratio, 3),
            "is_escalating": is_escalating
        }
