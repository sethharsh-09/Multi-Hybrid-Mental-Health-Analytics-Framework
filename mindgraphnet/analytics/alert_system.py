"""
MindGraphNet - Community Risk Alert & Human-in-the-Loop Triage System
"""
from typing import List, Dict, Any
import pandas as pd
from config.settings import HIGH_RISK_ALERT_THRESHOLD

class ClinicalAlertSystem:
    def __init__(self, high_risk_threshold: float = HIGH_RISK_ALERT_THRESHOLD):
        self.high_risk_threshold = high_risk_threshold

    def check_community_alerts(self, df_scored: pd.DataFrame) -> List[Dict[str, Any]]:
        alerts = []
        for topic, grp in df_scored.groupby("topic"):
            total_posts = len(grp)
            high_count = int((grp["risk_level"] == "High").sum())
            high_ratio = high_count / max(total_posts, 1)
            
            if high_ratio >= self.high_risk_threshold:
                alerts.append({
                    "severity": "CRITICAL" if high_ratio >= 0.50 else "WARNING",
                    "community": topic,
                    "total_posts": total_posts,
                    "high_risk_posts": high_count,
                    "high_risk_percentage": round(high_ratio * 100.0, 1),
                    "recommended_action": (
                        "Immediate human clinician triage & trigger automated mental health support resources."
                        if high_ratio >= 0.50 else
                        "Elevate monitoring cadence for this discussion cluster."
                    )
                })
        return sorted(alerts, key=lambda x: x["high_risk_percentage"], reverse=True)
