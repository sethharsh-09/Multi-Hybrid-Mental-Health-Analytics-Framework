"""
MindGraphNet - Social Media Mining Module
Implements Unsupervised Topic Modeling (LDA), Temporal Trajectory Tracking, and Community Behavioral Profiling.
"""
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation

class SocialMediaMiningEngine:
    def __init__(self, n_topics: int = 5, random_seed: int = 42):
        self.n_topics = n_topics
        self.random_seed = random_seed
        self.lda = LatentDirichletAllocation(n_components=n_topics, random_state=random_seed)
        self.is_fitted = False

    def fit_topics(self, tfidf_matrix: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, Dict[int, List[str]]]:
        topic_distributions = self.lda.fit_transform(tfidf_matrix)
        self.is_fitted = True
        
        top_words_per_topic = {}
        for topic_idx, comp in enumerate(self.lda.components_):
            top_word_indices = comp.argsort()[-8:][::-1]
            top_words = [feature_names[i] for i in top_word_indices]
            top_words_per_topic[topic_idx] = top_words
            
        return topic_distributions, top_words_per_topic

    @staticmethod
    def extract_temporal_trends(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["date"] = pd.to_datetime(df["created_at"]).dt.date
        daily_trends = df.groupby("date").agg({
            "sentiment_score": "mean",
            "distress_index": "mean",
            "post_id": "count"
        }).rename(columns={"post_id": "post_volume"}).reset_index()
        return daily_trends

    @staticmethod
    def extract_community_profiles(df: pd.DataFrame) -> pd.DataFrame:
        profiles = df.groupby("topic").agg({
            "sentiment_score": ["mean", "std"],
            "distress_index": "mean",
            "first_person_ratio": "mean",
            "cognitive_distortion_score": "mean",
            "post_id": "count"
        })
        profiles.columns = ["sentiment_mean", "sentiment_std", "distress_mean", 
                            "first_person_mean", "cognitive_distortion_mean", "post_count"]
        return profiles.reset_index().sort_values("distress_mean", ascending=False)
