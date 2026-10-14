"""
MindGraphNet - NLP Feature Extraction
Extracts TF-IDF representations, VADER sentiment vectors, and fine-grained clinical emotion signals.
"""
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config.settings import TFIDF_MAX_FEATURES, EMOTIONS

CLINICAL_EMOTION_LEXICON = {
    "sadness": [
        "sad", "depressed", "unhappy", "sorrow", "grief", "heavy", "miserable",
        "down", "crying", "tears", "empty", "hollow", "numb", "despair", "gloomy"
    ],
    "fear": [
        "anxious", "anxiety", "panic", "terrified", "scared", "afraid", "nervous",
        "dread", "worry", "worried", "heart racing", "shaking", "trembling", "phobia"
    ],
    "anger": [
        "angry", "mad", "furious", "rage", "irritated", "annoyed", "frustrated",
        "hate", "resentful", "bitter", "screaming", "disgusted"
    ],
    "joy": [
        "happy", "joy", "glad", "delighted", "grateful", "thankful", "peaceful",
        "calm", "relieved", "smiling", "content", "blessed", "hopeful", "optimistic"
    ],
    "fatigue": [
        "tired", "exhausted", "drained", "weary", "burnout", "burned out", "lethargic",
        "sleepy", "insomnia", "sleepless", "no energy", "fatigued", "sluggish"
    ],
    "hopelessness": [
        "hopeless", "pointless", "useless", "worthless", "trapped", "no way out",
        "give up", "giving up", "no point", "burden", "unwanted", "alone forever"
    ]
}

class NLPFeatureExtractor:
    def __init__(self, max_features: int = TFIDF_MAX_FEATURES):
        self.max_features = max_features
        self.vader = SentimentIntensityAnalyzer()
        self.tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
        self.is_fitted = False

    def fit_tfidf(self, corpus: List[str]):
        self.tfidf.fit(corpus)
        self.is_fitted = True

    def transform_tfidf(self, corpus: List[str]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("TF-IDF vectorizer must be fitted before transformation.")
        return self.tfidf.transform(corpus).toarray()

    def get_sentiment(self, text: str) -> Dict[str, float]:
        scores = self.vader.polarity_scores(text)
        compound = scores["compound"]
        label = "positive" if compound > 0.15 else ("negative" if compound < -0.15 else "neutral")
        return {
            "sentiment_score": float(compound),
            "sentiment_pos": float(scores["pos"]),
            "sentiment_neg": float(scores["neg"]),
            "sentiment_neu": float(scores["neu"]),
            "sentiment_label": label
        }

    def detect_emotions(self, tokens: List[str]) -> Dict[str, float]:
        text_joined = " ".join(tokens)
        counts = {emo: 0 for emo in EMOTIONS}
        for emo, keywords in CLINICAL_EMOTION_LEXICON.items():
            if emo not in counts:
                continue
            for kw in keywords:
                if " " in kw:
                    if kw in text_joined:
                        counts[emo] += 2
                else:
                    counts[emo] += tokens.count(kw)
        
        total = sum(counts.values()) or 1.0
        return {f"emo_{emo}": float(counts[emo] / total) for emo in EMOTIONS}
