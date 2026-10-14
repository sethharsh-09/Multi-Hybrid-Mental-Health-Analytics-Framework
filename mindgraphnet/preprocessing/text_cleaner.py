"""
MindGraphNet - Clinical Text Normalization & Preprocessing
Cleans text while strategically preserving negations and affective valence tokens.
"""
import re
from typing import List

# Preserving negations ('not', 'no', 'never', 'neither', 'hardly', 'barely')
SAFE_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "to", "of", "in", "on", "at",
    "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such",
    "than", "too", "very", "s", "t", "can", "will", "just", "don", "should"
}

CONTRACTION_MAP = {
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'t": " not",
    "'ve": " have",
    "'m": " am"
}

class ClinicalTextCleaner:
    def __init__(self, stopwords=SAFE_STOPWORDS):
        self.stopwords = stopwords

    def expand_contractions(self, text: str) -> str:
        for contraction, expansion in CONTRACTION_MAP.items():
            text = text.replace(contraction, expansion)
        return text

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = self.expand_contractions(text)
        # Remove markdown symbols and punctuation except apostrophes
        text = re.sub(r"[^a-z0-9'\s]", " ", text)
        # Collapse multi-space
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        return text.split()

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [t for t in tokens if t not in self.stopwords]
