"""
MindGraphNet - Psycholinguistic & Cognitive Distortion Analytics
Computes self-referential inward focus, cognitive distortion markers, and affective distress indices.
"""
from typing import List, Dict, Any
import numpy as np

FIRST_PERSON_PRONOUNS = {"i", "me", "my", "myself", "mine"}

# Cognitive Distortion markers: All-or-nothing thinking, catastrophizing, emotional reasoning
COGNITIVE_DISTORTION_TOKENS = {
    "always", "never", "everyone", "nobody", "everything", "nothing",
    "completely", "totally", "impossible", "ruined", "disaster", "worthless",
    "forever", "hate", "unbearable", "hopeless"
}

class PsycholinguisticExtractor:
    @staticmethod
    def extract_features(text_raw: str, tokens: List[str]) -> Dict[str, float]:
        n_tokens = max(len(tokens), 1)
        
        # Word length and density
        avg_word_len = float(np.mean([len(w) for w in tokens])) if tokens else 0.0
        exclamation_count = float(text_raw.count("!"))
        
        # First-person singular pronoun focus (Pennebaker metric for depression rumination)
        first_person_count = sum(1 for w in tokens if w in FIRST_PERSON_PRONOUNS)
        first_person_ratio = float(first_person_count / n_tokens)
        
        # Cognitive distortion density
        cd_count = sum(1 for w in tokens if w in COGNITIVE_DISTORTION_TOKENS)
        cd_score = float(cd_count / n_tokens)
        
        # Aggregate distress index based on cognitive & linguistic markers
        distress_index = float(np.clip(
            (first_person_ratio * 1.5) + (cd_score * 2.5) + (min(exclamation_count, 5) * 0.1),
            0.0, 1.0
        ))
        
        return {
            "word_count": float(len(tokens)),
            "avg_word_len": avg_word_len,
            "exclamation_count": exclamation_count,
            "first_person_ratio": first_person_ratio,
            "cognitive_distortion_score": cd_score,
            "distress_index": distress_index
        }
