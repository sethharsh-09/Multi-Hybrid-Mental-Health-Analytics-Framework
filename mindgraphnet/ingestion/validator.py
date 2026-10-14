"""
MindGraphNet - Data Ingestion Validation Gate
Enforces schema completeness, length thresholds, language heuristics, and duplicate suppression.
"""
from typing import Tuple, List, Dict, Any
import pandas as pd
from config.settings import MIN_TEXT_LENGTH

REQUIRED_COLUMNS = {"post_id", "author", "text", "topic", "created_at"}

class IngestionValidator:
    """
    Validates streaming and batch data before committing to the raw storage lake.
    """
    def __init__(self, min_length: int = MIN_TEXT_LENGTH, min_ascii_ratio: float = 0.85):
        self.min_length = min_length
        self.min_ascii_ratio = min_ascii_ratio

    def is_valid_text(self, text: str) -> bool:
        if not isinstance(text, str):
            return False
        clean = text.strip()
        if len(clean) < self.min_length:
            return False
        # Heuristic ASCII ratio for language sanity check
        ascii_count = sum(1 for c in clean if ord(c) < 128)
        ratio = ascii_count / max(len(clean), 1)
        return ratio >= self.min_ascii_ratio

    def validate_record(self, record: Dict[str, Any]) -> Tuple[bool, str]:
        for col in REQUIRED_COLUMNS:
            if col not in record or record[col] is None or record[col] == "":
                return False, f"Missing required field: '{col}'"
        if not self.is_valid_text(str(record["text"])):
            return False, "Text does not meet quality/length criteria"
        return True, "Valid"

    def validate_batch(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        missing_cols = REQUIRED_COLUMNS - set(df.columns)
        if missing_cols:
            raise ValueError(f"Schema validation failed. Missing required columns: {missing_cols}")

        initial_count = len(df)
        # 1. Non-empty & length validation
        length_mask = df["text"].astype(str).str.strip().str.len() >= self.min_length
        
        # 2. Language/ASCII sanity mask
        def check_ascii(t):
            t = str(t)
            return (sum(1 for c in t if ord(c) < 128) / max(len(t), 1)) >= self.min_ascii_ratio
            
        lang_mask = df["text"].apply(check_ascii)
        
        # 3. Duplicate text suppression
        dedup_mask = ~df["text"].duplicated(keep="first")

        valid_mask = length_mask & lang_mask & dedup_mask
        accepted_df = df[valid_mask].copy().reset_index(drop=True)
        
        stats = {
            "initial": initial_count,
            "accepted": len(accepted_df),
            "rejected_length": int((~length_mask).sum()),
            "rejected_lang": int((length_mask & ~lang_mask).sum()),
            "rejected_duplicates": int((length_mask & lang_mask & ~dedup_mask).sum())
        }
        return accepted_df, stats
