"""
MindGraphNet - Central Configuration and Hyperparameters
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models_store"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Random Seed
RANDOM_SEED = 42

# Preprocessing & Privacy
SALT = "mindgraphnet_research_salt_2026"
MIN_TEXT_LENGTH = 5
MAX_SEQ_LENGTH = 64

# Feature Engineering
TFIDF_MAX_FEATURES = 300
EMOTIONS = ["sadness", "fear", "anger", "joy", "fatigue", "hopelessness"]
LINGUISTIC_COLS = [
    "word_count", "avg_word_len", "exclamation_count",
    "first_person_ratio", "cognitive_distortion_score", "distress_index"
]
INTERACTION_COLS = ["replies_received", "author_activity", "thread_depth"]

ENGINEERED_COLS = ["sentiment_score"] + [f"emo_{e}" for e in EMOTIONS] + LINGUISTIC_COLS + INTERACTION_COLS
N_ENGINEERED_FEATURES = len(ENGINEERED_COLS)

# Transformer & Graph Dimensions
TRANSFORMER_NAME = "distilbert-base-uncased"
TRANSFORMER_EMB_DIM = 768
REDUCED_EMB_DIM = 64
GRAPH_EMB_DIM = 32

# Multi-Hybrid Fusion Network Dimensions
FUSED_INPUT_DIM = TRANSFORMER_EMB_DIM + GRAPH_EMB_DIM + N_ENGINEERED_FEATURES
NUM_CLASSES = 3
CLASS_NAMES = ["Low", "Moderate", "High"]

# Training Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 60

# Clinical & Monitoring Alert Thresholds
HIGH_RISK_ALERT_THRESHOLD = 0.30
CONFIDENCE_DRIFT_THRESHOLD = 0.55
CRISIS_SCORE_THRESHOLD = 0.70
