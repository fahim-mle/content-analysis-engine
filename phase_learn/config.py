# phase_learn/config.py
"""Model names, number of topics/clusters, evaluation settings."""

from pathlib import Path

# Directory Paths
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
MODELS_DIR = DATA_DIR / "models"
TOPIC_MODEL_DIR = MODELS_DIR / "topic_model"
CLASSIFIER_DIR = MODELS_DIR / "classifier"
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR = DATA_DIR / "logs"
LOG_FILE = LOGS_DIR / "learn.log"
ML_DATASET_FILE = DATA_DIR / "metadata" / "ml_dataset.json.gz"
EMBEDDINGS_FILE = EMBEDDINGS_DIR / "embeddings.npy"
TOPIC_MODEL_FILE = TOPIC_MODEL_DIR / "bertopic_model.pkl"

# Model Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOPIC_MODEL_TYPE = "bertopic"  # or "lda"
NUM_TOPICS = 10
NUM_CLUSTERS = 5

# Training Parameters
BATCH_SIZE = 32
MAX_EPOCHS = 100
LEARNING_RATE = 0.001

# Evaluation Settings
TEST_SIZE = 0.2
RANDOM_STATE = 42