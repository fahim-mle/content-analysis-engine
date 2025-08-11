# phase_learn/__init__.py
"""Machine learning and analysis phase - Phase 3 of the NLP pipeline."""

from .main_learn import run_learn_phase, MLPipelineOrchestrator
from .utils import get_logger, load_ml_dataset, save_model, load_model, save_json_results
from .config import (
    EMBEDDING_MODEL, 
    TOPIC_MODEL_TYPE, 
    NUM_TOPICS, 
    NUM_CLUSTERS,
    ML_DATASET_FILE,
    MODELS_DIR,
    REPORTS_DIR
)

__all__ = [
    'run_learn_phase',
    'MLPipelineOrchestrator', 
    'get_logger',
    'load_ml_dataset',
    'save_model',
    'load_model',
    'save_json_results',
    'EMBEDDING_MODEL',
    'TOPIC_MODEL_TYPE',
    'NUM_TOPICS', 
    'NUM_CLUSTERS',
    'ML_DATASET_FILE',
    'MODELS_DIR',
    'REPORTS_DIR'
]