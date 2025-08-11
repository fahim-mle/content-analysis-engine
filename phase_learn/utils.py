# phase_learn/utils.py
"""Model loading, metric calculation, result serialization."""

import gzip
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import LOG_FILE, LOGS_DIR, MODELS_DIR, REPORTS_DIR


def get_logger(name: str) -> logging.Logger:
    """
    Configure and return logger for learning phase.

    Args:
        name: Logger name

    Returns:
        Configured logger instance
    """
    LOGS_DIR.mkdir(exist_ok=True, parents=True)

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler for learn.log
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


logger = get_logger(__name__)


def load_ml_dataset(file_path: Path) -> Optional[Dict[str, List[Dict]]]:
    """
    Load the machine learning dataset from a JSON file (compressed or uncompressed).

    Args:
        file_path: Path to the ml_dataset.json or ml_dataset.json.gz file

    Returns:
        A dictionary containing the dataset or None if loading fails.
    """
    if not file_path.exists():
        logger.error(f"ML dataset file not found at: {file_path}")
        return None
    try:
        logger.info(f"Loading ML dataset from {file_path}...")
        
        # Try to load as gzipped file first
        if str(file_path).endswith('.gz'):
            with gzip.open(file_path, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            # Try regular JSON file
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
        logger.info(
            f"Dataset loaded successfully with {len(data.get('papers', []))} papers "
            f"and {len(data.get('chunks', []))} chunks."
        )
        return data
    except (json.JSONDecodeError, gzip.BadGzipFile) as e:
        logger.error(f"Failed to load or decode ML dataset file: {e}")
        return None


def save_model(model: Any, file_path: Path):
    """
    Save a model to a file using pickle.

    Args:
        model: The model object to save.
        file_path: The path to save the model file to.
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Model saved successfully to {file_path}")
    except pickle.PicklingError as e:
        logger.error(f"Failed to pickle model to {file_path}: {e}")
    except IOError as e:
        logger.error(f"Failed to write model file to {file_path}: {e}")


def load_model(file_path: Path) -> Optional[Any]:
    """
    Load a model from a pickle file.

    Args:
        file_path: The path to the model file.

    Returns:
        The loaded model object or None if loading fails.
    """
    if not file_path.exists():
        logger.error(f"Model file not found at: {file_path}")
        return None
    try:
        with open(file_path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Model loaded successfully from {file_path}")
        return model
    except pickle.UnpicklingError as e:
        logger.error(f"Failed to unpickle model from {file_path}: {e}")
        return None
    except IOError as e:
        logger.error(f"Failed to read model file from {file_path}: {e}")
        return None


def save_json_results(data: Dict, file_name: str):
    """
    Save dictionary of results to a JSON file in the reports directory.

    Args:
        data: The dictionary to save.
        file_name: The name of the file (e.g., 'topic_model_results.json').
    """
    try:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        file_path = REPORTS_DIR / file_name
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved successfully to {file_path}")
    except TypeError as e:
        logger.error(f"Data for {file_path} is not JSON serializable: {e}")
    except IOError as e:
        logger.error(f"Failed to write results to {file_path}: {e}")
