# phase_clean/utils.py
"""Shared cleaning utilities: text sanitization, language detection, etc."""

import logging
from pathlib import Path

from .config import LOG_FILE, LOGS_DIR


def get_logger(name: str) -> logging.Logger:
    """
    Configure and return logger for cleaning phase.

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

    # File handler for clean.log
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
