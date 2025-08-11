# src/utils.py
"""Shared utility functions for the ArXiv scraping system."""

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict

from .config import LOG_FILE, LOGS_DIR


def get_logger(name: str) -> logging.Logger:
    """
    Configure and return a logger instance with file and console handlers.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    # Ensure logs directory exists
    LOGS_DIR.mkdir(exist_ok=True)

    logger = logging.getLogger(name)

    # Avoid duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def hash_paper(paper: Dict[str, Any]) -> str:
    """
    Generate MD5 hash of paper based on title, first author, and year.

    Args:
        paper: Paper metadata dictionary

    Returns:
        MD5 hash string
    """
    title = paper.get("title", "").lower().strip()
    authors = paper.get("authors", [])
    first_author = authors[0] if authors else ""
    published = paper.get("published", "")
    year = published.year if hasattr(published, "year") else str(published)[:4]

    hash_input = f"{title}_{first_author}_{year}"
    return hashlib.md5(hash_input.encode("utf-8")).hexdigest()


def ensure_directory(path: Path) -> None:
    """
    Ensure directory exists, create if necessary.

    Args:
        path: Directory path
    """
    path.mkdir(parents=True, exist_ok=True)


def sanitize_filename(filename: str) -> str:
    """
    Remove or replace invalid characters for filenames.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "_")
    return filename.strip()
