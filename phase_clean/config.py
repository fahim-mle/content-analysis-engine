# phase_clean/config.py
"""Configuration for text cleaning phase."""

from pathlib import Path

# Directory Paths
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
RAW_TEXT_DIR = DATA_DIR / "extracted_text"
CLEANED_TEXT_DIR = DATA_DIR / "cleaned_text"
CHUNKS_DIR = DATA_DIR / "chunks"
METADATA_DIR = DATA_DIR / "metadata"
LOGS_DIR = DATA_DIR / "logs"
LOG_FILE = LOGS_DIR / "clean.log"

# Processing Configuration
CHUNK_SIZE = 1000  # characters per chunk
MIN_TEXT_LENGTH = 500  # minimum text length to process
MAX_TEXT_LENGTH = 100000  # maximum text length to process
SUPPORTED_LANGUAGES = ["en"]  # English only

# Quality Filter Configuration
MIN_QUALITY_SCORE = 0.1  # minimum quality score to accept papers
TARGET_CATEGORIES = {
    "cs.AI",
    "cs.CL",
    "cs.LG",
    "cs.CV",
    "cs.IR",
    "cs.NE",
    "cs.ML",
    "cs.IT",
    "stat.ML",
}

# Batch Processing Configuration
BATCH_SIZE = 25  # papers to process in each batch
CHUNK_BATCH_SIZE = 1000  # chunks per batch file
STORAGE_COMPRESSION = True  # use gzip compression for storage

# Memory Management
MAX_PAPERS_IN_MEMORY = 100  # maximum papers to keep in memory
MAX_CHUNKS_IN_MEMORY = 5000  # maximum chunks to keep in memory

# Cleaning Parameters
REMOVE_PATTERNS = [
    r"\n\s*\n\s*\n+",  # multiple consecutive newlines
    r"^\s*\d+\s*$",  # standalone page numbers
    r"arXiv:\d+\.\d+v\d+(?:\s*\[[\w\.-]+\])?(?:\s*\d+\s+\w+\s+\d{4})?",  # arXiv identifiers
    r"https?://[^\s]+",  # URLs
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # email addresses
]

# Unicode normalization replacements
UNICODE_REPLACEMENTS = {
    "´": "'",
    "`": "'",
    """: "'", """: "'",
    '"': '"',
    '"': '"',
    "–": "-",
    "—": "-",
    "…": "...",
    "¨": "",
    "¯": "-",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬀ": "ff",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "θ": "theta",
    "λ": "lambda",
    "μ": "mu",
    "π": "pi",
    "σ": "sigma",
    "τ": "tau",
    "φ": "phi",
    "≤": "<=",
    "≥": ">=",
    "≠": "!=",
    "≈": "~=",
    "∞": "infinity",
}
