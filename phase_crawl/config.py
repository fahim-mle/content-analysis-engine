# src/config.py
"""Configuration constants for the ArXiv scraping system."""

from pathlib import Path

# Database Configuration
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "arxiv_papers"
COLLECTION_NAME = "papers"

# ChromaDB Configuration
CHROMA_PATH = "./chroma_db"
CHROMA_COLLECTION = "arxiv_embeddings"

# ArXiv Search Configuration
CATEGORIES = ["cs.GR", "cs.HA", "cs.DB"]
MAX_RESULTS_PER_CATEGORY = 50  # Balanced download per category
DAYS_BACK = 1400  # Increased to get more papers

# PDF Processing Limits
MAX_PDF_COUNT = 200  # Total across all categories (100 per category)

# Directory Paths
BASE_DIR = Path(".")
PDF_DIR = BASE_DIR / "data" / "raw_pdfs"
TEXT_DIR = BASE_DIR / "data" / "extracted_text"
METADATA_DIR = BASE_DIR / "data" / "metadata"
METADATA_FILE = METADATA_DIR / "papers.jsonl"
LOGS_DIR = BASE_DIR / "data" / "logs"
LOG_FILE = LOGS_DIR / "crawl.log"

# Request Configuration
REQUEST_TIMEOUT = 30
RATE_LIMIT_DELAY = 2  # seconds between requests
USER_AGENT = "Mozilla/5.0 (compatible; research-scraper/1.0)"

# Text Processing
MAX_TEXT_LENGTH = 5000  # for embeddings
EMBEDDING_TEXT_LENGTH = 3000  # for ChromaDB
