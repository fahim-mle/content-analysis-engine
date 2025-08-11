# System Design Blueprint

This file defines the high-level structure, data flow, and module responsibilities for the NLP prototype. All code must follow this design.

## 🧱 Architecture Overview

```txt
main.py
├── fetch_papers.py     → Query ArXiv API
├── download_pdf.py     → Save PDF to disk
├── extract_text.py     → Extract text from PDF
├── store_mongodb.py    → Save metadata to MongoDB
├── store_chromadb.py   → Generate & store embeddings
├── utils.py            → Shared helpers (logging, hashing, etc.)
└── config.py           → Paths, DB URIs, model names
```

Each module returns `dict` or `bool`, never raises unhandled exceptions.

---

## 📥 Task 1: Data Collection Pipeline

### Step 1: Fetch Papers from ArXiv

```python
# fetch_papers.py
def build_query(categories: list, max_results: int, days_back: int) -> str:
    """Build ArXiv API URL with filters"""
    # Returns: http://export.arxiv.org/api/query?search_query=...

def fetch_papers(query_url: str) -> list[dict]:
    """Fetch and parse feedparser entries"""
    # Returns list of paper dicts with metadata
```

### Step 2: Download PDF

```python
# download_pdf.py
def download_pdf(paper: dict, pdf_dir: Path) -> str | None:
    """Download PDF from arXiv URL"""
    # Saves to: pdf_dir/{arxiv_id}.pdf
    # Returns file path or None on failure
```

### Step 3: Extract Text

```python
# extract_text.py
def extract_text(pdf_path: str) -> str | None:
    """Extract raw text from PDF"""
    # Returns full text or None
```

### Step 4: Store Metadata in MongoDB

```python
# store_mongodb.py
def connect_mongo(uri: str, db_name: str):
    # Returns db.collection

def upsert_paper_metadata(collection, paper_data: dict) -> bool:
    """Insert or update paper in MongoDB"""
```

### Step 5: Store Embedding in ChromaDB

```python
# store_chromadb.py
def get_chroma_collection(path: str, name: str):
    # Returns ChromaDB collection

def store_embedding(collection, arxiv_id: str, text: str, metadata: dict) -> bool:
    """Generate embedding and store in ChromaDB"""
```

### Step 6: Main Orchestration

```python
# main.py
from config import MONGO_URI, CHROMA_PATH, CATEGORIES, MAX_RESULTS, DAYS_BACK
from fetch_papers import fetch_papers
from download_pdf import download_pdf
from extract_text import extract_text
from store_mongodb import upsert_paper_metadata
from store_chromadb import store_embedding
import logging

def main():
    # Build query
    url = build_query(CATEGORIES, MAX_RESULTS, DAYS_BACK)
    papers = fetch_papers(url)

    for paper in papers:
        # Download PDF
        pdf_path = download_pdf(paper, Path("data/raw_pdfs"))
        if not pdf_path:
            continue

        # Extract text
        full_text = extract_text(pdf_path)
        if not full_text:
            continue

        # Enrich paper data
        paper["pdf_path"] = pdf_path
        paper["full_text"] = full_text[:5000]  # Truncate for embedding
        paper["pdf_downloaded"] = True
        paper["pdf_processed"] = True

        # Store in MongoDB
        upsert_paper_metadata(mongo_collection, paper)

        # Prepare embedding input
        embedding_text = f"{paper['title']} {paper['summary']} {full_text[:3000]}"
        metadata = {
            "arxiv_id": paper["arxiv_id"],
            "title": paper["title"],
            "categories": ",".join(paper["categories"]),
            "published": paper["published"].isoformat()
        }

        # Store in ChromaDB
        store_embedding(chroma_collection, paper["arxiv_id"], embedding_text, metadata)

        # Rate limit
        time.sleep(2)

if __name__ == "__main__":
    main()
```

---

## 🧠 Additional Functions

```python
# utils.py
def get_logger(name: str) -> logging.Logger:
    # Configure and return logger

def hash_paper(paper: dict) -> str:
    # MD5 hash of title + first author + year
```

```python
# config.py
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "arxiv_papers"
CHROMA_PATH = "./chroma_db"
CHROMA_COLLECTION = "arxiv_embeddings"

CATEGORIES = ["cs.AI", "cs.LG", "cs.CL"]
MAX_RESULTS = 100
DAYS_BACK = 90

PDF_DIR = "data/raw_pdfs"
TEXT_DIR = "data/extracted_text"
METADATA_FILE = "data/metadata/papers.jsonl"
```

> ✅ All future code must follow this structure exactly.
