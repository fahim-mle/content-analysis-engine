# FOLDER_STRUCTURE.md

## This file defines the complete folder structure with purpose and file responsibilities

```json
{
  "project_root": {
    "phase_crawl": {
      "fetch_papers.py": "Queries ArXiv API and parses metadata entries.",
      "download_pdf.py": "Downloads PDF from arXiv URL and saves to disk.",
      "extract_text.py": "Extracts raw text from downloaded PDF using PyPDF2.",
      "store_mongodb.py": "Stores paper metadata and processing status in MongoDB.",
      "store_chromadb.py": "Generates sentence embeddings and stores in ChromaDB.",
      "utils.py": "Shared utilities: logging setup, content hashing, error handling.",
      "config.py": "Centralized configuration: DB URIs, paths, categories, limits."
    },
    "phase_clean": {
      "clean_text.py": "Removes noise, fixes encoding, normalizes whitespace and line breaks.",
      "segment_sections.py": "Splits full text into logical sections (abstract, intro, etc.).",
      "filter_papers.py": "Filters papers by domain, language, length, or metadata.",
      "normalize_tokens.py": "Tokenizes, lemmatizes, and applies basic NLP normalization.",
      "generate_chunks.py": "Splits long documents into fixed-size or semantically meaningful chunks.",
      "utils.py": "Shared cleaning utilities: text sanitization, language detection, etc.",
      "config.py": "Paths to raw text, cleaned output, chunk size, filters."
    },
    "phase_learn": {
      "encode_embeddings.py": "Generates sentence/document embeddings using Sentence Transformers.",
      "train_topic_model.py": "Applies BERTopic or LDA to discover latent themes in corpus.",
      "classify_papers.py": "Trains classifier to categorize papers (e.g., NLP vs. CV).",
      "cluster_papers.py": "Performs unsupervised clustering using embeddings.",
      "evaluate_model.py": "Computes metrics (coherence, silhouette, accuracy) and generates plots.",
      "utils.py": "Model loading, metric calculation, result serialization.",
      "config.py": "Model names, number of topics/clusters, evaluation settings."
    },
    "main.py": "Orchestrates execution: runs phase_crawl → phase_clean → phase_learn in sequence.",
    "requirements.txt": "Python dependencies: feedparser, PyPDF2, pymongo, chromadb, scikit-learn, bertopic, sentence-transformers.",
    "CLAUDE.md": "Instructions for AI assistant — defines scope per phase and coding rules.",
    "PSEUDOCODE.md": "System design blueprint — function signatures and data flow across phases.",
    "FOLDER_STRUCTURE.md": "This file — defines full project layout and responsibilities.",
    "data": {
      "raw_pdfs": {},
      "extracted_text": {},
      "cleaned_text": {},
      "chunks": {},
      "embeddings": {},
      "models": {
        "topic_model/": "Saved BERTopic or LDA model files.",
        "classifier/": "Saved sklearn or transformer models."
      },
      "metadata": {
        "papers.jsonl": "Line-delimited JSON with full metadata for each paper."
      },
      "logs": {
        "crawl.log": "Logs from phase_crawl.",
        "clean.log": "Logs from phase_clean.",
        "learn.log": "Logs from phase_learn."
      }
    },
    "chroma_db": {
      ".gitignore": "Prevents ChromaDB binary files from being committed to git."
    },
    "reports": {
      "A3_firstname_lastname.pdf": "Final submitted report (written document)."
    },
    "tests": {
      "test_crawl/": {
        "test_fetch.py": "Tests ArXiv API query and parsing.",
        "test_download.py": "Verifies PDF download functionality."
      },
      "test_clean/": {
        "test_clean_text.py": "Tests text normalization pipeline.",
        "test_segment.py": "Validates section splitting logic."
      },
      "test_learn/": {
        "test_embedding.py": "Checks embedding generation.",
        "test_topic_model.py": "Validates topic model training."
      }
    }
  }
}
```

### `phase_crawl/`

- Core module for **data collection**: querying ArXiv, downloading PDFs, extracting text, storing metadata and embeddings
- Each file handles one responsibility:
  - `fetch_papers.py`: Queries ArXiv API and parses feed entries
  - `download_pdf.py`: Downloads PDF from `pdf_url` and saves to `data/raw_pdfs/`
  - `extract_text.py`: Extracts raw text using `PyPDF2`
  - `store_mongodb.py`: Inserts/updates paper metadata in MongoDB
  - `store_chromadb.py`: Generates and stores sentence embeddings in ChromaDB
  - `utils.py`: Shared helpers (logging setup, content hashing, retry logic)
  - `config.py`: Centralized settings (categories, rate limit, DB paths)

### `phase_clean/`

- Responsible for **text normalization and structuring**
  - `clean_text.py`: Removes line breaks, extra whitespace, control chars; fixes encoding
  - `segment_sections.py`: Splits full text into logical sections (e.g., Abstract, Introduction) using keyword matching
  - `filter_papers.py`: Filters by domain, language, length, or metadata (e.g., keep only `cs.CL`)
  - `normalize_tokens.py`: Tokenizes, removes stopwords, applies lemmatization (via `nltk` or `spacy`)
  - `generate_chunks.py`: Splits long texts into fixed-size (e.g., 512-token) chunks for embedding
  - `utils.py`: Text sanitization, language detection, error resilience
  - `config.py`: Paths, chunk size, filters, normalization rules

### `phase_learn/`

- Implements **machine learning tasks** on cleaned text
  - `encode_embeddings.py`: Uses `all-MiniLM-L6-v2` to generate document-level embeddings
  - `train_topic_model.py`: Applies BERTopic or LDA to discover research themes
  - `classify_papers.py`: Trains a classifier (e.g., SVM on TF-IDF) to categorize papers
  - `cluster_papers.py`: Performs K-Means or HDBSCAN clustering on embeddings
  - `evaluate_model.py`: Computes coherence, silhouette score, accuracy, F1; generates plots
  - `utils.py`: Model saving/loading, metric calculation, result serialization
  - `config.py`: Model names, topic count, cluster count, evaluation settings

### `main.py`

- **Orchestrator script** that runs the pipeline in sequence
- Imports modules from each phase
- Executes:

  ```python
  from phase_crawl.main import run_crawl
  from phase_clean.main import run_clean
  from phase_learn.main import run_learn

  if __name__ == "__main__":
      run_crawl()
      run_clean()
      run_learn()
  ```

- Handles global logging, timing, and error reporting

### `data/raw_pdfs/`

- Stores original PDFs downloaded from ArXiv
- Files named using ArXiv ID: `2401.12345.pdf`, `1906.05678.pdf`
- Never modified after download — serves as immutable source
- Used only for text extraction

### `data/extracted_text/`

- Contains raw text extracted from each PDF
- One `.txt` file per paper: `2401.12345.txt`
- UTF-8 encoded, with line breaks preserved as space
- May contain noise (headers, footers, equations) — to be cleaned in `phase_clean`

### `data/cleaned_text/`

- Stores cleaned, normalized versions of text
- Same naming convention: `2401.12345.txt`
- Output of `phase_clean` pipeline
- Ready for feature extraction or modeling

### `data/chunks/`

- Contains segmented text units (e.g., abstract, intro, or 512-token sliding windows)
- Stored as JSONL: each line is `{ "arxiv_id": "...", "chunk_id": 0, "text": "..." }`
- Input for embedding generation and topic modeling

### `data/embeddings/`

- Optional: stores precomputed embeddings as `.npy` or `.pkl` files
- Useful for offline analysis or large models
- Format: `embeddings/all-MiniLM-L6-v2.npy`, with matching metadata file

### `data/models/topic_model/`

- Persistent storage for trained topic models (e.g., BERTopic)
- Contains model files, vocabulary, and visualizations
- Enables reuse without retraining

### `data/models/classifier/`

- Stores trained classification models (e.g., `sklearn` pipeline with TF-IDF + LogisticRegression)
- Includes vectorizer, label encoder, and model binary

### `data/metadata/papers.jsonl`

- Line-delimited JSON file with one paper per line
- Fields:

  ```json
  {
    "arxiv_id": "2401.12345",
    "title": "...",
    "authors": ["..."],
    "categories": ["cs.AI", "cs.CL"],
    "published": "2024-01-15T00:00:00",
    "year": 2024,
    "primary_category": "cs.CL",
    "pdf_path": "data/raw_pdfs/2401.12345.pdf",
    "text_path": "data/extracted_text/2401.12345.txt",
    "cleaned_path": "data/cleaned_text/2401.12345.txt",
    "length_chars": 42350,
    "content_hash": "a1b2c3d4..."
  }
  ```

- Used for filtering, EDA, and linking across phases

### `chroma_db/`

- Persistent ChromaDB vector store
- Contains embeddings of paper content (e.g., abstract + intro)
- Enables fast semantic search via `search_similar_papers()`
- Collection name: `arxiv_embeddings`
- Metadata includes `arxiv_id`, `title`, `categories`, `published`

### `data/logs/crawl.log`

- Log output from `phase_crawl`
- Format: `[2025-04-05 10:30:00] INFO: Downloaded PDF: 2401.12345`
- Tracks success/failure of downloads, parsing, storage

### `data/logs/clean.log`

- Log output from `phase_clean`
- Tracks cleaning progress, errors, filtered papers

### `data/logs/learn.log`

- Log output from `phase_learn`
- Records model training, evaluation metrics, file saves

### `requirements.txt`

- Python dependencies required to run the project:

  ```txt
  feedparser
  requests
  PyPDF2
  pymongo
  chromadb
  sentence-transformers
  scikit-learn
  bertopic
  nltk
  tqdm
  python-dotenv
  ```

- Ensures reproducibility across environments

### `CLAUDE.md`, `PSEUDOCODE.md`, `FOLDER_STRUCTURE.md`

- Documentation files for AI coding assistants
- Define structure, behavior, and expectations
- Enable consistent, high-quality code generation

### `reports/A3_firstname_lastname.pdf`

- Final written report (PDF)
- Contains Task 1–4 discussions, visualizations, references
- Submitted to LearnJCU

### `tests/`

- Unit and integration tests for each phase
- Ensures reliability and correctness
- Example: `test_crawl/test_fetch.py`, `test_clean/test_clean_text.py`
