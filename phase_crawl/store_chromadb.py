# src/store_chromadb.py
"""Module for storing embeddings in ChromaDB."""

from typing import Any, Dict

import chromadb

from .config import CHROMA_COLLECTION, CHROMA_PATH
from .utils import get_logger

logger = get_logger(__name__)


def get_chroma_collection(path: str, name: str):
    """
    Get or create ChromaDB collection.

    Args:
        path: Path to ChromaDB storage
        name: Collection name

    Returns:
        ChromaDB collection object
    """
    try:
        client = chromadb.PersistentClient(path=path)
        collection = client.get_or_create_collection(
            name=name, metadata={"description": "ArXiv paper embeddings"}
        )

        logger.info(f"Connected to ChromaDB collection: {name}")
        return collection

    except Exception as e:
        logger.error(f"Error connecting to ChromaDB: {e}")
        raise


def store_embedding(
    collection, arxiv_id: str, text: str, metadata: Dict[str, Any]
) -> bool:
    """
    Generate embedding and store in ChromaDB.

    Args:
        collection: ChromaDB collection object
        arxiv_id: ArXiv paper ID
        text: Text content to embed
        metadata: Additional metadata dictionary

    Returns:
        True if operation successful, False otherwise
    """
    try:
        doc_id = f"arxiv_{arxiv_id}"

        # ChromaDB automatically generates embeddings
        collection.add(documents=[text], ids=[doc_id], metadatas=[metadata])

        logger.info(f"Stored embedding for: {arxiv_id}")
        return True

    except Exception as e:
        logger.error(f"Error storing embedding for {arxiv_id}: {e}")
        return False
