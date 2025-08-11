# src/store_chromadb.py
"""Module for storing embeddings in ChromaDB."""

from typing import Any, Dict

import chromadb
from chromadb.utils import embedding_functions

from .config import CHROMA_COLLECTION, CHROMA_PATH
from .utils import get_logger

logger = get_logger(__name__)


def get_chroma_collection(path: str, name: str):
    """
    Get or create ChromaDB collection with DistilBERT embeddings.

    Args:
        path: Path to ChromaDB storage
        name: Collection name

    Returns:
        ChromaDB collection object
    """
    try:
        client = chromadb.PersistentClient(path=path)
        
        # Use DistilBERT embedding function for consistency
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="distilbert-base-uncased"
        )
        
        collection = client.get_or_create_collection(
            name=name, 
            metadata={"description": "ArXiv paper embeddings with DistilBERT"},
            embedding_function=embedding_fn
        )

        logger.info(f"Connected to ChromaDB collection: {name} with DistilBERT")
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
