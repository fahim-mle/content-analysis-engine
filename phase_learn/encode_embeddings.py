# phase_learn/encode_embeddings.py
"""Generate and save embeddings for text data."""

import numpy as np
from sentence_transformers import SentenceTransformer
from .config import (DATA_DIR, EMBEDDINGS_DIR, ML_DATASET_FILE, 
                     EMBEDDING_MODEL, EMBEDDINGS_FILE)
from .utils import get_logger, load_ml_dataset

logger = get_logger(__name__)

def generate_embeddings(documents: list[str], model_name: str) -> np.ndarray:
    """
    Generate sentence embeddings for a list of documents.

    Args:
        documents: A list of text documents to embed.
        model_name: The name of the SentenceTransformer model to use.

    Returns:
        A numpy array of embeddings.
    """
    try:
        logger.info(f"Loading sentence transformer model: {model_name}")
        model = SentenceTransformer(model_name)
        logger.info("Model loaded. Generating embeddings...")
        embeddings = model.encode(documents, show_progress_bar=True)
        logger.info(f"Generated {len(embeddings)} embeddings.")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        return np.array([])

def save_embeddings(embeddings: np.ndarray, file_path: str):
    """
    Save embeddings to a file.

    Args:
        embeddings: A numpy array of embeddings.
        file_path: The path to save the embeddings to.
    """
    try:
        logger.info(f"Saving embeddings to {file_path}...")
        np.save(file_path, embeddings)
        logger.info("Embeddings saved successfully.")
    except IOError as e:
        logger.error(f"Failed to save embeddings to {file_path}: {e}")

def process_paper_embeddings():
    """
    Main function to generate and save paper embeddings.
    """
    logger.info("Starting embedding generation process...")
    dataset = load_ml_dataset(ML_DATASET_FILE)

    if not dataset or 'chunks' not in dataset or not dataset['chunks']:
        logger.error("ML dataset is empty or does not contain chunks. Aborting.")
        return

    # We'll use the text from the chunks for embedding
    documents = [chunk['text'] for chunk in dataset['chunks']]
    
    if not documents:
        logger.error("No documents found in chunks. Aborting.")
        return

    embeddings = generate_embeddings(documents, EMBEDDING_MODEL)

    if embeddings.size == 0:
        logger.error("Embedding generation failed. Aborting.")
        return

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    save_embeddings(embeddings, EMBEDDINGS_FILE)
    logger.info("Embedding generation process finished.")

if __name__ == '__main__':
    # This allows running the script directly for testing or reprocessing
    process_paper_embeddings()
