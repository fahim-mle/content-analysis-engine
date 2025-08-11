# phase_learn/train_topic_model.py
"""Train and save a BERTopic model."""

import numpy as np
from bertopic import BERTopic
from .config import EMBEDDINGS_FILE, TOPIC_MODEL_FILE
from .utils import get_logger, save_model, save_json_results

logger = get_logger(__name__)

def train_topic_model(embeddings: np.ndarray) -> BERTopic:
    """
    Train a BERTopic model on the given embeddings.

    Args:
        embeddings: A numpy array of document embeddings.

    Returns:
        A trained BERTopic model.
    """
    logger.info("Training BERTopic model...")
    # Parameters can be tuned for better performance
    topic_model = BERTopic(verbose=True, embedding_model='all-MiniLM-L6-v2')
    topics, _ = topic_model.fit_transform(np.random.rand(10,10), embeddings)
    logger.info("BERTopic model training complete.")
    return topic_model

def save_topic_results(topic_model: BERTopic):
    """
    Save the results of the topic model.

    Args:
        topic_model: A trained BERTopic model.
    """
    logger.info("Saving topic model results...")
    # Get topic information
    topic_info = topic_model.get_topic_info()
    # Get topics and their representative words
    topics = {str(k): v for k, v in topic_model.get_topics().items()}

    results = {
        'topic_info': topic_info.to_dict('records'),
        'topics': topics
    }

    save_json_results(results, 'topic_model_results.json')

def run_topic_modeling():
    """
    Main function to run the topic modeling process.
    """
    logger.info("Starting topic modeling process...")
    if not EMBEDDINGS_FILE.exists():
        logger.error(f"Embeddings file not found at {EMBEDDINGS_FILE}. Run encoding first.")
        return

    try:
        embeddings = np.load(EMBEDDINGS_FILE)
        logger.info(f"Loaded {len(embeddings)} embeddings.")
    except Exception as e:
        logger.error(f"Failed to load embeddings: {e}")
        return

    topic_model = train_topic_model(embeddings)
    save_model(topic_model, TOPIC_MODEL_FILE)
    save_topic_results(topic_model)
    logger.info("Topic modeling process finished.")

if __name__ == '__main__':
    run_topic_modeling()
