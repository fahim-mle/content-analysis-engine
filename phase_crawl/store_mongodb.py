# src/store_mongodb.py
"""Module for storing paper metadata in MongoDB."""

from typing import Any, Dict

import pymongo

from .config import COLLECTION_NAME, DB_NAME, MONGO_URI
from .utils import get_logger

logger = get_logger(__name__)


def connect_mongo(uri: str, db_name: str):
    """
    Connect to MongoDB and return collection.

    Args:
        uri: MongoDB connection URI
        db_name: Database name

    Returns:
        MongoDB collection object
    """
    try:
        client = pymongo.MongoClient(uri)
        db = client[db_name]
        collection = db[COLLECTION_NAME]

        # Create indexes for optimal query performance
        collection.create_index("arxiv_id", unique=True)
        collection.create_index("categories")
        collection.create_index("published")

        logger.info(f"Connected to MongoDB: {db_name}.{COLLECTION_NAME}")
        return collection

    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {e}")
        raise


def upsert_paper_metadata(collection, paper_data: Dict[str, Any]) -> bool:
    """
    Insert or update paper metadata in MongoDB.

    Args:
        collection: MongoDB collection object
        paper_data: Paper metadata dictionary

    Returns:
        True if operation successful, False otherwise
    """
    try:
        arxiv_id = paper_data["arxiv_id"]

        # Use upsert to insert new or update existing
        result = collection.update_one(
            {"arxiv_id": arxiv_id}, {"$set": paper_data}, upsert=True
        )

        if result.upserted_id:
            logger.info(f"Inserted new paper: {arxiv_id}")
        else:
            logger.info(f"Updated existing paper: {arxiv_id}")

        return True

    except Exception as e:
        logger.error(
            f"Error storing paper {paper_data.get('arxiv_id', 'unknown')}: {e}"
        )
        return False
