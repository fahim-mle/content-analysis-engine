# src/fetch_papers.py
"""Module for fetching papers from ArXiv API."""

from datetime import datetime, timedelta
from typing import Any, Dict, List

import feedparser
import requests

from .config import CATEGORIES, DAYS_BACK, MAX_RESULTS, REQUEST_TIMEOUT
from .utils import get_logger

logger = get_logger(__name__)


def build_query(categories: List[str], max_results: int, days_back: int) -> str:
    """
    Build ArXiv API URL with filters.

    Args:
        categories: List of ArXiv categories to search
        max_results: Maximum number of results to return
        days_back: Number of days back from today to search

    Returns:
        Complete ArXiv API query URL
    """
    base_url = "http://export.arxiv.org/api/query?"

    # Build category query
    if categories:
        cat_query = " OR ".join([f"cat:{cat}" for cat in categories])
        search_query = f"({cat_query})"
    else:
        search_query = "all"

    # Add date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    date_filter = f" AND submittedDate:[{start_date.strftime('%Y%m%d')} TO {end_date.strftime('%Y%m%d')}]"
    search_query += date_filter

    # Build parameters
    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    query_string = "&".join([f"{k}={v}" for k, v in params.items()])
    return f"{base_url}{query_string}"


def fetch_papers(query_url: str) -> List[Dict[str, Any]]:
    """
    Fetch and parse feedparser entries from ArXiv API.

    Args:
        query_url: Complete ArXiv API query URL

    Returns:
        List of paper dictionaries with metadata
    """
    logger.info(f"Fetching papers from: {query_url}")

    try:
        response = requests.get(query_url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        feed = feedparser.parse(response.text)
        papers = []

        for entry in feed.entries:
            try:
                paper = _parse_entry(entry)
                papers.append(paper)
            except Exception as e:
                logger.error(f"Error parsing entry: {e}")
                continue

        logger.info(f"Successfully parsed {len(papers)} papers")
        return papers

    except requests.RequestException as e:
        logger.error(f"Error fetching from ArXiv: {e}")
        return []


def _parse_entry(entry: feedparser.FeedParserDict) -> Dict[str, Any]:
    """
    Parse individual ArXiv feed entry into paper dictionary.

    Args:
        entry: Single feedparser entry

    Returns:
        Paper metadata dictionary
    """
    arxiv_id = entry.id.split("/abs/")[-1]
    categories = [tag.term for tag in entry.get("tags", [])]
    authors = [author.name for author in entry.get("authors", [])]
    published = datetime.strptime(entry.published, "%Y-%m-%dT%H:%M:%SZ")
    updated = datetime.strptime(entry.updated, "%Y-%m-%dT%H:%M:%SZ")
    pdf_url = entry.id.replace("/abs/", "/pdf/") + ".pdf"

    return {
        "arxiv_id": arxiv_id,
        "title": entry.title.replace("\n", " ").strip(),
        "summary": entry.summary.replace("\n", " ").strip(),
        "authors": authors,
        "categories": categories,
        "published": published,
        "updated": updated,
        "pdf_url": pdf_url,
        "scraped_at": datetime.utcnow(),
        "pdf_downloaded": False,
        "pdf_processed": False,
    }
