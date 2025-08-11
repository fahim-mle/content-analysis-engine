# src/download_pdf.py
"""Module for downloading PDF files from ArXiv."""

from pathlib import Path
from typing import Any, Dict, Optional

import requests

from .config import REQUEST_TIMEOUT, USER_AGENT
from .utils import ensure_directory, get_logger

logger = get_logger(__name__)


def download_pdf(paper: Dict[str, Any], pdf_dir: Path) -> Optional[str]:
    """
    Download PDF from arXiv URL.

    Args:
        paper: Paper metadata dictionary containing pdf_url and arxiv_id
        pdf_dir: Directory to save PDF files

    Returns:
        File path of downloaded PDF or None on failure
    """
    arxiv_id = paper["arxiv_id"]
    pdf_url = paper["pdf_url"]

    # Ensure directory exists
    ensure_directory(pdf_dir)

    # Generate file path
    pdf_path = pdf_dir / f"{arxiv_id}.pdf"

    # Check if already downloaded
    if pdf_path.exists():
        logger.info(f"PDF already exists: {arxiv_id}")
        return str(pdf_path)

    try:
        logger.info(f"Downloading PDF: {arxiv_id}")

        headers = {"User-Agent": USER_AGENT}
        response = requests.get(pdf_url, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        # Write PDF content to file
        with open(pdf_path, "wb") as f:
            f.write(response.content)

        logger.info(f"PDF downloaded: {arxiv_id}")
        return str(pdf_path)

    except Exception as e:
        logger.error(f"Error downloading PDF {arxiv_id}: {e}")
        return None
