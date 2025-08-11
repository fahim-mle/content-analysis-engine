# src/extract_text.py
"""Module for extracting text from PDF files."""

from typing import Optional

import PyPDF2

from .utils import get_logger

logger = get_logger(__name__)


def extract_text(pdf_path: str) -> Optional[str]:
    """
    Extract raw text from PDF file.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Extracted text content or None on failure
    """
    try:
        logger.info(f"Extracting text from: {pdf_path}")

        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""

            # Extract text from each page
            for page in reader.pages:
                text += page.extract_text() + "\n"

            extracted_text = text.strip()

            if extracted_text:
                logger.info(
                    f"Successfully extracted {len(extracted_text)} characters from {pdf_path}"
                )
                return extracted_text
            else:
                logger.warning(f"No text extracted from {pdf_path}")
                return None

    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        return None
