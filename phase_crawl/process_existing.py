# src/process_existing.py
"""Module for processing existing PDF files with different naming formats."""

import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from .config import PDF_DIR
from .extract_text import extract_text
from .store_mongodb import upsert_paper_metadata
from .store_chromadb import store_embedding
from .utils import get_logger

logger = get_logger(__name__)


def extract_arxiv_id_from_filename(filename: str) -> Optional[str]:
    """
    Extract ArXiv ID from different PDF filename formats.
    
    Args:
        filename: PDF filename
        
    Returns:
        ArXiv ID or None if not extractable
    """
    # Format 1: 2508.06492v1.pdf -> 2508.06492v1
    if re.match(r'^\d{4}\.\d{5}v\d+\.pdf$', filename):
        return filename.replace('.pdf', '')
    
    # Format 2: 10.48550_arXiv.2505.19489.pdf -> 2505.19489
    match = re.match(r'^10\.48550_arXiv\.(\d{4}\.\d{5})\.pdf$', filename)
    if match:
        return match.group(1)
    
    return None


def create_paper_from_pdf(pdf_path: Path) -> Optional[Dict[str, Any]]:
    """
    Create paper metadata from existing PDF file.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Paper metadata dictionary or None if processing fails
    """
    filename = pdf_path.name
    arxiv_id = extract_arxiv_id_from_filename(filename)
    
    if not arxiv_id:
        logger.warning(f"Could not extract ArXiv ID from: {filename}")
        return None
    
    # Create basic paper metadata
    paper_data = {
        "arxiv_id": arxiv_id,
        "title": f"Existing Paper {arxiv_id}",  # Will be updated if metadata exists
        "summary": "Summary to be extracted from full text",
        "authors": [],
        "categories": ["cs.AI"],  # Default category
        "published": datetime.now(),
        "updated": datetime.now(),
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        "scraped_at": datetime.utcnow(),
        "pdf_downloaded": True,
        "pdf_processed": False,
        "pdf_path": str(pdf_path),
    }
    
    return paper_data


def get_existing_pdfs() -> List[Path]:
    """
    Get list of all existing PDF files.
    
    Returns:
        List of PDF file paths
    """
    pdf_files = list(Path(PDF_DIR).glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} existing PDF files")
    return pdf_files


def process_existing_pdfs(mongo_collection, chroma_collection, max_process: int = 50) -> int:
    """
    Process existing PDF files for text extraction and embedding.
    
    Args:
        mongo_collection: MongoDB collection
        chroma_collection: ChromaDB collection  
        max_process: Maximum number of PDFs to process in this run
        
    Returns:
        Number of PDFs successfully processed
    """
    pdf_files = get_existing_pdfs()
    processed_count = 0
    
    for pdf_path in pdf_files[:max_process]:
        try:
            filename = pdf_path.name
            arxiv_id = extract_arxiv_id_from_filename(filename)
            
            if not arxiv_id:
                continue
            
            logger.info(f"Processing existing PDF: {filename}")
            
            # Check if already processed in MongoDB
            existing = mongo_collection.find_one({"arxiv_id": arxiv_id})
            if existing and existing.get("pdf_processed"):
                logger.info(f"Already processed: {arxiv_id}")
                continue
            
            # Create paper metadata
            paper_data = create_paper_from_pdf(pdf_path)
            if not paper_data:
                continue
            
            # Extract text
            full_text = extract_text(str(pdf_path))
            if not full_text:
                logger.warning(f"Text extraction failed for: {filename}")
                continue
            
            # Update paper data with extracted text
            paper_data["full_text"] = full_text[:5000]  # Truncate for storage
            paper_data["pdf_processed"] = True
            
            # Store in MongoDB
            if not upsert_paper_metadata(mongo_collection, paper_data):
                logger.error(f"Failed to store metadata for: {arxiv_id}")
                continue
            
            logger.debug(f"Successfully stored metadata for: {arxiv_id}")
            
            # Prepare embedding text
            embedding_text = f"{paper_data['title']} {paper_data['summary']} {full_text[:3000]}"
            metadata = {
                "arxiv_id": arxiv_id,
                "title": paper_data["title"],
                "categories": ",".join(paper_data["categories"]),
                "published": paper_data["published"].isoformat()
            }
            
            # Store embedding
            if store_embedding(chroma_collection, arxiv_id, embedding_text, metadata):
                processed_count += 1
                logger.info(f"Successfully processed: {arxiv_id}")
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            continue
    
    logger.info(f"Processed {processed_count} existing PDFs")
    return processed_count