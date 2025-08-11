# phase_crawl/main_crawl.py
"""Main entry point for crawling phase - refactored from original main.py"""

import time
from pathlib import Path

from .config import (
    CATEGORIES, CHROMA_COLLECTION, CHROMA_PATH, DAYS_BACK, DB_NAME,
    EMBEDDING_TEXT_LENGTH, MAX_PDF_COUNT, MAX_RESULTS, MAX_TEXT_LENGTH,
    MONGO_URI, PDF_DIR, RATE_LIMIT_DELAY,
)
from .download_pdf import download_pdf
from .extract_text import extract_text
from .fetch_papers import build_query, fetch_papers
from .process_existing import get_existing_pdfs, process_existing_pdfs
from .store_chromadb import get_chroma_collection, store_embedding
from .store_mongodb import connect_mongo, upsert_paper_metadata
from .utils import get_logger

logger = get_logger(__name__)


def run_crawl_phase() -> bool:
    """
    Execute the complete crawling phase.
    
    Returns:
        True if phase completed successfully, False otherwise
    """
    logger.info("Starting ArXiv scraping pipeline")
    
    try:
        # Initialize database connections
        mongo_collection = connect_mongo(MONGO_URI, DB_NAME)
        chroma_collection = get_chroma_collection(CHROMA_PATH, CHROMA_COLLECTION)
        
        # Check current PDF count
        existing_pdfs = get_existing_pdfs()
        current_pdf_count = len(existing_pdfs)
        logger.info(f"Current PDF count: {current_pdf_count}")
        
        # Process existing PDFs first
        if current_pdf_count > 0:
            logger.info("Processing existing PDFs for text extraction and embeddings")
            processed_count = process_existing_pdfs(
                mongo_collection, chroma_collection, max_process=400
            )
            logger.info(f"Processed {processed_count} existing PDFs")
        
        # Check if we should download more papers
        if current_pdf_count >= MAX_PDF_COUNT:
            logger.info(f"PDF limit reached ({current_pdf_count} >= {MAX_PDF_COUNT})")
            logger.info("Skipping new downloads, crawl phase complete")
            return True
        
        # Calculate how many new papers we can download
        remaining_slots = MAX_PDF_COUNT - current_pdf_count
        download_limit = min(MAX_RESULTS, remaining_slots)
        
        logger.info(f"Can download up to {download_limit} new papers")
        
        # Build query and fetch papers
        query_url = build_query(CATEGORIES, download_limit, DAYS_BACK)
        papers = fetch_papers(query_url)
        
        if not papers:
            logger.warning("No new papers found")
            return True
        
        logger.info(f"Processing {len(papers)} new papers")
        
        # Process each new paper
        downloaded_count = 0
        for i, paper in enumerate(papers):
            arxiv_id = paper["arxiv_id"]
            logger.info(f"Processing paper {i+1}/{len(papers)}: {arxiv_id}")
            
            # Check if we've hit the limit
            if current_pdf_count + downloaded_count >= MAX_PDF_COUNT:
                logger.info(f"Reached PDF limit ({MAX_PDF_COUNT}), stopping downloads")
                break
            
            try:
                # Download PDF
                pdf_path = download_pdf(paper, Path(PDF_DIR))
                if not pdf_path:
                    logger.warning(f"Skipping {arxiv_id}: PDF download failed")
                    continue
                
                downloaded_count += 1
                
                # Extract text
                full_text = extract_text(pdf_path)
                if not full_text:
                    logger.warning(f"Skipping {arxiv_id}: Text extraction failed")
                    continue
                
                # Enrich paper data
                paper["pdf_path"] = pdf_path
                paper["full_text"] = full_text[:MAX_TEXT_LENGTH]
                paper["pdf_downloaded"] = True
                paper["pdf_processed"] = True
                
                # Store in MongoDB
                if not upsert_paper_metadata(mongo_collection, paper):
                    logger.error(f"Failed to store metadata for {arxiv_id}")
                    continue
                
                # Prepare embedding input
                embedding_text = f"{paper['title']} {paper['summary']} {full_text[:EMBEDDING_TEXT_LENGTH]}"
                metadata = {
                    "arxiv_id": arxiv_id,
                    "title": paper["title"],
                    "categories": ",".join(paper["categories"]),
                    "published": paper["published"].isoformat(),
                }
                
                # Store in ChromaDB
                if not store_embedding(
                    chroma_collection, arxiv_id, embedding_text, metadata
                ):
                    logger.error(f"Failed to store embedding for {arxiv_id}")
                
                # Rate limiting
                time.sleep(RATE_LIMIT_DELAY)
                
            except Exception as e:
                logger.error(f"Error processing paper {arxiv_id}: {e}")
                continue
        
        final_pdf_count = current_pdf_count + downloaded_count
        logger.info(f"Crawl phase completed. Total PDFs: {final_pdf_count}")
        
        if final_pdf_count >= MAX_PDF_COUNT:
            logger.info("Ready to move to next phase - PDF limit reached")
        
        return True
        
    except Exception as e:
        logger.error(f"Crawl phase failed: {e}")
        return False