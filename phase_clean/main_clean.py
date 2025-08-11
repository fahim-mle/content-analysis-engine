# phase_clean/main_clean.py
"""Main entry point for text cleaning phase."""

import sys
import os
import time
from pathlib import Path
from typing import Any, Dict, List

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from phase_crawl.config import DB_NAME, MONGO_URI

# Import MongoDB connection from crawl phase
from phase_crawl.store_mongodb import connect_mongo

from phase_clean.citation_handler import CitationHandler, process_citations
from phase_clean.quality_filter import TextQualityFilter, filter_papers_by_quality
from phase_clean.section_parser import ResearchPaperSectionParser, parse_paper_sections
from phase_clean.storage_system import (
    CleanedTextStorage,
    create_chunks_from_papers,
    save_processed_papers,
)
from phase_clean.text_cleaner import ResearchPaperCleaner, process_paper_text
from phase_clean.utils import get_logger

logger = get_logger(__name__)


def load_papers_from_mongodb(
    collection, batch_size: int = 50, min_text_length: int = 100
) -> List[Dict[str, Any]]:
    """
    Load papers from MongoDB with full text extraction.

    Args:
        collection: MongoDB collection
        batch_size: Number of papers to load at once
        min_text_length: Minimum text length to consider

    Returns:
        List of paper documents with full text
    """
    papers = []
    total_papers = collection.count_documents({})
    logger.info(f"Loading {total_papers} papers from MongoDB")

    # Get papers with full text that meets minimum length
    query = {
        "full_text": {"$exists": True, "$ne": ""},
        "$where": f"this.full_text.length >= {min_text_length}",
    }

    cursor = collection.find(query).batch_size(batch_size)

    for paper in cursor:
        # Convert ObjectId to string for JSON serialization
        paper["_id"] = str(paper["_id"])

        # Ensure categories is a list
        if isinstance(paper.get("categories"), str):
            paper["categories"] = [paper["categories"]]

        papers.append(paper)

    logger.info(f"Loaded {len(papers)} papers with sufficient text content")
    return papers


def process_single_paper(
    paper_data: Dict[str, Any],
    cleaner: ResearchPaperCleaner,
    section_parser: ResearchPaperSectionParser,
    citation_handler: CitationHandler,
) -> Dict[str, Any]:
    """
    Process a single paper through the complete cleaning pipeline.

    Args:
        paper_data: Raw paper data from MongoDB
        cleaner: Text cleaner instance
        section_parser: Section parser instance
        citation_handler: Citation handler instance

    Returns:
        Fully processed paper data
    """
    arxiv_id = paper_data.get("arxiv_id", "unknown")

    try:
        # Step 1: Clean text
        paper_data = process_paper_text(paper_data, cleaner)

        if not paper_data.get("cleaned_text"):
            logger.warning(
                f"No cleaned text for {arxiv_id}, skipping further processing"
            )
            return paper_data

        # Step 2: Parse sections
        sections, section_summary = parse_paper_sections(
            paper_data["cleaned_text"], section_parser
        )
        paper_data["text_sections"] = {
            name: section.content for name, section in sections.items()
        }
        paper_data["section_summary"] = section_summary

        # Step 3: Process citations
        citations, citation_stats, cleaned_refs = process_citations(
            paper_data["cleaned_text"], citation_handler
        )
        paper_data["citations"] = citations
        paper_data["citation_stats"] = citation_stats
        paper_data["references_text"] = cleaned_refs

        # Step 4: Add processing metadata
        paper_data["processing_completed"] = True
        paper_data["processing_timestamp"] = time.time()

        logger.debug(
            f"Successfully processed {arxiv_id}: "
            f"{paper_data['text_quality']['length']} chars, "
            f"{section_summary['total_sections']} sections, "
            f"{citation_stats['total_citations']} citations"
        )

        return paper_data

    except Exception as e:
        logger.error(f"Failed to process paper {arxiv_id}: {e}")
        paper_data["processing_completed"] = False
        paper_data["processing_error"] = str(e)
        return paper_data


def run_clean_phase() -> bool:
    """
    Execute the complete text cleaning phase.

    Returns:
        True if phase completed successfully, False otherwise
    """
    logger.info("Starting comprehensive text cleaning phase")
    start_time = time.time()

    try:
        # Initialize components
        logger.info("Initializing text processing components...")
        cleaner = ResearchPaperCleaner()
        section_parser = ResearchPaperSectionParser()
        citation_handler = CitationHandler()
        quality_filter = TextQualityFilter()
        storage = CleanedTextStorage()

        # Step 1: Connect to MongoDB and load papers
        logger.info("Connecting to MongoDB and loading papers...")
        mongo_collection = connect_mongo(MONGO_URI, DB_NAME)
        papers = load_papers_from_mongodb(mongo_collection, batch_size=100)

        if not papers:
            logger.error("No papers found in MongoDB")
            return False

        logger.info(f"Processing {len(papers)} papers through cleaning pipeline...")

        # Step 2: Process papers in batches
        batch_size = 25  # Process papers in smaller batches for memory efficiency
        processed_papers = []
        failed_count = 0

        for i in range(0, len(papers), batch_size):
            batch = papers[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(papers) + batch_size - 1) // batch_size

            logger.info(
                f"Processing batch {batch_num}/{total_batches} ({len(batch)} papers)"
            )

            batch_processed = []
            for paper in batch:
                try:
                    processed_paper = process_single_paper(
                        paper, cleaner, section_parser, citation_handler
                    )
                    batch_processed.append(processed_paper)

                    if not processed_paper.get("processing_completed", False):
                        failed_count += 1

                except Exception as e:
                    logger.error(
                        f"Batch processing error for {paper.get('arxiv_id', 'unknown')}: {e}"
                    )
                    failed_count += 1

            processed_papers.extend(batch_processed)

            # Log progress
            success_count = len(
                [p for p in batch_processed if p.get("processing_completed", False)]
            )
            logger.info(
                f"Batch {batch_num} completed: {success_count}/{len(batch)} successful"
            )

        logger.info(
            f"Text processing completed: {len(processed_papers) - failed_count}/{len(processed_papers)} successful"
        )

        # Step 3: Apply quality filtering
        logger.info("Applying quality filters...")
        filtered_papers, filter_stats = filter_papers_by_quality(
            processed_papers, quality_filter
        )

        logger.info(
            f"Quality filtering completed: {filter_stats['accepted']}/{filter_stats['total']} papers accepted "
            f"({filter_stats['acceptance_rate']:.1%} acceptance rate)"
        )

        if filter_stats["rejection_reasons"]:
            logger.info("Rejection reasons:")
            for reason, count in filter_stats["rejection_reasons"].items():
                logger.info(f"  {reason}: {count} papers")

        # Step 4: Save processed papers to storage
        logger.info("Saving cleaned papers to storage...")
        save_stats = save_processed_papers(filtered_papers, storage)
        logger.info(
            f"Storage completed: {save_stats['saved']} papers saved, {save_stats['failed']} failed"
        )

        if save_stats["errors"]:
            logger.warning(
                f"Storage errors occurred: {len(save_stats['errors'])} errors"
            )
            for error in save_stats["errors"][:5]:  # Log first 5 errors
                logger.debug(f"Storage error: {error}")

        # Step 5: Create text chunks for ML processing
        logger.info("Creating text chunks for ML processing...")
        chunk_stats = create_chunks_from_papers(filtered_papers, storage)
        logger.info(
            f"Chunk creation completed: {chunk_stats['chunks_created']} chunks from "
            f"{chunk_stats['papers_processed']} papers in {chunk_stats['batches_saved']} batches"
        )

        # Step 6: Create ML dataset
        logger.info("Creating optimized ML dataset...")
        dataset_stats = storage.create_ml_dataset(
            filtered_papers, include_quality_filters=True
        )
        logger.info(
            f"ML dataset created: {dataset_stats.get('total_papers', 0)} papers, "
            f"{dataset_stats.get('total_chunks', 0)} chunks"
        )

        # Step 7: Generate final statistics and report
        storage_stats = storage.get_storage_stats()
        elapsed_time = time.time() - start_time

        logger.info("=== Text Cleaning Phase Summary ===")
        logger.info(f"Processing time: {elapsed_time:.1f} seconds")
        logger.info(f"Papers processed: {len(processed_papers)}")
        logger.info(f"Papers passed quality filter: {len(filtered_papers)}")
        logger.info(f"Papers saved to storage: {save_stats['saved']}")
        logger.info(f"Text chunks created: {chunk_stats['chunks_created']}")
        logger.info(f"Storage size: {storage_stats['storage_size_mb']:.2f} MB")
        logger.info(
            f"Categories found: {len(storage_stats['categories'])} ({', '.join(storage_stats['categories'][:10])})"
        )
        logger.info("=====================================")

        # Check if we have enough quality papers for ML phase
        min_papers_for_ml = 50
        if len(filtered_papers) < min_papers_for_ml:
            logger.warning(
                f"Only {len(filtered_papers)} quality papers found, "
                f"minimum {min_papers_for_ml} recommended for ML phase"
            )
            return False

        logger.info("Text cleaning phase completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Text cleaning phase failed: {e}")
        return False
