# main.py
"""Orchestrates execution: runs phase_crawl → phase_clean → phase_learn in sequence."""

import sys
from pathlib import Path

from phase_clean.main_clean import run_clean_phase

# Phase imports
from phase_crawl.main_crawl import run_crawl_phase

# Utility imports
from phase_crawl.utils import get_logger
from phase_learn.main_learn import run_learn_phase

logger = get_logger(__name__)


def main():
    """Main orchestration of all three phases."""
    logger.info("Starting multi-phase NLP pipeline")

    try:
        # Phase 1: Data Collection and Crawling
        logger.info("=" * 50)
        logger.info("PHASE 1: Data Collection and Crawling")
        logger.info("=" * 50)
        crawl_success = run_crawl_phase()

        if not crawl_success:
            logger.error("Phase 1 (crawl) failed. Stopping pipeline.")
            sys.exit(1)

        logger.info("Phase 1 (crawl) completed successfully")

        # Phase 2: Text Cleaning and Preprocessing
        logger.info("=" * 50)
        logger.info("PHASE 2: Text Cleaning and Preprocessing")
        logger.info("=" * 50)
        clean_success = run_clean_phase()

        if not clean_success:
            logger.error("Phase 2 (clean) failed. Stopping pipeline.")
            sys.exit(1)

        logger.info("Phase 2 (clean) completed successfully")

        # Phase 3: Machine Learning and Analysis
        logger.info("=" * 50)
        logger.info("PHASE 3: Machine Learning and Analysis")
        logger.info("=" * 50)
        learn_success = run_learn_phase()

        if not learn_success:
            logger.error("Phase 3 (learn) failed. Pipeline incomplete.")
            sys.exit(1)

        logger.info("Phase 3 (learn) completed successfully")
        logger.info("=" * 50)
        logger.info("ALL PHASES COMPLETED SUCCESSFULLY")
        logger.info("=" * 50)

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
