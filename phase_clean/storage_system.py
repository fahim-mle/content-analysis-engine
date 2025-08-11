# phase_clean/storage_system.py
"""Efficient storage system for cleaned text data optimized for ML phase."""

import gzip
import hashlib
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .config import CHUNKS_DIR, CLEANED_TEXT_DIR
from .utils import get_logger

logger = get_logger(__name__)


class CleanedTextStorage:
    """Efficient storage and retrieval system for cleaned research paper text."""

    def __init__(self, base_dir: Path = None):
        self.base_dir = base_dir or Path("data")
        self.cleaned_text_dir = self.base_dir / "cleaned_text"
        self.chunks_dir = self.base_dir / "chunks"
        self.metadata_dir = self.base_dir / "metadata"

        # Create directories
        for dir_path in [self.cleaned_text_dir, self.chunks_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Storage configuration
        self.use_compression = True
        self.chunk_size_limit = 1000  # characters
        self.batch_size = 100  # papers per batch file

    def _generate_file_hash(self, content: str) -> str:
        """Generate hash for content deduplication."""
        return hashlib.md5(content.encode("utf-8")).hexdigest()[:12]

    def _save_json_file(
        self, data: Any, file_path: Path, compress: bool = True
    ) -> bool:
        """Save data as JSON file with optional compression."""
        try:
            # Convert datetime objects to ISO strings for JSON serialization
            def datetime_converter(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

            content = json.dumps(
                data,
                indent=2 if not compress else None,
                ensure_ascii=False,
                default=datetime_converter,
            )

            if compress and self.use_compression:
                with gzip.open(f"{file_path}.gz", "wt", encoding="utf-8") as f:
                    f.write(content)
            else:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)

            return True
        except Exception as e:
            logger.error(f"Failed to save JSON file {file_path}: {e}")
            return False

    def _load_json_file(
        self, file_path: Path, compressed: bool = None
    ) -> Optional[Any]:
        """Load data from JSON file with optional compression."""
        try:
            # Auto-detect compression
            if compressed is None:
                compressed = file_path.suffix == ".gz"

            actual_path = file_path

            if compressed:
                with gzip.open(actual_path, "rt", encoding="utf-8") as f:
                    return json.load(f)
            else:
                with open(actual_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.debug(f"Failed to load JSON file {file_path}: {e}")
            return None

    def save_cleaned_paper(
        self, paper_data: Dict[str, Any], include_raw: bool = False
    ) -> bool:
        """
        Save cleaned paper data to storage.

        Args:
            paper_data: Processed paper data
            include_raw: Whether to include original raw text

        Returns:
            True if successful
        """
        arxiv_id = paper_data.get("arxiv_id")
        if not arxiv_id:
            logger.error("No arxiv_id found in paper data")
            return False

        try:
            # Prepare storage data
            storage_data = {
                "arxiv_id": arxiv_id,
                "title": paper_data.get("title", ""),
                "authors": paper_data.get("authors", []),
                "categories": paper_data.get("categories", []),
                "published": paper_data.get("published", ""),
                "updated": paper_data.get("updated", ""),
                "summary": paper_data.get("summary", ""),
                "cleaned_text": paper_data.get("cleaned_text", ""),
                "text_quality": paper_data.get("text_quality", {}),
                "text_sections": paper_data.get("text_sections", {}),
                "citations": paper_data.get("citations", []),
                "citation_stats": paper_data.get("citation_stats", {}),
                "quality_metrics": paper_data.get("quality_metrics", {}),
                "processing_metadata": {
                    "processed_at": datetime.now().isoformat(),
                    "text_hash": self._generate_file_hash(
                        paper_data.get("cleaned_text", "")
                    ),
                    "original_length": len(paper_data.get("full_text", "")),
                    "cleaned_length": len(paper_data.get("cleaned_text", "")),
                },
            }

            if include_raw:
                storage_data["full_text"] = paper_data.get("full_text", "")

            # Save main paper file
            paper_file = self.cleaned_text_dir / f"{arxiv_id}.json"
            if not self._save_json_file(storage_data, paper_file, compress=True):
                return False

            # Save text-only version for quick loading
            text_only_data = {
                "arxiv_id": arxiv_id,
                "title": paper_data.get("title", ""),
                "cleaned_text": paper_data.get("cleaned_text", ""),
                "categories": paper_data.get("categories", []),
                "quality_score": paper_data.get("quality_metrics", {}).get(
                    "overall_score", 0.0
                ),
            }

            text_file = self.cleaned_text_dir / f"{arxiv_id}_text.json"
            self._save_json_file(text_only_data, text_file, compress=True)

            logger.debug(f"Saved cleaned paper: {arxiv_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save paper {arxiv_id}: {e}")
            return False

    def load_cleaned_paper(
        self, arxiv_id: str, text_only: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Load cleaned paper data from storage.

        Args:
            arxiv_id: Paper identifier
            text_only: Load only text data for efficiency

        Returns:
            Paper data dictionary or None
        """
        try:
            if text_only:
                file_path = self.cleaned_text_dir / f"{arxiv_id}_text.json"
            else:
                file_path = self.cleaned_text_dir / f"{arxiv_id}.json"

            return self._load_json_file(file_path)

        except Exception as e:
            logger.error(f"Failed to load paper {arxiv_id}: {e}")
            return None

    def save_text_chunks(self, chunks: List[Dict], batch_id: str = None) -> bool:
        """
        Save text chunks for ML processing.

        Args:
            chunks: List of chunk dictionaries
            batch_id: Optional batch identifier

        Returns:
            True if successful
        """
        if not chunks:
            return True

        try:
            # Generate batch ID if not provided
            if not batch_id:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                batch_id = f"chunks_{timestamp}"

            # Prepare chunk data
            chunk_data = {
                "batch_id": batch_id,
                "created_at": datetime.now().isoformat(),
                "chunk_count": len(chunks),
                "chunks": chunks,
            }

            # Save chunks file
            chunks_file = self.chunks_dir / f"{batch_id}.json"
            success = self._save_json_file(chunk_data, chunks_file, compress=True)

            if success:
                logger.info(f"Saved {len(chunks)} chunks to batch {batch_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to save chunks: {e}")
            return False

    def load_text_chunks(
        self, batch_id: str = None, limit: int = None
    ) -> Iterator[Dict]:
        """
        Load text chunks for ML processing.

        Args:
            batch_id: Specific batch to load (loads all if None)
            limit: Maximum number of chunks to return

        Yields:
            Chunk dictionaries
        """
        chunks_loaded = 0

        try:
            if batch_id:
                # Load specific batch
                chunks_file = self.chunks_dir / f"{batch_id}.json"
                chunk_data = self._load_json_file(chunks_file)

                if chunk_data and "chunks" in chunk_data:
                    for chunk in chunk_data["chunks"]:
                        if limit and chunks_loaded >= limit:
                            break
                        yield chunk
                        chunks_loaded += 1
            else:
                # Load all chunk files
                for chunks_file in self.chunks_dir.glob("batch_*.json*"):
                    if limit and chunks_loaded >= limit:
                        break

                    chunk_data = self._load_json_file(chunks_file)
                    if chunk_data and "chunks" in chunk_data:
                        for chunk in chunk_data["chunks"]:
                            if limit and chunks_loaded >= limit:
                                break
                            yield chunk
                            chunks_loaded += 1

        except Exception as e:
            logger.error(f"Failed to load chunks: {e}")

    def create_ml_dataset(
        self,
        papers: List[Dict],
        output_file: Path = None,
        include_quality_filters: bool = True,
    ) -> Dict:
        """
        Create optimized dataset for ML phase.

        Args:
            output_file: Path to save dataset
            include_quality_filters: Filter low-quality papers

        Returns:
            Dataset statistics
        """
        logger.info("Starting ML dataset creation...")
        if not output_file:
            output_file = self.metadata_dir / "ml_dataset.json"
            logger.info(f"Output file set to: {output_file}")

        dataset = {
            "created_at": datetime.now().isoformat(),
            "papers": [],
            "chunks": [],
            "metadata": {
                "total_papers": 0,
                "total_chunks": 0,
                "quality_filtered": 0,
                "categories": set(),
                "avg_quality_score": 0.0,
            },
        }

        quality_scores = []

        try:
            # Use the provided papers directly
            logger.info(f"Processing {len(papers)} papers provided to create_ml_dataset.")

            for paper_data in papers:
                arxiv_id = paper_data.get("arxiv_id")
                quality_score = paper_data.get("quality_metrics", {}).get("overall_score", 0.0)
                
                # Apply quality filter
                if include_quality_filters and quality_score < 0.3:
                    dataset["metadata"]["quality_filtered"] += 1
                    logger.debug(f"Paper {arxiv_id} filtered out due to low quality score ({quality_score})")
                    continue

                # Add to dataset
                dataset["papers"].append({
                    "arxiv_id": arxiv_id,
                    "title": paper_data.get("title", ""),
                    "text": paper_data.get("cleaned_text", ""),
                    "categories": paper_data.get("categories", []),
                    "quality_score": quality_score,
                })
                dataset["metadata"]["categories"].update(paper_data.get("categories", []))
                quality_scores.append(quality_score)
                logger.debug(f"Added paper {arxiv_id} to dataset")

            # Load all chunks
            logger.info("Loading text chunks...")
            all_chunks = list(self.load_text_chunks())
            dataset["chunks"] = all_chunks
            logger.info(f"Loaded {len(all_chunks)} chunks.")

            # Finalize metadata
            dataset["metadata"]["total_papers"] = len(dataset["papers"])
            dataset["metadata"]["total_chunks"] = len(dataset["chunks"])
            dataset["metadata"]["categories"] = list(dataset["metadata"]["categories"])
            dataset["metadata"]["avg_quality_score"] = (
                sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
            )
            logger.info(
                f"Finalized metadata: {dataset['metadata']['total_papers']} papers, {dataset['metadata']['total_chunks']} chunks."
            )

            # Save dataset
            success = self._save_json_file(dataset, output_file, compress=True)

            if success:
                logger.info(
                    f"Created ML dataset: {dataset['metadata']['total_papers']} papers, "
                    f"{dataset['metadata']['total_chunks']} chunks"
                )

            return dataset["metadata"]

        except Exception as e:
            logger.error(f"Failed to create ML dataset: {e}", exc_info=True)
            return {}

    def get_storage_stats(self) -> Dict:
        """Get statistics about stored data."""
        stats = {
            "cleaned_papers": 0,
            "text_files": 0,
            "chunk_batches": 0,
            "total_chunks": 0,
            "storage_size_mb": 0.0,
            "categories": set(),
        }

        try:
            # Count files and sizes
            for paper_file in self.cleaned_text_dir.glob("*.json*"):
                stats["storage_size_mb"] += paper_file.stat().st_size / (1024 * 1024)

                if "_text.json" in paper_file.name:
                    stats["text_files"] += 1
                else:
                    stats["cleaned_papers"] += 1

            # Count chunks
            for chunks_file in self.chunks_dir.glob("batch_*.json*"):
                stats["chunk_batches"] += 1
                stats["storage_size_mb"] += chunks_file.stat().st_size / (1024 * 1024)

                chunk_data = self._load_json_file(chunks_file)
                if chunk_data and "chunk_count" in chunk_data:
                    stats["total_chunks"] += chunk_data["chunk_count"]

            # Sample categories
            sample_count = 0
            for paper_file in self.cleaned_text_dir.glob("*_text.json*"):
                if sample_count >= 10:  # Sample first 10 papers
                    break

                paper_data = self._load_json_file(paper_file)
                if paper_data:
                    stats["categories"].update(paper_data.get("categories", []))
                    sample_count += 1

            stats["categories"] = list(stats["categories"])
            stats["storage_size_mb"] = round(stats["storage_size_mb"], 2)

        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")

        return stats

    def cleanup_storage(self, keep_days: int = 30) -> int:
        """
        Clean up old temporary files.

        Args:
            keep_days: Keep files newer than this many days

        Returns:
            Number of files cleaned up
        """
        cleanup_count = 0
        cutoff_time = datetime.now().timestamp() - (keep_days * 24 * 3600)

        try:
            # Clean up old chunk files
            for chunks_file in self.chunks_dir.glob("batch_*.json*"):
                if chunks_file.stat().st_mtime < cutoff_time:
                    chunks_file.unlink()
                    cleanup_count += 1

            logger.info(f"Cleaned up {cleanup_count} old files")

        except Exception as e:
            logger.error(f"Failed to cleanup storage: {e}")

        return cleanup_count


# Convenience functions for integration with main pipeline


def save_processed_papers(
    papers: List[Dict], storage: CleanedTextStorage = None
) -> Dict:
    """Save multiple processed papers to storage."""
    if storage is None:
        storage = CleanedTextStorage()

    stats = {"saved": 0, "failed": 0, "errors": []}

    for paper in papers:
        try:
            if storage.save_cleaned_paper(paper):
                stats["saved"] += 1
            else:
                stats["failed"] += 1
        except Exception as e:
            stats["failed"] += 1
            stats["errors"].append(f"{paper.get('arxiv_id', 'unknown')}: {e}")

    return stats


def create_chunks_from_papers(
    papers: List[Dict], storage: CleanedTextStorage = None
) -> Dict:
    """Create and save text chunks from processed papers."""
    if storage is None:
        storage = CleanedTextStorage()

    from .section_parser import ResearchPaperSectionParser

    parser = ResearchPaperSectionParser()
    all_chunks = []
    stats = {"papers_processed": 0, "chunks_created": 0}

    for paper in papers:
        try:
            arxiv_id = paper.get("arxiv_id", "unknown")
            text = paper.get("cleaned_text", "")
            sections = paper.get("text_sections", {})

            if not text:
                continue

            # Convert sections to Section objects format expected by parser
            section_objects = {}
            for name, content in sections.items():
                # Create mock Section object
                class MockSection:
                    def __init__(self, name, content):
                        self.name = name
                        self.content = content
                        self.confidence = 0.8

                section_objects[name] = MockSection(name, content)

            # Create chunks
            chunks = parser.create_chunked_sections(section_objects)

            # Add paper metadata to chunks
            for chunk in chunks:
                chunk["arxiv_id"] = arxiv_id
                chunk["paper_title"] = paper.get("title", "")
                chunk["categories"] = paper.get("categories", [])
                chunk["quality_score"] = paper.get("quality_metrics", {}).get(
                    "overall_score", 0.0
                )

            all_chunks.extend(chunks)
            stats["papers_processed"] += 1
            stats["chunks_created"] += len(chunks)

        except Exception as e:
            logger.error(f"Failed to create chunks for {paper.get('arxiv_id')}: {e}")

    # Save chunks in batches
    batch_size = 1000
    batch_count = 0

    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i : i + batch_size]
        batch_id = f"batch_{batch_count:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if storage.save_text_chunks(batch_chunks, batch_id):
            batch_count += 1

    stats["batches_saved"] = batch_count
    return stats
