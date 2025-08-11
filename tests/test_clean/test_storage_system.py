# tests/test_clean/test_storage_system.py
"""Tests for storage system functionality."""

import unittest
import tempfile
import shutil
import json
import gzip
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase_clean.storage_system import CleanedTextStorage, save_processed_papers, create_chunks_from_papers


class TestCleanedTextStorage(unittest.TestCase):
    """Test cases for CleanedTextStorage class."""
    
    def setUp(self):
        """Set up test fixtures with temporary directory."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.storage = CleanedTextStorage(self.temp_dir)
        
        # Sample paper data
        self.sample_paper = {
            'arxiv_id': '2024.12345v1',
            'title': 'Test Paper on Machine Learning',
            'authors': ['John Doe', 'Jane Smith'],
            'categories': ['cs.AI', 'cs.LG'],
            'published': '2024-08-11',
            'updated': '2024-08-11',
            'summary': 'This paper presents novel ML techniques.',
            'full_text': 'Original full text content.',
            'cleaned_text': 'Cleaned text content for ML processing.',
            'text_quality': {'score': 0.85, 'issues': [], 'length': 100, 'word_count': 20},
            'text_sections': {'abstract': 'Abstract content', 'introduction': 'Intro content'},
            'citations': [{'id': 1, 'authors': ['Smith'], 'year': 2023}],
            'citation_stats': {'total_citations': 1, 'valid_citations': 1},
            'quality_metrics': {'overall_score': 0.85, 'issues': []}
        }
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_and_load_cleaned_paper(self):
        """Test saving and loading cleaned paper data."""
        # Save paper
        success = self.storage.save_cleaned_paper(self.sample_paper)
        self.assertTrue(success)
        
        # Check files were created
        paper_file = self.storage.cleaned_text_dir / "2024.12345v1.json.gz"
        text_file = self.storage.cleaned_text_dir / "2024.12345v1_text.json.gz"
        self.assertTrue(paper_file.exists())
        self.assertTrue(text_file.exists())
        
        # Load full paper data
        loaded_paper = self.storage.load_cleaned_paper('2024.12345v1')
        self.assertIsNotNone(loaded_paper)
        self.assertEqual(loaded_paper['arxiv_id'], '2024.12345v1')
        self.assertEqual(loaded_paper['title'], 'Test Paper on Machine Learning')
        self.assertIn('processing_metadata', loaded_paper)
        
        # Load text-only data
        text_only = self.storage.load_cleaned_paper('2024.12345v1', text_only=True)
        self.assertIsNotNone(text_only)
        self.assertEqual(text_only['arxiv_id'], '2024.12345v1')
        self.assertIn('cleaned_text', text_only)
        # Text-only should not have full metadata
        self.assertNotIn('citations', text_only)
    
    def test_save_paper_without_arxiv_id(self):
        """Test saving paper without arxiv_id fails gracefully."""
        paper_no_id = self.sample_paper.copy()
        del paper_no_id['arxiv_id']
        
        success = self.storage.save_cleaned_paper(paper_no_id)
        self.assertFalse(success)
    
    def test_save_text_chunks(self):
        """Test saving and loading text chunks."""
        # Sample chunks
        chunks = [
            {
                'chunk_id': 0,
                'section': 'abstract',
                'text': 'This is the abstract content.',
                'char_count': 30,
                'word_count': 6,
                'arxiv_id': '2024.12345v1'
            },
            {
                'chunk_id': 1,
                'section': 'introduction',
                'text': 'This is the introduction content.',
                'char_count': 35,
                'word_count': 6,
                'arxiv_id': '2024.12345v1'
            }
        ]
        
        # Save chunks
        success = self.storage.save_text_chunks(chunks, 'test_batch_001')
        self.assertTrue(success)
        
        # Check file was created
        chunks_file = self.storage.chunks_dir / "test_batch_001.json.gz"
        self.assertTrue(chunks_file.exists())
        
        # Load chunks
        loaded_chunks = list(self.storage.load_text_chunks('test_batch_001'))
        self.assertEqual(len(loaded_chunks), 2)
        self.assertEqual(loaded_chunks[0]['text'], 'This is the abstract content.')
        self.assertEqual(loaded_chunks[1]['section'], 'introduction')
    
    def test_load_text_chunks_with_limit(self):
        """Test loading text chunks with limit."""
        # Create multiple chunks
        chunks = [{'chunk_id': i, 'text': f'Chunk {i}'} for i in range(10)]
        self.storage.save_text_chunks(chunks, 'limit_test')
        
        # Load with limit
        limited_chunks = list(self.storage.load_text_chunks('limit_test', limit=3))
        self.assertEqual(len(limited_chunks), 3)
    
    def test_create_ml_dataset(self):
        """Test ML dataset creation."""
        # Save some papers first
        papers = [self.sample_paper.copy() for _ in range(3)]
        for i, paper in enumerate(papers):
            paper['arxiv_id'] = f'test_{i}'
            self.storage.save_cleaned_paper(paper)
        
        # Create some chunks
        chunks = [{'chunk_id': i, 'text': f'Chunk {i}'} for i in range(5)]
        self.storage.save_text_chunks(chunks, 'ml_test')
        
        # Create ML dataset
        dataset_file = self.temp_dir / "test_dataset.json"
        stats = self.storage.create_ml_dataset(dataset_file)
        
        # Check statistics
        self.assertIn('total_papers', stats)
        self.assertIn('total_chunks', stats)
        self.assertGreater(stats['total_papers'], 0)
        
        # Check file was created
        self.assertTrue(dataset_file.with_suffix('.json.gz').exists())
    
    def test_get_storage_stats(self):
        """Test storage statistics."""
        # Add some data
        self.storage.save_cleaned_paper(self.sample_paper)
        chunks = [{'chunk_id': i, 'text': f'Chunk {i}'} for i in range(3)]
        self.storage.save_text_chunks(chunks, 'stats_test')
        
        # Get stats
        stats = self.storage.get_storage_stats()
        
        # Should have expected fields
        self.assertIn('cleaned_papers', stats)
        self.assertIn('text_files', stats)
        self.assertIn('chunk_batches', stats)
        self.assertIn('storage_size_mb', stats)
        
        # Should count files correctly
        self.assertGreater(stats['cleaned_papers'], 0)
        self.assertGreater(stats['text_files'], 0)
        self.assertGreater(stats['chunk_batches'], 0)
        self.assertGreater(stats['storage_size_mb'], 0)
    
    def test_cleanup_storage(self):
        """Test storage cleanup."""
        # Create some test files
        old_file = self.storage.chunks_dir / "chunks_old.json"
        old_file.write_text('{"old": "data"}')
        
        # Mock file modification time to be old
        import os
        import time
        old_time = time.time() - (40 * 24 * 3600)  # 40 days ago
        os.utime(old_file, (old_time, old_time))
        
        # Cleanup (keep 30 days)
        cleaned_count = self.storage.cleanup_storage(keep_days=30)
        
        # Should have cleaned up old file
        self.assertGreater(cleaned_count, 0)
        self.assertFalse(old_file.exists())
    
    def test_json_compression(self):
        """Test JSON compression functionality."""
        test_data = {'test': 'data', 'numbers': [1, 2, 3, 4, 5]}
        test_file = self.temp_dir / "test.json"
        
        # Save with compression
        success = self.storage._save_json_file(test_data, test_file, compress=True)
        self.assertTrue(success)
        
        # Check compressed file exists
        compressed_file = test_file.with_suffix('.json.gz')
        self.assertTrue(compressed_file.exists())
        
        # Load compressed file
        loaded_data = self.storage._load_json_file(test_file, compressed=True)
        self.assertEqual(loaded_data, test_data)


class TestStorageHelperFunctions(unittest.TestCase):
    """Test cases for storage helper functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        self.sample_papers = [
            {
                'arxiv_id': 'helper_test_1',
                'cleaned_text': 'Content 1',
                'title': 'Paper 1'
            },
            {
                'arxiv_id': 'helper_test_2',
                'cleaned_text': 'Content 2',
                'title': 'Paper 2'
            }
        ]
    
    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('phase_clean.storage_system.logger')
    def test_save_processed_papers(self, mock_logger):
        """Test save_processed_papers function."""
        storage = CleanedTextStorage(self.temp_dir)
        
        stats = save_processed_papers(self.sample_papers, storage)
        
        # Should have statistics
        self.assertIn('saved', stats)
        self.assertIn('failed', stats)
        self.assertIn('errors', stats)
        
        # Should save papers successfully
        self.assertEqual(stats['saved'], 2)
        self.assertEqual(stats['failed'], 0)
    
    @patch('phase_clean.storage_system.logger')
    def test_create_chunks_from_papers(self, mock_logger):
        """Test create_chunks_from_papers function."""
        storage = CleanedTextStorage(self.temp_dir)
        
        # Add section data to papers
        papers_with_sections = []
        for paper in self.sample_papers:
            paper_copy = paper.copy()
            paper_copy['text_sections'] = {
                'abstract': 'Abstract content',
                'introduction': 'Introduction content'
            }
            papers_with_sections.append(paper_copy)
        
        stats = create_chunks_from_papers(papers_with_sections, storage)
        
        # Should have statistics
        self.assertIn('papers_processed', stats)
        self.assertIn('chunks_created', stats)
        self.assertIn('batches_saved', stats)
        
        # Should process papers and create chunks
        self.assertEqual(stats['papers_processed'], 2)
        self.assertGreater(stats['chunks_created'], 0)


if __name__ == '__main__':
    unittest.main()