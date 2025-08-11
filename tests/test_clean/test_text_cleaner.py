# tests/test_clean/test_text_cleaner.py
"""Tests for text cleaning functionality."""

import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase_clean.text_cleaner import ResearchPaperCleaner, process_paper_text


class TestResearchPaperCleaner(unittest.TestCase):
    """Test cases for ResearchPaperCleaner class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cleaner = ResearchPaperCleaner()
        
        # Sample problematic text
        self.sample_text = """
Effective Training Data Synthesis for Improving MLLM Chart Understanding
Yuwei Yang1, Zeyu Zhang1, Yunzhong Hou1
1Australian National University


arXiv:2508.06492v1 [cs.AI] 12 Aug 2025


1

Abstract

This paper presents a compre-
hensive approach to train- ing data synthesis for multi-
modal large language models.


    1   Introduction

Recent advances in machine learn-
ing have shown that...

2

Figure 1: Sample chart

References

[1] Smith, J. (2024). "Machine Learning Advances." 
    Nature Machine Intelligence, 1(1):1-10.
"""
    
    def test_normalize_unicode(self):
        """Test Unicode normalization."""
        text_with_unicode = "This is a "test" with various—characters…"
        normalized = self.cleaner.normalize_unicode(text_with_unicode)
        
        self.assertNotIn('"', normalized)
        self.assertNotIn('"', normalized)
        self.assertNotIn('—', normalized)
        self.assertNotIn('…', normalized)
        self.assertIn('"test"', normalized)
        self.assertIn('-', normalized)
        self.assertIn('...', normalized)
    
    def test_remove_artifacts(self):
        """Test removal of PDF artifacts."""
        result = self.cleaner.remove_artifacts(self.sample_text)
        
        # Should remove arXiv identifier
        self.assertNotIn('arXiv:2508.06492v1', result)
        
        # Should remove page numbers (standalone numbers)
        lines = result.split('\n')
        standalone_numbers = [line.strip() for line in lines if line.strip().isdigit()]
        self.assertEqual(len(standalone_numbers), 0)
    
    def test_fix_hyphenation(self):
        """Test fixing hyphenated words across lines."""
        hyphenated_text = "This is a compre-\nhensive approach to train-\ning data"
        fixed = self.cleaner.fix_hyphenation(hyphenated_text)
        
        self.assertIn('comprehensive', fixed)
        self.assertIn('training', fixed)
        self.assertNotIn('compre-\nhensive', fixed)
    
    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        messy_text = "This   has    excessive\n\n\n\nwhitespace"
        normalized = self.cleaner.normalize_whitespace(messy_text)
        
        # Should not have excessive spaces
        self.assertNotIn('   ', normalized)
        # Should not have excessive newlines
        self.assertNotIn('\n\n\n', normalized)
    
    def test_assess_quality(self):
        """Test text quality assessment."""
        # Good quality text
        good_text = "This is a well-written academic paper with sufficient content and proper structure."
        quality = self.cleaner.assess_quality(good_text)
        
        self.assertGreater(quality['score'], 0.5)
        self.assertGreater(quality['word_count'], 0)
        self.assertGreater(quality['length'], 0)
        
        # Poor quality text
        poor_text = "abc def"
        poor_quality = self.cleaner.assess_quality(poor_text)
        self.assertLess(poor_quality['score'], 0.5)
        self.assertIn('too_short', poor_quality['issues'])
    
    def test_clean_text_pipeline(self):
        """Test complete text cleaning pipeline."""
        cleaned_text, quality = self.cleaner.clean_text(self.sample_text)
        
        # Should not be empty
        self.assertGreater(len(cleaned_text), 0)
        
        # Should have quality metrics
        self.assertIn('score', quality)
        self.assertIn('issues', quality)
        self.assertIn('length', quality)
        self.assertIn('word_count', quality)
        
        # Should remove artifacts but preserve content
        self.assertNotIn('arXiv:2508', cleaned_text)
        self.assertIn('Abstract', cleaned_text)
        self.assertIn('Introduction', cleaned_text)
    
    def test_extract_and_clean_sections(self):
        """Test section extraction."""
        sections = self.cleaner.extract_and_clean_sections(self.sample_text)
        
        # Should identify key sections
        self.assertIn('abstract', sections)
        self.assertIn('introduction', sections)
        self.assertIn('references', sections)
        
        # Sections should have content
        self.assertGreater(len(sections['abstract']), 10)
        self.assertGreater(len(sections['introduction']), 10)


class TestProcessPaperText(unittest.TestCase):
    """Test cases for process_paper_text function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_paper = {
            'arxiv_id': '2508.06492v1',
            'title': 'Test Paper',
            'full_text': 'This is a sample research paper with some content.'
        }
    
    @patch('phase_clean.text_cleaner.logger')
    def test_process_paper_with_text(self, mock_logger):
        """Test processing paper with valid text."""
        result = process_paper_text(self.sample_paper.copy())
        
        # Should have cleaned text
        self.assertIn('cleaned_text', result)
        self.assertIn('text_quality', result)
        self.assertIn('text_sections', result)
        
        # Should preserve original data
        self.assertEqual(result['arxiv_id'], '2508.06492v1')
        self.assertEqual(result['title'], 'Test Paper')
    
    @patch('phase_clean.text_cleaner.logger')
    def test_process_paper_without_text(self, mock_logger):
        """Test processing paper without text."""
        paper_no_text = {'arxiv_id': 'test123', 'title': 'No Text Paper'}
        result = process_paper_text(paper_no_text)
        
        # Should handle missing text gracefully
        self.assertEqual(result['cleaned_text'], '')
        self.assertIn('no_text', result['text_quality']['issues'])
        mock_logger.warning.assert_called()
    
    @patch('phase_clean.text_cleaner.logger')
    def test_process_paper_with_error(self, mock_logger):
        """Test error handling in paper processing."""
        # Create a paper that will cause an error
        problematic_paper = {
            'arxiv_id': 'error_test',
            'full_text': None  # This should cause an error
        }
        
        result = process_paper_text(problematic_paper)
        
        # Should handle error gracefully
        self.assertIn('processing_error', result['text_quality']['issues'])
        mock_logger.error.assert_called()


if __name__ == '__main__':
    unittest.main()