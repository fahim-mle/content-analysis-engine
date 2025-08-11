# tests/test_clean/test_section_parser.py
"""Tests for section parsing functionality."""

import unittest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase_clean.section_parser import ResearchPaperSectionParser, parse_paper_sections, Section


class TestResearchPaperSectionParser(unittest.TestCase):
    """Test cases for ResearchPaperSectionParser class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parser = ResearchPaperSectionParser()
        
        # Sample academic paper text
        self.sample_paper = """
Title: Advanced Machine Learning Techniques

Abstract
This paper presents novel approaches to machine learning that improve accuracy and efficiency.
The methods show significant improvements over baseline approaches.

1. Introduction
Machine learning has become increasingly important in recent years.
This paper addresses key challenges in the field.

2. Related Work
Previous studies have explored various approaches to this problem.
Smith et al. (2023) proposed a different method.

3. Methodology
We propose a novel algorithm based on transformer architectures.
The method consists of three main components.

4. Experiments
We evaluate our approach on standard benchmarks.
Results show significant improvements.

5. Results
Our method achieves state-of-the-art performance.
Table 1 shows detailed results.

6. Discussion
The results demonstrate the effectiveness of our approach.
Several limitations should be noted.

7. Conclusion
We have presented a novel approach to machine learning.
Future work will explore extensions to other domains.

References
[1] Smith, J., Jones, M. (2023). Machine Learning Advances. Nature.
[2] Brown, A. (2022). Deep Learning Methods. ICML.
"""
    
    def test_detect_section_headers(self):
        """Test detection of section headers."""
        headers = self.parser.detect_section_headers(self.sample_paper)
        
        # Should detect major sections
        section_names = [header[0] for header in headers]
        
        self.assertIn('abstract', section_names)
        self.assertIn('introduction', section_names)
        self.assertIn('related_work', section_names)
        self.assertIn('methodology', section_names)
        self.assertIn('experiments', section_names)
        self.assertIn('results', section_names)
        self.assertIn('discussion', section_names)
        self.assertIn('conclusion', section_names)
        self.assertIn('references', section_names)
        
        # Headers should have reasonable confidence scores
        for _, _, _, confidence in headers:
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)
    
    def test_extract_sections(self):
        """Test section extraction."""
        sections = self.parser.extract_sections(self.sample_paper)
        
        # Should extract key sections
        self.assertIn('abstract', sections)
        self.assertIn('introduction', sections)
        self.assertIn('conclusion', sections)
        
        # Sections should be Section objects
        self.assertIsInstance(sections['abstract'], Section)
        
        # Sections should have content
        self.assertGreater(len(sections['abstract'].content), 20)
        self.assertIn('novel approaches', sections['abstract'].content)
        
        # Introduction should have expected content
        self.assertIn('machine learning', sections['introduction'].content.lower())
    
    def test_calculate_header_confidence(self):
        """Test header confidence calculation."""
        # Test high-confidence header
        high_conf = self.parser._calculate_header_confidence(
            "Abstract", "abstract", 5, 100  # Early in document
        )
        self.assertGreater(high_conf, 0.5)
        
        # Test numbered header
        numbered_conf = self.parser._calculate_header_confidence(
            "1. Introduction", "introduction", 10, 100
        )
        self.assertGreater(numbered_conf, 0.6)  # Should get bonus for numbering
        
        # Test low-confidence header (wrong position)
        low_conf = self.parser._calculate_header_confidence(
            "abstract", "abstract", 90, 100  # Very late in document
        )
        self.assertLess(low_conf, 0.5)
    
    def test_create_chunked_sections(self):
        """Test creation of chunked sections."""
        sections = self.parser.extract_sections(self.sample_paper)
        chunks = self.parser.create_chunked_sections(sections, chunk_size=200)
        
        # Should create chunks
        self.assertGreater(len(chunks), 0)
        
        # Each chunk should have required fields
        for chunk in chunks:
            self.assertIn('chunk_id', chunk)
            self.assertIn('section', chunk)
            self.assertIn('text', chunk)
            self.assertIn('char_count', chunk)
            self.assertIn('word_count', chunk)
            
            # Chunk size should be reasonable
            self.assertLessEqual(chunk['char_count'], 250)  # Allow some margin
            self.assertGreater(len(chunk['text']), 0)
    
    def test_get_section_summary(self):
        """Test section summary generation."""
        sections = self.parser.extract_sections(self.sample_paper)
        summary = self.parser.get_section_summary(sections)
        
        # Should have expected summary fields
        self.assertIn('total_sections', summary)
        self.assertIn('total_words', summary)
        self.assertIn('sections_found', summary)
        self.assertIn('avg_confidence', summary)
        
        # Should detect key sections
        self.assertTrue(summary['has_abstract'])
        self.assertTrue(summary['has_introduction'])
        self.assertTrue(summary['has_conclusion'])
        self.assertTrue(summary['has_references'])
        
        # Should count sections correctly
        self.assertGreater(summary['total_sections'], 5)
        self.assertGreater(summary['total_words'], 50)
    
    def test_empty_text_handling(self):
        """Test handling of empty or invalid text."""
        # Empty text
        sections = self.parser.extract_sections("")
        self.assertEqual(len(sections), 0)
        
        # None text
        sections = self.parser.extract_sections(None)
        self.assertEqual(len(sections), 0)
        
        # Very short text
        sections = self.parser.extract_sections("Short text")
        self.assertIn('body', sections)  # Should default to body section
    
    def test_section_without_clear_headers(self):
        """Test handling text without clear section headers."""
        unstructured_text = """
        This is a research paper that doesn't have clear section headers.
        It discusses machine learning and presents some results.
        The methodology is described in the middle.
        Finally, we conclude with some observations.
        """
        
        sections = self.parser.extract_sections(unstructured_text)
        
        # Should still create a body section
        self.assertIn('body', sections)
        self.assertGreater(len(sections['body'].content), 50)


class TestParsePaperSections(unittest.TestCase):
    """Test cases for parse_paper_sections function."""
    
    def test_parse_paper_sections_success(self):
        """Test successful section parsing."""
        sample_text = """
        Abstract
        This is the abstract section.
        
        Introduction
        This is the introduction section.
        """
        
        sections, summary = parse_paper_sections(sample_text)
        
        # Should return sections and summary
        self.assertIsInstance(sections, dict)
        self.assertIsInstance(summary, dict)
        
        # Should have parsed sections
        self.assertIn('abstract', sections)
        self.assertIn('introduction', sections)
    
    def test_parse_paper_sections_empty(self):
        """Test parsing empty text."""
        sections, summary = parse_paper_sections("")
        
        self.assertEqual(len(sections), 0)
        self.assertEqual(summary['total_sections'], 0)
    
    def test_parse_paper_sections_with_error(self):
        """Test error handling in section parsing."""
        # This should not raise an exception
        sections, summary = parse_paper_sections(None)
        
        # Should return fallback results
        self.assertIsInstance(sections, dict)
        self.assertIsInstance(summary, dict)


if __name__ == '__main__':
    unittest.main()