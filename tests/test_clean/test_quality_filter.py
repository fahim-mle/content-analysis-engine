# tests/test_clean/test_quality_filter.py
"""Tests for quality filtering functionality."""

import unittest
from unittest.mock import patch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase_clean.quality_filter import TextQualityFilter, filter_papers_by_quality, QualityMetrics


class TestTextQualityFilter(unittest.TestCase):
    """Test cases for TextQualityFilter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.filter = TextQualityFilter()
        
        # Sample texts for testing
        self.good_text = """
        This is a well-written academic research paper that discusses advanced machine learning 
        techniques. The paper presents a comprehensive methodology for improving natural language 
        processing systems. Our experiments demonstrate significant improvements over baseline 
        methods. The results show consistent performance gains across multiple datasets. 
        We conclude that our approach offers substantial benefits for practical applications.
        """
        
        self.short_text = "This is too short."
        
        self.gibberish_text = "a b c d e f g h i j k l m n o p q r s t u v w x y z " * 20
        
        self.non_english_text = "Este es un texto en español que no debería pasar el filtro."
    
    def test_assess_length_quality(self):
        """Test length quality assessment."""
        # Good length text
        score, issues = self.filter.assess_length_quality(self.good_text)
        self.assertGreater(score, 0.5)
        self.assertEqual(len(issues), 0)
        
        # Too short text
        score, issues = self.filter.assess_length_quality(self.short_text)
        self.assertLess(score, 0.5)
        self.assertTrue(any('short' in issue for issue in issues))
        
        # Empty text
        score, issues = self.filter.assess_length_quality("")
        self.assertEqual(score, 0.0)
        self.assertIn('text_too_short_0_chars', issues)
    
    def test_assess_readability_quality(self):
        """Test readability quality assessment."""
        # Good readability
        score, issues = self.filter.assess_readability_quality(self.good_text)
        self.assertGreater(score, 0.6)
        
        # Gibberish text
        score, issues = self.filter.assess_readability_quality(self.gibberish_text)
        self.assertLess(score, 0.7)
        self.assertTrue(any('gibberish' in issue for issue in issues))
        
        # Empty text
        score, issues = self.filter.assess_readability_quality("")
        self.assertEqual(score, 0.0)
        self.assertIn('empty_text', issues)
    
    def test_assess_structure_quality(self):
        """Test structure quality assessment."""
        # Text with academic indicators
        academic_text = """
        This study proposes a novel method for analyzing data. We evaluate our approach 
        using experiments on benchmark datasets. The results demonstrate significant 
        improvements over existing techniques. Our methodology is based on recent 
        advances in machine learning research.
        """
        
        score, issues = self.filter.assess_structure_quality(academic_text)
        self.assertGreater(score, 0.5)
        
        # Text with sections
        sections = {
            'abstract': 'This is the abstract.',
            'introduction': 'This is the introduction.',
            'methodology': 'This is the methodology.',
            'results': 'These are the results.',
            'conclusion': 'This is the conclusion.'
        }
        
        score, issues = self.filter.assess_structure_quality(academic_text, sections)
        self.assertGreater(score, 0.7)  # Should get bonus for having sections
    
    def test_assess_completeness_quality(self):
        """Test completeness quality assessment."""
        # Complete text ending properly
        complete_text = "This is a complete research paper that ends properly."
        score, issues = self.filter.assess_completeness_quality(complete_text)
        self.assertGreater(score, 0.8)
        
        # Text with truncation indicator
        truncated_text = "This text is truncated... [truncated]"
        score, issues = self.filter.assess_completeness_quality(truncated_text)
        self.assertLess(score, 0.5)
        self.assertTrue(any('truncation_indicator' in issue for issue in issues))
        
        # Very short text
        short_text = "Too short."
        score, issues = self.filter.assess_completeness_quality(short_text)
        self.assertLess(score, 0.5)
        self.assertIn('likely_incomplete', issues)
    
    def test_assess_language_match(self):
        """Test language matching assessment."""
        # English text
        score, issues = self.filter.assess_language_match(self.good_text, 'en')
        self.assertGreater(score, 0.8)
        self.assertEqual(len(issues), 0)
        
        # Non-English text (simulated)
        score, issues = self.filter.assess_language_match(self.non_english_text, 'en')
        self.assertLess(score, 0.8)  # May still pass if it has some English words
    
    def test_filter_by_categories(self):
        """Test category filtering."""
        # Target categories
        target_cats = ['cs.AI', 'cs.CL']
        should_include, issues = self.filter.filter_by_categories(target_cats)
        self.assertTrue(should_include)
        self.assertEqual(len(issues), 0)
        
        # Unrelated categories
        unrelated_cats = ['physics.gen-ph', 'math.CO']
        should_include, issues = self.filter.filter_by_categories(unrelated_cats)
        self.assertFalse(should_include)
        self.assertTrue(any('unrelated_categories' in issue for issue in issues))
        
        # No categories
        should_include, issues = self.filter.filter_by_categories([])
        self.assertFalse(should_include)
        self.assertIn('no_categories', issues)
    
    def test_comprehensive_quality_assessment(self):
        """Test comprehensive quality assessment."""
        quality_metrics = self.filter.comprehensive_quality_assessment(self.good_text)
        
        # Should return QualityMetrics object
        self.assertIsInstance(quality_metrics, QualityMetrics)
        
        # Should have all required scores
        self.assertGreaterEqual(quality_metrics.length_score, 0.0)
        self.assertGreaterEqual(quality_metrics.readability_score, 0.0)
        self.assertGreaterEqual(quality_metrics.structure_score, 0.0)
        self.assertGreaterEqual(quality_metrics.completeness_score, 0.0)
        self.assertGreaterEqual(quality_metrics.language_score, 0.0)
        self.assertGreaterEqual(quality_metrics.overall_score, 0.0)
        
        # Good text should have high overall score
        self.assertGreater(quality_metrics.overall_score, 0.5)
        
        # Should have issue and recommendation lists
        self.assertIsInstance(quality_metrics.issues, list)
        self.assertIsInstance(quality_metrics.recommendations, list)
    
    def test_should_include_paper(self):
        """Test paper inclusion decision."""
        # High quality paper
        good_metrics = self.filter.comprehensive_quality_assessment(self.good_text)
        should_include, reason = self.filter.should_include_paper(good_metrics, ['cs.AI'])
        self.assertTrue(should_include)
        self.assertIn('Passed', reason)
        
        # Low quality paper
        poor_metrics = self.filter.comprehensive_quality_assessment(self.short_text)
        should_include, reason = self.filter.should_include_paper(poor_metrics, ['cs.AI'])
        self.assertFalse(should_include)
        self.assertIn('Quality score', reason)
        
        # Wrong category
        should_include, reason = self.filter.should_include_paper(good_metrics, ['physics.gen-ph'])
        self.assertFalse(should_include)
        self.assertIn('Category filter', reason)


class TestFilterPapersByQuality(unittest.TestCase):
    """Test cases for filter_papers_by_quality function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_papers = [
            {
                'arxiv_id': 'good_paper_1',
                'cleaned_text': """This is a high-quality research paper with substantial content. 
                                 The methodology is clearly described and the experiments are comprehensive. 
                                 Results demonstrate significant improvements over baseline methods.""",
                'categories': ['cs.AI', 'cs.CL'],
                'text_sections': {'abstract': 'Good abstract', 'introduction': 'Good intro'}
            },
            {
                'arxiv_id': 'short_paper_1',
                'cleaned_text': 'Too short.',
                'categories': ['cs.AI'],
                'text_sections': {}
            },
            {
                'arxiv_id': 'wrong_category_1',
                'cleaned_text': 'This is a decent paper but in the wrong category.',
                'categories': ['physics.gen-ph'],
                'text_sections': {}
            },
            {
                'arxiv_id': 'good_paper_2',
                'cleaned_text': """Another high-quality paper discussing machine learning techniques. 
                                 The approach is novel and the evaluation is thorough. The results 
                                 show consistent improvements across multiple datasets.""",
                'categories': ['cs.LG'],
                'text_sections': {'abstract': 'Good abstract', 'methodology': 'Good methods'}
            }
        ]
    
    @patch('phase_clean.quality_filter.logger')
    def test_filter_papers_success(self, mock_logger):
        """Test successful paper filtering."""
        filtered_papers, stats = filter_papers_by_quality(self.sample_papers)
        
        # Should filter out low-quality papers
        self.assertLess(len(filtered_papers), len(self.sample_papers))
        
        # Should have statistics
        self.assertIn('total', stats)
        self.assertIn('accepted', stats)
        self.assertIn('rejected', stats)
        self.assertIn('acceptance_rate', stats)
        self.assertIn('rejection_reasons', stats)
        
        # All accepted papers should have quality metrics
        for paper in filtered_papers:
            self.assertIn('quality_metrics', paper)
            self.assertIn('overall_score', paper['quality_metrics'])
            self.assertGreaterEqual(paper['quality_metrics']['overall_score'], 0.3)
    
    def test_filter_empty_list(self):
        """Test filtering empty paper list."""
        filtered_papers, stats = filter_papers_by_quality([])
        
        self.assertEqual(len(filtered_papers), 0)
        self.assertEqual(stats['total'], 0)
        self.assertEqual(stats['accepted'], 0)
    
    @patch('phase_clean.quality_filter.logger')
    def test_filter_with_custom_filter(self, mock_logger):
        """Test filtering with custom quality filter."""
        # Create a very strict filter
        strict_filter = TextQualityFilter()
        strict_filter.min_quality_score = 0.8  # Very high threshold
        
        filtered_papers, stats = filter_papers_by_quality(self.sample_papers, strict_filter)
        
        # Should accept fewer papers with strict filter
        self.assertLessEqual(len(filtered_papers), len(self.sample_papers))
        self.assertLessEqual(stats['acceptance_rate'], 1.0)


if __name__ == '__main__':
    unittest.main()