# tests/test_crawl/test_fetch.py
"""Tests ArXiv API query and parsing."""

import unittest
from unittest.mock import patch, MagicMock

from phase_crawl.fetch_papers import build_query, fetch_papers


class TestFetchPapers(unittest.TestCase):
    """Test suite for paper fetching functionality."""
    
    def test_build_query_with_categories(self):
        """Test query building with specific categories."""
        categories = ["cs.AI", "cs.LG"]
        max_results = 50
        days_back = 30
        
        url = build_query(categories, max_results, days_back)
        
        self.assertIn("cat:cs.AI", url)
        self.assertIn("cat:cs.LG", url)
        self.assertIn("max_results=50", url)
        
    def test_build_query_no_categories(self):
        """Test query building without categories."""
        url = build_query(None, 10, 7)
        self.assertIn("search_query=all", url)
        
    @patch('phase_crawl.fetch_papers.requests.get')
    @patch('phase_crawl.fetch_papers.feedparser.parse')
    def test_fetch_papers_success(self, mock_parse, mock_get):
        """Test successful paper fetching."""
        # Mock response
        mock_response = MagicMock()
        mock_response.text = "<xml>mock response</xml>"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Mock feedparser
        mock_entry = MagicMock()
        mock_entry.id = "http://arxiv.org/abs/2024.12345v1"
        mock_entry.title = "Test Paper"
        mock_entry.summary = "Test summary"
        mock_entry.published = "2024-01-01T00:00:00Z"
        mock_entry.updated = "2024-01-01T00:00:00Z"
        mock_entry.get.return_value = []
        
        mock_feed = MagicMock()
        mock_feed.entries = [mock_entry]
        mock_parse.return_value = mock_feed
        
        papers = fetch_papers("http://test-url.com")
        
        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0]["arxiv_id"], "2024.12345v1")


if __name__ == "__main__":
    unittest.main()