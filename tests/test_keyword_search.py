"""Tests for keyword search function."""

import unittest
from dotenv import load_dotenv
from kgnode import search_entities_by_keywords, KGConfig

# Load environment variables
load_dotenv()

# Import shared fixtures
from fixtures import get_test_config


class TestKeywordSearch(unittest.TestCase):
    """Test keyword-based entity search."""

    def setUp(self):
        """Set up test config."""
        self.config = get_test_config()

    def test_single_keyword(self):
        """Test search_entities_by_keywords with single keyword."""
        results = search_entities_by_keywords(["hinton"], limit=5, config=self.config)
        self.assertIsInstance(results, list)
        if len(results) > 0:
            self.assertIsInstance(results[0], dict)
            self.assertIn("entity", results[0])
            self.assertIn("label", results[0])

    def test_multiple_keywords(self):
        """Test search_entities_by_keywords with multiple keywords."""
        results = search_entities_by_keywords(
            ["neural", "network"], limit=10, config=self.config
        )
        self.assertIsInstance(results, list)
        if len(results) > 0:
            self.assertIn("entity", results[0])

    def test_limit_parameter(self):
        """Test search_entities_by_keywords respects limit."""
        results = search_entities_by_keywords(["learning"], limit=3, config=self.config)
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 3)

    def test_no_results(self):
        """Test search_entities_by_keywords with non-existent keywords."""
        results = search_entities_by_keywords(
            ["xyzabc123nonexistent"], limit=5, config=self.config
        )
        self.assertIsInstance(results, list)


if __name__ == "__main__":
    # Setup fixtures before running standalone
    from fixtures import setup_global_fixtures, cleanup_global_fixtures

    setup_global_fixtures()
    try:
        unittest.main()
    finally:
        cleanup_global_fixtures()
