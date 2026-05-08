"""Tests for seed node finding functions."""

import unittest

from dotenv import load_dotenv

from kgnode import SearchMode, citable, get_seed_nodes
from kgnode.seed_finder import _fuzzy_match_score

# Load environment variables
load_dotenv()

# Import shared fixtures
from fixtures import get_test_chromadb, get_test_config


class TestSeedFinderUnit(unittest.TestCase):
    """Unit tests for seed finder - no external dependencies."""

    def test_fuzzy_match_score_exact_match(self):
        """Test fuzzy match returns 1.0 for identical strings."""
        score = _fuzzy_match_score("Geoffrey Hinton", "Geoffrey Hinton")
        self.assertEqual(score, 1.0)

    def test_fuzzy_match_score_case_insensitive(self):
        """Test fuzzy match is case insensitive."""
        score = _fuzzy_match_score("GEOFFREY HINTON", "geoffrey hinton")
        self.assertEqual(score, 1.0)

    def test_fuzzy_match_score_partial_match(self):
        """Test fuzzy match returns score between 0 and 1 for partial match."""
        score = _fuzzy_match_score("Geoffrey Hinton", "Geoffrey")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_fuzzy_match_score_no_match(self):
        """Test fuzzy match returns low score for unrelated strings."""
        score = _fuzzy_match_score("Geoffrey Hinton", "ICML 2024")
        self.assertLess(score, 0.3)

    def test_search_mode_enum_values(self):
        """Test SearchMode enum has correct values."""
        self.assertEqual(SearchMode.hybrid.value, "hybrid")
        self.assertEqual(SearchMode.keyword.value, "keyword")
        self.assertEqual(SearchMode.semantic.value, "semantic")

    def test_search_mode_enum_string_equality(self):
        """Test SearchMode enum works with string comparison."""
        self.assertEqual(SearchMode.hybrid, "hybrid")
        self.assertEqual(SearchMode.semantic, "semantic")


class TestSeedFinder(unittest.TestCase):
    """Test seed node finding and citability check."""

    def setUp(self):
        """Set up test - uses shared fixtures."""
        self.config = get_test_config()
        self.collection = get_test_chromadb()

    def test_get_seed_nodes_returns_list(self):
        """Test get_seed_nodes returns list of dicts."""
        results, extracted_entities = get_seed_nodes(
            query="papers by Geoffrey Hinton about neural networks",
            n_results=2,
            config=self.config,
        )
        self.assertIsInstance(results, list)
        self.assertIsInstance(extracted_entities, list)
        if len(results) > 0:
            self.assertIsInstance(results[0], dict)
            self.assertIn("entity_uri", results[0])
            self.assertIn("label", results[0])
            self.assertIn("source", results[0])

    def test_get_seed_nodes_with_different_query(self):
        """Test get_seed_nodes with academic query."""
        results, extracted_entities = get_seed_nodes(
            query="Find publications about deep learning in ICML",
            n_results=3,
            config=self.config,
        )
        self.assertIsInstance(results, list)
        self.assertIsInstance(extracted_entities, list)

    def test_citable_returns_tuple(self):
        """Test citable returns (bool, dict) tuple."""
        is_citable, entity_nodes = citable(
            query="What is Geoffrey Hinton's research about?", config=self.config
        )
        self.assertIsInstance(is_citable, bool)
        self.assertIsInstance(entity_nodes, dict)

    def test_citable_with_specific_author(self):
        """Test citable with specific author query."""
        is_citable, entity_nodes = citable(
            query="Robert Schober publications", config=self.config
        )
        self.assertIsInstance(is_citable, bool)
        self.assertIsInstance(entity_nodes, dict)

    def test_get_seed_nodes_semantic_mode(self):
        """Test get_seed_nodes with semantic-only mode returns semantic sources."""
        results, extracted_entities = get_seed_nodes(
            query="neural networks research",
            n_results=3,
            config=self.config,
            search_mode=SearchMode.semantic,
        )
        self.assertIsInstance(results, list)
        self.assertIsInstance(extracted_entities, list)
        # All results should be from semantic search
        for result in results:
            self.assertEqual(result["source"], "semantic")


if __name__ == "__main__":
    # Setup fixtures before running standalone
    from fixtures import cleanup_global_fixtures, setup_global_fixtures

    setup_global_fixtures()
    try:
        unittest.main()
    finally:
        cleanup_global_fixtures()
