"""Tests for subgraph extraction function."""

import unittest
from dotenv import load_dotenv
from kgnode import get_subgraphs, KGConfig

# Load environment variables
load_dotenv()

# Import shared fixtures
from fixtures import get_test_config


class TestSubgraphExtraction(unittest.TestCase):
    """Test subgraph extraction using path-aware Markov chain."""

    def setUp(self):
        """Set up test config."""
        self.config = get_test_config()

    def test_get_subgraphs_returns_list(self):
        """Test get_subgraphs returns list of subgraph dicts."""
        # Use a known DBLP entity
        seed_node = "https://dblp.org/pid/h/GeoffreyEHinton"
        query = "What papers has Geoffrey Hinton written?"

        subgraphs, template_text = get_subgraphs(
            seed_node=seed_node,
            query=query,
            config=self.config,
            max_hops=3,  # Small for faster testing
            max_k=2,
        )
        self.assertIsInstance(subgraphs, list)
        self.assertIsInstance(template_text, str)

    def test_get_subgraphs_structure(self):
        """Test get_subgraphs returns correct structure."""
        seed_node = "https://dblp.org/pid/95/2265"
        query = "Show publications by Robert Schober"

        subgraphs, template_text = get_subgraphs(
            seed_node=seed_node, query=query, config=self.config, max_hops=2, max_k=2
        )
        self.assertIsInstance(subgraphs, list)
        self.assertIsInstance(template_text, str)

        if len(subgraphs) > 0:
            subgraph = subgraphs[0]
            self.assertIn("triplet_uris", subgraph)
            self.assertIn("path_with_label", subgraph)
            self.assertIn("probability", subgraph)
            self.assertIn("path_depth", subgraph)
            self.assertIsInstance(subgraph["triplet_uris"], list)
            self.assertIsInstance(subgraph["path_with_label"], list)
            self.assertIsInstance(subgraph["probability"], (int, float))
            self.assertIsInstance(subgraph["path_depth"], int)

    def test_get_subgraphs_with_different_query(self):
        """Test get_subgraphs with different query type."""
        seed_node = "https://dblp.org/pid/b/YoshuaBengio"
        query = "Find collaborators and venues"

        subgraphs, template_text = get_subgraphs(
            seed_node=seed_node, query=query, config=self.config, max_hops=2, max_k=1
        )
        self.assertIsInstance(subgraphs, list)
        self.assertIsInstance(template_text, str)


if __name__ == "__main__":
    # Setup fixtures before running standalone
    from fixtures import setup_global_fixtures, cleanup_global_fixtures

    setup_global_fixtures()
    try:
        unittest.main()
    finally:
        cleanup_global_fixtures()
