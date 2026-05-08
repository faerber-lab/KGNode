"""Tests for subgraph validation function."""

import unittest
from dotenv import load_dotenv
from kgnode import validate_subgraph, get_subgraphs, KGConfig

# Load environment variables
load_dotenv()

# Import shared fixtures
from fixtures import get_test_config


class TestValidator(unittest.TestCase):
    """Test subgraph validation against ground truth SPARQL."""

    def setUp(self):
        """Set up test config."""
        self.config = get_test_config()

    def test_validate_subgraph_returns_bool(self):
        """Test validate_subgraph returns boolean."""
        # First get a subgraph
        seed_node = "https://dblp.org/pid/h/GeoffreyEHinton"
        query = "What are Geoffrey Hinton's publications?"

        subgraphs, template_text = get_subgraphs(
            seed_node=seed_node, query=query, config=self.config, max_hops=2, max_k=1
        )

        if len(subgraphs) > 0:
            # Create a simple SPARQL query as ground truth
            answer_sparql = f"""
            SELECT ?pub WHERE {{
                <{seed_node}> ?rel ?pub .
            }} LIMIT 5
            """

            is_valid = validate_subgraph(
                subgraph=subgraphs[0],
                answer_sparql=answer_sparql,
                config=self.config,
            )
            self.assertIsInstance(is_valid, bool)

    def test_validate_subgraph_with_different_query(self):
        """Test validate_subgraph with author query."""
        seed_node = "https://dblp.org/pid/95/2265"
        query = "Show Robert Schober's publications"

        subgraphs, template_text = get_subgraphs(
            seed_node=seed_node, query=query, config=self.config, max_hops=2, max_k=1
        )

        if len(subgraphs) > 0:
            answer_sparql = f"""
            SELECT ?o WHERE {{
                <{seed_node}> ?p ?o .
            }} LIMIT 10
            """

            is_valid = validate_subgraph(
                subgraph=subgraphs[0],
                answer_sparql=answer_sparql,
                config=self.config,
            )
            self.assertIsInstance(is_valid, bool)


if __name__ == "__main__":
    # Setup fixtures before running standalone
    from fixtures import setup_global_fixtures, cleanup_global_fixtures

    setup_global_fixtures()
    try:
        unittest.main()
    finally:
        cleanup_global_fixtures()
