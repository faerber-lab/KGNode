"""Tests for answer generation functions."""

import unittest
from dotenv import load_dotenv
from kgnode import (
    generate_sparql,
    kg_retrieve,
    generate_answer,
    generate_answer_using_subgraph,
    get_subgraphs,
    get_entities_collection,
    KGConfig,
    SPARQLGenerationError,
    AnswerGenerationError,
)

# Load environment variables
load_dotenv()

# Import shared fixtures
from fixtures import get_test_config, get_test_chromadb


class TestGenerator(unittest.TestCase):
    """Test SPARQL and answer generation functions."""

    def setUp(self):
        """Set up test - uses shared fixtures."""
        self.config = get_test_config()
        self.collection = get_test_chromadb()

    def test_generate_sparql_returns_string(self):
        """Test generate_sparql returns valid SPARQL string."""
        # First get subgraphs
        seed_node = "https://dblp.org/pid/h/GeoffreyEHinton"
        query = "What papers has Geoffrey Hinton published?"

        subgraphs, template_text = get_subgraphs(
            seed_node=seed_node, query=query, config=self.config, max_hops=2, max_k=1
        )

        if len(subgraphs) > 0:
            # Should always return a string or raise SPARQLGenerationError
            sparql = generate_sparql(
                query=query, subgraphs=subgraphs, config=self.config, max_retries=2
            )
            self.assertIsInstance(sparql, str)
            self.assertIn("SELECT", sparql.upper())

    def test_kg_retrieve_returns_dict(self):
        """Test kg_retrieve returns dict with expected keys."""
        query = "Find publications by Geoffrey Hinton"

        result = kg_retrieve(
            query=query,
            config=self.config,
            n_seed_results=2,
            max_hops=2,
            max_k=1,
            max_sparql_retries=2,
        )

        if result is not None:
            self.assertIsInstance(result, dict)
            self.assertIn("sparql", result)
            self.assertIn("results", result)
            self.assertIn("subgraphs", result)

    def test_generate_answer_returns_dict(self):
        """Test generate_answer returns dict with answer."""
        query = "What research has Geoffrey Hinton done?"

        # Should always return a dict or raise AnswerGenerationError
        result = generate_answer(
            query=query,
            config=self.config,
            n_seed_results=2,
            max_hops=2,
            max_k=1,
            max_sparql_retries=2,
        )

        self.assertIsInstance(result, dict)
        self.assertIn("answer", result)
        self.assertIn("confidence", result)
        self.assertIsInstance(result["answer"], str)
        self.assertIsInstance(result["confidence"], (int, float))

    def test_generate_answer_using_subgraph_returns_dict(self):
        """Test generate_answer_using_subgraph with auto-extraction."""
        query = "Show Robert Schober's publications"

        # Should always return a dict or raise AnswerGenerationError
        result = generate_answer_using_subgraph(
            query=query,
            config=self.config,
            n_seed_results=2,
            max_hops=2,
            max_k=1,
        )

        self.assertIsInstance(result, dict)
        self.assertIn("answer", result)
        self.assertIn("confidence", result)
        self.assertIn("subgraphs_used", result)

    def test_generate_answer_using_subgraph_with_provided_subgraphs(self):
        """Test generate_answer_using_subgraph with pre-extracted subgraphs."""
        seed_node = "https://dblp.org/pid/95/2265"
        query = "What is Robert Schober's research about?"

        # First get subgraphs
        subgraphs, template_text = get_subgraphs(
            seed_node=seed_node, query=query, config=self.config, max_hops=2, max_k=1
        )

        if len(subgraphs) > 0:
            # Should always return a dict or raise AnswerGenerationError
            result = generate_answer_using_subgraph(
                query=query, config=self.config, subgraphs=subgraphs
            )

            self.assertIsInstance(result, dict)
            self.assertIn("answer", result)

    def test_generate_sparql_raises_exception_on_failure(self):
        """Test generate_sparql raises SPARQLGenerationError when it fails."""
        query = "Invalid query that should fail"
        # Empty subgraphs or malformed data should cause failure
        empty_subgraphs = []

        with self.assertRaises(SPARQLGenerationError):
            generate_sparql(
                query=query,
                subgraphs=empty_subgraphs,
                config=self.config,
                max_retries=1  # Low retries to fail faster
            )


if __name__ == "__main__":
    # Setup fixtures before running standalone
    from fixtures import setup_global_fixtures, cleanup_global_fixtures

    setup_global_fixtures()
    try:
        unittest.main()
    finally:
        cleanup_global_fixtures()
