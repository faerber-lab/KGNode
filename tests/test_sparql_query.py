"""Tests for execute_sparql_query function."""

import unittest
from dotenv import load_dotenv
from kgnode import execute_sparql_query, KGConfig

# Load environment variables
load_dotenv()

# Import shared fixtures
from fixtures import get_test_config


class TestSparqlQuery(unittest.TestCase):
    """Test SPARQL query execution."""

    def setUp(self):
        """Set up test config."""
        self.config = get_test_config()

    def test_basic_select_query(self):
        """Test execute_sparql_query with basic SELECT."""
        query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 5"
        results = execute_sparql_query(query, self.config)
        self.assertIsInstance(results, list)
        if len(results) > 0:
            self.assertIsInstance(results[0], dict)
            self.assertIn("s", results[0])

    def test_query_with_filter(self):
        """Test execute_sparql_query with FILTER clause."""
        query = """
        SELECT ?entity ?label WHERE {
            ?entity <http://www.w3.org/2000/01/rdf-schema#label> ?label .
            FILTER(CONTAINS(LCASE(?label), "hinton"))
        } LIMIT 5
        """
        results = execute_sparql_query(query, self.config)
        self.assertIsInstance(results, list)

    def test_empty_results(self):
        """Test execute_sparql_query returns empty list for no matches."""
        query = """
        SELECT ?s WHERE {
            ?s <http://nonexistent/predicate/12345> "impossible_value_xyz"
        }
        """
        results = execute_sparql_query(query, self.config)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 0)


if __name__ == "__main__":
    # Setup fixtures before running standalone
    from fixtures import setup_global_fixtures, cleanup_global_fixtures

    setup_global_fixtures()
    try:
        unittest.main()
    finally:
        cleanup_global_fixtures()
