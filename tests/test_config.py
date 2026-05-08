"""Tests for KGConfig class."""

import unittest
import os
from dotenv import load_dotenv
from kgnode.core.kg_config import KGConfig

# Load environment variables
load_dotenv()

# Import shared fixtures
from fixtures import get_test_config


class TestKGConfig(unittest.TestCase):
    """Test KGConfig configuration class."""

    def test_default_config(self):
        """Test KGConfig.default() creates valid config."""
        config = get_test_config()
        self.assertIsNotNone(config)
        self.assertIsNotNone(config.sparql_endpoint)
        self.assertIsNotNone(config.embedding_model)

    def test_custom_config(self):
        """Test KGConfig with custom parameters."""
        config = KGConfig(
            sparql_endpoint="http://custom:7878/query",
            embedding_model="test-model",
            semantic_similarity_threshold=0.7,
        )
        self.assertEqual(config.sparql_endpoint, "http://custom:7878/query")
        self.assertEqual(config.embedding_model, "test-model")
        self.assertEqual(config.semantic_similarity_threshold, 0.7)

    def test_describe_entity(self):
        """Test describe_entity returns string."""
        config = KGConfig.default()
        # Use a real DBLP entity URI
        description = config.describe_entity("https://dblp.org/pid/95/2265")
        self.assertIsInstance(description, str)
        self.assertGreater(len(description), 0)

    def test_describe_entities_batch(self):
        """Test describe_entities_batch returns dict."""
        config = KGConfig.default()
        entity_uris = [
            "https://dblp.org/pid/95/2265",
            "https://dblp.org/pid/h/GeoffreyEHinton",
        ]
        descriptions = config.describe_entities_batch(entity_uris)
        self.assertIsInstance(descriptions, dict)
        self.assertGreater(len(descriptions), 0)
        for uri in entity_uris:
            if uri in descriptions:
                self.assertIsInstance(descriptions[uri], str)

    def test_describe_relation(self):
        """Test describe_relation returns string."""
        config = KGConfig.default()
        # Use a common DBLP relation
        description = config.describe_relation(
            "https://dblp.org/rdf/schema#authoredBy"
        )
        self.assertIsInstance(description, str)
        self.assertGreater(len(description), 0)


if __name__ == "__main__":
    # Setup fixtures before running standalone
    from fixtures import setup_global_fixtures, cleanup_global_fixtures

    setup_global_fixtures()
    try:
        unittest.main()
    finally:
        cleanup_global_fixtures()
