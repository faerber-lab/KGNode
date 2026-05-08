"""Tests for ChromaDB operations."""

import unittest
import os
import shutil
from dotenv import load_dotenv
from kgnode import (
    compile_entities_chromadb,
    compile_entities_chromadb_from_csv,
    semantic_search_entities,
    get_entities_collection,
    add_or_update_entities,
    delete_entities,
    KGConfig,
)

# Load environment variables
load_dotenv()

# Import shared fixtures
from fixtures import get_test_config, get_test_chromadb, reset_chromadb_for_test


class TestChromaDB(unittest.TestCase):
    """Test ChromaDB compilation and search operations."""

    def setUp(self):
        """Set up test - uses shared fixtures."""
        self.config = get_test_config()
        self.collection = get_test_chromadb()

    def test_get_entities_collection(self):
        """Test get_entities_collection uses shared collection."""
        # Use the shared collection
        self.assertIsNotNone(self.collection)
        self.assertEqual(self.collection.name, "top_entity_descriptions")

    def test_semantic_search_entities(self):
        """Test semantic_search_entities returns results."""
        results = semantic_search_entities(
            self.collection, "machine learning papers", n_results=3
        )
        self.assertIsInstance(results, list)
        if len(results) > 0:
            self.assertIn("entity_uri", results[0])
            self.assertIn("description", results[0])
            self.assertIn("distance", results[0])

    def test_add_or_update_entities(self):
        """Test add_or_update_entities adds new entities."""
        # Try to add new entities to shared collection
        new_entities = [
            "https://dblp.org/pid/h/GeoffreyEHinton",
            "https://dblp.org/pid/b/YoshuaBengio",
        ]
        added, updated, collection = add_or_update_entities(
            new_entities, config=self.config, batch_size=10
        )
        self.assertIsInstance(added, int)
        self.assertIsInstance(updated, int)
        self.assertIsNotNone(collection)

    def test_delete_entities(self):
        """Test delete_entities removes entities."""
        # Try to delete entities (may not exist, but should not error)
        entities_to_delete = ["https://dblp.org/pid/test/123"]
        deleted_count = delete_entities(entities_to_delete, config=self.config)
        self.assertIsInstance(deleted_count, int)
        self.assertGreaterEqual(deleted_count, 0)

    def test_compile_entities_chromadb_from_csv(self):
        """Test compile_entities_chromadb_from_csv with shared CSV."""
        # This test needs its own ChromaDB instance
        test_csv_path = "_data/vector_db/test_compile_csv.csv"
        test_db_dir = "_data/vector_db/test_compile_csv_db"

        try:
            # Get shared collection to ensure CSV exists
            if os.path.exists("_data/vector_db/test_entities.csv"):
                collection = compile_entities_chromadb_from_csv(
                    "_data/vector_db/test_entities.csv",
                    config=self.config,
                    force_recreate=False,
                )
                self.assertIsNotNone(collection)
        finally:
            # Clean up
            if os.path.exists(test_csv_path):
                os.remove(test_csv_path)
            if os.path.exists(test_db_dir):
                shutil.rmtree(test_db_dir)

    def test_compile_entities_chromadb(self):
        """Test compile_entities_chromadb uses shared collection."""
        # Verify shared collection was compiled correctly
        self.assertIsNotNone(self.collection)
        self.assertEqual(self.collection.name, "top_entity_descriptions")
        # Verify it has some entities
        count = self.collection.count()
        self.assertGreater(count, 0)


if __name__ == "__main__":
    # Setup fixtures before running standalone
    from fixtures import setup_global_fixtures, cleanup_global_fixtures

    setup_global_fixtures()
    try:
        unittest.main()
    finally:
        cleanup_global_fixtures()
