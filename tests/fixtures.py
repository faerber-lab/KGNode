"""Shared test fixtures for kgnode test suite.

This module provides global fixtures that are created once before all tests
and cleaned up once after all tests complete. This significantly speeds up
test execution by avoiding redundant ChromaDB compilation.
"""

import os
import shutil
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import kgnode modules
from kgnode import get_entities_collection, KGConfig

# Global fixtures (singleton pattern)
_global_chromadb_collection = None
_global_config = None
_test_data_paths = {
    "csv_path": "_data/vector_db/test_entities.csv",
    "db_dir": "_data/vector_db/test_chroma_db",
}


def get_test_config() -> KGConfig:
    """Get shared KGConfig instance for all tests.

    Returns:
        KGConfig: Shared configuration object with test-specific settings.
    """
    global _global_config

    if _global_config is None:
        _global_config = KGConfig.default()
        # Use test-specific persist directory
        _global_config.chroma_persist_directory = _test_data_paths["db_dir"]

    return _global_config


def get_test_chromadb():
    """Get shared ChromaDB collection for all tests.

    Creates the collection once on first call, returns cached version on
    subsequent calls. This avoids expensive recompilation for each test.

    Returns:
        chromadb.Collection: Shared ChromaDB collection.
    """
    global _global_chromadb_collection

    if _global_chromadb_collection is None:
        print("  Creating shared ChromaDB collection (one-time setup)...")
        config = get_test_config()

        # Create ChromaDB with small entity limit for faster testing
        _global_chromadb_collection = get_entities_collection(
            config=config,
            csv_path=_test_data_paths["csv_path"],
            entity_limit=100,  # Small limit for fast testing
            force_recreate=False,  # Reuse existing if available
        )
        print("  ✓ ChromaDB collection ready")

    return _global_chromadb_collection


def setup_global_fixtures():
    """Initialize global test fixtures before any tests run.

    This is called once by test_runner.py before the test suite starts.
    It pre-creates ChromaDB and schema ChromaDB to avoid redundant work.
    """
    print("Setting up global test fixtures...")

    # Pre-load config
    get_test_config()

    # Pre-create ChromaDB (this triggers schema_chromadb creation too)
    get_test_chromadb()

    print("✓ Global fixtures ready\n")


def cleanup_global_fixtures():
    """Clean up global test fixtures after all tests complete.

    This is called once by test_runner.py after the entire test suite finishes.
    It removes temporary test files and directories.
    """
    global _global_chromadb_collection, _global_config

    print("\nCleaning up global test fixtures...")

    # Clean up test ChromaDB files
    if os.path.exists(_test_data_paths["csv_path"]):
        try:
            os.remove(_test_data_paths["csv_path"])
        except Exception:
            pass

    if os.path.exists(_test_data_paths["db_dir"]):
        try:
            shutil.rmtree(_test_data_paths["db_dir"])
        except Exception:
            pass

    # Reset globals
    _global_chromadb_collection = None
    _global_config = None

    print("✓ Cleanup complete")


def reset_chromadb_for_test():
    """Force recreation of ChromaDB for tests that need fresh data.

    Most tests should NOT call this - it defeats the performance optimization.
    Only use for tests that specifically need to test ChromaDB creation/deletion.
    """
    global _global_chromadb_collection

    # Clean up existing
    if os.path.exists(_test_data_paths["db_dir"]):
        shutil.rmtree(_test_data_paths["db_dir"])

    # Force recreation
    _global_chromadb_collection = None
    return get_test_chromadb()
