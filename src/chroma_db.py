"""ChromaDB operations facade - re-exports from specialized modules.

This module serves as the public API for ChromaDB operations, re-exporting functions
from specialized entity and relation modules for convenience and backward compatibility.
"""

# Entity-specific operations
from kgnode._entity_chromadb import (
    add_descriptions_to_csv_batched,
    add_or_update_entities,
    compile_entities_chromadb,
    compile_entities_chromadb_from_csv,
    delete_entities,
    get_entities_collection,
    semantic_search_entities,
)

# Relation-specific operations
from kgnode._relation_chromadb import (
    add_or_update_relations,
    compile_relations_chromadb,
    compile_relations_chromadb_from_csv,
    delete_relations,
    get_relations_collection,
    semantic_search_relations,
)

# Shared utilities (re-export for convenience)
from kgnode._chromadb_shared import (
    uri_to_label,
)


__all__ = [
    # Entity operations
    "add_descriptions_to_csv_batched",
    "add_or_update_entities",
    "compile_entities_chromadb",
    "compile_entities_chromadb_from_csv",
    "delete_entities",
    "get_entities_collection",
    "semantic_search_entities",

    # Relation operations
    "add_or_update_relations",
    "compile_relations_chromadb",
    "compile_relations_chromadb_from_csv",
    "delete_relations",
    "get_relations_collection",
    "semantic_search_relations",

    # Shared utilities
    "uri_to_label",
]
