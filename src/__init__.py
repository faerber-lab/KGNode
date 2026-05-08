"""kgnode - Knowledge Graph Agnostic Node for Knowledge-Aware LLM Applications.

Public API for knowledge graph retrieval and answer generation.
"""

__version__ = "0.2.0"

# Exceptions
# Main Pipeline APIs
# VectorDB Operations
from kgnode.chroma_db import (
    add_or_update_entities,
    add_or_update_relations,
    compile_entities_chromadb,
    compile_entities_chromadb_from_csv,
    compile_relations_chromadb,
    compile_relations_chromadb_from_csv,
    delete_entities,
    delete_relations,
    get_entities_collection,
    get_relations_collection,
    semantic_search_entities,
    semantic_search_relations,
)

# Core Configuration
from kgnode.core.kg_config import KGConfig
from kgnode.core.sparql_query import execute_sparql_query
from kgnode.exceptions import (
    AnswerGenerationError,
    ConfigurationError,
    EntityNotFoundError,
    KGNodeError,
    PipelineError,
    SchemaError,
    SeedNodeError,
    SPARQLExecutionError,
    SPARQLGenerationError,
    SubgraphExtractionError,
    ValidationError,
    VectorDBError,
)

# Search Operations
from kgnode.keyword_search import search_entities_by_keywords
from kgnode.pipeline import (
    generate_answer,
    kg_retrieve,
)
from kgnode.seed_finder import SearchMode, citable, get_seed_nodes
from kgnode.sparql_answer_generator import generate_sparql
from kgnode.subgraph_answer_generator import generate_answer_using_subgraph
from kgnode.subgraph_extraction import get_subgraphs

# Validation
from kgnode.validator import validate_subgraph


__all__ = [
    # Version
    "__version__",
    # Exceptions
    "KGNodeError",
    "ConfigurationError",
    "PipelineError",
    "SeedNodeError",
    "SubgraphExtractionError",
    "SPARQLGenerationError",
    "SPARQLExecutionError",
    "AnswerGenerationError",
    "EntityNotFoundError",
    "ValidationError",
    "VectorDBError",
    "SchemaError",
    # Main Pipeline APIs
    "citable",
    "get_seed_nodes",
    "SearchMode",
    "get_subgraphs",
    "generate_sparql",
    "kg_retrieve",
    "generate_answer",
    "generate_answer_using_subgraph",
    # Validation
    "validate_subgraph",
    # Search Operations
    "search_entities_by_keywords",
    # VectorDB Operations - Entities
    "compile_entities_chromadb",
    "compile_entities_chromadb_from_csv",
    "semantic_search_entities",
    "get_entities_collection",
    "add_or_update_entities",
    "delete_entities",
    # VectorDB Operations - Relations
    "compile_relations_chromadb",
    "compile_relations_chromadb_from_csv",
    "semantic_search_relations",
    "get_relations_collection",
    "add_or_update_relations",
    "delete_relations",
    # Core Configuration
    "KGConfig",
    "execute_sparql_query",
]
