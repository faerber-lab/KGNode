"""Custom exception hierarchy for kgnode.

This module defines domain-specific exceptions for the kgnode package,
providing clear error semantics for different failure modes in the
knowledge graph retrieval pipeline.

Exception Hierarchy:
    KGNodeError (base)
    ├── ConfigurationError
    ├── PipelineError (base for pipeline stages)
    │   ├── SeedNodeError
    │   ├── SubgraphExtractionError
    │   ├── SPARQLGenerationError
    │   ├── SPARQLExecutionError
    │   └── AnswerGenerationError
    ├── EntityNotFoundError
    ├── ValidationError
    ├── VectorDBError
    └── SchemaError

Example:
    >>> from kgnode.exceptions import SPARQLGenerationError
    >>> raise SPARQLGenerationError("Failed to generate valid SPARQL query")
"""


class KGNodeError(Exception):
    """Base exception for all kgnode errors.

    All custom exceptions in the kgnode package inherit from this base class,
    allowing users to catch all kgnode-specific errors with a single except clause.

    Example:
        >>> try:
        ...     # kgnode operations
        ... except KGNodeError as e:
        ...     # Handle any kgnode error
        ...     logger.error(f"KGNode error: {e}")
    """

    pass


class ConfigurationError(KGNodeError):
    """Configuration-related errors.

    Raised when there are issues with KGConfig setup, invalid parameters,
    or missing required configuration values.

    Example:
        >>> raise ConfigurationError("SPARQL endpoint URL is required")
    """

    pass


class PipelineError(KGNodeError):
    """Base exception for pipeline stage errors.

    This serves as a base class for errors occurring in specific pipeline
    stages (seed finding, subgraph extraction, SPARQL generation, etc.).

    Users can catch this to handle any pipeline-related error, or catch
    specific subclasses for fine-grained error handling.

    Example:
        >>> try:
        ...     result = kg_retrieve(query, config)
        ... except PipelineError as e:
        ...     # Handle any pipeline stage error
        ...     logger.error(f"Pipeline failed: {e}")
    """

    pass


class SeedNodeError(PipelineError):
    """Seed node finding failed.

    Raised when the seed finder cannot identify suitable seed nodes for a query.
    This may occur when:
    - No entities can be extracted from the query
    - Extracted entities are not found in the knowledge graph
    - Search operations fail

    Example:
        >>> raise SeedNodeError("No seed nodes found for query: 'example query'")
    """

    pass


class SubgraphExtractionError(PipelineError):
    """Subgraph extraction failed.

    Raised when the subgraph extraction process encounters errors, such as:
    - Path exploration failures
    - Invalid probability calculations
    - Graph traversal errors

    Example:
        >>> raise SubgraphExtractionError("Failed to extract subgraph from seed nodes")
    """

    pass


class SPARQLGenerationError(PipelineError):
    """SPARQL generation failed.

    Raised when the system cannot generate a valid SPARQL query from the
    extracted subgraph and user query. This may occur when:
    - LLM fails to generate SPARQL
    - Generated SPARQL has syntax errors
    - Maximum retry attempts exceeded

    Example:
        >>> raise SPARQLGenerationError("Failed to generate valid SPARQL after 3 retries")
    """

    pass


class SPARQLExecutionError(PipelineError):
    """SPARQL execution failed.

    Raised when a SPARQL query fails to execute against the knowledge graph endpoint.
    This may occur due to:
    - Endpoint connection errors
    - Query timeout
    - Malformed queries
    - Server errors

    Example:
        >>> raise SPARQLExecutionError("SPARQL endpoint returned 500 error")
    """

    pass


class AnswerGenerationError(PipelineError):
    """Answer generation failed.

    Raised when the system cannot generate a natural language answer from
    SPARQL results. This may occur when:
    - LLM fails to process results
    - Results are in unexpected format
    - Answer synthesis fails

    Example:
        >>> raise AnswerGenerationError("Failed to generate answer from SPARQL results")
    """

    pass


class EntityNotFoundError(KGNodeError):
    """Entity not found in knowledge graph.

    Raised when a requested entity does not exist in the KG or vector database.

    Example:
        >>> raise EntityNotFoundError("Entity 'http://example.org/entity123' not found")
    """

    pass


class ValidationError(KGNodeError):
    """Validation failed.

    Raised when validation of data structures, subgraphs, or results fails.
    This includes:
    - Invalid subgraph structure
    - Failed consistency checks
    - Schema validation errors

    Example:
        >>> raise ValidationError("Subgraph validation failed: missing required edges")
    """

    pass


class VectorDBError(KGNodeError):
    """Vector database operation failed.

    Raised when ChromaDB or other vector database operations encounter errors:
    - Database initialization failures
    - Query errors
    - Index corruption
    - Connection issues

    Example:
        >>> raise VectorDBError("Failed to connect to ChromaDB")
    """

    pass


class SchemaError(KGNodeError):
    """Schema-related errors.

    Raised when there are issues with KG schema operations:
    - Schema extraction failures
    - Invalid schema format
    - Schema selection errors
    - Missing schema information

    Example:
        >>> raise SchemaError("Failed to extract schema from SPARQL endpoint")
    """

    pass
