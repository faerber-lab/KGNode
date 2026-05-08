from typing import Any, Callable, Dict, List, Optional

from kgnode._chromadb_shared import uri_to_label
from kgnode.core.kg_config import KGConfig
from kgnode.core.logging_config import get_logger
from kgnode.core.sparql_query import execute_sparql_query


logger = get_logger(__name__)


# =============================================================================
# DBLP Predicate URIs
# =============================================================================
DBLP_SCHEMA = "https://dblp.org/rdf/schema#"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"

DBLP_PRIMARY_FULL_CREATOR_NAME = f"{DBLP_SCHEMA}primaryFullCreatorName"
DBLP_OTHER_FULL_CREATOR_NAME = f"{DBLP_SCHEMA}otherFullCreatorName"
DBLP_TITLE = f"{DBLP_SCHEMA}title"


def _default_entity_descriptor_logic(entity_uri: str, triples: List[Dict[str, Any]]) -> str:
    """Default DBLP-specific entity descriptor logic.

    Creates entity descriptor: 'Type1, Type2: Name, Alt1, Alt2.'

    Args:
        entity_uri: URI of the entity (cleaned, no brackets).
        triples: List of dicts with 'predicate' and 'object' keys.

    Returns:
        Description in format: "Type1, Type2: Name, Alt1, Alt2."
    """
    if not triples:
        return uri_to_label(entity_uri)

    entity_types: List[str] = []
    primary_name = None
    rdfs_label = None
    alt_names: List[str] = []
    title = None

    for triple in triples:
        pred = triple["predicate"]
        obj = triple["object"]

        if pred == RDF_TYPE:
            type_name = uri_to_label(obj).title()
            if type_name not in ["Thing", "Resource"]:
                entity_types.append(type_name)
        elif pred == DBLP_PRIMARY_FULL_CREATOR_NAME:
            primary_name = obj
        elif pred == RDFS_LABEL:
            rdfs_label = obj
        elif pred == DBLP_OTHER_FULL_CREATOR_NAME:
            alt_names.append(obj)
        elif pred == DBLP_TITLE:
            title = obj

    # Build descriptor: "Type1, Type2: Name, Alt1, Alt2."
    types_str = ", ".join(entity_types) if entity_types else "Entity"
    name = primary_name or title or rdfs_label or uri_to_label(entity_uri)

    descriptor = f"{types_str}: {name}"

    # Add alternative names if present and different from primary
    unique_alts = [n for n in alt_names if n != name]
    if unique_alts:
        descriptor += f", {', '.join(unique_alts[:3])}"

    return descriptor


def create_entity_description(entity_uri: str, config: Optional[KGConfig] = None) -> str:
    """
    Create a focused natural language description optimized for semantic search.
    Prioritizes key identifying information: names, titles, venues, years.

    Note: This function uses default DBLP logic. For custom logic, use EntityDescriptorWrapper
    with KGConfig.

    Args:
        entity_uri: URI of the entity (without angle brackets)
        config: Optional KGConfig instance for configuration.
            If None, uses default KGConfig with environment variables or built-in defaults.

    Returns:
        Natural language description optimized for search
    """
    # Initialize config if not provided
    if config is None:
        config = KGConfig.default()

    # Remove angle brackets if present
    entity_uri = entity_uri.strip()
    if entity_uri.startswith('<') and entity_uri.endswith('>'):
        entity_uri = entity_uri[1:-1]

    query = f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX dblp: <https://dblp.org/rdf/schema#>
    SELECT ?predicate ?object
    WHERE {{
        <{entity_uri}> ?predicate ?object .
    }}
    """

    triples = execute_sparql_query(query, config=config)

    return _default_entity_descriptor_logic(entity_uri, triples)

def create_entity_descriptions_batch(entity_uris: List[str], config: Optional[KGConfig] = None) -> Dict[str, str]:
    """
    Create focused descriptions for multiple entities in batch.
    Optimized for semantic search of author names and paper titles.

    Args:
        entity_uris: List of entity URIs to describe.
        config: Optional KGConfig instance for configuration.
            If None, uses default KGConfig with environment variables or built-in defaults.

    Returns:
        Dictionary mapping entity URIs to their descriptions.
    """
    # Initialize config if not provided
    if config is None:
        config = KGConfig.default()

    if not entity_uris:
        return {}

    # Clean URIs
    cleaned_uris = []
    for uri in entity_uris:
        uri = uri.strip()
        if not uri:
            continue
        if uri.startswith('<') and uri.endswith('>'):
            uri = uri[1:-1]
        cleaned_uris.append(uri)

    if not cleaned_uris:
        return {}

    # Build VALUES clause
    values_clause = " ".join([f"<{uri}>" for uri in cleaned_uris])

    query = f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX dblp: <https://dblp.org/rdf/schema#>

    SELECT ?entity ?predicate ?object
    WHERE {{
        VALUES ?entity {{ {values_clause} }}
        ?entity ?predicate ?object .
    }}
    """

    triples = execute_sparql_query(query, config=config)

    # Group triples by entity
    entity_triples = {}
    for triple in triples:
        entity = triple['entity']
        if entity not in entity_triples:
            entity_triples[entity] = []
        entity_triples[entity].append(triple)

    # Create descriptions using the same logic as _default_entity_descriptor_logic
    descriptions = {}
    for original_uri, cleaned_uri in zip(entity_uris, cleaned_uris):
        if cleaned_uri not in entity_triples:
            descriptions[original_uri] = uri_to_label(original_uri)
        else:
            entity_types: List[str] = []
            primary_name = None
            rdfs_label = None
            alt_names: List[str] = []
            title = None

            for triple in entity_triples[cleaned_uri]:
                pred = triple["predicate"]
                obj = triple["object"]

                if pred == RDF_TYPE:
                    type_name = uri_to_label(obj).title()
                    if type_name not in ["Thing", "Resource"]:
                        entity_types.append(type_name)
                elif pred == DBLP_PRIMARY_FULL_CREATOR_NAME:
                    primary_name = obj
                elif pred == RDFS_LABEL:
                    rdfs_label = obj
                elif pred == DBLP_OTHER_FULL_CREATOR_NAME:
                    alt_names.append(obj)
                elif pred == DBLP_TITLE:
                    title = obj

            # Build descriptor: "Type1, Type2: Name, Alt1, Alt2."
            types_str = ", ".join(entity_types) if entity_types else "Entity"
            name = primary_name or title or rdfs_label or uri_to_label(cleaned_uri)

            descriptor = f"{types_str}: {name}"

            # Add alternative names if present and different from primary
            unique_alts = [n for n in alt_names if n != name]
            if unique_alts:
                descriptor += f", {', '.join(unique_alts[:3])}"

            descriptions[original_uri] = descriptor

    return descriptions


class EntityDescriptorWrapper:
    """Wrapper class that handles SPARQL queries and applies descriptor logic.

    This class provides both single and batch entity description functionality,
    using either user-provided descriptor logic or default DBLP logic.
    """

    def __init__(
        self,
        descriptor_function: Optional[Callable[[str, List[Dict[str, Any]]], str]] = None,
        config: Optional[KGConfig] = None
    ):
        """Initialize the wrapper with descriptor function.

        Args:
            descriptor_function: Function that takes (entity_uri, triples) and returns
                description string. If None, uses default DBLP logic.
            config: Optional KGConfig instance for configuration.
                If None, uses default KGConfig with environment variables or built-in defaults.
        """
        # Initialize config if not provided
        if config is None:
            from kgnode.core.kg_config import KGConfig
            config = KGConfig.default()

        self.descriptor_function = descriptor_function or _default_entity_descriptor_logic
        self.config = config

    def describe_single(self, entity_uri: str) -> str:
        """Create description for a single entity.

        Args:
            entity_uri: URI of the entity (with or without angle brackets).

        Returns:
            Natural language description of the entity.
        """
        # Clean URI
        entity_uri = entity_uri.strip()
        if entity_uri.startswith('<') and entity_uri.endswith('>'):
            entity_uri = entity_uri[1:-1]

        # Query triples for this entity
        query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX dblp: <https://dblp.org/rdf/schema#>
        SELECT ?predicate ?object
        WHERE {{
            <{entity_uri}> ?predicate ?object .
        }}
        """

        try:
            triples = execute_sparql_query(query, config=self.config)
            return self.descriptor_function(entity_uri, triples)
        except Exception as e:
            logger.warning(f" Error describing entity {entity_uri}: {e}")
            return uri_to_label(entity_uri)

    def describe_batch(self, entity_uris: List[str]) -> Dict[str, str]:
        """Create descriptions for multiple entities in batch (optimized).

        Uses a single SPARQL query with VALUES clause to fetch all triples,
        then applies descriptor logic to each entity. Batch size limited to
        80 entities to stay within 8KB SPARQL query size limit.

        Args:
            entity_uris: List of entity URIs (with or without angle brackets).

        Returns:
            Dictionary mapping original URIs to their descriptions.
        """
        if not entity_uris:
            return {}

        # Clean URIs
        cleaned_uris = []
        for uri in entity_uris:
            uri = uri.strip()
            if not uri:
                continue
            if uri.startswith('<') and uri.endswith('>'):
                uri = uri[1:-1]
            cleaned_uris.append(uri)

        if not cleaned_uris:
            return {}

        # Build VALUES clause (limited to 80 URIs to stay within 8KB query limit)
        if len(cleaned_uris) > 80:
            logger.warning(f" Batch size {len(cleaned_uris)} exceeds limit of 80. "
                  f"Consider splitting into multiple batches.")
            cleaned_uris = cleaned_uris[:80]

        values_clause = " ".join([f"<{uri}>" for uri in cleaned_uris])

        query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX dblp: <https://dblp.org/rdf/schema#>

        SELECT ?entity ?predicate ?object
        WHERE {{
            VALUES ?entity {{ {values_clause} }}
            ?entity ?predicate ?object .
        }}
        """

        try:
            triples = execute_sparql_query(query, config=self.config)
        except Exception as e:
            logger.warning(f" Error fetching triples for batch: {e}")
            # Fallback to URI labels
            return {uri: uri_to_label(uri) for uri in entity_uris}

        # Group triples by entity
        entity_triples = {}
        for triple in triples:
            entity = triple['entity']
            if entity not in entity_triples:
                entity_triples[entity] = []
            entity_triples[entity].append(triple)

        # Create descriptions using descriptor function
        descriptions = {}
        for original_uri, cleaned_uri in zip(entity_uris[:len(cleaned_uris)], cleaned_uris):
            if cleaned_uri not in entity_triples:
                descriptions[original_uri] = uri_to_label(original_uri)
            else:
                try:
                    descriptions[original_uri] = self.descriptor_function(
                        cleaned_uri,
                        entity_triples[cleaned_uri]
                    )
                except Exception as e:
                    logger.warning(f" Error applying descriptor function to {cleaned_uri}: {e}")
                    descriptions[original_uri] = uri_to_label(original_uri)

        return descriptions


if __name__ == "__main__":
    # print(create_entity_description("https://dblp.org/rdf/schema#Publication"))
    print(create_entity_descriptions_batch(["<https://dblp.org/rec/journals/nature/RheinbayNAWSTHH20>", "https://dblp.org/rdf/schema#Person"]))

    # Example usage:
    # entities = ["http://example.org/entity1", "http://example.org/entity2", "http://example.org/entity3"]
    # descriptions = create_entity_descriptions_batch(entities)
    #
    # for entity, desc in descriptions.items():
    #     print(f"{entity}: {desc}")