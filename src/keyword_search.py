import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from kgnode.core.kg_config import ExtractedEntity, KGConfig
from kgnode.core.logging_config import get_logger
from kgnode.core.sparql_query import execute_sparql_query


logger = get_logger(__name__)


def generate_search_variants(query: str) -> List[str]:
    """Generate multiple search variants to match different punctuation styles.

    Creates variants by selectively removing common punctuation (periods, commas)
    to match database labels regardless of punctuation format.

    Args:
        query: The search query string.

    Returns:
        List of lowercase search variants with different punctuation removed.
    """
    variants = set()
    query_lower = query.lower().strip()

    # Collapse multiple spaces
    query_lower = re.sub(r"\s+", " ", query_lower)

    # Original query (keep all punctuation)
    variants.add(query_lower)

    # Remove periods only (for "S." vs "S")
    no_periods = query_lower.replace(".", "")
    no_periods = re.sub(r"\s+", " ", no_periods).strip()
    if no_periods and no_periods != query_lower:
        variants.add(no_periods)

    # Remove commas only (for "Last, First" vs "Last First")
    no_commas = query_lower.replace(",", "")
    no_commas = re.sub(r"\s+", " ", no_commas).strip()
    if no_commas and no_commas != query_lower:
        variants.add(no_commas)

    # Remove both periods and commas
    no_periods_commas = query_lower.replace(".", "").replace(",", "")
    no_periods_commas = re.sub(r"\s+", " ", no_periods_commas).strip()
    if no_periods_commas and no_periods_commas not in variants:
        variants.add(no_periods_commas)

    return list(variants)


def _query_single_keyword(
    keyword: str,
    entity_type: Optional[str],
    limit: int,
    config: KGConfig,
    timeout: Optional[int],
    variant: bool,
) -> List[dict]:
    """Execute a single SPARQL query for one keyword with optional type filtering.

    Args:
        keyword: The keyword to search for.
        entity_type: Optional entity type for filtering (e.g., "Creator").
                    If None, no type filtering is applied.
        limit: Maximum number of results for this query.
        config: KGConfig instance for SPARQL endpoint.
        timeout: Optional timeout in seconds for the query.
        variant: If True, generates punctuation variants for matching.

    Returns:
        List of query results with 'entity' and 'label' keys.
    """
    # Generate keyword variants
    if variant:
        variants = generate_search_variants(keyword)
    else:
        variants = [keyword.lower()]

    # Build keyword filter conditions
    keyword_conditions = []
    for v in variants:
        if v:
            condition = f'CONTAINS(LCASE(?label), "{v}")'
            keyword_conditions.append(condition)

    if not keyword_conditions:
        return []

    # Combine keyword variants with OR
    if len(keyword_conditions) == 1:
        filter_clause = keyword_conditions[0]
    else:
        filter_clause = "(" + " || ".join(keyword_conditions) + ")"

    # Build SPARQL query
    if entity_type:
        # With type filtering
        sparql_query = f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX dblp: <https://dblp.org/rdf/schema#>

    SELECT DISTINCT ?entity ?label
    WHERE {{
      ?entity rdf:type dblp:{entity_type} .
      ?entity rdfs:label ?label .
      FILTER({filter_clause})
    }}
    LIMIT {limit}
    """
    else:
        # Without type filtering
        sparql_query = f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX dblp: <https://dblp.org/rdf/schema#>

    SELECT DISTINCT ?entity ?label
    WHERE {{
      ?entity rdfs:label ?label .
      FILTER({filter_clause})
    }}
    LIMIT {limit}
    """

    return execute_sparql_query(sparql_query, config=config, timeout=timeout)


def search_entities_by_keywords(
    keywords: List[ExtractedEntity],
    limit: int = 10,
    config: Optional[KGConfig] = None,
    timeout: Optional[int] = None,
    variant: bool = False,
    use_type_filter: bool = True,
):
    """Search for entities in the knowledge graph based on keywords with type filtering.

    Executes separate SPARQL queries per keyword in parallel for better performance.
    Each query is limited individually, so total results may exceed the limit parameter.

    Args:
        keywords: List of ExtractedEntity objects with 'name' and 'types' fields.
            Example: [ExtractedEntity(name="John Smith", types=["Creator", "Person"])]
        limit: Maximum results per keyword query (not total). Default is 10.
        config: Optional KGConfig instance for configuration.
            If None, uses default KGConfig with env vars or built-in defaults.
        timeout: Optional timeout in seconds per SPARQL query.
            If None, no timeout is set. Returns empty list on timeout.
        variant: If True, generates punctuation variants for better matching.
            If False, uses only the exact keyword. Default is False.
        use_type_filter: If True, filters by types[0]. If False, ignores types. Default is True.

    Returns:
        List[List[Dict]]: List of result lists, one per keyword. Each sublist contains
                         query results with 'entity' and 'label' keys for that keyword.
                         Each sublist can have up to 'limit' results.
    """
    # Initialize config if not provided
    if config is None:
        config = KGConfig.default()

    if not keywords:
        return []

    # Prepare query parameters for each keyword, maintaining order
    query_params = []
    for extracted_entity in keywords:
        keyword = extracted_entity["name"].strip()
        if not keyword:
            # Add empty list for empty keywords to maintain index alignment
            query_params.append(None)
            continue

        # Get top type only if use_type_filter is True
        entity_types = extracted_entity.get("types", [])
        top_type = entity_types[0] if (use_type_filter and entity_types) else None
        query_params.append((keyword, top_type))

    if not any(query_params):
        return []

    # Execute all queries in parallel
    # Use dictionary to maintain keyword index -> results mapping
    results_by_index = {}

    with ThreadPoolExecutor(max_workers=min(len(query_params), 10)) as executor:
        # Submit all queries with per-query limit, tracking index
        future_to_index = {}
        for idx, params in enumerate(query_params):
            if params is None:
                results_by_index[idx] = []
                continue
            keyword, entity_type = params
            future = executor.submit(
                _query_single_keyword, keyword, entity_type, limit, config, timeout, variant
            )
            future_to_index[future] = idx

        # Collect results as they complete
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                results = future.result()
                results_by_index[idx] = results
            except Exception as e:
                # Log error but continue with other queries
                keyword = query_params[idx][0] if query_params[idx] else "unknown"
                logger.warning(f"Query failed for keyword '{keyword}': {e}")
                results_by_index[idx] = []

    # Return results in original keyword order
    return [results_by_index[idx] for idx in sorted(results_by_index.keys())]


if __name__ == "__main__":
    # print(search_entities_by_keywords(["publication", "author"]))
    # print(search_entities_by_keywords(["Robert Schober"]))
    # print(search_entities_by_keywords(["Rodríguez, S."]))
    print(search_entities_by_keywords(["Silva-Faundez"]))
    print(search_entities_by_keywords(["Pablo G. Tahoces"]))
    print(search_entities_by_keywords(["O'Brien"]))
    print(search_entities_by_keywords(["O'Brien"]))
    print(search_entities_by_keywords(["Geoffrey Hinton"]))
