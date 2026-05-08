"""Subgraph validation utilities for knowledge graph node extraction.

This module provides validation functions to check if extracted subgraphs
contain the correct answer paths by comparing against ground truth SPARQL queries.
"""

import os
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv

from kgnode.core.kg_config import KGConfig
from kgnode.core.sparql_query import execute_sparql_query


def validate_subgraph(subgraph: Dict[str, Any],
                      answer_sparql: str,
                      config: Optional[KGConfig] = None,) -> bool:
    """
    Returns True if subgraph contains answer path.

    Args:
        subgraph: Dictionary with 'triplet_uris' and 'path_with_label' keys
        answer_sparql: SPARQL query that retrieves the correct answer
        config: Optional KGConfig instance for configuration.
            If None, uses default KGConfig with environment variables or built-in defaults.
        kg_connection: Optional SPARQL query function. If None, creates from config.

    Returns:
        True if the subgraph's path leads to the correct answer, False otherwise

    Example:
        answer_sparql = "SELECT DISTINCT ?answer WHERE {
            <https://dblp.org/pid/95/2265> <https://dblp.org/rdf/schema#wikidata> ?answer
        }"
        If subgraph contains this edge, returns True
    """
    # Initialize config if not provided
    if config is None:
        config = KGConfig.default()

    # Execute the answer SPARQL query to get ground truth answer(s)
    results = execute_sparql_query(answer_sparql, config=config)

    if not results:
        return False

    # Extract answer URIs from SPARQL results
    answer_uris = set()
    for result in results:
        # The answer variable is typically named '?answer' in the query
        if 'answer' in result:
            answer_uris.add(result['answer'])

    if not answer_uris:
        return False

    # Check if any node in the subgraph path matches an answer URI
    for uri, label, is_node in subgraph['path_with_label']:
        if is_node and uri in answer_uris:
            return True

    return False


# ============================================================================
# New validation system using pre-computed answers from answers.json
# ============================================================================

def _extract_uris_from_sparql(sparql: str) -> Set[str]:
    """Extract entity URIs from SPARQL query using regex.

    Args:
        sparql: SPARQL query string

    Returns:
        Set of entity URIs (dblp.org/pid/* and dblp.org/rec/*)
    """
    # Pattern: <https://dblp.org/pid/...> or <https://dblp.org/rec/...>
    pattern = r'<(https://dblp\.org/(?:pid|rec)/[^>]+)>'
    matches = re.findall(pattern, sparql)
    return set(matches)


def _extract_subgraph_uris(subgraph: Dict[str, Any]) -> Set[str]:
    """Extract all node URIs from subgraph.

    Args:
        subgraph: Subgraph dict with 'path_with_label' key

    Returns:
        Set of node URIs in the subgraph
    """
    uris = set()
    for uri, label, is_node in subgraph.get('path_with_label', []):
        if is_node:
            uris.add(uri)
    return uris


def _extract_answer_uris(answer_data: Dict[str, Any], var_names: List[str]) -> Set[str]:
    """Extract URIs from answer bindings for given variable names.

    Args:
        answer_data: Answer data from answers.json
        var_names: List of SPARQL variable names to extract (e.g., ['answer', 'firstanswer'])

    Returns:
        Set of answer URIs
    """
    uris = set()
    if 'results' in answer_data:
        for binding in answer_data['results']['bindings']:
            for var_name in var_names:
                if var_name in binding:
                    if binding[var_name].get('type') == 'uri':
                        uris.add(binding[var_name]['value'])
    return uris


def _extract_predicates_from_sparql(sparql: str) -> Set[str]:
    """Extract all DBLP predicate URIs from SPARQL query.

    Args:
        sparql: SPARQL query string

    Returns:
        Set of predicate URIs like 'https://dblp.org/rdf/schema#primaryAffiliation'
    """
    pattern = r'<(https://dblp\.org/rdf/schema#[^>]+)>'
    matches = re.findall(pattern, sparql)
    return set(matches)


def _detect_answer_type(answer_data: Dict[str, Any]) -> str:
    """Detect if answer is 'uri' or 'literal'.

    Args:
        answer_data: Answer data from answers.json

    Returns:
        'uri', 'literal', or 'unknown'
    """
    bindings = answer_data.get('results', {}).get('bindings', [])
    if bindings and 'answer' in bindings[0]:
        return bindings[0]['answer'].get('type', 'unknown')
    return 'unknown'


def _validate_literal_based(
    subgraph: Dict[str, Any],
    sparql: str
) -> Tuple[bool, str]:
    """Validate literal answers by checking predicate presence.

    Args:
        subgraph: Subgraph dictionary
        sparql: SPARQL query string

    Returns:
        (is_valid, reason): Validation result and explanation
    """
    required_predicates = _extract_predicates_from_sparql(sparql)
    if not required_predicates:
        return False, "Literal: No predicates in SPARQL"

    # Extract predicates from subgraph
    subgraph_predicates = set()
    for (source, predicate, target) in subgraph.get('triplet_uris', []):
        subgraph_predicates.add(predicate)

    # Check for matches
    matches = required_predicates & subgraph_predicates
    if matches:
        return True, f"Literal: {len(matches)}/{len(required_predicates)} predicates present"
    return False, "Literal: No required predicates found"


def _validate_uri_based(
    subgraph: Dict[str, Any],
    answer_data: Dict[str, Any],
    var_names: List[str] = None
) -> Tuple[bool, str]:
    """Validate queries with URI answers (SINGLE_FACT, MULTI_FACT, UNION, DISAMBIGUATION, DOUBLE_INTENT).

    Args:
        subgraph: Subgraph dictionary
        answer_data: Answer data from answers.json
        var_names: Variable names to check (default: ['answer'])

    Returns:
        (is_valid, reason): Validation result and explanation
    """
    if var_names is None:
        var_names = ['answer']

    answer_uris = _extract_answer_uris(answer_data, var_names)
    if not answer_uris:
        return False, "No URI answers found"

    subgraph_uris = _extract_subgraph_uris(subgraph)
    if not subgraph_uris:
        return False, "Empty subgraph"

    # Success if ANY answer URI appears in subgraph
    matches = answer_uris & subgraph_uris
    if matches:
        return True, f"URI match: {list(matches)[0][:50]}..."
    return False, "No answer URIs in subgraph"


def _validate_boolean(
    subgraph: Dict[str, Any],
    answer_data: Dict[str, Any],
    sparql: str
) -> Tuple[bool, str]:
    """Validate ASK queries by checking entity presence (BOOLEAN, NEGATION, DOUBLE_NEGATION).

    Args:
        subgraph: Subgraph dictionary
        answer_data: Answer data from answers.json (contains boolean result)
        sparql: SPARQL query string

    Returns:
        (is_valid, reason): Validation result and explanation
    """
    # Extract entities from SPARQL query
    sparql_uris = _extract_uris_from_sparql(sparql)
    if not sparql_uris:
        return False, "No entities in SPARQL to validate"

    subgraph_uris = _extract_subgraph_uris(subgraph)
    if not subgraph_uris:
        return False, "Empty subgraph"

    # Success if URIs from SPARQL appear in subgraph
    matches = sparql_uris & subgraph_uris
    if matches:
        return True, f"Boolean: {len(matches)}/{len(sparql_uris)} entities present"
    return False, "Boolean: Query entities not in subgraph"


def _validate_count(
    subgraph: Dict[str, Any],
    answer_data: Dict[str, Any],
    sparql: str
) -> Tuple[bool, str]:
    """Validate COUNT queries with lenient criteria.

    Args:
        subgraph: Subgraph dictionary
        answer_data: Answer data from answers.json (contains count result)
        sparql: SPARQL query string

    Returns:
        (is_valid, reason): Validation result and explanation
    """
    # Very lenient: just check if subgraph has ANY query entity
    sparql_uris = _extract_uris_from_sparql(sparql)
    subgraph_uris = _extract_subgraph_uris(subgraph)

    if not subgraph_uris:
        return False, "Empty subgraph"

    matches = sparql_uris & subgraph_uris
    if matches:
        return True, f"COUNT: Contains {len(matches)} query entities (lenient pass)"
    return False, "COUNT: No query entities found"


def _validate_superlative(
    subgraph: Dict[str, Any],
    answer_data: Dict[str, Any],
    sparql: str
) -> Tuple[bool, str]:
    """Validate SUPERLATIVE+COMPARATIVE queries.

    Args:
        subgraph: Subgraph dictionary
        answer_data: Answer data from answers.json
        sparql: SPARQL query string

    Returns:
        (is_valid, reason): Validation result and explanation
    """
    # Try URI-based validation first (for queries with ?answer variable)
    answer_uris = _extract_answer_uris(answer_data, ['answer', 'count'])
    if answer_uris:
        subgraph_uris = _extract_subgraph_uris(subgraph)
        matches = answer_uris & subgraph_uris
        if matches:
            return True, f"Superlative: URI match found"

    # Fallback to lenient validation (similar to COUNT)
    sparql_uris = _extract_uris_from_sparql(sparql)
    subgraph_uris = _extract_subgraph_uris(subgraph)

    if not subgraph_uris:
        return False, "Empty subgraph"

    matches = sparql_uris & subgraph_uris
    if matches:
        return True, f"Superlative: Contains query entities (lenient pass)"
    return False, "Superlative: No entities found"


def validate_subgraph_v2(
    subgraph: Dict[str, Any],
    question_id: str,
    answers_dict: Dict[str, Dict[str, Any]],
    question_data: Dict[str, Any]
) -> Tuple[bool, str]:
    """Validate subgraph using pre-computed answers from answers.json.

    This function handles all 10 query types in DBLP-QuAD dataset with
    appropriate validation strategies for each type.

    Args:
        subgraph: Subgraph dictionary with 'path_with_label' and 'triplet_uris'
        question_id: Question ID (e.g., 'Q0001')
        answers_dict: Dictionary mapping question_id to answer data
        question_data: Dict with 'sparql' and 'query_type' keys

    Returns:
        (is_valid, reason): Tuple of validation result and explanation string

    Example:
        answers_dict = {ans['id']: ans['answer'] for ans in answers_data}
        is_valid, reason = validate_subgraph_v2(
            subgraph=subgraph,
            question_id='Q0001',
            answers_dict=answers_dict,
            question_data={'sparql': query, 'query_type': 'SINGLE_FACT'}
        )
    """
    # Get answer data
    answer_data = answers_dict.get(question_id)
    if answer_data is None:
        return False, f"No answer in answers.json for {question_id}"

    # Validate subgraph structure
    if not subgraph or not subgraph.get('path_with_label'):
        return False, "Empty subgraph"

    # Get query info
    sparql = question_data.get('sparql', '')
    query_type = question_data.get('query_type', '')

    # Dispatch to appropriate handler based on query type
    try:
        if query_type in ['SINGLE_FACT', 'MULTI_FACT', 'UNION', 'DISAMBIGUATION']:
            answer_type = _detect_answer_type(answer_data)
            if answer_type == 'literal':
                return _validate_literal_based(subgraph, sparql)
            return _validate_uri_based(subgraph, answer_data, ['answer'])

        elif query_type == 'DOUBLE_INTENT':
            answer_type = _detect_answer_type(answer_data)
            if answer_type == 'literal':
                return _validate_literal_based(subgraph, sparql)
            return _validate_uri_based(subgraph, answer_data, ['firstanswer', 'secondanswer', 'answer'])

        elif query_type in ['BOOLEAN', 'NEGATION', 'DOUBLE_NEGATION']:
            return _validate_boolean(subgraph, answer_data, sparql)

        elif query_type == 'COUNT':
            return _validate_count(subgraph, answer_data, sparql)

        elif query_type == 'SUPERLATIVE+COMPARATIVE':
            return _validate_superlative(subgraph, answer_data, sparql)

        else:
            # Unknown query type - try URI-based validation as default
            return _validate_uri_based(subgraph, answer_data, ['answer'])

    except (KeyError, TypeError, ValueError) as e:
        return False, f"Validation error: {type(e).__name__}"


if __name__ == '__main__':
    """
    Example usage demonstrating subgraph validation.

    This example validates subgraphs extracted for the question:
    "Show the Wikidata ID of the person Robert Schober."

    It checks if the extracted subgraphs contain the correct answer path
    leading to Robert Schober's Wikidata ID.
    """
    from kgnode.core.kg_config import KGConfig
    from kgnode.subgraph_extraction import get_subgraphs

    # Define the test case
    seed_node = 'https://dblp.org/pid/95/2265'
    query = "Show the Wikidata ID of the person Robert Schober."
    answer_sparql = """
    SELECT DISTINCT ?answer WHERE {
        <https://dblp.org/pid/95/2265> <https://dblp.org/rdf/schema#wikidata> ?answer
    }
    """

    print("=" * 80)
    print("SUBGRAPH VALIDATION EXAMPLE")
    print("=" * 80)
    print(f"\nQuestion: {query}")
    print(f"Seed Node: {seed_node}")
    print(f"\nExtracting subgraphs...")

    # Extract subgraphs using the path-aware Markov chain algorithm (uses default config)
    subgraphs, template_text = get_subgraphs(
        seed_node=seed_node,
        query=query,
        max_hops=3,
        max_k=2
    )

    print(f"Template text: {template_text}")
    print(f"Total subgraphs found: {len(subgraphs)}")
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)

    # Validate each subgraph (uses default config)
    valid_count = 0
    for i, subgraph in enumerate(subgraphs):
        is_valid = validate_subgraph(subgraph, answer_sparql)
        if is_valid:
            valid_count += 1
            print(f"\n Subgraph {i + 1} is VALID!")
            print(f"  Number of edges: {len(subgraph['triplet_uris'])}")
            print(f"  Edges: {subgraph['triplet_uris']}")
            print(f"  Path:")
            for uri, label, is_node in subgraph['path_with_label']:
                if is_node:
                    print(f"    Node: {label}")
                    print(f"          ({uri})")
                else:
                    print(f"    --> Relation: {label}")
            print("-" * 80)
        else:
            print(f"\n Subgraph {i + 1} is INVALID (does not contain answer)")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Valid subgraphs: {valid_count}/{len(subgraphs)}")
    print(f"Accuracy: {(valid_count / len(subgraphs) * 100):.2f}%")
    print("=" * 80)