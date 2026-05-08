"""SPARQL generation from natural language queries and KG subgraphs.

This module provides:
1. SPARQL generation from filtered seed nodes and merged graphs using DSpy
2. SPARQL validation via execution
3. Formatting utilities for seed nodes and merged graphs
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import dspy
from dotenv import load_dotenv

from kgnode.core.kg_config import KGConfig
from kgnode.core.logging_config import get_logger
from kgnode.core.schema_selector import (
    _format_schema_for_prompt,
    _select_relevant_schema,
)
from kgnode.core.sparql_query import execute_sparql_query
from kgnode.exceptions import SPARQLGenerationError


logger = get_logger(__name__)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _validate_sparql(
    sparql_query: str,
    kg_connection: Callable
) -> Tuple[bool, Any, Optional[str]]:
    """Execute SPARQL and return validity, results, and any error.

    Args:
        sparql_query: SPARQL query string to execute.
        kg_connection: SPARQL query executor function (already configured with config).

    Returns:
        Tuple of (is_valid, results, error_message).
            - is_valid: True if execution succeeded
            - results: query results if valid, None otherwise
            - error_message: None if valid, error string if invalid
    """
    try:
        results = kg_connection(sparql_query)
        return True, results, None
    except Exception as e:
        return False, None, f"SPARQL execution failed: {str(e)}"


def _format_seed_nodes_for_prompt(filtered_seeds: List[Dict[str, Any]]) -> str:
    """Format filtered seed nodes as a numbered list for the LLM prompt.

    Args:
        filtered_seeds: List of dicts with at least 'seed_uri' and 'seed_label'.

    Returns:
        Formatted string, e.g.:
            1. Robert Schober | <https://dblp.org/pid/95/2265>
            2. Sameh S. Askar | <https://dblp.org/pid/141/4165>
    """
    lines = []
    for i, seed in enumerate(filtered_seeds, 1):
        lines.append(f"{i}. {seed['seed_label']} | <{seed['seed_uri']}>")
    return "\n".join(lines)


def _format_relation_nodes_for_prompt(relation_nodes: List[Dict[str, Any]]) -> str:
    """Format relation nodes as a numbered list for the LLM prompt.

    Args:
        relation_nodes: List of relation dicts with 'relation_uri' and 'label'.

    Returns:
        Formatted string, e.g.:
            1. wikidata <https://dblp.org/rdf/schema#wikidata>
            2. authoredBy <https://dblp.org/rdf/schema#authoredBy>
    """
    if not relation_nodes:
        return "(No relations found)"

    lines = []
    for i, rel in enumerate(relation_nodes, 1):
        label = rel.get('label', rel.get('relation_uri', '').split('#')[-1].split('/')[-1])
        uri = rel.get('relation_uri', '')
        lines.append(f"{i}. {label} <{uri}>")
    return "\n".join(lines)


def _format_merged_graphs_for_prompt(merged_graphs: List[Dict[str, Any]]) -> str:
    """Format merged KG graphs as labeled triples for the LLM prompt.

    Args:
        merged_graphs: List of merged graph dicts, each with 'triples',
            'probability', 'path_depth', 'merged_from'.
            Each triple has 'subject', 'subject_uri', 'predicate',
            'predicate_uri', 'object', 'object_uri'.

    Returns:
        Formatted string showing each graph and its labeled triples.
    """
    lines = []
    for i, graph in enumerate(merged_graphs, 1):
        prob = graph.get("probability", 0.0)
        depth = graph.get("path_depth", 0)
        merged = graph.get("merged_from", 1)
        lines.append(
            f"=== Graph {i} (confidence: {prob:.2f}, depth: {depth}, merged: {merged}) ==="
        )
        for triple in graph.get("triples", []):
            s_label = triple.get("subject", "")
            s_uri = triple.get("subject_uri", "")
            p_label = triple.get("predicate", "")
            p_uri = triple.get("predicate_uri", "")
            o_label = triple.get("object", "")
            o_uri = triple.get("object_uri", "")
            lines.append(
                f"  [{s_label}] <{s_uri}>"
                f" --[{p_label} <{p_uri}>]-->"
                f" [{o_label}] <{o_uri}>"
            )
        lines.append("")
    return "\n".join(lines)


def _extract_predicates_from_merged_graphs(
    merged_graphs: List[Dict[str, Any]],
) -> str:
    """Extract unique predicate label→URI pairs from merged graphs.

    Returns a formatted string enumerating every predicate URI present in the
    graphs so the LLM cannot hallucinate predicates that are absent.

    Returns empty string if no predicates are found.
    """
    # authorOf is the person→paper direction and is never correct in SPARQL patterns.
    # Always use authoredBy (?paper authoredBy ?person) instead.
    _NEVER_USE = {"https://dblp.org/rdf/schema#authorOf"}

    seen: Dict[str, str] = {}  # uri -> label
    for graph in merged_graphs:
        for triple in graph.get("triples", []):
            label = triple.get("predicate", "")
            uri = triple.get("predicate_uri", "")
            if uri and label and uri not in _NEVER_USE:
                seen[uri] = label

    if not seen:
        return ""

    lines = [
        "AVAILABLE PREDICATES — copy these URIs verbatim, do NOT invent others:",
    ]
    for uri, label in sorted(seen.items(), key=lambda x: x[1]):
        lines.append(f"  - {label} -> <{uri}>")
    return "\n".join(lines)


def _subgraphs_to_merged_graphs(subgraphs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert raw subgraph dicts (from get_subgraphs) to merged_graphs format.

    Used when calling generate_sparql from the live pipeline where
    pre-computed merged_graphs are not available.

    Args:
        subgraphs: List of subgraph dicts with 'path_with_label', 'triplet_uris',
            'probability', 'path_depth'.

    Returns:
        List of merged graph dicts compatible with _format_merged_graphs_for_prompt.
    """
    merged = []
    for subgraph in subgraphs:
        path = subgraph.get("path_with_label", [])
        nodes = [(uri, label) for uri, label, is_node in path if is_node]
        edges = [(uri, label) for uri, label, is_node in path if not is_node]

        triples = []
        for idx, (p_uri, p_label) in enumerate(edges):
            if idx < len(nodes) and idx + 1 < len(nodes):
                s_uri, s_label = nodes[idx]
                o_uri, o_label = nodes[idx + 1]
                triples.append({
                    "subject": s_label,
                    "subject_uri": s_uri,
                    "predicate": p_label,
                    "predicate_uri": p_uri,
                    "object": o_label,
                    "object_uri": o_uri,
                })

        merged.append({
            "triples": triples,
            "probability": subgraph.get("probability", 0.0),
            "path_depth": subgraph.get("path_depth", 1),
            "merged_from": 1,
        })
    return merged


# ============================================================================
# SPARQL GENERATION
# ============================================================================

def generate_sparql(
    query: str,
    filtered_seeds: List[Dict[str, Any]],
    merged_graphs: List[Dict[str, Any]],
    config: Optional[KGConfig] = None,
    kg_connection: Optional[Callable] = None,
    max_retries: int = 3,
    verbose: bool = False,
    relation_nodes: Optional[List[Dict[str, Any]]] = None
) -> Tuple[str, Any]:
    """Generate SPARQL query from seed entities and merged KG graphs.

    Args:
        query: User's natural language query.
        filtered_seeds: List of seed dicts with 'seed_uri' and 'seed_label'.
        merged_graphs: List of merged graph dicts with 'triples', 'probability',
            'path_depth', 'merged_from'. Each triple has subject/predicate/object
            with both label and URI fields.
        config: Optional KGConfig. Defaults to KGConfig.default().
        kg_connection: Optional SPARQL executor. Defaults to execute_sparql_query.
        max_retries: Max retry attempts for invalid SPARQL.
        verbose: Whether to print verbose output.
        relation_nodes: Optional list of validated relation nodes from ChromaDB search.

    Returns:
        Tuple of (sparql_string, results) where results is the output of executing
        the generated SPARQL against the KG.

    Raises:
        SPARQLGenerationError: If unable to generate valid SPARQL after max_retries.
        ValueError: If LM API key not found.
    """
    if config is None:
        config = KGConfig.default()

    load_dotenv()

    api_key = config.lm_api_key
    if not api_key:
        raise ValueError("LM API key not found in config or environment")

    lm = dspy.LM(model=config.openai_model, api_key=api_key)
    dspy.configure(lm=lm)

    if kg_connection is None:
        kg_connection = lambda q: execute_sparql_query(q, config=config)

    # Format relation nodes (prefer validated nodes over schema selection)
    if relation_nodes is not None and len(relation_nodes) > 0:
        # Use validated relation nodes from ChromaDB search
        formatted_relation_nodes = _format_relation_nodes_for_prompt(relation_nodes)
    else:
        # Fallback: select relevant schema if relation_nodes not provided
        try:
            selected_schema = _select_relevant_schema(
                query=query,
                config=config,
                top_k_entities=10,
                top_k_relations=20,
                include_uris=True
            )
            # Extract only relations from schema
            relations_from_schema = []
            for rel in selected_schema.get('relations', []):
                relations_from_schema.append({
                    'relation_uri': rel.get('uri', ''),
                    'label': rel.get('label', '')
                })
            formatted_relation_nodes = _format_relation_nodes_for_prompt(relations_from_schema)
        except Exception:
            formatted_relation_nodes = ""

    # Format inputs for prompt
    formatted_seeds = _format_seed_nodes_for_prompt(filtered_seeds)
    formatted_graphs = _format_merged_graphs_for_prompt(merged_graphs)

    # P2: Extract predicates from the actual merged graphs — grounded, no hallucination
    predicate_hint = _extract_predicates_from_merged_graphs(merged_graphs)

    # Build few-shot examples block
    example_text = ""
    if config.sparql_few_shot_examples:
        example_text = "\n\n" + "=" * 80 + "\nFEW-SHOT EXAMPLES:\n" + "=" * 80
        for i, ex in enumerate(config.sparql_few_shot_examples, 1):
            example_text += f"\n\n--- Example {i} ---\n"
            example_text += f"Query: {ex['query']}\n\n"
            example_text += f"Seed Nodes:\n{ex['seed_nodes']}\n\n"
            example_text += f"Merged Graphs:\n{ex['merged_graphs']}\n\n"
            example_text += f"Generated SPARQL:\n{ex['sparql']}\n"
        example_text += (
            "\n" + "=" * 80 + "\nNOW GENERATE SPARQL FOR THE CURRENT QUERY:\n" + "=" * 80 + "\n"
        )

    last_error = None

    for attempt in range(max_retries):
        try:
            current_instruction = config.sparql_generation_instruction
            if predicate_hint:
                current_instruction += "\n\n" + predicate_hint
            current_instruction += example_text

            if last_error and attempt > 0:
                current_instruction += (
                    f"\n\nIMPORTANT - Previous attempt failed:\n{last_error}\nPlease fix this."
                )

            signature_with_instructions = config.sparql_generation_signature.with_instructions(
                current_instruction
            )

            if config.chain_of_thought:
                sparql_generator = dspy.ChainOfThought(signature_with_instructions)
            else:
                sparql_generator = dspy.Predict(signature_with_instructions)

            result = sparql_generator(
                query=query,
                seed_nodes=formatted_seeds,
                merged_graphs=formatted_graphs,
                relation_nodes=formatted_relation_nodes
            )
            sparql_query = result.sparql.strip()

            # Strip markdown code fences if present
            if sparql_query.startswith("```"):
                lines = sparql_query.split("\n")
                sparql_query = "\n".join(lines[1:-1]) if len(lines) > 2 else sparql_query
                sparql_query = sparql_query.replace("```sparql", "").replace("```", "").strip()

            is_valid, results, error_msg = _validate_sparql(sparql_query, kg_connection)

            if is_valid:
                return sparql_query, results
            else:
                last_error = error_msg

        except Exception as e:
            last_error = f"Exception during SPARQL generation: {str(e)}"
            continue

    logger.error(
        f"Failed to generate valid SPARQL after {max_retries} retries. Last error: {last_error}"
    )
    raise SPARQLGenerationError(
        f"Failed to generate valid SPARQL after {max_retries} retries. Last error: {last_error}"
    )
