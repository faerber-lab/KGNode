"""Natural language answer generation directly from KG subgraphs.

This module generates answers by reasoning over subgraph paths without
going through SPARQL generation and execution.
"""

import json
from typing import Any, Callable, Dict, List, Optional

import dspy
from dotenv import load_dotenv

from kgnode.core.kg_config import KGConfig
from kgnode.core.sparql_query import execute_sparql_query
from kgnode.seed_finder import SearchMode, get_seed_and_relation_nodes
from kgnode.subgraph_extraction import get_subgraphs


def generate_answer_using_subgraph(
    query: str,
    config: Optional[KGConfig] = None,
    subgraphs: Optional[List[Dict[str, Any]]] = None,
    n_seed_results: int = 3,
    n_relation_results: int = 3,
    max_hops: int = 10,
    max_k: int = 2,
    kg_connection: Optional[Callable] = None,
    search_mode: SearchMode = SearchMode.semantic,
    keyword_timeout: int = 25
) -> Dict[str, Any]:
    """Generate natural language answer directly from query and subgraphs.

    Unlike generate_answer(), this function works directly with subgraphs
    without going through SPARQL generation and execution. If subgraphs are
    not provided, it will automatically extract them from the query. Uses DSpy for generation.

    Args:
        query: User's natural language query.
        config: Optional KGConfig instance for configuration.
            If None, uses default KGConfig with environment variables or built-in defaults.
            The config provides lm_api_key and embedding_model.
        subgraphs: Optional list of subgraph dicts. If None, will extract from query.
        n_seed_results: Number of seed nodes to retrieve (used if subgraphs is None).
        max_hops: Maximum hops for subgraph extraction (used if subgraphs is None).
        max_k: Maximum neighbors per hop (used if subgraphs is None).
        kg_connection: Optional SPARQL query executor function. If None, creates from config.
        search_mode: Search mode for seed finding (semantic, keyword, or hybrid).
        keyword_timeout: Timeout in seconds for keyword search.

    Returns:
        Dictionary with structure:
        {
            'answer': str,                  # Natural language answer
            'citations': List[str],         # Entity URIs cited
            'confidence': float,            # Confidence score (0-1)
            'subgraphs_used': List[int],    # Indices of subgraphs used
            'raw_subgraphs': List[Dict]     # Original subgraphs
        }
        Returns None if generation fails.
    """
    if config is None:
        config = KGConfig.default()

    load_dotenv()

    api_key = config.lm_api_key
    if not api_key:
        raise ValueError("LM API key not found in config or environment")

    lm = dspy.LM(model=config.openai_model, api_key=api_key)
    dspy.configure(lm=lm)

    # Extract subgraphs if not provided
    if subgraphs is None:

        if kg_connection is None:
            def kg_connection(q):
                return execute_sparql_query(q, config=config)

        seed_nodes, extracted_entities, relation_nodes, extracted_relations = get_seed_and_relation_nodes(
            query=query,
            config=config,
            n_results_entity=n_seed_results,
            n_results_relation=n_relation_results,
            search_mode=search_mode
        )

        if not seed_nodes:
            return None

        subgraphs = []

        for seed_node in seed_nodes:
            seed_uri = seed_node['entity_uri']

            try:
                extracted_subgraphs, template_text = get_subgraphs(
                    seed_node=seed_uri,
                    query=query,
                    config=config,
                    kg_connection=kg_connection,
                    seed_nodes=seed_nodes,
                    relation_nodes=relation_nodes,
                    max_hops=max_hops,
                    max_k=max_k,
                )
                subgraphs.extend(extracted_subgraphs)

            except Exception:
                continue

        if not subgraphs:
            return None

    if not subgraphs:
        return None

    # Format all subgraphs for the prompt
    formatted_subgraphs = []

    for i, subgraph in enumerate(subgraphs, 1):
        formatted_subgraphs.append(f"Knowledge Graph Path {i}:")
        formatted_subgraphs.append("-" * 60)

        for uri, label, is_node in subgraph['path_with_label']:
            if is_node:
                formatted_subgraphs.append(f"  Node: {label}")
                formatted_subgraphs.append(f"        URI: <{uri}>")
            else:
                formatted_subgraphs.append(f"    --[{label}]-->")

        formatted_subgraphs.append("")

    subgraphs_text = "\n".join(formatted_subgraphs)

    signature_with_instructions = config.answer_from_subgraph_signature.with_instructions(
        config.answer_from_subgraph_instruction
    )

    if config.chain_of_thought:
        answer_generator = dspy.ChainOfThought(signature_with_instructions)
    else:
        answer_generator = dspy.Predict(signature_with_instructions)

    try:
        result = answer_generator(query=query, subgraphs=subgraphs_text)

        citations = json.loads(result.citations) if isinstance(result.citations, str) else result.citations
        subgraphs_used = json.loads(result.subgraphs_used) if isinstance(result.subgraphs_used, str) else result.subgraphs_used

        return {
            'answer': result.answer,
            'citations': citations,
            'confidence': float(result.confidence),
            'subgraphs_used': subgraphs_used,
            'raw_subgraphs': subgraphs
        }

    except json.JSONDecodeError:
        return None
    except Exception:
        return None