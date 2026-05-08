"""Answer generation pipeline for knowledge graph retrieval.

This module provides the complete pipeline for:
1. SPARQL generation from subgraphs
2. Full KG retrieval (seed nodes -> subgraphs -> SPARQL -> results)
3. Natural language answer generation from retrieval results or subgraphs
"""

import json
import time
from typing import Any, Callable, Dict, Optional

import dspy
from dotenv import load_dotenv

from kgnode.core.kg_config import KGConfig
from kgnode.core.logging_config import get_logger
from kgnode.core.sparql_query import execute_sparql_query
from kgnode.exceptions import AnswerGenerationError
from kgnode.seed_finder import SearchMode, citable, get_seed_and_relation_nodes
from kgnode.sparql_answer_generator import _subgraphs_to_merged_graphs, generate_sparql
from kgnode.subgraph_extraction import get_subgraphs


logger = get_logger(__name__)


# ============================================================================
# KG RETRIEVAL PIPELINE (Depends on generate_sparql + existing modules)
# ============================================================================

def kg_retrieve(
    query: str,
    config: Optional[KGConfig] = None,
    check_citable: bool = False,
    n_seed_results: int = 3,
    n_relation_results: int = 3,
    max_hops: int = 10,
    max_k: int = 2,
    kg_connection: Optional[Callable] = None,
    max_sparql_retries: int = 3,
    search_mode: SearchMode = SearchMode.semantic,
    keyword_timeout: int = 25,
    verbose: bool = False
) -> Dict[str, Any]:
    """Full retrieval pipeline: query -> seed nodes -> subgraphs -> SPARQL -> results.

    This function orchestrates the complete knowledge graph retrieval pipeline:
    1. Optional citability check
    2. Seed node finding (keyword + semantic search)
    3. Subgraph extraction using path-aware Markov chain
    4. SPARQL generation from ALL subgraphs (single LLM call)
    5. SPARQL execution to retrieve results

    Args:
        query: User's natural language query.
        config: Optional KGConfig instance for configuration.
            If None, uses default KGConfig with environment variables or built-in defaults.
            The config provides lm_api_key and embedding_model.
        check_citable: Whether to check citability before retrieval.
        n_seed_results: Number of seed nodes to retrieve per entity.
        max_hops: Maximum hops for subgraph extraction.
        max_k: Maximum neighbors per hop in subgraph extraction.
        kg_connection: Optional SPARQL query executor function. If None, creates from config.
        max_sparql_retries: Max retries for SPARQL generation.
        search_mode: Search mode for seed finding (semantic, keyword, or hybrid).
        keyword_timeout: Timeout in seconds for keyword search.

    Returns:
        Dictionary with structure on success:
        {
            'success': True,
            'sparql': str,              # Generated SPARQL query
            'results': List[Dict],      # SPARQL execution results
            'subgraphs': List[Dict],    # All extracted subgraphs used
            'seed_nodes': List[Dict],   # Seed nodes found
            'timing': Dict[str, float], # Timing information per stage
            'failure_stage': None
        }

        Dictionary with structure on failure:
        {
            'success': False,
            'failure_stage': str,       # Stage where failure occurred
            'failure_reason': str,      # Error message
            'seed_nodes': List[Dict] | None,
            'subgraphs': List[Dict] | None,
            'timing': Dict[str, float]
        }
    """
    # Initialize timing
    start_time = time.time()
    timing = {}

    def elapsed():
        return time.time() - start_time

    # Initialize config if not provided
    if config is None:
        config = KGConfig.default()

    load_dotenv()

    # Initialize DSpy LM
    api_key = config.lm_api_key
    if not api_key:
        error_msg = "LM API key not found in config or environment"

        timing['total'] = elapsed()
        return {
            'success': False,
            'failure_stage': 'initialization',
            'failure_reason': error_msg,
            'seed_nodes': None,
            'subgraphs': None,
            'timing': timing
        }

    lm = dspy.LM(model=config.openai_model, api_key=api_key)
    dspy.configure(lm=lm)

    # Initialize kg_connection if not provided
    if kg_connection is None:
        def kg_connection(q):
            return execute_sparql_query(q, config=config)




    # Step 1: Optional citability check
    if check_citable:

        is_citable, entity_nodes = citable(
            query=query,
            config=config,
            n_results_per_entity=n_seed_results,
            search_mode=search_mode,
            keyword_timeout=keyword_timeout
        )

        if not is_citable:
            error_msg = "Query is not citable - no suitable seed nodes found in knowledge graph"

            timing['citability_check'] = elapsed()
            timing['total'] = elapsed()
            return {
                'success': False,
                'failure_stage': 'citability',
                'failure_reason': error_msg,
                'seed_nodes': None,
                'subgraphs': None,
                'timing': timing
            }
        timing['citability_check'] = elapsed()


    # Step 2: Seed node and relation finding

    seed_nodes, extracted_entities, relation_nodes, extracted_relations = get_seed_and_relation_nodes(
        query=query,
        config=config,
        n_results_entity=n_seed_results,
        n_results_relation=n_relation_results,
        search_mode=search_mode
    )

    if not seed_nodes:
        error_msg = "No seed nodes found"

        timing['seed_finding'] = elapsed() - timing.get('citability_check', 0)
        timing['total'] = elapsed()
        return {
            'success': False,
            'failure_stage': 'seed_finding',
            'failure_reason': error_msg,
            'seed_nodes': None,
            'subgraphs': None,
            'timing': timing
        }

    timing['seed_finding'] = elapsed() - timing.get('citability_check', 0)

    # Step 3: Subgraph extraction

    all_subgraphs = []

    for seed_node in seed_nodes:
        seed_uri = seed_node['entity_uri']

        try:
            subgraphs, template_text = get_subgraphs(
                seed_node=seed_uri,
                query=query,
                config=config,
                kg_connection=kg_connection,
                max_hops=max_hops,
                max_k=max_k,
                verbose=verbose,
                seed_nodes=seed_nodes,
                relation_nodes=relation_nodes
            )

            all_subgraphs.extend(subgraphs)

        except Exception:

            continue

    if not all_subgraphs:
        error_msg = "No valid subgraphs extracted from any seed node"

        timing['subgraph_extraction'] = elapsed() - timing.get('citability_check', 0) - timing['seed_finding']
        timing['total'] = elapsed()
        return {
            'success': False,
            'failure_stage': 'subgraph_extraction',
            'failure_reason': error_msg,
            'seed_nodes': seed_nodes,
            'subgraphs': None,
            'timing': timing
        }

    timing['subgraph_extraction'] = elapsed() - timing.get('citability_check', 0) - timing['seed_finding']

    # Step 4: SPARQL generation using filtered seeds + merged graphs

    # Build filtered_seeds from seed_nodes (entity_uri + label)
    filtered_seeds = [
        {"seed_uri": n["entity_uri"], "seed_label": n.get("label", n["entity_uri"])}
        for n in seed_nodes
    ]
    # Convert subgraphs to merged_graphs format for the prompt
    merged_graphs_for_sparql = _subgraphs_to_merged_graphs(all_subgraphs)

    sparql_query, sparql_results = generate_sparql(
        query=query,
        filtered_seeds=filtered_seeds,
        merged_graphs=merged_graphs_for_sparql,
        config=config,
        kg_connection=kg_connection,
        max_retries=max_sparql_retries,
        verbose=verbose,
        relation_nodes=relation_nodes
    )

    timing['sparql_generation'] = elapsed() - timing.get('citability_check', 0) - timing['seed_finding'] - timing['subgraph_extraction']

    # Step 5: Results already returned from generate_sparql (no separate execution needed)

    if not sparql_results:
        error_msg = "SPARQL executed but returned no results"

        timing['sparql_execution'] = 0.0
        timing['total'] = elapsed()
        return {
            'success': False,
            'failure_stage': 'sparql_execution',
            'failure_reason': error_msg,
            'seed_nodes': seed_nodes,
            'subgraphs': all_subgraphs,
            'timing': timing
        }

    timing['sparql_execution'] = 0.0
    timing['total'] = elapsed()

    return {
        'success': True,
        'sparql': sparql_query,
        'results': sparql_results,
        'subgraphs': all_subgraphs,
        'seed_nodes': seed_nodes,
        'timing': timing,
        'failure_stage': None
    }



# ============================================================================
# ANSWER GENERATION (Depends on kg_retrieve)
# ============================================================================

def generate_answer(
    query: str,
    config: Optional[KGConfig] = None,
    check_citable: bool = False,
    n_seed_results: int = 3,
    n_relation_results: int = 3,
    max_hops: int = 10,
    max_k: int = 2,
    kg_connection: Optional[Callable] = None,
    max_sparql_retries: int = 3
) -> Dict[str, Any]:
    """End-to-end natural language answer generation using full KG retrieval pipeline.

    Runs the complete pipeline (kg_retrieve) and generates a natural language
    answer from the retrieved results using DSpy.

    Args:
        query: User's natural language query.
        config: Optional KGConfig instance for configuration.
            If None, uses default KGConfig with environment variables or built-in defaults.
            The config provides lm_api_key and embedding_model.
        check_citable: Whether to check citability before retrieval.
        n_seed_results: Number of seed nodes per entity.
        n_relation_results: Number of relation nodes per relation.
        max_hops: Maximum hops for subgraph extraction.
        max_k: Maximum neighbors per hop.
        kg_connection: Optional SPARQL query executor function. If None, creates from config.
        max_sparql_retries: Max retries for SPARQL generation.

    Returns:
        Dictionary with structure:
        {
            'answer': str,                  # Natural language answer
            'citations': List[str],         # Entity URIs cited
            'confidence': float,            # Confidence score (0-1)
            'sparql_used': str,             # SPARQL query used
            'raw_results': Dict             # Raw kg_retrieve result
        }

    Raises:
        AnswerGenerationError: If KG retrieval fails, JSON parsing fails, or answer generation fails.
        ValueError: If LM API key not found in config or environment.
    """
    # Initialize config if not provided
    if config is None:
        config = KGConfig.default()

    load_dotenv()

    # Initialize DSpy LM
    api_key = config.lm_api_key
    if not api_key:
        raise ValueError("LM API key not found in config or environment")

    lm = dspy.LM(model=config.openai_model, api_key=api_key)
    dspy.configure(lm=lm)



    # Step 1: Run KG retrieval pipeline
    retrieval_result = kg_retrieve(
        query=query,
        config=config,
        check_citable=check_citable,
        n_seed_results=n_seed_results,
        n_relation_results=n_relation_results,
        max_hops=max_hops,
        max_k=max_k,
        kg_connection=kg_connection,
        max_sparql_retries=max_sparql_retries,
    )

    # Check if retrieval succeeded
    if not retrieval_result.get('success', False):
        failure_reason = retrieval_result.get('failure_reason', 'Unknown error')
        failure_stage = retrieval_result.get('failure_stage', 'unknown')
        logger.error(f"KG retrieval failed at stage '{failure_stage}': {failure_reason}")
        raise AnswerGenerationError(
            f"KG retrieval failed at stage '{failure_stage}': {failure_reason}"
        )

    # Step 2: Format results for LLM

    formatted_results = []
    formatted_results.append(f"SPARQL Query: {retrieval_result['sparql']}")
    formatted_results.append(f"Results ({len(retrieval_result['results'])} items):")

    for j, row in enumerate(retrieval_result['results'][:10], 1):
        formatted_results.append(f"  {j}. {row}")

    if len(retrieval_result['results']) > 10:
        formatted_results.append(f"  ... and {len(retrieval_result['results']) - 10} more results")

    results_text = "\n".join(formatted_results)

    # Step 3: Generate answer using DSpy

    # Create DSpy module based on chain_of_thought setting with instructions
    signature_with_instructions = config.answer_generation_signature.with_instructions(
        config.answer_generation_instruction
    )

    if config.chain_of_thought:
        answer_generator = dspy.ChainOfThought(signature_with_instructions)
    else:
        answer_generator = dspy.Predict(signature_with_instructions)

    try:
        # Call DSpy
        result = answer_generator(query=query, results=results_text)

        # Parse citations from JSON string
        citations = json.loads(result.citations) if isinstance(result.citations, str) else result.citations

        # Build final result
        final_result = {
            'answer': result.answer,
            'citations': citations,
            'confidence': float(result.confidence),
            'sparql_used': retrieval_result['sparql'],
            'raw_results': retrieval_result
        }

        return final_result

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse citations JSON: {e}")
        raise AnswerGenerationError(f"Failed to parse citations JSON: {e}") from e
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        raise AnswerGenerationError(f"Answer generation failed: {e}") from e

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    # Example 1: Test generate_sparql
    # print("\n" + "=" * 80)
    # print("EXAMPLE 1: Generate SPARQL from subgraphs")
    # print("=" * 80)
    #
    # # Sample subgraphs for testing
    # sample_subgraphs = [
    #     {
    #         'triplet_uris': [
    #             ('https://dblp.org/pid/95/2265', 'https://dblp.org/rdf/schema#wikidata',
    #              'https://www.wikidata.org/entity/Q55238282')
    #         ],
    #         'path_with_label': [
    #             ('https://dblp.org/pid/95/2265', 'Robert Schober', True),
    #             ('https://dblp.org/rdf/schema#wikidata', 'wikidata', False),
    #             ('https://www.wikidata.org/entity/Q55238282', 'Q55238282', True)
    #         ]
    #     }
    # ]
    #
    # sparql = generate_sparql(
    #     query="What is the Wikidata ID of Robert Schober?",
    #     subgraphs=sample_subgraphs
    # )
    # if sparql:
    #     print(f"\nGenerated SPARQL:\n{sparql}")

    # Example 2: Test kg_retrieve
    # print("\n" + "=" * 80)
    # print("EXAMPLE 2: Full KG retrieval pipeline")
    # print("=" * 80)
    #
    # result = kg_retrieve(
    #     query="What is the Wikidata ID of Robert Schober?",
    #     check_citable=True,
    #     n_seed_results=2,
    #     max_hops=3,
    #     max_k=2
    # )
    # if result:
    #     print(f"\nSPARQL: {result['sparql']}")
    #     print(result)
    #     print(f"Results: {len(result['results'])} items")
    #     print(f"Subgraphs: {len(result['subgraphs'])} paths")

    # Example 3: Test generate_answer
    print("\n" + "=" * 80)
    print("EXAMPLE 3: End-to-end answer generation")
    print("=" * 80)

    answer = generate_answer(
        query="What is the Wikidata ID of Robert Schober?",
        check_citable=True,
        n_seed_results=2,
        max_hops=3,
        max_k=2
    )
    if answer:
        print(f"\nFinal Answer: {answer['answer']}")
        print(f"Confidence: {answer['confidence']:.2f}")
        print(f"Citations: {answer['citations']}")
        print(f"\nSPARQL Used: {answer['sparql_used']}")

    # Example 4a: Test generate_answer_using_subgraph with provided subgraphs
    # print("\n" + "=" * 80)
    # print("EXAMPLE 4a: Answer generation from provided subgraphs")
    # print("=" * 80)
    #
    # # You can use subgraphs from kg_retrieve result or extract_subgraphs_bfs
    # # For this example, we'll use the sample subgraphs
    # sample_subgraphs = [
    #     {
    #         'triplet_uris': [
    #             ('https://dblp.org/pid/95/2265', 'https://dblp.org/rdf/schema#wikidata',
    #              'https://www.wikidata.org/entity/Q55238282')
    #         ],
    #         'path_with_label': [
    #             ('https://dblp.org/pid/95/2265', 'Robert Schober', True),
    #             ('https://dblp.org/rdf/schema#wikidata', 'wikidata', False),
    #             ('https://www.wikidata.org/entity/Q55238282', 'Q55238282', True)
    #         ]
    #     }
    # ]
    #
    # answer = generate_answer_using_subgraph(
    #     query="What is the Wikidata ID of Robert Schober?",
    #     subgraphs=sample_subgraphs
    # )
    # if answer:
    #     print(f"\nFinal Answer: {answer['answer']}")
    #     print(f"Confidence: {answer['confidence']:.2f}")
    #     print(f"Citations: {answer['citations']}")
    #     print(f"Subgraphs used: {answer['subgraphs_used']}")

    # Example 4b: Test generate_answer_using_subgraph without subgraphs (auto-extract)
    # print("\n" + "=" * 80)
    # print("EXAMPLE 4b: Answer generation with automatic subgraph extraction")
    # print("=" * 80)
    #
    # # When subgraphs are not provided, the function will automatically:
    # # 1. Find seed nodes from the query
    # # 2. Extract subgraphs from those seed nodes
    # # 3. Generate answer from the extracted subgraphs
    # answer = generate_answer_using_subgraph(
    #     query="What is the Wikidata ID of Robert Schober?",
    #     n_seed_results=2,
    #     max_hops=3,
    #     max_k=2
    # )
    # if answer:
    #     print(f"\nFinal Answer: {answer['answer']}")
    #     print(f"Confidence: {answer['confidence']:.2f}")
    #     print(f"Citations: {answer['citations']}")
    #     print(f"Subgraphs used: {answer['subgraphs_used']}")
    #
    # print("\nAll examples defined. Uncomment the code blocks to test each function.")
