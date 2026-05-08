"""Seed node finding functionality using hybrid search (keyword + semantic)."""

import json
import time
from difflib import SequenceMatcher
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import dspy
from dotenv import load_dotenv

from kgnode.chroma_db import get_entities_collection, semantic_search_entities
from kgnode.core.kg_config import ExtractedEntity, KGConfig
from kgnode.core.schema_chromadb import (
    _compile_schema_collections,
    _load_schema_collections,
)
from kgnode.core.schema_selector import (
    _format_schema_for_prompt,
    _select_relevant_schema,
)
from kgnode.keyword_search import search_entities_by_keywords


class SearchMode(str, Enum):
    """Search mode for seed node finding."""

    hybrid = "hybrid"
    keyword = "keyword"
    semantic = "semantic"


def _extract_entities_from_query(
    query: str,
    config: KGConfig
) -> List[ExtractedEntity]:
    """Extract entity names WITH types from a natural language query using DSpy.

    Args:
        query: Natural language query from user.
        config: KGConfig instance with DSpy signature configuration.

    Returns:
        List of ExtractedEntity objects with 'name' and 'types' fields.

    Raises:
        ValueError: If LLM response cannot be parsed as JSON array.
        Exception: If DSpy call fails.
    """
    try:
        # Get relevant schema types for entity extraction
        try:
            # Try to load existing schema collections first
            try:
                _load_schema_collections(config)
            except ValueError:
                # Collections don't exist - compile them (one-time operation)
                _compile_schema_collections(
                    config=config,
                    ontology_path=config.ontology_path,
                    force_recreate=config.force_schema_refresh
                )

            # Select relevant entity types (we only need entity types, not relations)
            relevant_schema = _select_relevant_schema(
                query=query,
                config=config,
                top_k_entities=config.schema_top_k_entities,
                top_k_relations=0  # Don't need relations for entity extraction
            )

            schema_context = _format_schema_for_prompt(relevant_schema)
        except Exception:
            schema_context = "Entity Types: Creator, Person, Publication, Article, Inproceedings"  # Fallback

        # Create DSpy module based on chain_of_thought setting with instructions
        signature_with_instructions = config.entity_extraction_signature.with_instructions(
            config.entity_extraction_instruction
        )

        if config.chain_of_thought:
            entity_extractor = dspy.ChainOfThought(signature_with_instructions)
        else:
            entity_extractor = dspy.Predict(signature_with_instructions)

        # Call DSpy with query and schema context
        result = entity_extractor(query=query, schema_context=schema_context)

        # Parse JSON array of entity objects
        entities_raw = json.loads(result.entities)

        if not isinstance(entities_raw, list):
            raise ValueError(f"Expected JSON array, got: {type(entities_raw)}")

        # Convert to ExtractedEntity format
        extracted_entities: List[ExtractedEntity] = []
        for entity_obj in entities_raw:
            if isinstance(entity_obj, dict):
                name = str(entity_obj.get("name", "")).strip()
                types = entity_obj.get("types", [])

                # Validate types is a list
                if not isinstance(types, list):
                    types = [str(types)] if types else []

                # Convert types to strings and filter empties
                types = [str(t).strip() for t in types if str(t).strip()]

                # Limit to 3 types max (user requirement)
                types = types[:3]

                if name:
                    extracted_entities.append(ExtractedEntity(name=name, types=types))
            elif isinstance(entity_obj, str):
                # Backward compatibility: handle old format gracefully
                name = str(entity_obj).strip()
                if name:
                    extracted_entities.append(ExtractedEntity(name=name, types=[]))

        return extracted_entities

    except json.JSONDecodeError:
        # Fallback: return query as single entity without types
        return [ExtractedEntity(name=query, types=[])]
    except Exception:
        raise


def _extract_entities_and_relations_from_query(
    query: str,
    config: KGConfig
) -> Tuple[List[ExtractedEntity], List[ExtractedEntity]]:
    """Extract BOTH entities and relations in a SINGLE LLM call.

    This function extracts entities (people, publications, venues) and relations
    (predicates like "published", "authored by") from a query simultaneously,
    minimizing LLM API calls.

    Args:
        query: Natural language query from user.
        config: KGConfig instance with DSpy signature configuration.

    Returns:
        Tuple of (extracted_entities, extracted_relations)
        Both use ExtractedEntity format: {name: str, types: List[str]}

    Raises:
        ValueError: If LLM response cannot be parsed as JSON.
        Exception: If DSpy call fails.
    """
    try:
        # Get relevant schema types for both entities and relations
        try:
            try:
                _load_schema_collections(config)
            except ValueError:
                _compile_schema_collections(
                    config=config,
                    ontology_path=config.ontology_path,
                    force_recreate=config.force_schema_refresh
                )

            # Select relevant schema (both entity types AND relation types)
            relevant_schema = _select_relevant_schema(
                query=query,
                config=config,
                top_k_entities=config.schema_top_k_entities,
                top_k_relations=config.schema_top_k_relations  # Include relations!
            )

            schema_context = _format_schema_for_prompt(relevant_schema)
        except Exception:
            # Fallback schema context
            schema_context = "Entity Types: Creator, Person, Publication, Article\nRelation Types: authoredBy, publishedIn, yearOfPublication"

        # Single DSpy call for BOTH entities and relations
        signature_with_instructions = config.entity_relation_extraction_signature.with_instructions(
            config.entity_relation_extraction_instruction
        )

        if config.chain_of_thought:
            extractor = dspy.ChainOfThought(signature_with_instructions)
        else:
            extractor = dspy.Predict(signature_with_instructions)

        result = extractor(query=query, schema_context=schema_context)

        # Parse JSON with both entities and relations
        extraction_data = json.loads(result.extraction)

        # Extract entities
        entities_raw = extraction_data.get("entities", [])
        extracted_entities = []
        for ent in entities_raw:
            if isinstance(ent, dict):
                name = str(ent.get("name", "")).strip()
                types = ent.get("types", [])
                if not isinstance(types, list):
                    types = [str(types)] if types else []
                types = [str(t).strip() for t in types if str(t).strip()][:3]
                if name:
                    extracted_entities.append(ExtractedEntity(name=name, types=types))

        # Extract relations
        relations_raw = extraction_data.get("relations", [])
        extracted_relations = []
        for rel in relations_raw:
            if isinstance(rel, dict):
                name = str(rel.get("name", "")).strip()
                # Try "predicted_relations" first (new format), fall back to "types" for backward compatibility
                predicted_relations = rel.get("predicted_relations") or rel.get("types", [])
                if not isinstance(predicted_relations, list):
                    predicted_relations = [str(predicted_relations)] if predicted_relations else []
                predicted_relations = [str(t).strip() for t in predicted_relations if str(t).strip()][:3]
                if name:
                    # Store in ExtractedEntity.types field (internal structure unchanged)
                    extracted_relations.append(ExtractedEntity(name=name, types=predicted_relations))

        return extracted_entities, extracted_relations

    except json.JSONDecodeError:
        # Fallback: return query as entity, no relations
        return [ExtractedEntity(name=query, types=[])], []
    except Exception:
        raise


def _fuzzy_match_score(text1: str, text2: str) -> float:
    """Calculate fuzzy string similarity score between two texts.

    Uses SequenceMatcher from difflib for similarity calculation.

    Args:
        text1: First text string.
        text2: Second text string.

    Returns:
        Similarity score between 0.0 (no match) and 1.0 (exact match).
    """
    # Normalize: lowercase and strip whitespace
    text1_norm = text1.lower().strip()
    text2_norm = text2.lower().strip()

    # Calculate similarity
    return SequenceMatcher(None, text1_norm, text2_norm).ratio()


def _perform_hybrid_search(
    extracted_entity: ExtractedEntity,
    n_results: int,
    chromadb_collection: chromadb.Collection,
    config: Optional[KGConfig] = None,
    search_mode: SearchMode = SearchMode.hybrid,
    timeout: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Perform search using specified mode with entity type information.

    Search modes:
    - "hybrid": 66% semantic + 33% keyword (default)
    - "semantic": 100% semantic search only
    - "keyword": 100% keyword search only

    Uses entity type information to enhance both searches.

    Args:
        extracted_entity: ExtractedEntity with 'name' and 'types' fields.
        n_results: Total number of results to return.
        chromadb_collection: ChromaDB collection for semantic search.
        config: Optional KGConfig instance for configuration.
            If None, uses default KGConfig with env vars or built-in defaults.
        search_mode: Search strategy - "hybrid", "keyword", or "semantic".
        timeout: Optional timeout in seconds for keyword search. Default None.

    Returns:
        List of dictionaries with keys: entity_uri, label, source, score, description.
    """
    # Initialize config if not provided
    if config is None:
        config = KGConfig.default()

    # Extract entity name and types
    entity_name = extracted_entity["name"]
    entity_types = extracted_entity.get("types", [])

    # Calculate split based on search_mode
    if search_mode == SearchMode.hybrid:
        n_semantic = round(n_results * 0.6666)
        n_keyword = n_results - n_semantic
    elif search_mode == SearchMode.semantic:
        n_semantic = n_results
        n_keyword = 0
    else:  # keyword
        n_semantic = 0
        n_keyword = n_results

    # Perform semantic search if needed
    semantic_results = []
    semantic_time = 0.0
    if n_semantic > 0:
        semantic_start = time.time()
        semantic_results = semantic_search_entities(
            collection=chromadb_collection,
            query=entity_name,
            n_results=n_semantic,
            entity_types=entity_types
        )
        semantic_time = (time.time() - semantic_start) * 1000

    # Perform keyword search if needed
    keyword_results = []
    keyword_time = 0.0
    if n_keyword > 0:
        keyword_start = time.time()
        results_by_keyword = search_entities_by_keywords(
            keywords=[extracted_entity],
            limit=n_keyword,
            config=config,
            timeout=timeout
        )
        # Flatten list of lists into single list of dicts
        keyword_results = []
        for sublist in results_by_keyword:
            keyword_results.extend(sublist)
        keyword_time = (time.time() - keyword_start) * 1000

    # Format results
    formatted_results = []
    seen_uris = set()

    # Add semantic results first (priority)
    for result in semantic_results:
        entity_uri = result['entity_uri']
        if entity_uri not in seen_uris:
            seen_uris.add(entity_uri)
            # Extract label from description or use URI
            desc = result.get('description', '')
            if ':' in desc:
                label = desc.split(':')[1].split('.')[0].strip()
            else:
                label = entity_uri.split('/')[-1]

            formatted_results.append({
                'entity_uri': entity_uri,
                'label': label,
                'source': 'semantic',
                'score': result.get('distance', 0.0),
                'description': desc
            })

    # Add keyword results (only if not duplicate)
    for result in keyword_results:
        entity_uri = result['entity']
        if entity_uri not in seen_uris:
            seen_uris.add(entity_uri)
            formatted_results.append({
                'entity_uri': entity_uri,
                'label': result.get('label', entity_uri.split('/')[-1]),
                'source': 'keyword',
                'score': 0.0,  # Keyword search doesn't have distance score
                'description': result.get('label', '')
            })

    return formatted_results[:n_results]


def get_seed_nodes(
    query: str,
    n_results: int = 3,
    config: Optional[KGConfig] = None,
    search_mode: SearchMode = SearchMode.hybrid,
    keyword_timeout: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], List[ExtractedEntity]]:
    """Find seed nodes using specified search mode for extracted entities.

    This function:
    1. Extracts entities from the query using DSpy
    2. Performs search for each entity using the specified mode
    3. Returns all results with entity tracking

    Args:
        query: Natural language query from user.
        n_results: Number of results to return per entity.
        config: Optional KGConfig instance for configuration.
            If None, uses default KGConfig with env vars or built-in defaults.
            The config provides lm_api_key for entity extraction.
        search_mode: Search strategy - "hybrid", "keyword", or "semantic".
            Default is "hybrid" (66% semantic + 33% keyword).
        keyword_timeout: Optional timeout in seconds for keyword search.

    Returns:
        Tuple of (seed_nodes, extracted_entities):
            - seed_nodes: List of dictionaries with keys:
                - entity_uri: KG entity URI
                - label: Entity label
                - source: "semantic" or "keyword"
                - score: Similarity score (for semantic) or 0.0 (for keyword)
                - description: Entity description
                - entity_from: Original extracted entity name
            - extracted_entities: List of ExtractedEntity objects with 'name' and 'types' fields
    """
    # Initialize config if not provided
    if config is None:
        config = KGConfig.default()

    # Load .env file
    load_dotenv()

    # Initialize DSpy LM
    api_key = config.lm_api_key
    if not api_key:
        raise ValueError(
            "LM API key not found in config or environment"
        )

    lm = dspy.LM(model=config.openai_model, api_key=api_key)
    dspy.configure(lm=lm)

    # Extract entities from query
    extracted_entities = _extract_entities_from_query(query, config)

    # Load or create ChromaDB collection using config
    collection = get_entities_collection(config=config)

    # Perform hybrid search for each entity
    all_results = []
    for extracted_entity in extracted_entities:
        entity_name = extracted_entity["name"]
        entity_types = extracted_entity.get("types", [])

        results = _perform_hybrid_search(
            extracted_entity=extracted_entity,
            n_results=n_results,
            chromadb_collection=collection,
            config=config,
            search_mode=search_mode,
            timeout=keyword_timeout
        )

        # Add entity_from field for backward compatibility
        for result in results:
            result['entity_from'] = entity_name
            result['entity_types'] = entity_types

        all_results.extend(results)

    return all_results, extracted_entities


def citable(
    query: str,
    config: Optional[KGConfig] = None,
    n_results_per_entity: int = 3,
    search_mode: SearchMode = SearchMode.hybrid,
    keyword_timeout: Optional[int] = None
) -> Tuple[bool, Dict[str, List[str]]]:
    """Check if a query is citable by finding good quality seed nodes.

    A query is citable if we can find seed nodes in the KG that:
    - For semantic search: have similarity score < config.semantic_similarity_threshold
    - For keyword search: have fuzzy match score > config.fuzzy_match_threshold

    Args:
        query: Natural language query from user.
        config: Optional KGConfig instance for configuration.
            If None, uses default KGConfig with env vars or built-in defaults.
            The config provides semantic_similarity_threshold and fuzzy_match_threshold.
        n_results_per_entity: Number of results to check per entity.
        search_mode: Search strategy - "hybrid", "keyword", or "semantic".
            Default is "hybrid" (66% semantic + 33% keyword).
        keyword_timeout: Optional timeout in seconds for keyword search.

    Returns:
        Tuple of (is_citable, entity_to_nodes):
            - is_citable: True if at least one entity has citable nodes
            - entity_to_nodes: Dict mapping entity name to list of citable node URIs
    """
    # Initialize config if not provided
    if config is None:
        config = KGConfig.default()

    # Load .env file
    load_dotenv()

    # Initialize DSpy LM
    api_key = config.lm_api_key
    if not api_key:
        raise ValueError(
            "LM API key not found in config or environment"
        )

    lm = dspy.LM(model=config.openai_model, api_key=api_key)
    dspy.configure(lm=lm)

    # Extract entities from query
    extracted_entities = _extract_entities_from_query(query, config)

    # Load or create ChromaDB collection using config
    collection = get_entities_collection(config=config)

    # Check citability for each entity
    entity_to_nodes: Dict[str, List[str]] = {}

    for extracted_entity in extracted_entities:
        entity_name = extracted_entity["name"]
        entity_types = extracted_entity.get("types", [])

        results = _perform_hybrid_search(
            extracted_entity=extracted_entity,
            n_results=n_results_per_entity,
            chromadb_collection=collection,
            config=config,
            search_mode=search_mode,
            timeout=keyword_timeout
        )

        citable_nodes = []
        for result in results:
            is_citable_node = False

            # Check semantic results
            if result['source'] == 'semantic' and result['score'] < config.semantic_similarity_threshold:
                is_citable_node = True

            # Check keyword results with fuzzy matching
            elif result['source'] == 'keyword':
                fuzzy_score = _fuzzy_match_score(entity_name, result['label'])
                if fuzzy_score > config.fuzzy_match_threshold:
                    is_citable_node = True

            if is_citable_node:
                citable_nodes.append(result['entity_uri'])

        if citable_nodes:
            entity_to_nodes[entity_name] = citable_nodes

    # Determine overall citability
    is_citable = len(entity_to_nodes) > 0 and any(len(nodes) > 0 for nodes in entity_to_nodes.values())

    return is_citable, entity_to_nodes


def get_seed_and_relation_nodes(
    query: str,
    n_results_entity: int = 3,
    n_results_relation: int = 3,
    config: Optional[KGConfig] = None,
    search_mode: SearchMode = SearchMode.hybrid
) -> Tuple[List[Dict], List[ExtractedEntity], List[Dict], List[ExtractedEntity]]:
    """Extract entities and relations in ONE LLM call, then search both collections in parallel.

    This is the most efficient function for getting both seed entities and relations,
    minimizing LLM API calls and performing searches in parallel.

    Args:
        query: Natural language query from user.
        n_results_entity: Number of results to return per entity (default: 3).
        n_results_relation: Number of results to return per relation (default: 3).
        config: Optional KGConfig instance for configuration.
            If None, uses default KGConfig with env vars or built-in defaults.
        search_mode: Search strategy - "hybrid", "keyword", or "semantic".
            Default is "hybrid" (66% semantic + 33% keyword).

    Returns:
        Tuple of (entity_nodes, extracted_entities, relation_nodes, extracted_relations):
            - entity_nodes: List of entity node dicts with entity_uri, label, score, etc.
            - extracted_entities: List of ExtractedEntity dicts with name and types
            - relation_nodes: List of relation node dicts with relation_uri, label, distance, etc.
            - extracted_relations: List of ExtractedEntity dicts with name and types

    Example:
        >>> entity_nodes, entities, relation_nodes, relations = get_seed_and_relation_nodes(
        ...     "Did A. Hertz publish paper X in 2020?"
        ... )
        >>> print(f"Found {len(entities)} entities, {len(relations)} relations")
        Found 2 entities, 2 relations
    """
    if config is None:
        config = KGConfig.default()

    load_dotenv()

    # Initialize DSpy ONCE
    api_key = config.lm_api_key
    if not api_key:
        raise ValueError("LM API key not found in config or environment")

    lm = dspy.LM(model=config.openai_model, api_key=api_key)
    dspy.configure(lm=lm)

    # ===== SINGLE LLM CALL FOR BOTH ENTITIES AND RELATIONS =====
    extracted_entities, extracted_relations = _extract_entities_and_relations_from_query(query, config)

    # Load both ChromaDB collections
    from kgnode.chroma_db import get_entities_collection, get_relations_collection, semantic_search_relations

    entity_collection = get_entities_collection(config=config)
    relation_collection = get_relations_collection(config=config)

    # ===== PARALLEL SEARCH (using ThreadPoolExecutor) =====
    from concurrent.futures import ThreadPoolExecutor

    def search_entities():
        """Search entities in parallel thread."""
        entity_results = []
        for entity in extracted_entities:
            results = _perform_hybrid_search(
                extracted_entity=entity,
                n_results=n_results_entity,
                chromadb_collection=entity_collection,
                config=config,
                search_mode=search_mode
            )
            for r in results:
                r['entity_from'] = entity["name"]
                r['entity_types'] = entity.get("types", [])
            entity_results.extend(results)
        return entity_results

    def search_relations():
        """Search relations in parallel thread."""
        relation_results = []
        for relation in extracted_relations:
            results = semantic_search_relations(
                collection=relation_collection,
                query=relation["name"],
                n_results=n_results_relation,
                predicted_relations=relation.get("types", [])
            )
            for r in results:
                r['relation_from'] = relation["name"]
                r['relation_types'] = relation.get("types", [])
            relation_results.extend(results)
        return relation_results

    # Execute searches in parallel (significant speedup!)
    with ThreadPoolExecutor(max_workers=2) as executor:
        entity_future = executor.submit(search_entities)
        relation_future = executor.submit(search_relations)

        entity_nodes = entity_future.result()
        relation_nodes = relation_future.result()

    return entity_nodes, extracted_entities, relation_nodes, extracted_relations


if __name__ == "__main__":
    print("=" * 80)
    print("SEED FINDER EXAMPLES")
    print("=" * 80)

    # Example 1: Simple query with citable()
    # print("\n" + "=" * 80)
    # print("Example 1: Check citability for a simple query")
    # print("=" * 80)
    #
    # query1 = "What is the Wikidata ID of Robert Schober?"
    # print(f"\nQuery: {query1}\n")
    #
    # is_citable1, entity_nodes1 = citable(query1, n_results_per_entity=3)
    #
    # print(f"\n{'=' * 80}")
    # print(f"Result: {'CITABLE' if is_citable1 else 'NOT CITABLE'}")
    # print(f"{'=' * 80}")
    # for entity, nodes in entity_nodes1.items():
    #     print(f"\nEntity: {entity}")
    #     print(f"Citable nodes ({len(nodes)}):")
    #     for i, node in enumerate(nodes, 1):
    #         print(f"  {i}. {node}")

    # Example 2: Complex multi-entity query with citable()
    # print("\n\n" + "=" * 80)
    # print("Example 2: Check citability for a complex query")
    # print("=" * 80)
    #
    # query2 = "Show papers by John Smith published in ICML about machine learning"
    # print(f"\nQuery: {query2}\n")
    #
    # is_citable2, entity_nodes2 = citable(query2, n_results_per_entity=3)
    #
    # print(f"\n{'=' * 80}")
    # print(f"Result: {'CITABLE' if is_citable2 else 'NOT CITABLE'}")
    # print(f"{'=' * 80}")
    # for entity, nodes in entity_nodes2.items():
    #     print(f"\nEntity: {entity}")
    #     print(f"Citable nodes ({len(nodes)}):")
    #     for i, node in enumerate(nodes, 1):
    #         print(f"  {i}. {node}")
    #
    # Example 3: Get seed nodes with default parameters
    print("\n\n" + "=" * 80)
    print("Example 3: Get seed nodes with default parameters (n=3)")
    print("=" * 80)

    query3 = "Find papers about neural networks by Geoffrey Hinton"
    print(f"\nQuery: {query3}\n")

    seed_nodes3, extracted_entities3 = get_seed_nodes(query3, n_results=3)

    print(f"\n{'=' * 80}")
    print(f"Found {len(seed_nodes3)} seed nodes total from {len(extracted_entities3)} extracted entities")
    print(f"{'=' * 80}")
    #
    # # Group by entity_from
    # from collections import defaultdict
    # grouped = defaultdict(list)
    # for node in seed_nodes3:
    #     grouped[node['entity_from']].append(node)
    #
    # for entity, nodes in grouped.items():
    #     print(f"\n--- Entity: {entity} ({len(nodes)} nodes) ---")
    #     for i, node in enumerate(nodes, 1):
    #         print(f"{i}. [{node['source']}] {node['label']}")
    #         print(f"   URI: {node['entity_uri']}")
    #         print(f"   Score: {node['score']:.4f}")
    #         print()
    #
    # # Example 4: Get seed nodes with custom n_results
    # print("\n" + "=" * 80)
    # print("Example 4: Get seed nodes with n_results=5")
    # print("=" * 80)
    #
    # query4 = "somatic drivers in cancer genomes"
    # print(f"\nQuery: {query4}\n")
    #
    # seed_nodes4 = get_seed_nodes(query4, n_results=5)
    #
    # print(f"\n{'=' * 80}")
    # print(f"Found {len(seed_nodes4)} seed nodes total")
    # print(f"{'=' * 80}")
    #
    # # Group by entity_from
    # grouped4 = defaultdict(list)
    # for node in seed_nodes4:
    #     grouped4[node['entity_from']].append(node)
    #
    # for entity, nodes in grouped4.items():
    #     print(f"\n--- Entity: {entity} ({len(nodes)} nodes) ---")
    #     semantic_count = sum(1 for n in nodes if n['source'] == 'semantic')
    #     keyword_count = sum(1 for n in nodes if n['source'] == 'keyword')
    #     print(f"Breakdown: {semantic_count} semantic, {keyword_count} keyword")
    #     for i, node in enumerate(nodes, 1):
    #         print(f"{i}. [{node['source']}] {node['label']}")
    #         print(f"   URI: {node['entity_uri']}")
    #         if node['source'] == 'semantic':
    #             print(f"   Distance: {node['score']:.4f}")
    #         print()
