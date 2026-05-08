# Seed node: <https://dblp.org/pid/95/2265>
# Question: Show the Wikidata ID of the person Robert Schober.
# Subgraph: ??
# Sparql: SELECT DISTINCT ?answer WHERE { <https://dblp.org/pid/95/2265> <https://dblp.org/rdf/schema#wikidata> ?answer }
# Answer: https://www.wikidata.org/entity/Q55238282
import heapq
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, LiteralString, Optional, Set, Tuple

import dspy
import numpy as np
from dotenv import load_dotenv

from kgnode.core.kg_config import KGConfig
from kgnode.core.logging_config import get_logger


logger = get_logger(__name__)


def _get_allowed_relations(config: KGConfig, verbose: bool = False) -> Set[str]:
    """Get set of allowed relation URIs from schema, minus restricted relations.

    Args:
        config: KGConfig instance with schema and filtering configuration

    Returns:
        Set of allowed relation URIs (empty = allow all relations)

    Logic:
        - If schema filtering is OFF: Return empty set (allow all)
        - If schema filtering is ON and restricted filtering is OFF: Return schema relations
        - If schema filtering is ON and restricted filtering is ON: Return schema MINUS restricted
    """
    if not config.enable_schema_based_filtering:
        return set()  # Empty = allow all

    try:
        # Try loading from ChromaDB first (FAST!)
        from kgnode.core.schema_chromadb import _get_all_relation_uris

        relation_uris = _get_all_relation_uris(config)

        if not relation_uris:
            raise ValueError("No relation URIs found in ChromaDB")

        allowed_relations = set(relation_uris)

    except (ValueError, Exception):
        # ChromaDB schema collections don't exist - compile them first

        try:
            from kgnode.core.schema_chromadb import _compile_schema_collections

            _compile_schema_collections(
                config=config, ontology_path=config.ontology_path, force_recreate=False
            )

            # Try again after compilation
            from kgnode.core.schema_chromadb import _get_all_relation_uris

            relation_uris = _get_all_relation_uris(config)

            if not relation_uris:
                raise ValueError("No relation URIs found after compilation")

            allowed_relations = set(relation_uris)

        except Exception:
            return set()

    # Apply restricted relationship filtering if enabled
    if config.enable_restricted_relation_filtering:
        allowed_relations = allowed_relations - config.restricted_relations

    return allowed_relations


def get_neighbor_nodes(
    node_id: str,
    sparql_query: Callable,
    allowed_relations: Optional[Set[str]] = None,
    parallel: bool = True,
    neighbor_limit: Optional[int] = None,
) -> List[Tuple[str, str, str]]:
    """Get all neighbors of a node in the knowledge graph (both outgoing and incoming edges).

    This function performs bidirectional traversal:
    1. Outgoing edges: node → relation → neighbor
    2. Incoming edges: neighbor → relation → node (traversed inversely)

    For incoming edges, if an inverse property is declared in the ontology
    (e.g., authorOf ↔ authoredBy), the inverse relation URI is returned to
    maintain semantic correctness.

    Args:
        node_id: URI of the node (e.g., 'https://dblp.org/pid/95/2265')
        sparql_query: sparql_query method for querying the SPARQL endpoint.
        allowed_relations: Optional set of allowed relation URIs. If provided and non-empty,
            only neighbors connected via these relations will be returned.
            If None or empty, all relations are allowed.

    Returns:
        List of tuples: (relation, neighbor_node_id, neighbor_label)
        - relation: predicate URI connecting node_id to neighbor
        - neighbor_node_id: URI of the neighbor node
        - neighbor_label: rdfs:label of neighbor node (empty string if not available)
    """
    # Build VALUES clause for relation filtering
    if allowed_relations:
        values_list = " ".join([f"<{r}>" for r in allowed_relations])
        filter_clause = f"VALUES ?relation {{ {values_list} }}"
    else:
        filter_clause = ""

    # Build LIMIT clause for neighbor limiting
    limit_clause = f"LIMIT {neighbor_limit}" if neighbor_limit else ""

    # Query 1: Outgoing edges (node → relation → neighbor)
    outgoing_query = f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT DISTINCT ?relation ?neighbor ?label
    WHERE {{
      {filter_clause}
      <{node_id}> ?relation ?neighbor .

      # Optionally get the label of the neighbor
      OPTIONAL {{
        ?neighbor rdfs:label ?label .
      }}
    }}
    {limit_clause}
    """

    # Query 2: Incoming edges (neighbor → relation → node)
    # We traverse these inversely to find nodes pointing TO the current node
    incoming_query = f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>

    SELECT DISTINCT ?relation ?neighbor ?label ?inverse_relation
    WHERE {{
      {filter_clause}
      ?neighbor ?relation <{node_id}> .

      # Try to get inverse relation if declared (e.g., authorOf ↔ authoredBy)
      OPTIONAL {{
        ?relation owl:inverseOf ?inverse_relation .
      }}

      # Optionally get the label of the neighbor
      OPTIONAL {{
        ?neighbor rdfs:label ?label .
      }}
    }}
    {limit_clause}
    """

    # Execute both queries (parallel or sequential based on parameter)
    if parallel:
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_outgoing = executor.submit(sparql_query, outgoing_query)
            future_incoming = executor.submit(sparql_query, incoming_query)

            outgoing_results = future_outgoing.result()
            incoming_results = future_incoming.result()
    else:
        outgoing_results = sparql_query(outgoing_query)
        incoming_results = sparql_query(incoming_query)

    # Convert results to list of tuples
    neighbors = []
    seen = set()  # Track (relation, neighbor) pairs to avoid duplicates

    # Process outgoing edges
    for result in outgoing_results:
        relation = result["relation"]
        neighbor_node_id = result["neighbor"]
        neighbor_label = result.get("label", "")

        key = (relation, neighbor_node_id)
        if key not in seen:
            seen.add(key)
            neighbors.append((relation, neighbor_node_id, neighbor_label))

    # Process incoming edges
    for result in incoming_results:
        original_relation = result["relation"]
        neighbor_node_id = result["neighbor"]
        neighbor_label = result.get("label", "")

        # Use inverse relation if declared, otherwise use original relation
        # For example: if we traverse Paper → authoredBy → Author inversely,
        # we represent it as Author → authorOf → Paper
        inverse_relation = result.get("inverse_relation", "")
        relation_to_use = inverse_relation if inverse_relation else original_relation

        key = (relation_to_use, neighbor_node_id)
        if key not in seen:
            seen.add(key)
            neighbors.append((relation_to_use, neighbor_node_id, neighbor_label))

    return neighbors


def get_node_label(
    node_id: str, sparql_query: Callable, label_cache: Dict[str, str]
) -> str:
    """Get the rdfs:label of a node from the knowledge graph.

    Args:
        node_id: URI of the node (e.g., 'https://dblp.org/pid/95/2265')
        sparql_query: sparql_query method for querying the SPARQL endpoint
        label_cache: Dictionary to cache node labels for performance

    Returns:
        The label of the node, or the last part of the URI if no label exists
    """
    # Check cache first
    if node_id in label_cache:
        return label_cache[node_id]

    query = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?label
    WHERE {{
      <{node_id}> rdfs:label ?label .
    }}
    LIMIT 1
    """

    results = sparql_query(query)

    if results and len(results) > 0:
        label = results[0].get("label", "")
        label_cache[node_id] = label
        return label
    else:
        # Fallback: extract last part of URI
        fallback_label = node_id.split("/")[-1].split("#")[-1]
        label_cache[node_id] = fallback_label
        return fallback_label


def extract_relation_name(relation_uri: str) -> str:
    """Extract human-readable name from relation URI.

    Args:
        relation_uri: Full URI of the relation (e.g., 'https://dblp.org/rdf/schema#wikidata')

    Returns:
        Human-readable relation name (e.g., 'wikidata')

    Examples:
        'https://dblp.org/rdf/schema#wikidata' -> 'wikidata'
        'http://xmlns.com/foaf/0.1/name' -> 'name'
        'https://dblp.org/rdf/schema#authoredBy' -> 'authoredBy'
    """
    # Split by '#' first (common in RDF URIs)
    if "#" in relation_uri:
        return relation_uri.split("#")[-1]

    # Otherwise split by '/' and take the last part
    if "/" in relation_uri:
        return relation_uri.split("/")[-1]

    # If no separators, return as-is
    return relation_uri


def _format_seed_nodes_for_template_prompt(seed_nodes: List[Dict[str, Any]]) -> str:
    """Format seed nodes as a numbered list for template creation prompt.

    Args:
        seed_nodes: List of seed node dicts with 'entity_uri' and 'label' fields.

    Returns:
        Formatted string, e.g.:
            1. Robert Schober | <https://dblp.org/pid/95/2265>
            2. Sameh S. Askar | <https://dblp.org/pid/141/4165>
    """
    if not seed_nodes:
        return "(No seed nodes found)"

    lines = []
    for i, node in enumerate(seed_nodes, 1):
        label = node.get('label', node.get('entity_uri', '').split('/')[-1])
        uri = node.get('entity_uri', '')
        lines.append(f"{i}. {label} | <{uri}>")
    return "\n".join(lines)


def _format_relation_nodes_for_template_prompt(relation_nodes: List[Dict[str, Any]]) -> str:
    """Format relation nodes as a numbered list for template creation prompt.

    Args:
        relation_nodes: List of relation node dicts with 'relation_uri' and 'label' fields.

    Returns:
        Formatted string, e.g.:
            1. wikidata <https://dblp.org/rdf/schema#wikidata>
            2. authoredBy <https://dblp.org/rdf/schema#authoredBy>
    """
    if not relation_nodes:
        return "(No relations found)"

    lines = []
    for i, node in enumerate(relation_nodes, 1):
        label = node.get('label', extract_relation_name(node.get('relation_uri', '')))
        uri = node.get('relation_uri', '')
        lines.append(f"{i}. {label} <{uri}>")
    return "\n".join(lines)


def create_template(
    query: str,
    config: KGConfig,
    embedding_model,
    seed_nodes: Optional[List[Dict]] = None,
    relation_nodes: Optional[List[Dict]] = None
) -> tuple[Any, Any]:
    """Creates template embedding from query using DSpy rephrasing.

    If schema-aware templates are enabled and seed_nodes/relation_nodes are provided,
    uses validated nodes from ChromaDB search. Otherwise falls back to schema selection.

    Args:
        query: The original question (e.g., "Show the Wikidata ID of the person Robert Schober.")
        config: KGConfig instance with DSpy signature configuration
        embedding_model: Sentence transformer model for creating embeddings
        seed_nodes: Optional list of validated seed entity nodes from ChromaDB search
        relation_nodes: Optional list of validated relation nodes from ChromaDB search

    Returns:
        Tuple of (template_embedding, template_text)

    Example (without schema):
        query = "Show the Wikidata ID of the person Robert Schober."
        -> Rephrased to: "Robert Schober wikidata ID"
        -> Returns: (embedding vector, "Robert Schober wikidata ID")

    Example (with schema-aware templates):
        query = "Show papers by John Smith published in ICML"
        -> Uses relevant schema: Entity Types: Person, Publication, Venue; Relations: authoredBy, publishedIn
        -> Rephrased to: "John Smith authoredBy publications publishedIn ICML"
        -> Returns: (embedding vector, "John Smith authoredBy publications publishedIn ICML")
    """
    # Check if schema-aware templates are enabled
    if config.use_schema_aware_templates:
        try:
            # Prefer using seed_nodes and relation_nodes if provided (validated from ChromaDB)
            if seed_nodes is not None and relation_nodes is not None:
                # Format seed nodes and relation nodes for prompt
                formatted_seed_nodes = _format_seed_nodes_for_template_prompt(seed_nodes)
                formatted_relation_nodes = _format_relation_nodes_for_template_prompt(relation_nodes)

                # Use schema-aware signature and instruction with seed/relation nodes
                signature_with_instructions = config.schema_aware_template_creation_signature.with_instructions(
                    config.schema_aware_template_creation_instruction
                )

                if config.chain_of_thought:
                    template_creator = dspy.ChainOfThought(signature_with_instructions)
                else:
                    template_creator = dspy.Predict(signature_with_instructions)

                # Call DSpy with query, seed_nodes, and relation_nodes
                result = template_creator(
                    query=query,
                    seed_nodes=formatted_seed_nodes,
                    relation_nodes=formatted_relation_nodes
                )

            else:
                # Fallback: select relevant schema if seed/relation nodes not provided
                from kgnode.core.schema_chromadb import (
                    _compile_schema_collections,
                    _load_schema_collections,
                )

                # Try to load existing collections first
                try:
                    _load_schema_collections(config)
                except ValueError:
                    # Collections don't exist - compile them (one-time operation)
                    _compile_schema_collections(
                        config=config,
                        ontology_path=config.ontology_path,
                        force_recreate=config.force_schema_refresh,
                    )

                # Select relevant schema for this query
                from kgnode.core.schema_selector import (
                    _format_schema_for_prompt,
                    _select_relevant_schema,
                )

                relevant_schema = _select_relevant_schema(
                    query=query,
                    config=config,
                    top_k_entities=config.schema_top_k_entities,
                    top_k_relations=config.schema_top_k_relations,
                )

                formatted_seed_nodes = _format_schema_for_prompt(relevant_schema)
                formatted_relation_nodes = ""  # Not available in fallback mode

                # Use schema-aware signature and instruction
                signature_with_instructions = config.schema_aware_template_creation_signature.with_instructions(
                    config.schema_aware_template_creation_instruction
                )

                if config.chain_of_thought:
                    template_creator = dspy.ChainOfThought(signature_with_instructions)
                else:
                    template_creator = dspy.Predict(signature_with_instructions)

                # Call DSpy with formatted schema as seed_nodes (legacy format)
                result = template_creator(
                    query=query,
                    seed_nodes=formatted_seed_nodes,
                    relation_nodes=formatted_relation_nodes
                )

        except Exception:
            # If schema-aware processing fails, fall back to standard template creation
            signature_with_instructions = config.template_creation_signature.with_instructions(
                config.template_creation_instruction
            )

            if config.chain_of_thought:
                template_creator = dspy.ChainOfThought(signature_with_instructions)
            else:
                template_creator = dspy.Predict(signature_with_instructions)

            result = template_creator(query=query)
    else:
        # Standard template creation (original behavior)
        signature_with_instructions = config.template_creation_signature.with_instructions(
            config.template_creation_instruction
        )

        if config.chain_of_thought:
            template_creator = dspy.ChainOfThought(signature_with_instructions)
        else:
            template_creator = dspy.Predict(signature_with_instructions)

        result = template_creator(query=query)

    template_text = result.path_pattern.strip()

    # Create embedding
    template_embedding = embedding_model.encode(template_text, convert_to_numpy=True)

    return template_embedding, template_text


def _build_path_text(
    path_nodes: List[str],
    path_relations: List[str],
    label_cache: Dict[str, str],
    sparql_query: Callable,
) -> str:
    """Build path text without computing embedding.

    Args:
        path_nodes: List of node URIs in the path
        path_relations: List of relation URIs connecting nodes
        label_cache: Dictionary to cache node labels for performance
        sparql_query: sparql_query method for querying the SPARQL endpoint

    Returns:
        Path text string (e.g., "Robert Schober wikidata Q55238282")
    """
    # Handle empty path (just seed node)
    if len(path_nodes) == 1 and len(path_relations) == 0:
        return get_node_label(path_nodes[0], sparql_query, label_cache)

    # Build path text: "label_a relation1 label_b relation2 label_c ..."
    path_parts = []
    for i, node in enumerate(path_nodes):
        node_label = get_node_label(node, sparql_query, label_cache)
        path_parts.append(node_label)

        # Add relation if not the last node
        if i < len(path_relations):
            relation_name = extract_relation_name(path_relations[i])
            path_parts.append(relation_name)

    return " ".join(path_parts)


def compute_path_embedding(
    path_nodes: List[str],
    path_relations: List[str],
    embedding_model,
    label_cache: Dict[str, str],
    sparql_query: Callable,
) -> tuple[Any, str | LiteralString]:
    """Returns embedding vector for current path.

    Args:
        path_nodes: List of node URIs in the path (e.g., [a, b, v, w])
        path_relations: List of relation URIs connecting nodes (e.g., [r1, r2, r3])
        embedding_model: Sentence transformer model for creating embeddings
        label_cache: Dictionary to cache node labels for performance
        sparql_query: sparql_query method for querying the SPARQL endpoint

    Returns:
        Path embedding as numpy array

    Example:
        path_nodes = ['https://dblp.org/pid/95/2265', 'https://www.wikidata.org/entity/Q55238282']
        path_relations = ['https://dblp.org/rdf/schema#wikidata']
        -> Creates text: "Robert Schober wikidata Q55238282"
        -> Returns: embedding vector
    """
    path_text = _build_path_text(path_nodes, path_relations, label_cache, sparql_query)
    path_embedding = embedding_model.encode(path_text, convert_to_numpy=True)
    return path_embedding, path_text


def compute_transition_probability(
    path_embedding: np.ndarray, template_embedding: np.ndarray
) -> float:
    """Returns probability score using cosine similarity and softmax.

    Args:
        path_embedding: Embedding vector of the current path
        template_embedding: Embedding vector of the query template

    Returns:
        Probability score (between 0 and 1)

    Note:
        Uses cosine similarity followed by softmax normalization.
        Since we're computing probability for a single transition,
        we use exponential of cosine similarity as the probability.
    """
    # Compute cosine similarity
    # cosine_sim = (A · B) / (||A|| × ||B||)
    dot_product = np.dot(path_embedding, template_embedding)
    norm_path = np.linalg.norm(path_embedding)
    norm_template = np.linalg.norm(template_embedding)

    cosine_sim = dot_product / (norm_path * norm_template)

    # Apply exponential (softmax-like transformation)
    # This gives us a probability-like score
    probability = np.exp(cosine_sim)

    return float(probability)


def select_next_nodes(
    current_node: str,
    path_history: List[str],
    relation_history: List[str],
    template_embedding: np.ndarray,
    previous_max_prob: float,
    embedding_model,
    kg_connection,
    label_cache: Dict[str, str],
    max_k: int = 2,
    allowed_relations: Optional[Set[str]] = None,
    config: Optional[KGConfig] = None,
    path_prob_print: bool = False,
    seed_relation_uris: Optional[Set[str]] = None,
    current_hop: int = 0,
    path_embedding_cache: Optional[OrderedDict] = None,
    max_cache_size: int = 5000,
) -> List[Tuple[str, str, float]]:
    """Returns top neighbors selected based on relation type.

    Args:
        current_node: URI of the current node to expand
        path_history: List of node URIs visited so far (not including current_node)
        relation_history: List of relation URIs used so far
        template_embedding: Embedding vector of the query template
        previous_max_prob: Maximum probability from the previous hop
        embedding_model: Sentence transformer model for creating embeddings
        kg_connection: SPARQL query function
        label_cache: Dictionary to cache node labels
        max_k: Maximum number of neighbors to select for non-seed relations (default: 2)
        allowed_relations: Optional set of allowed relation URIs for filtering
        path_prob_print: print the path and probability
        seed_relation_uris: Optional set of seed relation URIs. Paths using these relations
            get a 1.5x probability boost and can select up to seed_relation_max_k neighbors.
        current_hop: Current hop count (kept for compatibility, not used)
        path_embedding_cache: Optional LRU cache (OrderedDict) for path embeddings to avoid
            recomputation of embeddings for paths sharing common prefixes.
        max_cache_size: Maximum number of entries in the path embedding cache (default: 5000)

    Returns:
        List of tuples: [(neighbor_node_id, relation, probability), ...]
        Sorted by probability in descending order.
        - Up to seed_relation_max_k (default 5) neighbors via seed relations
        - Up to max_k (default 2) neighbors via other relations
        Only includes neighbors where probability > previous_max_prob.
    """
    # Get neighbor limit from config (default 100)
    neighbor_limit = (
        config.neighbor_limit if config and config.enable_neighbor_limiting else None
    )
    parallel = config.parallel_neighbor_queries if config else True

    # Get neighbors with optimizations
    neighbors = get_neighbor_nodes(
        current_node,
        kg_connection,
        allowed_relations,
        parallel=parallel,
        neighbor_limit=neighbor_limit,
    )

    if not neighbors:
        return []

    # Store neighbor labels in cache for later use
    for relation, neighbor_id, neighbor_label in neighbors:
        if neighbor_label and neighbor_id not in label_cache:
            label_cache[neighbor_id] = neighbor_label

    # Prepare data for batch embedding
    candidate_paths = []
    candidate_info = []  # Store (neighbor_id, relation) for each candidate

    for relation, neighbor_id, neighbor_label in neighbors:
        # Skip literals - don't add them to candidate paths for traversal
        # Literals will still be in the final subgraph via triplet_uris
        if not (neighbor_id.startswith("http://") or neighbor_id.startswith("https://")):
            continue

        # Cycle detection: skip if neighbor is already in current path
        # This prevents cycles like: seed → A → B → A
        if neighbor_id in path_history or neighbor_id == current_node:
            continue

        # Build new path: path_history + [current_node] + [neighbor_id]
        new_path_nodes = path_history + [current_node, neighbor_id]
        new_path_relations = relation_history + [relation]

        candidate_paths.append((new_path_nodes, new_path_relations))
        candidate_info.append((neighbor_id, relation))

    # Build path texts and check cache (with LRU eviction)
    path_texts_to_encode = []
    uncached_indices = []  # Track which candidates need encoding
    path_embeddings = []

    for idx, (path_nodes, path_relations) in enumerate(candidate_paths):
        # Build path text
        path_text = _build_path_text(
            path_nodes, path_relations, label_cache, kg_connection
        )

        # Check cache if available
        cache_key = tuple(path_nodes)
        if path_embedding_cache is not None and cache_key in path_embedding_cache:
            # Cache hit - use cached embedding and mark as recently used
            path_embedding_cache.move_to_end(cache_key)
            cached_emb = path_embedding_cache[cache_key]
            path_embeddings.append((cached_emb, path_text))
        else:
            # Cache miss - need to encode
            path_texts_to_encode.append(path_text)
            uncached_indices.append(idx)
            # Placeholder - will be filled after batch encoding
            path_embeddings.append(None)

    # Batch encode only uncached paths (MUCH faster than sequential!)
    if path_texts_to_encode:
        new_embeddings_array = embedding_model.encode(
            path_texts_to_encode, convert_to_numpy=True, show_progress_bar=False
        )

        # Add to cache and update results
        for i, new_emb in enumerate(new_embeddings_array):
            original_idx = uncached_indices[i]
            path_nodes = candidate_paths[original_idx][0]
            path_text = path_texts_to_encode[i]

            # Update result at correct position
            path_embeddings[original_idx] = (new_emb, path_text)

            # Add to cache with LRU eviction if cache is enabled
            if path_embedding_cache is not None:
                cache_key = tuple(path_nodes)
                path_embedding_cache[cache_key] = new_emb

                # Evict oldest entry if cache is full (LRU eviction)
                if len(path_embedding_cache) > max_cache_size:
                    path_embedding_cache.popitem(last=False)  # Remove oldest (first) item

    # Compute probabilities for all candidates
    probabilities = []
    for path_emb, path_text in path_embeddings:
        prob = compute_transition_probability(path_emb, template_embedding)
        if path_prob_print:
            logger.debug(f"........... path_text: {path_text} --> prob: {prob}")
        probabilities.append(prob)

    # Apply probability boosting for seed relations (Option 1: Probability Boosting)
    # This prefers paths using seed relations without hard filtering
    # Get seed relation boost from config (default 1.2)
    seed_relation_boost = 1.2
    if config and hasattr(config, 'seed_relation_boost'):
        seed_relation_boost = config.seed_relation_boost

    boosted_probabilities = []
    for (neighbor_id, relation), prob in zip(candidate_info, probabilities):
        if seed_relation_uris and relation in seed_relation_uris:
            boosted_prob = prob * seed_relation_boost
            if path_prob_print:
                logger.debug(
                    f"Boosted seed relation {extract_relation_name(relation)}: "
                    f"{prob:.4f} → {boosted_prob:.4f} (boost={seed_relation_boost}x)"
                )
        else:
            boosted_prob = prob
        boosted_probabilities.append(boosted_prob)

    # Filter candidates where probability > previous_max_prob
    valid_candidates = []
    for (neighbor_id, relation), prob in zip(candidate_info, boosted_probabilities):
        if prob > previous_max_prob:
            valid_candidates.append((neighbor_id, relation, prob))

    # Get seed_relation_max_k from config (default 5)
    seed_relation_max_k = 5
    if config and hasattr(config, 'seed_relation_max_k'):
        seed_relation_max_k = config.seed_relation_max_k

    # Split candidates by seed relation vs non-seed relation
    seed_relation_candidates = []
    other_candidates = []

    for neighbor_id, relation, prob in valid_candidates:
        if seed_relation_uris and relation in seed_relation_uris:
            seed_relation_candidates.append((neighbor_id, relation, prob))
        else:
            other_candidates.append((neighbor_id, relation, prob))

    # Sort each group by probability (descending)
    seed_relation_candidates.sort(key=lambda x: x[2], reverse=True)
    other_candidates.sort(key=lambda x: x[2], reverse=True)

    # Take top-K from each group
    selected = []
    selected.extend(seed_relation_candidates[:seed_relation_max_k])  # Top-5 seed relations
    selected.extend(other_candidates[:max_k])  # Top-2 others

    # Sort combined result by probability (descending)
    selected.sort(key=lambda x: x[2], reverse=True)

    return selected


def _check_adaptive_stopping(
    subgraphs: List[Dict[str, Any]],
    queue: List,
    min_subgraphs: int,
    absolute_prob_threshold: float,
    quality_threshold_ratio: float
) -> bool:
    """Check if adaptive stopping conditions are met.

    Args:
        subgraphs: List of collected subgraphs so far
        queue: Priority queue of paths to explore
        min_subgraphs: Minimum number of subgraphs required before stopping
        absolute_prob_threshold: Absolute probability threshold (e.g., 1.5)
        quality_threshold_ratio: Relative quality ratio (e.g., 0.65)

    Returns:
        True if we should stop exploration, False otherwise
    """
    # Check if we have at least one depth >= 1 subgraph
    has_valid_subgraph = any(sg['path_depth'] >= 1 for sg in subgraphs)

    # Adaptive stopping check (only if we have at least one valid subgraph)
    if has_valid_subgraph and len(subgraphs) >= min_subgraphs and queue:
        # Peek at next best path in queue
        next_priority = -queue[0][0]

        # Absolute threshold check
        if next_priority < absolute_prob_threshold:
            logger.info(
                f"Adaptive stop (absolute): next_prob={next_priority:.3f} < "
                f"threshold={absolute_prob_threshold}"
            )
            return True

        # Relative quality check
        collected_probs = [sg['probability'] for sg in subgraphs]
        median_prob = np.median(collected_probs)
        relative_threshold = median_prob * quality_threshold_ratio

        if next_priority < relative_threshold:
            logger.info(
                f"Adaptive stop (relative): next_prob={next_priority:.3f} < "
                f"threshold={relative_threshold:.3f} (median={median_prob:.3f}, "
                f"ratio={quality_threshold_ratio})"
            )
            return True

    return False


def get_subgraphs(
    seed_node: str,
    query: str,
    config: Optional[KGConfig] = None,
    kg_connection=None,
    max_hops: int = 10,
    max_k: int = 2,
    verbose: bool = False,
    seed_nodes: Optional[List[Dict]] = None,
    relation_nodes: Optional[List[Dict]] = None
) -> Tuple[List[Dict[str, Any]], str]:
    """Returns list of subgraphs and template text.

    Args:
        seed_node: URI of the starting node
        query: The original query/question
        config: Optional KGConfig instance for configuration.
            If None, uses default KGConfig with environment variables or built-in defaults.
        kg_connection: Optional SPARQL query function. If None, creates from config.
        max_hops: Maximum number of hops to explore (default: 10)
        max_k: Maximum number of neighbors to select per node (default: 2)
        verbose: Whether to print verbose output
        seed_nodes: Optional list of validated seed nodes from ChromaDB search
        relation_nodes: Optional list of validated relation nodes from ChromaDB search

    Returns:
        Tuple of (subgraphs, template_text):
        - subgraphs: List of subgraphs, each as:
          {
              'nodes': [node_uri1, node_uri2, ...],
              'edges': [(node_uri1, relation_uri, node_uri2), ...],
              'path': [(node_uri, node_label), (relation_name, None), ...],
              'probability': float,
              'path_depth': int
          }
        - template_text: The rephrased query template
    """
    # Initialize config if not provided
    if config is None:
        config = KGConfig.default()

    # Initialize embedding model from config
    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer(config.embedding_model)

    # Initialize DSpy LM
    load_dotenv()
    api_key = config.lm_api_key
    if not api_key:
        raise ValueError("LM API key not found in config or environment")

    lm = dspy.LM(model=config.openai_model, api_key=api_key)
    dspy.configure(lm=lm)

    # Initialize kg_connection if not provided
    if kg_connection is None:
        from kgnode.core.sparql_query import execute_sparql_query

        kg_connection = lambda q: execute_sparql_query(q, config=config)

    # Initialize label cache for this query
    label_cache = {}

    # Initialize allowed relations for schema-based filtering (Phase 1)
    allowed_relations = _get_allowed_relations(config, verbose=verbose)

    # Extract seed relation URIs for probability boosting
    seed_relation_uris = None
    if relation_nodes:
        seed_relation_uris = {node['relation_uri'] for node in relation_nodes}
        logger.debug(f"Extracted {len(seed_relation_uris)} seed relation URIs for boosting")

    # Create template embedding
    template_embedding, template_text = create_template(
        query, config, embedding_model, seed_nodes, relation_nodes
    )

    # Initialize priority queue (max-heap using negative probability)
    # Format: (negative_priority, counter, current_node, path_nodes, path_relations, previous_max_prob, hop_count)
    queue = []
    counter = 0  # For tie-breaking to maintain stable ordering

    # Start with seed node at hop 0
    heapq.heappush(queue, (-0.0, counter, seed_node, [], [], 0.0, 0))
    counter += 1

    # Store completed subgraphs
    subgraphs = []

    # Track seen triplet combinations to avoid duplicate subgraphs
    seen_triplet_sets = set()

    # Initialize LRU cache for path embeddings (Optimization #2)
    # Cache maps path node sequences to their embeddings to avoid recomputation
    path_embedding_cache = OrderedDict()
    MAX_CACHE_SIZE = 5000

    # Get adaptive stopping parameters from config
    min_subgraphs = config.min_subgraphs if config else 3
    max_subgraphs_limit = config.max_subgraphs if config else 15
    quality_threshold_ratio = config.quality_threshold_ratio if config else 0.65
    absolute_prob_threshold = config.absolute_prob_threshold if config else 1.5

    while queue and len(subgraphs) < max_subgraphs_limit:
        neg_priority, _, current_node, path_nodes, path_relations, previous_max_prob, current_hop = (
            heapq.heappop(queue)
        )

        # Check stopping conditions
        if current_hop >= max_hops:
            # Only store subgraphs with path_depth >= 1 (skip isolated seed nodes)
            if current_hop >= 1:
                subgraph = _build_subgraph(
                    path_nodes + [current_node],
                    path_relations,
                    label_cache,
                    kg_connection,
                    template_embedding,
                    embedding_model,
                    probability=previous_max_prob,
                    path_depth=current_hop
                )
                # Deduplicate: only add if triplets are unique
                triplet_tuple = tuple(sorted(subgraph['triplet_uris']))
                if triplet_tuple not in seen_triplet_sets:
                    seen_triplet_sets.add(triplet_tuple)
                    subgraphs.append(subgraph)

            # Check adaptive stopping conditions
            if _check_adaptive_stopping(
                subgraphs, queue, min_subgraphs, absolute_prob_threshold, quality_threshold_ratio
            ):
                break

            continue

        # Select next nodes to explore
        selected_neighbors = select_next_nodes(
            current_node=current_node,
            path_history=path_nodes,
            relation_history=path_relations,
            template_embedding=template_embedding,
            previous_max_prob=previous_max_prob,
            embedding_model=embedding_model,
            kg_connection=kg_connection,
            label_cache=label_cache,
            max_k=max_k,
            allowed_relations=allowed_relations,
            config=config,
            seed_relation_uris=seed_relation_uris,  # Probability Boosting for seed relations
            current_hop=current_hop,  # Pass hop count to use seed_node_max_k for hop 0
            path_embedding_cache=path_embedding_cache,  # LRU cache for path embeddings
            max_cache_size=MAX_CACHE_SIZE,  # Maximum cache size
        )

        # If no valid neighbors found (all probabilities <= previous_max_prob), store subgraph
        if not selected_neighbors:
            # Only store subgraphs with path_depth >= 1 (skip isolated seed nodes)
            if current_hop >= 1:
                subgraph = _build_subgraph(
                    path_nodes + [current_node],
                    path_relations,
                    label_cache,
                    kg_connection,
                    template_embedding,
                    embedding_model,
                    probability=previous_max_prob,
                    path_depth=current_hop
                )
                # Deduplicate: only add if triplets are unique
                triplet_tuple = tuple(sorted(subgraph['triplet_uris']))
                if triplet_tuple not in seen_triplet_sets:
                    seen_triplet_sets.add(triplet_tuple)
                    subgraphs.append(subgraph)

            # Check adaptive stopping conditions
            if _check_adaptive_stopping(
                subgraphs, queue, min_subgraphs, absolute_prob_threshold, quality_threshold_ratio
            ):
                break

            continue

        # Add selected neighbors to priority queue (explore high-probability paths first)
        # Each neighbor uses its own probability as threshold for its children
        for neighbor_id, relation, prob in selected_neighbors:
            new_path_nodes = path_nodes + [current_node]
            new_path_relations = path_relations + [relation]

            heapq.heappush(
                queue,
                (
                    -prob,              # Use individual neighbor's probability for priority
                    counter,            # Tie-breaker for stable ordering
                    neighbor_id,
                    new_path_nodes,
                    new_path_relations,
                    prob,               # Pass individual prob as threshold for children
                    current_hop + 1,
                )
            )
            counter += 1

    # Sort subgraphs by probability (descending) before returning
    subgraphs.sort(key=lambda sg: sg.get("probability", 0.0), reverse=True)

    # Log cache statistics for performance monitoring
    if path_embedding_cache:
        cache_size = len(path_embedding_cache)
        logger.debug(f"Path embedding cache final size: {cache_size}/{MAX_CACHE_SIZE}")

    return (subgraphs, template_text)


def _build_subgraph(
    path_nodes: List[str],
    path_relations: List[str],
    label_cache: Dict[str, str],
    kg_connection: Callable,
    template_embedding: np.ndarray,
    embedding_model,
    probability: float = 0.0,
    path_depth: int = 0,
) -> Dict[str, Any]:
    """Helper function to build subgraph dictionary from path.

    Args:
        path_nodes: List of node URIs in the path
        path_relations: List of relation URIs in the path
        label_cache: Dictionary containing cached node labels
        kg_connection: SPARQL query function
        template_embedding: Template embedding for computing true path probability
        embedding_model: Sentence transformer model for creating embeddings
        probability: Path probability/confidence score (DEPRECATED - computed from full path)
        path_depth: Number of hops in the path (default 0)

    Returns:
        Dictionary with 'triplet_uris', 'path_with_label', 'probability', and 'path_depth' keys
        - triplet_uris: [(source_uri, relation_uri, target_uri), ...]
        - path_with_label: [(uri, label, is_node), ...] where is_node is True for nodes, False for relations
        - probability: Float confidence score computed from full path semantic similarity to query
        - path_depth: Number of hops in the path
    """
    # Build edges list: [(source, relation, target), ...]
    # Only include the actual path triplets, no additional metadata
    edges = []
    for i in range(len(path_relations)):
        source = path_nodes[i]
        relation = path_relations[i]
        target = path_nodes[i + 1]
        edges.append((source, relation, target))

    # Build path with (uri, label, is_node) format
    path = []
    for i, node in enumerate(path_nodes):
        node_label = get_node_label(node, kg_connection, label_cache)
        path.append((node, node_label, True))  # True = is_node

        # Add relation if not the last node
        if i < len(path_relations):
            relation_uri = path_relations[i]
            relation_name = extract_relation_name(relation_uri)
            path.append((relation_uri, relation_name, False))  # False = is_relation

    # Compute TRUE probability: full path semantic similarity to query template
    path_embedding, path_text = compute_path_embedding(
        path_nodes, path_relations, embedding_model, label_cache, kg_connection
    )
    true_probability = compute_transition_probability(path_embedding, template_embedding)

    return {
        "triplet_uris": edges,
        "path_with_label": path,
        "probability": true_probability,
        "path_depth": path_depth
    }


if __name__ == "__main__":
    # neighbors = get_neighbor_nodes("https://dblp.org/pid/95/2265", _sparql_query)
    # print(neighbors)
    # print(get_node_label("https://dblp.org/pid/95/2265", _sparql_query, {}))
    # print(extract_relation_name("https://dblp.org/rdf/schema#wikidata"))

    # ---------------------------- template embedding --------------------------
    # from openai import OpenAI
    # from sentence_transformers import SentenceTransformer
    #
    # # Initialize models
    # openai_client = OpenAI(api_key="api_key")
    # embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    #
    # # Test create_template
    # query = "Show the Wikidata ID of the person Robert Schober."
    # template_emb, template_text = create_template(query, openai_client, embedding_model)
    # print(f"Template embedding shape: {template_emb.shape}")
    # print(f"Template text : {template_text}")
    #
    # # Test with multiple entities
    # query2 = "Did John Smith and Mary Johnson collaborate on any papers?"
    # template_emb2, template_text_2 = create_template(query2, openai_client, embedding_model)
    # print(f"\nTemplate embedding 2 shape: {template_emb2.shape}")
    # print(f"Template text 2: {template_text_2}")

    # --------------------------------- path embedding -----------------------------
    # from sentence_transformers import SentenceTransformer
    # from kgnode._sparql_query import _sparql_query
    #
    # # Initialize model
    # embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    # label_cache = {}
    #
    # # Test 1: Single node (seed node)
    # path_nodes = ['https://dblp.org/pid/95/2265']
    # path_relations = []
    # path_emb, path_text = compute_path_embedding(path_nodes, path_relations, embedding_model, label_cache, _sparql_query)
    # print(f"Single node embedding shape: {path_emb.shape}")
    # print(f"Cached label: {label_cache.get('https://dblp.org/pid/95/2265', 'Not found')}")
    # print(f"Path embedding text: {path_text}")
    #
    # # Test 2: Two nodes with one relation
    # path_nodes = ['https://dblp.org/pid/95/2265', 'https://www.wikidata.org/entity/Q55238282']
    # path_relations = ['https://dblp.org/rdf/schema#wikidata']
    # path_emb_2, path_text_2 = compute_path_embedding(path_nodes, path_relations, embedding_model, label_cache, _sparql_query)
    # print(f"\nTwo node path embedding shape: {path_emb_2.shape}")
    # print(f"Path embedding text: {path_text_2}")

    # --------------------------------- compute transitional probability -----------------------------
    # from sentence_transformers import SentenceTransformer
    # from openai import OpenAI
    # from kgnode._sparql_query import _sparql_query
    # import numpy as np
    #
    # # Initialize models
    # embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    # openai_client = OpenAI(api_key="api_key")
    # label_cache = {}
    #
    # # Create template embedding
    # query = "Show the Wikidata ID of the person Robert Schober."
    # template_emb, template_text = create_template(query, openai_client, embedding_model)
    #
    # # Create a path embedding
    # path_nodes = ['https://dblp.org/pid/95/2265', 'https://www.wikidata.org/entity/Q55238282']
    # path_relations = ['https://dblp.org/rdf/schema#wikidata']
    # path_emb, path_text = compute_path_embedding(path_nodes, path_relations, embedding_model, label_cache, _sparql_query)
    #
    # # Compute transition probability
    # prob = compute_transition_probability(path_emb, template_emb)
    # print(f"Transition probability: {prob}")
    # print(f"Probability type: {type(prob)}")
    #
    # # Test with a less relevant path (should have lower probability)
    # irrelevant_path_nodes = ['https://dblp.org/pid/95/2265']
    # irrelevant_path_relations = []
    # irrelevant_path_emb, irrelevant_path_text = compute_path_embedding(irrelevant_path_nodes, irrelevant_path_relations,
    #                                              embedding_model, label_cache, _sparql_query)
    # prob2 = compute_transition_probability(irrelevant_path_emb, template_emb)
    # print(f"\nIrrelevant path probability: {prob2}")
    # print(f"template_text: {template_text} \npath_text: {path_text} \nirrelevant_path_text: {irrelevant_path_text}")

    # --------------------------------- select next node -----------------------------
    # from sentence_transformers import SentenceTransformer
    # from openai import OpenAI
    # from kgnode._sparql_query import _sparql_query
    #
    # # Initialize models
    # embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    # openai_client = OpenAI(api_key="api_key")
    # label_cache = {}
    #
    # # Create template embedding
    # query = "Show the Wikidata ID of the person Robert Schober."
    # template_emb, template_text = create_template(query, openai_client, embedding_model)
    #
    # # Test select_next_nodes
    # seed_node = 'https://dblp.org/pid/95/2265'
    # path_history = []
    # relation_history = []
    # previous_max_prob = 0.0  # Starting probability
    #
    # selected_nodes = select_next_nodes(
    #     current_node=seed_node,
    #     path_history=path_history,
    #     relation_history=relation_history,
    #     template_embedding=template_emb,
    #     previous_max_prob=previous_max_prob,
    #     embedding_model=embedding_model,
    #     kg_connection=_sparql_query,
    #     label_cache=label_cache,
    #     max_k=2
    # )
    #
    # print(f"Number of selected neighbors: {len(selected_nodes)}")
    # for neighbor_id, relation, prob in selected_nodes:
    #     print(f"\nNeighbor: {neighbor_id}")
    #     print(f"Relation: {extract_relation_name(relation)}")
    #     print(f"Probability: {prob:.6f}")

    # --------------------------------- select next node -----------------------------------
    from openai import OpenAI
    from sentence_transformers import SentenceTransformer

    # Initialize models
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    openai_client = OpenAI(api_key="api_key")

    # Test extract_subgraphs_bfs
    seed_node = "https://dblp.org/pid/95/2265"
    query = "Show the Wikidata ID of the person Robert Schober."

    subgraphs, template_text = get_subgraphs(
        seed_node=seed_node,
        query=query,
        max_hops=5,  # Keep it small for testing
        max_k=4,
    )

    print(f"Template text: {template_text}")
    print(f"Number of subgraphs found: {len(subgraphs)}")
    print("\n" + "=" * 80)

    for i, subgraph in enumerate(subgraphs[:3]):  # Show first 3 subgraphs
        print(f"\nSubgraph {i + 1}:")
        print(f"Triplet count: {len(subgraph['triplet_uris'])}")
        print(f"Path with label count: {len(subgraph['path_with_label'])}")
        print(f"Triplet uris: {subgraph['triplet_uris']}")
        print(f"Path with labels: {subgraph['path_with_label']}")
        print("\n  Path:")
        for uri, label, is_node in subgraph["path_with_label"]:
            if is_node:  # It's a node
                print(f"    Node: {label} ({uri})")
            else:  # It's a relation
                print(f"    relation --> {label} {uri} -->")
        print("\n" + "-" * 80)

    # --------------------------------- validate subgraph -----------------------------------
    # from kgnode.core.sparql_query import execute_sparql_query
    # from kgnode.validator import validate_subgraph
    # from kgnode.core.kg_config import KGConfig
    #
    # # Initialize config
    # load_dotenv()
    # config = KGConfig.default()
    #
    # # Test full pipeline
    # seed_node = 'https://dblp.org/pid/95/2265'
    # query = "Show the Wikidata ID of the person Robert Schober."
    # answer_sparql = """
    # SELECT DISTINCT ?answer WHERE {
    #     <https://dblp.org/pid/95/2265> <https://dblp.org/rdf/schema#wikidata> ?answer
    # }
    # """
    #
    # # Extract subgraphs
    # subgraphs = get_subgraphs(
    #     seed_node=seed_node,
    #     query=query,
    #     config=config,
    #     kg_connection=execute_sparql_query,
    #     max_hops=3,
    #     max_k=2
    # )
    #
    # print(subgraphs)
    # print(f"Total subgraphs found: {len(subgraphs)}")
    # print("\n" + "=" * 80)
    #
    # # Validate each subgraph
    # valid_count = 0
    # for i, subgraph in enumerate(subgraphs):
    #     is_valid = validate_subgraph(subgraph, answer_sparql, execute_sparql_query)
    #     if is_valid:
    #         valid_count += 1
    #         print(f"\n✓ Subgraph {i + 1} is VALID!")
    #         print(f"  Edges: {subgraph['triplet_uris']}")
    #         print(f"  Path:")
    #         for uri, label, is_node in subgraph['path_with_label']:
    #             if is_node:
    #                 print(f"    Node: {label} ({uri})")
    #             else:
    #                 print(f"    --> Relation: {label} ({uri})")
    #         print("-" * 80)
    #
    # print(f"\n\nSummary: {valid_count}/{len(subgraphs)} subgraphs contain the correct answer")
