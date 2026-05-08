"""Filter and merge subgraphs to reduce LLM context token usage.

This script:
1. Filters subgraphs by probability threshold
2. Merges linear paths and connected subgraphs into larger graphs
3. Evaluates impact on golden entity/relation coverage
"""

import json
import os
import sys
from typing import Any, Dict, List, Set, Tuple


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logging_util import setup_dual_output


def normalize_uri(uri: str) -> str:
    """Remove angle brackets from URI."""
    return uri.strip("<>").strip()


def extract_nodes_and_edges(subgraph: Dict[str, Any]) -> Tuple[Set[str], List[Tuple]]:
    """Extract nodes and edges from a subgraph.

    Supports both old and new formats:
    - Old: triplet_uris format
    - New: inline labeled triples format

    Args:
        subgraph: Subgraph dictionary

    Returns:
        Tuple of (node_set, edge_list)
    """
    nodes = set()
    edges = []

    # New format: inline labeled triples
    if "triples" in subgraph:
        for triple in subgraph["triples"]:
            subject_uri = triple.get("subject_uri")
            predicate_uri = triple.get("predicate_uri")
            object_uri = triple.get("object_uri")

            if subject_uri and object_uri:
                nodes.add(normalize_uri(subject_uri))
                nodes.add(normalize_uri(object_uri))

            if subject_uri and predicate_uri and object_uri:
                edges.append((
                    normalize_uri(subject_uri),
                    normalize_uri(predicate_uri),
                    normalize_uri(object_uri)
                ))

    # Old format: triplet_uris
    elif "triplet_uris" in subgraph:
        triplets = subgraph.get("triplet_uris", [])
        for triple in triplets:
            if len(triple) >= 3:
                subject, predicate, obj = triple[0], triple[1], triple[2]
                nodes.add(normalize_uri(subject))
                nodes.add(normalize_uri(obj))
                edges.append((
                    normalize_uri(subject),
                    normalize_uri(predicate),
                    normalize_uri(obj)
                ))

    return nodes, edges


def can_merge_subgraphs(sg1_nodes: Set[str], sg2_nodes: Set[str]) -> bool:
    """Check if two subgraphs can be merged (share nodes).

    Args:
        sg1_nodes: Node set from subgraph 1
        sg2_nodes: Node set from subgraph 2

    Returns:
        True if subgraphs share at least one node
    """
    return len(sg1_nodes & sg2_nodes) > 0


def create_merged_graph(cluster: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create a single merged graph from a cluster of subgraphs.

    Args:
        cluster: List of subgraphs to merge

    Returns:
        Single merged graph with inline labeled triples
    """
    if not cluster:
        return None

    # Collect all triples and uri->label mappings
    all_triples = set()
    uri_to_label_map = {}  # Map uri -> label

    for sg in cluster:
        # Add triples
        for triple in sg.get('triplet_uris', []):
            if len(triple) >= 3:
                all_triples.add(tuple(triple))

        # Collect uri -> label mappings
        for path_entry in sg.get('path_with_label', []):
            if len(path_entry) >= 3:
                uri, label, is_entity = path_entry[0], path_entry[1], path_entry[2]
                # Keep first occurrence (they should be consistent)
                if uri not in uri_to_label_map:
                    uri_to_label_map[uri] = label

    # Build inline labeled triples
    inline_triples = []
    for triple in all_triples:
        subject_uri, predicate_uri, object_uri = triple

        inline_triple = {
            "subject": uri_to_label_map.get(subject_uri, subject_uri),
            "subject_uri": subject_uri,
            "predicate": uri_to_label_map.get(predicate_uri, predicate_uri),
            "predicate_uri": predicate_uri,
            "object": uri_to_label_map.get(object_uri, object_uri),
            "object_uri": object_uri
        }
        inline_triples.append(inline_triple)

    # Calculate merged probability (max = best) and depth (max = longest path)
    probabilities = [sg.get('probability', 0) for sg in cluster]
    depths = [sg.get('path_depth', 0) for sg in cluster]

    merged_graph = {
        'triples': inline_triples,
        'probability': max(probabilities),  # Use best (highest) probability in cluster
        'path_depth': max(depths),
        'merged_from': len(cluster)
    }

    return merged_graph


def merge_subgraphs(
    subgraphs: List[Dict[str, Any]],
    probability_threshold: float = float('inf')
) -> List[Dict[str, Any]]:
    """Merge connected subgraphs into unified graphs.

    Note: This function is kept for backward compatibility but is not used
    in the main pipeline. Use apply_topk_and_merge() instead.

    Args:
        subgraphs: List of subgraph dictionaries
        probability_threshold: Minimum probability to keep subgraph

    Returns:
        List of merged graphs (actual unified graphs, not clusters of subgraphs)
    """
    # Filter by probability (higher is better quality - exp(cosine_similarity))
    filtered = [
        sg for sg in subgraphs
        if sg.get('probability', 0.0) >= probability_threshold
    ]

    if not filtered:
        return []

    # Extract nodes for each subgraph
    subgraph_nodes = []
    for sg in filtered:
        nodes, _ = extract_nodes_and_edges(sg)
        subgraph_nodes.append((sg, nodes))

    # Union-Find / Graph clustering approach
    # Group subgraphs that share nodes
    clusters = []
    used = set()

    for i, (sg1, nodes1) in enumerate(subgraph_nodes):
        if i in used:
            continue

        # Start a new cluster
        cluster = [sg1]
        cluster_nodes = nodes1.copy()
        used.add(i)

        # Keep expanding cluster by finding connected subgraphs
        changed = True
        while changed:
            changed = False
            for j, (sg2, nodes2) in enumerate(subgraph_nodes):
                if j in used:
                    continue

                # Check if this subgraph connects to current cluster
                if can_merge_subgraphs(cluster_nodes, nodes2):
                    cluster.append(sg2)
                    cluster_nodes.update(nodes2)
                    used.add(j)
                    changed = True

        clusters.append(cluster)

    # Convert each cluster into a single merged graph
    merged_graphs = []
    for cluster in clusters:
        merged_graph = create_merged_graph(cluster)
        if merged_graph:
            merged_graphs.append(merged_graph)

    return merged_graphs


def apply_topk_and_merge(
    filtered_seeds: List[Dict[str, Any]],
    k: int = 3
) -> List[Dict[str, Any]]:
    """Apply top-K filtering per seed, then merge connected subgraphs.

    Args:
        filtered_seeds: List of seed dicts with subgraphs
        k: Number of best subgraphs to keep per seed (default: 3)

    Returns:
        List of merged graphs after top-K filtering
    """
    # Step 1: Apply top-K filtering per seed
    all_filtered_subgraphs = []

    for seed_info in filtered_seeds:
        seed_subgraphs = seed_info.get('subgraphs', [])

        # Sort by probability (higher is better - exp(cosine_similarity))
        sorted_subgraphs = sorted(
            seed_subgraphs,
            key=lambda x: x.get('probability', 0.0),
            reverse=True  # Descending: highest probability first
        )

        # Keep top K
        top_k = sorted_subgraphs[:k]
        all_filtered_subgraphs.extend(top_k)

    # Step 2: Merge connected subgraphs
    if not all_filtered_subgraphs:
        return []

    # Extract nodes for each subgraph
    subgraph_nodes = []
    for sg in all_filtered_subgraphs:
        nodes, _ = extract_nodes_and_edges(sg)
        subgraph_nodes.append((sg, nodes))

    # Group subgraphs that share nodes
    clusters = []
    used = set()

    for i, (sg1, nodes1) in enumerate(subgraph_nodes):
        if i in used:
            continue

        # Start a new cluster
        cluster = [sg1]
        cluster_nodes = nodes1.copy()
        used.add(i)

        # Keep expanding cluster by finding connected subgraphs
        changed = True
        while changed:
            changed = False
            for j, (sg2, nodes2) in enumerate(subgraph_nodes):
                if j in used:
                    continue

                # Check if this subgraph connects to current cluster
                if can_merge_subgraphs(cluster_nodes, nodes2):
                    cluster.append(sg2)
                    cluster_nodes.update(nodes2)
                    used.add(j)
                    changed = True

        clusters.append(cluster)

    # Convert each cluster into a single merged graph
    merged_graphs = []
    for cluster in clusters:
        merged_graph = create_merged_graph(cluster)
        if merged_graph:
            merged_graphs.append(merged_graph)

    return merged_graphs


def evaluate_subgraph_filtering(
    file_path: str,
    probability_threshold: float = float('inf'),
    output_path: str = None
):
    """Evaluate subgraph filtering and merging.

    Args:
        file_path: Path to JSONL file with subgraph_extraction data
        probability_threshold: Minimum probability threshold
        output_path: Optional path for output file
    """
    # Set default output path (same file - in-place update)
    if output_path is None:
        output_path = file_path

    # Load questions
    print(f"Loading questions from {file_path}...")
    questions = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    print(f"Loaded {len(questions)} questions\n")

    # Statistics
    stats = {
        "total_questions": len(questions),
        "questions_processed": 0,
        "top_k": 3,  # Keep top-3 subgraphs per seed
        "probability_threshold": probability_threshold,
        "total_subgraphs_before": 0,
        "total_subgraphs_after_filter": 0,
        "total_merged_clusters": 0,
        "total_golden_entities": 0,
        "golden_in_subgraphs_before": 0,
        "golden_in_subgraphs_after": 0,
        "total_golden_relations": 0,
        "relations_in_subgraphs_before": 0,
        "relations_in_subgraphs_after": 0,
        "questions_with_golden_before": 0,
        "questions_with_golden_after": 0,
        "filtration_cases": [],
    }

    # Process each question
    for i, question in enumerate(questions, 1):
        question_id = question.get("id", f"Q{i:04d}")

        # Skip if no subgraph_extraction
        if "subgraph_extraction" not in question:
            continue

        stats["questions_processed"] += 1

        # Get golden entities and relations
        golden_entities = [
            normalize_uri(e) for e in question.get("golden_entities", [])
        ]
        golden_relations = [
            normalize_uri(r) for r in question.get("golden_relations", [])
        ]

        stats["total_golden_entities"] += len(golden_entities)
        stats["total_golden_relations"] += len(golden_relations)

        # Get all subgraphs
        subgraph_extraction = question["subgraph_extraction"]
        filtered_seeds = subgraph_extraction.get("filtered_seeds", [])

        all_subgraphs = []
        for seed in filtered_seeds:
            all_subgraphs.extend(seed.get("subgraphs", []))

        stats["total_subgraphs_before"] += len(all_subgraphs)

        # Extract all nodes/URIs from all subgraphs (before filtering)
        all_nodes_before = set()
        for sg in all_subgraphs:
            nodes, _ = extract_nodes_and_edges(sg)
            all_nodes_before.update(nodes)

        # Check golden entities before
        golden_before = [g for g in golden_entities if g in all_nodes_before]
        if golden_before:
            stats["questions_with_golden_before"] += 1
            stats["golden_in_subgraphs_before"] += len(golden_before)

        # Check golden relations before
        relations_before = 0
        for relation in golden_relations:
            found = False
            for sg in all_subgraphs:
                _, edges = extract_nodes_and_edges(sg)
                # Check if relation appears in edges
                for _, pred, _ in edges:
                    if relation == pred:
                        found = True
                        break
                if found:
                    break
            if found:
                relations_before += 1
        stats["relations_in_subgraphs_before"] += relations_before

        # Apply top-K filtering per seed and merging (K=3)
        merged_graphs = apply_topk_and_merge(filtered_seeds, k=3)

        # Count original subgraphs that were merged
        total_filtered = sum(mg.get('merged_from', 1) for mg in merged_graphs)
        stats["total_subgraphs_after_filter"] += total_filtered
        stats["total_merged_clusters"] += len(merged_graphs)

        # Extract all nodes from merged graphs (after filtering)
        all_nodes_after = set()
        for mg in merged_graphs:
            nodes, _ = extract_nodes_and_edges(mg)
            all_nodes_after.update(nodes)

        # Check golden entities after
        golden_after = [g for g in golden_entities if g in all_nodes_after]
        if golden_after:
            stats["questions_with_golden_after"] += 1
            stats["golden_in_subgraphs_after"] += len(golden_after)

        # Check golden relations after
        relations_after = 0
        for relation in golden_relations:
            found = False
            for mg in merged_graphs:
                _, edges = extract_nodes_and_edges(mg)
                for _, pred, _ in edges:
                    if relation == pred:
                        found = True
                        break
                if found:
                    break
            if found:
                relations_after += 1
        stats["relations_in_subgraphs_after"] += relations_after

        # Calculate reduction
        reduction = 1 - (len(merged_graphs) / len(all_subgraphs)) if all_subgraphs else 0

        # Record case
        case = {
            "question_id": question_id,
            "subgraphs_before": len(all_subgraphs),
            "subgraphs_after_filter": total_filtered,
            "merged_graphs": len(merged_graphs),
            "reduction_ratio": reduction,
            "golden_before": len(golden_before),
            "golden_after": len(golden_after),
            "status": "PASS" if len(golden_after) >= len(golden_before) else "FAIL",
        }

        stats["filtration_cases"].append(case)

        # Add filtered subgraph info to question
        question["filtered_subgraphs_info"] = {
            "top_k_per_seed": 3,
            "filtering_method": "top_k_then_merge",
            "original_subgraph_count": len(all_subgraphs),
            "filtered_subgraph_count": total_filtered,
            "merged_graph_count": len(merged_graphs),
            "merged_graphs": merged_graphs,
            "reduction_ratio": f"{reduction:.2%}",
            "golden_entities_before": len(golden_before),
            "golden_entities_after": len(golden_after),
        }

        # Print progress
        if case["status"] == "FAIL":
            print(
                f"[{i}/{len(questions)}] {question_id}: ✗ FAIL - "
                f"Lost {len(golden_before) - len(golden_after)} golden entities"
            )
        elif i % 10 == 0:
            print(
                f"[{i}/{len(questions)}] {question_id}: "
                f"{len(all_subgraphs)}→{len(merged_graphs)} graphs "
                f"({reduction*100:.1f}% reduction)"
            )

    # Save updated data
    print(f"\nSaving results to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for question in questions:
            f.write(json.dumps(question, ensure_ascii=False) + "\n")
    print("Done!")

    # Print summary
    print("\n" + "=" * 70)
    print("SUBGRAPH FILTERING & MERGING EVALUATION")
    print("=" * 70)

    print(f"\nConfiguration:")
    print(f"  Top-K per seed: {stats['top_k']}")
    print(f"  Probability threshold: {probability_threshold} (not used with top-K)")

    print(f"\nDataset Statistics:")
    print(f"  Total questions: {stats['total_questions']}")
    print(f"  Questions processed: {stats['questions_processed']}")

    print(f"\nSubgraph Reduction:")
    print(f"  Subgraphs before: {stats['total_subgraphs_before']}")
    print(f"  Subgraphs after filter: {stats['total_subgraphs_after_filter']}")
    print(f"  Merged graphs: {stats['total_merged_clusters']}")

    if stats["total_subgraphs_before"] > 0:
        avg_before = stats["total_subgraphs_before"] / stats["questions_processed"]
        avg_after = stats["total_merged_clusters"] / stats["questions_processed"]
        overall_reduction = (
            1 - stats["total_merged_clusters"] / stats["total_subgraphs_before"]
        ) * 100
        print(f"  Avg before: {avg_before:.1f} subgraphs/question")
        print(f"  Avg after: {avg_after:.1f} merged graphs/question")
        print(f"  Overall reduction: {overall_reduction:.1f}%")

    print(f"\nGolden Entity Coverage:")
    print(f"  Total golden entities: {stats['total_golden_entities']}")
    print(
        f"  Found before: {stats['golden_in_subgraphs_before']} "
        f"({stats['golden_in_subgraphs_before']/stats['total_golden_entities']*100:.1f}%)"
    )
    print(
        f"  Found after: {stats['golden_in_subgraphs_after']} "
        f"({stats['golden_in_subgraphs_after']/stats['total_golden_entities']*100:.1f}%)"
    )

    print(f"\nGolden Relation Coverage:")
    print(f"  Total golden relations: {stats['total_golden_relations']}")
    print(
        f"  Found before: {stats['relations_in_subgraphs_before']} "
        f"({stats['relations_in_subgraphs_before']/stats['total_golden_relations']*100:.1f}%)"
    )
    print(
        f"  Found after: {stats['relations_in_subgraphs_after']} "
        f"({stats['relations_in_subgraphs_after']/stats['total_golden_relations']*100:.1f}%)"
    )

    print(f"\nQuestion-Level Success:")
    print(f"  Questions with golden before: {stats['questions_with_golden_before']}")
    print(f"  Questions with golden after: {stats['questions_with_golden_after']}")

    if stats["questions_with_golden_before"] > 0:
        success_rate = (
            stats["questions_with_golden_after"]
            / stats["questions_with_golden_before"]
            * 100
        )
        print(f"  Success rate: {success_rate:.1f}%")

    # Show failures
    failures = [c for c in stats["filtration_cases"] if c["status"] == "FAIL"]
    if failures:
        print(f"\nFailed Cases ({len(failures)}):")
        for fail in failures[:10]:
            print(
                f"  {fail['question_id']}: "
                f"{fail['subgraphs_before']}→{fail['merged_graphs']} graphs, "
                f"golden: {fail['golden_before']}→{fail['golden_after']}"
            )
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more")

    # Show high reduction cases
    print(f"\nTop Reduction Cases:")
    high_reduction = sorted(
        stats["filtration_cases"], key=lambda x: x["reduction_ratio"], reverse=True
    )[:5]
    for case in high_reduction:
        print(
            f"  {case['question_id']}: "
            f"{case['subgraphs_before']}→{case['merged_graphs']} graphs "
            f"({case['reduction_ratio']*100:.1f}% reduction) - {case['status']}"
        )

    print("=" * 70)

    return stats


if __name__ == "__main__":
    setup_dual_output(__file__)

    # Default paths
    input_path = "./data/balanced_400.jsonl"
    output_path = None  # Will default to same file (in-place update)
    threshold = float('inf')  # Will be set to optimal value after finding it

    # Allow command line override
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    if len(sys.argv) > 2:
        threshold = float(sys.argv[2])

    print("=" * 70)
    print("SUBGRAPH FILTERING & MERGING (TOP-K=3 PER SEED)")
    print("=" * 70)
    print(f"File: {input_path} (in-place update)")
    print(f"Method: Keep top-3 subgraphs per seed, then merge")
    print("=" * 70 + "\n")

    # Run evaluation
    evaluate_subgraph_filtering(input_path, threshold, output_path)