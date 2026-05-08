"""Add filtered relation information to evaluation dataset.

This script reads extracted_relation_nodes from the dataset, applies adaptive
filtering logic, and adds filtered_relations to each question.
"""

import json
import os
import sys
from difflib import SequenceMatcher
from typing import Any, Dict, List


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logging_util import setup_dual_output


def normalize_uri(uri: str) -> str:
    """Remove angle brackets from URI."""
    return uri.strip("<>").strip()


def calculate_fuzzy_score(label: str, relation_name: str) -> float:
    """Calculate fuzzy match score between label and relation name.

    Args:
        label: Relation label (e.g., "authoredBy")
        relation_name: Extracted relation name (e.g., "authored by")

    Returns:
        Fuzzy match score between 0 and 1 (higher is better match)
    """
    label_lower = label.lower()
    relation_lower = relation_name.lower()

    # Check for exact substring match
    if label_lower in relation_lower or relation_lower in label_lower:
        return 1.0

    # Word-level matching - find best match between any words
    label_words = label_lower.split()
    relation_words = relation_lower.split()

    if not label_words or not relation_words:
        return 0.0

    max_ratio = 0.0
    for label_word in label_words:
        for relation_word in relation_words:
            ratio = SequenceMatcher(None, label_word, relation_word).ratio()
            max_ratio = max(max_ratio, ratio)

    return max_ratio


def filter_relations(
    extracted_relation_nodes: List[Dict[str, Any]],
    use_fuzzy_matching: bool = False,  # CHANGED: Disabled by default based on analysis
) -> List[Dict[str, Any]]:
    """Filter relation nodes with adaptive threshold.

    Logic:
    1. Group by relation_from (which extracted relation each node came from)
    2. Per relation:
       - Sort by distance ONLY (not fuzzy score) - analysis shows fuzzy hurts
       - Apply threshold based on gap from best distance
       - Cap maximum relations per extracted relation at 1-3

    NOTE: We retrieve max 3 nodes per relation from ChromaDB, so max_relations ≤ 3
    Analysis showed: 86.7% of losses due to fuzzy sorting when distances are equal

    Args:
        extracted_relation_nodes: List of relation dicts from search with keys:
            'relation_uri', 'label', 'distance', 'relation_from', 'relation_types'
        use_fuzzy_matching: Whether to use fuzzy label matching for ranking (default: False)
            Disabled by default - analysis shows avg fuzzy=0.38, causes losses

    Returns:
        List of filtered relation nodes (1-3 relations per extracted relation)
    """
    # Group by relation_from
    nodes_by_relation = {}
    for node in extracted_relation_nodes:
        relation = node['relation_from']
        if relation not in nodes_by_relation:
            nodes_by_relation[relation] = []
        nodes_by_relation[relation].append(node)

    # Filter relations per extracted relation
    filtered_relations = []
    for relation, nodes in nodes_by_relation.items():
        if not nodes:
            continue

        # Add fuzzy match scores
        if use_fuzzy_matching:
            for node in nodes:
                node['fuzzy_score'] = calculate_fuzzy_score(
                    node['label'], relation
                )

        # Sort by fuzzy score (desc), then distance (asc)
        if use_fuzzy_matching:
            sorted_nodes = sorted(
                nodes, key=lambda x: (-x.get('fuzzy_score', 0), x['distance'])
            )
        else:
            sorted_nodes = sorted(nodes, key=lambda x: x['distance'])

        # Get the best (lowest) distance
        best_distance = sorted_nodes[0]['distance']

        # ADAPTIVE THRESHOLD based on confidence
        # Updated based on failure analysis:
        # - Disabled fuzzy matching (was causing ties to lose on fuzzy score)
        # - Keep 1-2 relations per group (we only retrieve 3 max from ChromaDB)
        # - Relaxed threshold to 0.15 to save GAP_TOO_LARGE cases
        # - Most losses (gap=0.0) will be fixed by removing fuzzy sort bias
        if best_distance < 0.15:
            # Very confident - exact match likely, keep just 1
            threshold = 0.10
            max_relations = 1
        elif best_distance < 0.25:
            # Confident - good match, keep top 2
            threshold = 0.15
            max_relations = 2
        elif best_distance < 0.35:
            # Moderate confidence - keep top 2
            threshold = 0.15
            max_relations = 2
        else:
            # Low confidence - be conservative
            threshold = 0.15
            max_relations = 3

        # Apply threshold with cap
        count = 0
        for node in sorted_nodes:
            if count >= max_relations:
                break

            gap = node['distance'] - best_distance
            if gap < threshold:
                filtered_relations.append(node)
                count += 1
            elif not use_fuzzy_matching:
                # Without fuzzy matching, nodes are sorted by distance, can break
                break

    return filtered_relations


def deduplicate_relation_nodes_by_types(
    extracted_relation_nodes: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Remove duplicate relation nodes that come from extracted_relations with same types.

    When multiple extracted_relations have the same types (e.g., "not publish in" and
    "published in" both have types ["publishedIn"]), we only keep nodes from the first one.

    Example:
        extracted_relations: [
            {"name": "not publish in", "types": ["publishedIn"]},    # First
            {"name": "published in", "types": ["publishedIn"]}       # Duplicate types - skip
        ]

        extracted_relation_nodes: [
            {..., "relation_from": "not publish in", "relation_types": ["publishedIn"]},   # Keep
            {..., "relation_from": "not publish in", "relation_types": ["publishedIn"]},   # Keep
            {..., "relation_from": "published in", "relation_types": ["publishedIn"]},     # Skip
            {..., "relation_from": "published in", "relation_types": ["publishedIn"]}      # Skip
        ]

        Result: Only keeps nodes from "not publish in" (first occurrence of types ["publishedIn"])

    Args:
        extracted_relation_nodes: List of relation nodes with 'relation_from' and 'relation_types'

    Returns:
        Deduplicated list of relation nodes
    """
    # Track which types combinations we've seen
    seen_types = set()
    # Track which relation_from values to keep (first occurrence only)
    keep_relation_from = set()

    # First pass: identify which relation_from to keep for each unique types
    for node in extracted_relation_nodes:
        relation_types = tuple(sorted(node.get('relation_types', [])))  # Convert to tuple for hashing

        if relation_types not in seen_types:
            seen_types.add(relation_types)
            keep_relation_from.add(node['relation_from'])

    # Second pass: filter nodes
    deduplicated = []
    for node in extracted_relation_nodes:
        if node['relation_from'] in keep_relation_from:
            deduplicated.append(node)

    return deduplicated


def add_filtered_relation_info(file_path: str):
    """Add filtered relation information to each question in JSONL file.

    Reads extracted_relation_nodes from each question, deduplicates by types,
    applies filtering logic, and adds filtered_relations field.

    Args:
        file_path: Path to JSONL file to process
    """
    # Read all questions
    questions = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    print(f"Processing {len(questions)} questions...")

    # Process each question
    processed = 0
    skipped = 0
    total_golden_before = 0
    total_golden_after = 0
    total_golden_lost = 0

    for i, question in enumerate(questions, 1):
        question_id = question.get("id", f"Q{i:04d}")

        # Skip if no extracted_relation_nodes
        if "extracted_relation_nodes" not in question or not question["extracted_relation_nodes"]:
            print(f"[{i}/{len(questions)}] {question_id}: SKIP - No extracted_relation_nodes")
            question["filtered_relation_info"] = {
                "filtered_relations": [],
                "stats": {
                    "original_count": 0,
                    "filtered_count": 0,
                    "reduction_ratio": "0.00%",
                },
            }
            skipped += 1
            continue

        # Deduplicate relation nodes by types (keep only first occurrence)
        extracted_relation_nodes = question["extracted_relation_nodes"]
        deduplicated_nodes = deduplicate_relation_nodes_by_types(extracted_relation_nodes)

        # Filter relations (fuzzy matching disabled - analysis shows it hurts)
        filtered_relations = filter_relations(deduplicated_nodes, use_fuzzy_matching=False)

        # Calculate golden relation metrics (if available)
        golden_relations = [
            normalize_uri(r) for r in question.get("golden_relations", [])
        ]

        if golden_relations:
            # Check before/after filtering (use deduplicated nodes as baseline)
            extracted_uris = {node["relation_uri"] for node in deduplicated_nodes}
            filtered_uris = {node["relation_uri"] for node in filtered_relations}

            golden_before = [g for g in golden_relations if g in extracted_uris]
            golden_after = [g for g in golden_relations if g in filtered_uris]
            golden_lost = set(golden_before) - set(golden_after)

            reduction_ratio = (
                1 - (len(filtered_relations) / len(deduplicated_nodes))
                if deduplicated_nodes
                else 0
            )

            # Add filtered relation info
            question["filtered_relation_info"] = {
                "filtered_relations": filtered_relations,
                "stats": {
                    "original_count": len(deduplicated_nodes),  # After deduplication
                    "filtered_count": len(filtered_relations),
                    "reduction_ratio": f"{reduction_ratio:.2%}",
                    "golden_relations_before": len(golden_before),
                    "golden_relations_after": len(golden_after),
                    "golden_relations_lost": list(golden_lost),
                },
            }

            # Track totals
            total_golden_before += len(golden_before)
            total_golden_after += len(golden_after)
            total_golden_lost += len(golden_lost)

            status = "PASS" if len(golden_after) == len(golden_before) else "FAIL"
            if len(deduplicated_nodes) < len(extracted_relation_nodes):
                # Show deduplication happened
                print(
                    f"[{i}/{len(questions)}] {question_id}: {status} - "
                    f"{len(extracted_relation_nodes)}→{len(deduplicated_nodes)}(dedup)→{len(filtered_relations)}(filter) relations, "
                    f"golden: {len(golden_before)}→{len(golden_after)}"
                )
            else:
                # No deduplication
                print(
                    f"[{i}/{len(questions)}] {question_id}: {status} - "
                    f"{len(deduplicated_nodes)}→{len(filtered_relations)} relations, "
                    f"golden: {len(golden_before)}→{len(golden_after)}"
                )
        else:
            # No golden relations - just add filtered relations
            reduction_ratio = (
                1 - (len(filtered_relations) / len(deduplicated_nodes))
                if deduplicated_nodes
                else 0
            )

            question["filtered_relation_info"] = {
                "filtered_relations": filtered_relations,
                "stats": {
                    "original_count": len(deduplicated_nodes),  # After deduplication
                    "filtered_count": len(filtered_relations),
                    "reduction_ratio": f"{reduction_ratio:.2%}",
                },
            }

            print(
                f"[{i}/{len(questions)}] {question_id}: OK - "
                f"{len(deduplicated_nodes)}→{len(filtered_relations)} relations (after dedup)"
            )

        processed += 1

    # Save results
    print(f"\nSaving results to {file_path}...")
    with open(file_path, "w", encoding="utf-8") as f:
        for question in questions:
            f.write(json.dumps(question, ensure_ascii=False) + "\n")

    print(f"✓ Results saved!")
    print(f"\nSummary:")
    print(f"  Processed: {processed}/{len(questions)}")
    print(f"  Skipped: {skipped}/{len(questions)}")

    if total_golden_before > 0:
        success_rate = (total_golden_after / total_golden_before) * 100
        print(f"\nGolden Relation Filtering:")
        print(f"  Before filtering: {total_golden_before}")
        print(f"  After filtering:  {total_golden_after}")
        print(f"  Lost:             {total_golden_lost}")
        print(f"  Success rate:     {success_rate:.1f}%")


if __name__ == "__main__":
    setup_dual_output(__file__)

    # Default file path
    file_path = os.path.join(os.path.dirname(__file__), "./data/balanced_400.jsonl")

    if len(sys.argv) > 1:
        file_path = sys.argv[1]

    print("=" * 70)
    print("ADD FILTERED RELATION INFORMATION")
    print("=" * 70)
    print(f"File: {file_path}")
    print("=" * 70 + "\n")

    add_filtered_relation_info(file_path)
