"""Add filtered entity (seed node) information to evaluation dataset.

This script reads extracted_nodes from the dataset, applies adaptive filtering
logic, and adds filtered_seeds to each question.
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


def calculate_fuzzy_score(label: str, entity_name: str) -> float:
    """Calculate fuzzy match score between label and entity name.

    Args:
        label: Node label (e.g., "Matti Rossi")
        entity_name: Extracted entity name (e.g., "Matti R.")

    Returns:
        Fuzzy match score between 0 and 1 (higher is better match)
    """
    label_lower = label.lower()
    entity_lower = entity_name.lower()

    # Check for exact substring match
    if label_lower in entity_lower or entity_lower in label_lower:
        return 1.0

    # Word-level matching - find best match between any words
    label_words = label_lower.split()
    entity_words = entity_lower.split()

    if not label_words or not entity_words:
        return 0.0

    max_ratio = 0.0
    for label_word in label_words:
        for entity_word in entity_words:
            ratio = SequenceMatcher(None, label_word, entity_word).ratio()
            max_ratio = max(max_ratio, ratio)

    return max_ratio


def filter_seeds(
    extracted_nodes: List[Dict[str, Any]],
    use_fuzzy_matching: bool = True,
) -> List[Dict[str, Any]]:
    """Filter seed nodes with adaptive threshold and per-entity cap.

    Logic:
    1. Exclude nodes with "Literal" in entity_types
    2. Per entity:
       - Calculate fuzzy match scores (label vs entity_name)
       - Sort by fuzzy score (descending), then embedding score (ascending)
       - Apply adaptive threshold based on confidence:
         * Very confident (score < 0.15): threshold=0.05, max=2 seeds
         * Confident (score 0.15-0.25): threshold=0.06, max=2 seeds
         * Moderate (score 0.25-0.35): threshold=0.05, max=3 seeds
         * Low confidence (score >= 0.35): threshold=0.04, max=2 seeds
       - Cap maximum seeds per entity

    Args:
        extracted_nodes: List of node dicts with 'entity_from', 'score',
            'entity_uri', 'label', 'entity_types'
        use_fuzzy_matching: Whether to use fuzzy label matching for ranking
            (default: True)

    Returns:
        List of filtered seed nodes (adaptive: 1-3 seeds per entity)
    """
    # First filter: Remove nodes with "Literal" in entity_types
    non_literal_nodes = []
    for node in extracted_nodes:
        entity_types = node.get('entity_types', [])
        if 'Literal' not in entity_types:
            non_literal_nodes.append(node)

    # If all nodes are literals, return empty list
    if not non_literal_nodes:
        return []

    # Group by entity_from
    nodes_by_entity = {}
    for node in non_literal_nodes:
        entity = node['entity_from']
        if entity not in nodes_by_entity:
            nodes_by_entity[entity] = []
        nodes_by_entity[entity].append(node)

    # Filter seeds per entity with adaptive threshold
    filtered_seeds = []
    for entity, nodes in nodes_by_entity.items():
        if not nodes:
            continue

        # Add fuzzy match scores to each node
        if use_fuzzy_matching:
            for node in nodes:
                node['fuzzy_score'] = calculate_fuzzy_score(
                    node['label'], entity
                )

        # Sort by fuzzy score (descending), then by embedding score (ascending)
        if use_fuzzy_matching:
            sorted_nodes = sorted(
                nodes, key=lambda x: (-x.get('fuzzy_score', 0), x['score'])
            )
        else:
            sorted_nodes = sorted(nodes, key=lambda x: x['score'])

        # Get the best (lowest) embedding score
        best_score = sorted_nodes[0]['score']

        # ADAPTIVE THRESHOLD based on confidence
        if best_score < 0.15:
            # Very confident - exact match likely
            threshold = 0.05
            max_seeds = 2
        elif best_score < 0.25:
            # Confident - good match
            threshold = 0.06
            max_seeds = 2
        elif best_score < 0.35:
            # Moderate confidence - keep more candidates
            threshold = 0.05
            max_seeds = 3
        else:
            # Low confidence - be more inclusive
            threshold = 0.04
            max_seeds = 2

        # Apply threshold with cap
        count = 0
        for node in sorted_nodes:
            if count >= max_seeds:
                break

            gap = node['score'] - best_score
            if gap < threshold:
                filtered_seeds.append(node)
                count += 1
            elif not use_fuzzy_matching:
                # Without fuzzy matching, nodes are sorted by score, can break
                break

    return filtered_seeds


def add_filtered_entity_info(file_path: str):
    """Add filtered entity information to each question in JSONL file.

    Reads extracted_nodes from each question, applies filtering logic,
    and adds filtered_seeds field.

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

        # Skip if no extracted_nodes
        if "extracted_nodes" not in question or not question["extracted_nodes"]:
            print(f"[{i}/{len(questions)}] {question_id}: SKIP - No extracted_nodes")
            skipped += 1
            continue

        # Filter seeds
        extracted_nodes = question["extracted_nodes"]
        filtered_seeds = filter_seeds(extracted_nodes, use_fuzzy_matching=True)

        # Calculate golden entity metrics (if available)
        golden_entities = [
            normalize_uri(e) for e in question.get("golden_entities", [])
        ]

        if golden_entities:
            # Check before/after filtering
            extracted_uris = {node["entity_uri"] for node in extracted_nodes}
            filtered_uris = {node["entity_uri"] for node in filtered_seeds}

            golden_before = [g for g in golden_entities if g in extracted_uris]
            golden_after = [g for g in golden_entities if g in filtered_uris]
            golden_lost = set(golden_before) - set(golden_after)

            reduction_ratio = 1 - (len(filtered_seeds) / len(extracted_nodes)) if extracted_nodes else 0

            # Add filtered entity info
            question["filtered_entity_info"] = {
                "filtered_seeds": filtered_seeds,
                "stats": {
                    "original_count": len(extracted_nodes),
                    "filtered_count": len(filtered_seeds),
                    "reduction_ratio": f"{reduction_ratio:.2%}",
                    "golden_entities_before": len(golden_before),
                    "golden_entities_after": len(golden_after),
                    "golden_entities_lost": list(golden_lost),
                },
            }

            # Track totals
            total_golden_before += len(golden_before)
            total_golden_after += len(golden_after)
            total_golden_lost += len(golden_lost)

            status = "PASS" if len(golden_after) == len(golden_before) else "FAIL"
            print(
                f"[{i}/{len(questions)}] {question_id}: {status} - "
                f"{len(extracted_nodes)}→{len(filtered_seeds)} seeds, "
                f"golden: {len(golden_before)}→{len(golden_after)}"
            )
        else:
            # No golden entities - just add filtered seeds
            reduction_ratio = 1 - (len(filtered_seeds) / len(extracted_nodes)) if extracted_nodes else 0

            question["filtered_entity_info"] = {
                "filtered_seeds": filtered_seeds,
                "stats": {
                    "original_count": len(extracted_nodes),
                    "filtered_count": len(filtered_seeds),
                    "reduction_ratio": f"{reduction_ratio:.2%}",
                },
            }

            print(
                f"[{i}/{len(questions)}] {question_id}: OK - "
                f"{len(extracted_nodes)}→{len(filtered_seeds)} seeds"
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
        print(f"\nGolden Entity Filtering:")
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
    print("ADD FILTERED ENTITY INFORMATION")
    print("=" * 70)
    print(f"File: {file_path}")
    print("=" * 70 + "\n")

    add_filtered_entity_info(file_path)
