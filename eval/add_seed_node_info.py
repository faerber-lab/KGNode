"""Add seed node information to evaluation dataset."""

import json
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logging_util import setup_dual_output

from kgnode import KGConfig, SearchMode
from kgnode.seed_finder import get_seed_and_relation_nodes


def normalize_uri(uri: str) -> str:
    """Remove angle brackets from URI.

    Args:
        uri: URI string, possibly with angle brackets

    Returns:
        Normalized URI string
    """
    return uri.strip("<>").strip()


def add_seed_node_info(file_path: str):
    """Add seed node information to each question in JSONL file.

    Reads a JSONL file, runs get_seed_nodes() for each question,
    and adds extracted_entities, seed_nodes, seed_node_found flag,
    and match_ratio to each entry.

    Args:
        file_path: Path to JSONL file to process
    """
    # Initialize config
    config = KGConfig.default()

    # Read all questions from file
    questions = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    print(f"Processing {len(questions)} questions...")

    # Process each question
    for i, question in enumerate(questions, 1):
        question_id = question["id"]
        # Use first version of question
        question_text = question["question"]["1"]
        golden_entities = [normalize_uri(e) for e in question.get("golden_entities", [])]

        print(f"[{i}/{len(questions)}] Processing {question_id}...")

        # Get seed nodes and relation nodes with semantic search
        seed_nodes, extracted_entities, relation_nodes, extracted_relations = get_seed_and_relation_nodes(
            query=question_text,
            n_results_entity=10,
            n_results_relation=3,
            config=config,
            search_mode=SearchMode.semantic
        )

        # Normalize seed node URIs
        seed_node_uris = [normalize_uri(node.get("entity_uri", "")) for node in seed_nodes]

        # Normalize relation URIs
        relation_uris = [normalize_uri(node.get("relation_uri", "")) for node in relation_nodes]
        golden_relations = [normalize_uri(r) for r in question.get("golden_relations", [])]

        # Check which golden entities were found
        matched_golden_entities = []
        for golden_uri in golden_entities:
            if golden_uri in seed_node_uris:
                matched_golden_entities.append(golden_uri)

        # Check which golden relations were found
        matched_golden_relations = []
        for golden_rel in golden_relations:
            if golden_rel in relation_uris:
                matched_golden_relations.append(golden_rel)

        # Calculate metrics
        seed_node_found = len(matched_golden_entities) > 0
        match_ratio = len(matched_golden_entities) / len(golden_entities) if golden_entities else 0.0

        relation_found = len(matched_golden_relations) > 0
        relation_match_ratio = len(matched_golden_relations) / len(golden_relations) if golden_relations else 0.0

        # Add information to question
        question["extracted_entities"] = [
            {"name": e["name"], "types": e.get("types", [])}
            for e in extracted_entities
        ]
        # TODO changing authorOf to authoredBy
        # question["extracted_relations"] = [
        #      {"name": r["name"], "types": r.get("types", [])}
        #     for r in extracted_relations]
        # Convert authorOf to authoredBy in relation types
        processed_relations = []
        for r in extracted_relations:
            types = r.get("types", [])
            # Replace authorOf with authoredBy
            types = ["authoredBy" if t == "authorOf" else t for t in types]
            processed_relations.append({"name": r["name"], "types": types})

        question["extracted_relations"] = processed_relations
        question["extracted_nodes"] = seed_nodes
        question["extracted_relation_nodes"] = relation_nodes
        question["seed_node_found"] = seed_node_found
        question["seed_node_match_ratio"] = f"{len(matched_golden_entities)}/{len(golden_entities)}"
        question["relation_found"] = relation_found
        question["relation_match_ratio"] = f"{len(matched_golden_relations)}/{len(golden_relations)}"

        print(f"  ✓ Found {len(seed_nodes)} seed nodes, {len(relation_nodes)} relation nodes")
        print(f"    Entities: {len(matched_golden_entities)}/{len(golden_entities)} ({match_ratio:.1%}), "
              f"Relations: {len(matched_golden_relations)}/{len(golden_relations)} ({relation_match_ratio:.1%})")

    # Write results back to file
    with open(file_path, "w", encoding="utf-8") as f:
        for question in questions:
            f.write(json.dumps(question, ensure_ascii=False) + "\n")

    print(f"\n✓ Results saved to {file_path}")

    # Print summary statistics
    total_found = sum(1 for q in questions if q["seed_node_found"])

    # Calculate average match ratio from seed_node_match_ratio strings
    match_ratios = []
    for q in questions:
        matched, total = q["seed_node_match_ratio"].split("/")
        ratio = int(matched) / int(total) if int(total) > 0 else 0.0
        match_ratios.append(ratio)
    avg_ratio = sum(match_ratios) / len(match_ratios) if match_ratios else 0

    print("\nSummary:")
    print(f"  Questions with at least one golden entity found: {total_found}/{len(questions)} ({total_found/len(questions):.1%})")
    print(f"  Average match ratio: {avg_ratio:.1%}")


if __name__ == "__main__":
    setup_dual_output(__file__)

    # Default file path
    file_path = os.path.join(os.path.dirname(__file__), "./data/balanced_400.jsonl")

    if len(sys.argv) > 1:
        file_path = sys.argv[1]

    add_seed_node_info(file_path)


# Next task: Seed node need to be ranked ....
#  If any nodes have semantic_score < 0.30 → take top 2 with lowest scores, if  difference between the top
#  two score is < .15 otherwise take only the top one
#  If all nodes have semantic_score >= 0.30 → take only one (This logic need to be changed)
#  Filter out seed nodes with types Literal.
# Provide the extracted entity for sparql generation (it may help)