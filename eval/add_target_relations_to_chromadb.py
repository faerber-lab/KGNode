"""Script to add relations from first N questions of DBLP-QuAD to ChromaDB."""

import json
import os
from typing import Any, Dict, List, Tuple

from chromadb import Collection

from kgnode._relation_chromadb import add_or_update_relations
from kgnode.core.kg_config import KGConfig

# =============================================================================
# Helper Functions
# =============================================================================


def strip_relation_brackets(relation: str) -> str:
    """Strip angle brackets from relation URI."""
    return relation.strip("<>")


def is_valid_relation_uri(uri: str) -> bool:
    """Check if URI is a valid relation URL."""
    if not uri or not isinstance(uri, str):
        return False
    # Accept http/https URIs
    return uri.startswith("http://") or uri.startswith("https://")


def add_relations_from_questions(
        count: int,
        json_path: str | None = None,
        batch_size: int = 50,
        save_csv: bool = True,
) -> Tuple[int, int, Collection]:
    """Add relations from the first N questions to ChromaDB.

    Args:
        count: Number of questions to process from the beginning
        json_path: Path to questions JSONL file. Defaults to balanced_200.jsonl.
        batch_size: Batch size for add_or_update_relations (default: 50)
        save_csv: Whether to save relation descriptions to CSV (default: True)

    Returns:
        Tuple of (num_added, num_updated, collection)
    """
    if json_path is None:
        json_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            "balanced_400.jsonl",
        )

    # Load questions from JSONL
    questions = []
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    questions = questions[:count]

    unique_relations = set()
    invalid_relations = set()

    for question in questions:
        relations = question.get("golden_relations", [])
        for relation in relations:
            clean_uri = strip_relation_brackets(relation)
            if is_valid_relation_uri(clean_uri):
                unique_relations.add(clean_uri)
            else:
                invalid_relations.add(clean_uri)

    relation_uris = list(unique_relations)

    print(f"Processing first {count} questions from {json_path}")
    print(f"Found {len(relation_uris)} valid relation URIs")
    if invalid_relations:
        print(f"Filtered out {len(invalid_relations)} invalid relations")
        print(f"Examples of filtered relations: {list(invalid_relations)[:5]}")

    if not relation_uris:
        print("No relations found to add.")
        raise ValueError("No relations found in the specified questions")

    config = KGConfig.default()

    print(f"Adding/updating {len(relation_uris)} relations to ChromaDB...")
    num_added, num_updated, collection = add_or_update_relations(
        relation_uris=relation_uris,
        config=config,
        batch_size=batch_size,
        save_csv=save_csv,
    )

    return num_added, num_updated, collection


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ADD RELATIONS FROM DBLP-QUAD QUESTIONS")
    print("=" * 60)

    QUESTION_COUNT = 2000

    num_added, num_updated, collection = add_relations_from_questions(
        count=QUESTION_COUNT,
        batch_size=50,
        save_csv=True,
    )

    print("\nResults:")
    print(f"  - Questions processed: {QUESTION_COUNT}")
    print(f"  - Relations added: {num_added}")
    print(f"  - Relations updated: {num_updated}")
    print(f"  - Total relations in collection: {collection.count()}")
