"""Script to add entities from first N questions of DBLP-QuAD to ChromaDB."""

import json
import os
import re
from typing import Any, Dict, List, Tuple

from chromadb import Collection

from kgnode.chroma_db import add_or_update_entities
from kgnode.core.kg_config import KGConfig

# =============================================================================
# DBLP Predicate URIs
# =============================================================================
DBLP_SCHEMA = "https://dblp.org/rdf/schema#"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"

DBLP_PRIMARY_FULL_CREATOR_NAME = f"{DBLP_SCHEMA}primaryFullCreatorName"
DBLP_OTHER_FULL_CREATOR_NAME = f"{DBLP_SCHEMA}otherFullCreatorName"
DBLP_TITLE = f"{DBLP_SCHEMA}title"


# =============================================================================
# Helper Functions
# =============================================================================


def _uri_to_label(uri: str) -> str:
    """Convert URI to human-readable label."""
    uri = uri.strip("<>")

    if "#" in uri:
        label = uri.split("#")[-1]
    elif "/" in uri:
        label = uri.split("/")[-1]
    else:
        label = uri

    label = re.sub(r"([a-z])([A-Z])", r"\1 \2", label)
    label = label.replace("_", " ").replace("-", " ")

    return label.lower().strip()


def dblp_relation_descriptor(relation_uri: str) -> str:
    """Extract the last part of relation URI (e.g., 'authoredBy' from full URI)."""
    clean_uri = relation_uri.strip("<>")

    if "#" in clean_uri:
        return clean_uri.split("#")[-1]
    elif "/" in clean_uri:
        return clean_uri.split("/")[-1]
    else:
        return clean_uri


def dblp_entity_descriptor(entity_uri: str, triples: List[Dict[str, Any]]) -> str:
    """Create entity descriptor: 'Type1, Type2: Name, Alt1, Alt2.'

    Args:
        entity_uri: URI of the entity (cleaned, no brackets)
        triples: List of dicts with 'predicate' and 'object' keys

    Returns:
        Description in format: "Type1, Type2: Name, Alt1, Alt2."
    """
    if not triples:
        return _uri_to_label(entity_uri)

    entity_types: List[str] = []
    primary_name = None
    rdfs_label = None
    alt_names: List[str] = []
    title = None

    for triple in triples:
        pred = triple["predicate"]
        obj = triple["object"]

        if pred == RDF_TYPE:
            type_name = _uri_to_label(obj).title()
            if type_name not in ["Thing", "Resource"]:
                entity_types.append(type_name)
        elif pred == DBLP_PRIMARY_FULL_CREATOR_NAME:
            primary_name = obj
        elif pred == RDFS_LABEL:
            rdfs_label = obj
        elif pred == DBLP_OTHER_FULL_CREATOR_NAME:
            alt_names.append(obj)
        elif pred == DBLP_TITLE:
            title = obj

    # Build descriptor: "Type1, Type2: Name, Alt1, Alt2."
    types_str = ", ".join(entity_types) if entity_types else "Entity"
    name = primary_name or title or rdfs_label or _uri_to_label(entity_uri)

    descriptor = f"{types_str}: {name}"

    # Add alternative names if present and different from primary
    unique_alts = [n for n in alt_names if n != name]
    if unique_alts:
        descriptor += f", {', '.join(unique_alts[:3])}"

    return descriptor


def strip_entity_brackets(entity: str) -> str:
    """Strip angle brackets from entity URI."""
    return entity.strip("<>")


def is_valid_dblp_uri(uri: str) -> bool:
    """Check if URI is a valid DBLP URL."""
    if not uri or not isinstance(uri, str):
        return False
    return uri.startswith("https://dblp.org/")


def add_entities_from_questions(
        count: int,
        json_path: str | None = None,
        batch_size: int = 80,
        save_csv: bool = True,
        use_custom_descriptor: bool = True,
) -> Tuple[int, int, Collection]:
    """Add entities from the first N questions to ChromaDB.

    Args:
        count: Number of questions to process from the beginning
        json_path: Path to questions JSON file. Defaults to test questions.
        batch_size: Batch size for add_or_update_entities (default: 80)
        save_csv: Whether to save entity descriptions to CSV (default: True)
        use_custom_descriptor: Use improved DBLP descriptors (default: True)

    Returns:
        Tuple of (num_added, num_updated, collection)
    """
    if json_path is None:
        json_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "_data",
            "dblp-quad",
            "test",
            "questions.json",
        )

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data["questions"][:count]

    unique_entities = set()
    invalid_entities = set()
    for question in questions:
        entities = question.get("entities", [])
        for entity in entities:
            clean_uri = strip_entity_brackets(entity)
            if is_valid_dblp_uri(clean_uri):
                unique_entities.add(clean_uri)
            else:
                invalid_entities.add(clean_uri)

    entity_uris = list(unique_entities)

    print(f"Processing first {count} questions")
    print(f"Found {len(entity_uris)} valid DBLP entities")
    if invalid_entities:
        print(f"Filtered out {len(invalid_entities)} invalid entities (non-DBLP URLs)")
        print(f"Examples of filtered entities: {list(invalid_entities)[:5]}")

    if not entity_uris:
        print("No entities found to add.")
        raise ValueError("No entities found in the specified questions")

    config = None
    if use_custom_descriptor:
        config = KGConfig(
            entity_descriptor=dblp_entity_descriptor,
            relation_descriptor=dblp_relation_descriptor,
        )
        print("Using custom DBLP entity descriptor")

    num_added, num_updated, collection = add_or_update_entities(
        entity_uris=entity_uris,
        config=config,
        batch_size=batch_size,
        save_csv=save_csv,
    )

    return num_added, num_updated, collection


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ADD ENTITIES FROM DBLP-QUAD QUESTIONS")
    print("=" * 60)

    QUESTION_COUNT = 2000

    num_added, num_updated, collection = add_entities_from_questions(
        count=QUESTION_COUNT,
        batch_size=80,
        save_csv=True,
    )

    print("\nResults:")
    print(f"  - Questions processed: {QUESTION_COUNT}")
    print(f"  - Entities added: {num_added}")
    print(f"  - Entities updated: {num_updated}")
    print(f"  - Total in collection: {collection.count()}")