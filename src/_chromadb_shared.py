"""Shared ChromaDB utilities for both entities and relations."""

import re
from typing import Optional

import chromadb

from kgnode.core.kg_config import KGConfig
from kgnode.core.logging_config import get_logger


logger = get_logger(__name__)


def _get_next_entity_id(collection: chromadb.Collection) -> int:
    """
    Get the next available entity ID number from a collection.

    Args:
        collection: ChromaDB collection

    Returns:
        Next available entity ID number (e.g., if max is entity_99, returns 100)
    """
    all_ids = collection.get()['ids']
    if not all_ids:
        return 0

    # Extract numeric parts from entity_N format
    id_numbers = []
    for id_str in all_ids:
        if id_str.startswith('entity_'):
            try:
                num = int(id_str.split('_')[1])
                id_numbers.append(num)
            except (ValueError, IndexError):
                continue

    return max(id_numbers) + 1 if id_numbers else 0


def _chromadb_exists(
        config: Optional[KGConfig] = None,
        collection_name: Optional[str] = None
) -> bool:
    """
    Check if ChromaDB collection exists.

    Args:
        config: Optional KGConfig instance for collection settings.
            If None, uses default KGConfig with environment variables or built-in defaults.
        collection_name: Optional collection name override. If None, uses config.collection_name

    Returns:
        True if collection exists, False otherwise
    """
    # Initialize config if not provided
    if config is None:
        config = KGConfig.default()

    # Use provided collection_name or fall back to config
    collection_name = collection_name or config.collection_name
    persist_directory = config.chroma_persist_dir

    try:
        client = chromadb.PersistentClient(path=persist_directory)
        client.get_collection(name=collection_name)
        return True
    except:
        return False


def _load_chromadb(
        config: Optional[KGConfig] = None,
        collection_name: Optional[str] = None
) -> chromadb.Collection:
    """
    Load an existing ChromaDB collection (private function).

    Args:
        config: Optional KGConfig instance for collection settings.
            If None, uses default KGConfig with environment variables or built-in defaults.
        collection_name: Optional collection name override. If None, uses config.collection_name

    Returns:
        ChromaDB collection object

    Raises:
        chromadb.errors.NotFoundError: If collection does not exist
    """
    # Initialize config if not provided
    if config is None:
        config = KGConfig.default()

    # Use provided collection_name or fall back to config
    collection_name = collection_name or config.collection_name
    persist_directory = config.chroma_persist_dir
    embedding_model = config.embedding_model

    client = chromadb.PersistentClient(path=persist_directory)

    collection = client.get_collection(
        name=collection_name,
        embedding_function=chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
    )

    logger.info(f"Loaded collection '{collection_name}' with {collection.count()} items")
    return collection


def uri_to_label(uri: str) -> str:
    """
    Convert URI to human-readable label.
    Works for any knowledge graph by extracting and formatting the URI fragment.

    Args:
        uri: Full URI (e.g., "http://example.org/ontology#hasName" or "<http://example.org/ontology#hasName>")

    Returns:
        Human-readable label (e.g., "has name")
    """
    # Remove SPARQL brackets if present
    uri = uri.strip('<>')

    # Extract the fragment/local name from URI
    if '#' in uri:
        label = uri.split('#')[-1]
    elif '/' in uri:
        label = uri.split('/')[-1]
    else:
        label = uri

    # Convert camelCase or PascalCase to spaces
    # hasName -> has Name -> has name
    label = re.sub(r'([a-z])([A-Z])', r'\1 \2', label)

    # Convert snake_case or kebab-case to spaces
    label = label.replace('_', ' ').replace('-', ' ')

    # Lowercase and clean up
    label = label.lower().strip()

    return label
