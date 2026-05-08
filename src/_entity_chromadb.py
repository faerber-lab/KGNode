"""Entity-specific ChromaDB operations."""

import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import chromadb
import pandas as pd

from kgnode._chromadb_shared import _get_next_entity_id, _chromadb_exists, _load_chromadb, uri_to_label
from kgnode._entity_ranker import get_top_entities_by_degree
from kgnode.core.kg_config import KGConfig
from kgnode.core.logging_config import get_logger

logger = get_logger(__name__)


def add_descriptions_to_csv_batched(
    csv_path: str,
    config: KGConfig,
    output_path: str = None,
    batch_size: int = 80
) -> pd.DataFrame:
    """
    Read CSV and add descriptions using batched queries.

    Args:
        csv_path: Path to input CSV file
        config: KGConfig instance with descriptor functions
        output_path: Path to save output CSV
        batch_size: Number of entities to query at once, query should fit in 8kb, so batch size max 100.

    Returns:
        DataFrame with added 'description' column
    """
    df = pd.read_csv(csv_path)

    if 'entity' not in df.columns:
        raise ValueError("CSV must have an 'entity' column")

    entity_uris = df['entity'].unique().tolist()

    logger.info(f"Processing {len(entity_uris)} unique entities in batches of {batch_size}...")

    # Process in batches using config's descriptor
    all_descriptions = {}
    for i in range(0, len(entity_uris), batch_size):
        batch = entity_uris[i:i + batch_size]
        logger.info(f"Processing batch {i // batch_size + 1}/{(len(entity_uris) - 1) // batch_size + 1}...")
        batch_descriptions = config.describe_entities_batch(batch)
        all_descriptions.update(batch_descriptions)

    # Map descriptions
    df['description'] = df['entity'].map(all_descriptions)
    df['description'] = df['description'].fillna('')

    if output_path is None:
        output_path = csv_path

    df.to_csv(output_path, index=False)
    logger.info(f"Saved results to {output_path}")

    return df


def compile_entities_chromadb(
        csv_path: str,
        config: Optional[KGConfig] = None,
        entity_limit: int = 10000,
        description_batch_size: int = 80,
        db_batch_size: int = 1000,
        force_recreate: bool = False
) -> chromadb.Collection:
    """
    Complete pipeline to compile a ChromaDB collection from knowledge graph entities.

    This function orchestrates the entire process:
    1. Retrieves top entities from KG by degree
    2. Adds descriptions to entities using config's descriptor functions
    3. Compiles ChromaDB collection

    Args:
        csv_path: Path for intermediate CSV file (will be created/overwritten).
            If not provided, uses config.csv_path (default: ~/.kgnode/data/top_entities.csv).
        config: Optional KGConfig instance with descriptor functions and settings.
            If None, uses default KGConfig with environment variables or built-in defaults.
        entity_limit: Number of top entities to retrieve from KG
        description_batch_size: Batch size for entity description queries (max 100)
        db_batch_size: Number of documents to insert per batch into ChromaDB
        force_recreate: If True, delete existing collection and create new one

    Returns:
        ChromaDB collection object
    """
    # Initialize config if not provided
    if config is None:
        config = KGConfig.default()
    logger.info("=" * 60)
    logger.info("STEP 1: Retrieving top entities by degree")
    logger.info("=" * 60)

    # Step 1: Get top entities by degree
    get_top_entities_by_degree(limit=entity_limit, output_file=csv_path, config=config)

    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Adding descriptions to entities")
    logger.info("=" * 60)

    # Step 2: Add descriptions to the CSV using config
    add_descriptions_to_csv_batched(
        csv_path=csv_path,
        config=config,
        output_path=csv_path,
        batch_size=description_batch_size
    )

    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Compiling ChromaDB collection")
    logger.info("=" * 60)

    # Step 3: Compile ChromaDB from the CSV with descriptions
    collection = compile_entities_chromadb_from_csv(
        csv_path=csv_path,
        config=config,
        batch_size=db_batch_size,
        force_recreate=force_recreate
    )

    logger.info("\n" + "=" * 60)
    logger.info("✓ PIPELINE COMPLETE")
    logger.info("=" * 60)

    return collection


def compile_entities_chromadb_from_csv(
        csv_path: str,
        config: Optional[KGConfig] = None,
        batch_size: int = 1000,
        force_recreate: bool = False
) -> chromadb.Collection:
    """
    Compile a ChromaDB collection from a CSV file with entity descriptions.
    Uses batch insertion for fast performance.

    Args:
        csv_path: Path to CSV file with 'entity' and 'description' columns
        config: Optional KGConfig instance for collection settings.
            If None, uses default KGConfig with environment variables or built-in defaults.
        batch_size: Number of documents to insert per batch (larger = faster but more memory)
        force_recreate: If True, delete existing collection and create new one

    Returns:
        ChromaDB collection object
    """
    # Initialize config if not provided
    if config is None:
        config = KGConfig.default()

    # Extract config values
    collection_name = config.collection_name
    persist_directory = config.chroma_persist_dir
    embedding_model = config.embedding_model
    # Read the CSV file
    logger.info(f"Reading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Validate required columns
    if 'entity' not in df.columns or 'description' not in df.columns:
        raise ValueError("CSV must have 'entity' and 'description' columns")

    # Remove rows with empty descriptions
    df = df.dropna(subset=['description'])
    df = df[df['description'].str.strip() != '']

    logger.info(f"Found {len(df)} entities with descriptions")

    # Initialize ChromaDB client with persistence
    logger.info(f"Initializing ChromaDB at {persist_directory}...")
    client = chromadb.PersistentClient(path=persist_directory)

    # Delete existing collection if force_recreate is True
    if force_recreate:
        try:
            client.delete_collection(name=collection_name)
            logger.info(f"Deleted existing collection '{collection_name}'")
        except:
            pass

    # Create or get collection with the specified embedding model
    logger.info(f"Creating collection '{collection_name}' with model '{embedding_model}'...")
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={
            "hnsw:space": "cosine",
            "description": "Entity descriptions from knowledge graph"
        },
        embedding_function=chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
    )

    # Check if collection already has data
    existing_count = collection.count()
    if existing_count > 0 and not force_recreate:
        logger.info(f"Collection already has {existing_count} documents. Use force_recreate=True to rebuild.")
        return collection

    # Prepare data for batch insertion
    entities = df['entity'].tolist()
    descriptions = df['description'].tolist()

    # Insert in batches for performance
    total_batches = (len(entities) + batch_size - 1) // batch_size
    logger.info(f"\nInserting {len(entities)} entities in {total_batches} batches...")

    for i in range(0, len(entities), batch_size):
        batch_entities = entities[i:i + batch_size]
        batch_descriptions = descriptions[i:i + batch_size]

        # Use entity URIs as IDs (ChromaDB requires string IDs)
        # If URIs are too long or have special chars, create simpler IDs
        batch_ids = [f"entity_{j}" for j in range(i, min(i + batch_size, len(entities)))]

        # Store original entity URI in metadata
        batch_metadatas = [{"entity_uri": entity} for entity in batch_entities]

        collection.add(
            ids=batch_ids,
            documents=batch_descriptions,
            metadatas=batch_metadatas
        )

        batch_num = (i // batch_size) + 1
        logger.info(f"Batch {batch_num}/{total_batches} completed ({len(batch_entities)} entities)")

    final_count = collection.count()
    logger.info(f"\n✓ ChromaDB compilation complete!")
    logger.info(f"✓ Total entities indexed: {final_count}")
    logger.info(f"✓ Collection: {collection_name}")
    logger.info(f"✓ Persisted at: {persist_directory}")

    return collection


def _enhanced_query_entity(entity_name: str, entity_types: List[str]) -> str:
    """Build type-enhanced query string for semantic search.

    Incorporates entity type information into the query string to improve
    semantic search relevance by matching the entity descriptor format.

    Args:
        entity_name: Original entity name (e.g., "Geoffrey Hinton")
        entity_types: Predicted entity types (e.g., ["Creator", "Person"])

    Returns:
        Enhanced query string matching descriptor format

    Examples:
        entity_name="Geoffrey Hinton", types=["Creator", "Person"]
        -> "Creator Person: Geoffrey Hinton."

        entity_name="Deep Learning", types=["Article", "Publication"]
        -> "Article Publication: Deep Learning."

        entity_name="neural networks", types=[]
        -> "neural networks"
    """
    if not entity_types:
        return entity_name

    # Match exact descriptor format: "{Type1} {Type2}: {Name}."
    types_str = " ".join(entity_types)
    enhanced_query = f"{types_str}: {entity_name}."

    return enhanced_query


def semantic_search_entities(
        collection: chromadb.Collection,
        query: str,
        n_results: int = 5,
        entity_types: Optional[List[str]] = None
) -> List[Dict[str, any]]:
    """Search for similar entities based on query with optional type enhancement.

    Args:
        collection: ChromaDB collection
        query: Query text to search for (entity name)
        n_results: Number of results to return
        entity_types: Optional list of entity types to enhance the search query.
            If provided, types are incorporated into the query for better semantic matching.

    Returns:
        List of dictionaries with entity URIs, descriptions, and similarity scores
    """
    # Enhance query with entity types if provided
    if entity_types:
        enhanced_query = _enhanced_query_entity(query, entity_types)
        logger.debug(f"  Semantic search enhanced: '{query}' -> '{enhanced_query}'")
    else:
        enhanced_query = query

    results = collection.query(
        query_texts=[enhanced_query],
        n_results=n_results
    )

    # Format results
    formatted_results = []
    for i in range(len(results['ids'][0])):
        formatted_results.append({
            'entity_uri': results['metadatas'][0][i]['entity_uri'],
            'description': results['documents'][0][i],
            'distance': results['distances'][0][i] if 'distances' in results else None,
            'id': results['ids'][0][i]
        })

    return formatted_results


def get_entities_collection(
        config: Optional[KGConfig] = None,
        csv_path: Optional[str] = None,
        entity_limit: int = 10000,
        description_batch_size: int = 80,
        db_batch_size: int = 1000,
        force_recreate: bool = False
) -> chromadb.Collection:
    """Get or create ChromaDB collection for entities.

    This function intelligently handles ChromaDB collection lifecycle:
    - If force_recreate=True: Recreates collection from scratch
    - If collection exists: Loads existing collection
    - If collection doesn't exist: Creates new collection via compile_entities_chromadb()

    Args:
        config: Optional KGConfig instance for collection settings.
            If None, uses default KGConfig with environment variables or built-in defaults.
        csv_path: Optional path for intermediate CSV file. If None, uses default:
            "_data/vector_db/top_entities.csv"
        entity_limit: Number of top entities to retrieve from KG (used only if creating)
        description_batch_size: Batch size for entity description queries (used only if creating)
        db_batch_size: Number of documents to insert per batch (used only if creating)
        force_recreate: If True, delete existing collection and create new one

    Returns:
        ChromaDB collection for entities
    """
    if config is None:
        config = KGConfig.default()

    # Set default csv_path if not provided
    if csv_path is None:
        csv_path = "../../_data/vector_db/top_entities.csv"

    # If force_recreate, always compile
    if force_recreate:
        logger.info("Force recreate enabled - compiling ChromaDB from scratch...")
        return compile_entities_chromadb(
            csv_path=csv_path,
            config=config,
            entity_limit=entity_limit,
            description_batch_size=description_batch_size,
            db_batch_size=db_batch_size,
            force_recreate=True
        )

    # Check if collection exists
    if _chromadb_exists(config=config):
        logger.info("ChromaDB collection found - loading existing collection...")
        return _load_chromadb(config=config)
    else:
        logger.info("ChromaDB collection not found - creating new collection...")
        return compile_entities_chromadb(
            csv_path=csv_path,
            config=config,
            entity_limit=entity_limit,
            description_batch_size=description_batch_size,
            db_batch_size=db_batch_size,
            force_recreate=False
        )


def add_or_update_entities(
        entity_uris: List[str],
        config: Optional[KGConfig] = None,
        batch_size: int = 80,
        save_csv: bool = False
) -> tuple[int, int, chromadb.Collection]:
    """
    Add new entities to ChromaDB or update existing ones with new descriptions.

    This function generates descriptions for the provided entity URIs using the config's
    descriptor functions, then either adds them as new entries or updates existing ones.
    ChromaDB automatically re-embeds updated descriptions.

    Args:
        entity_uris: List of entity URIs to add or update
        config: Optional KGConfig instance for generating descriptions and loading collection.
            If None, uses default KGConfig with environment variables or built-in defaults.
        batch_size: Batch size for description generation and DB operations
        save_csv: If True, saves the processed entities to a timestamped CSV file
            in config.chroma_persist_dir with columns: entity_uri, description, status

    Returns:
        Tuple of (num_added, num_updated, collection)

    Raises:
        RuntimeError: If ChromaDB collection does not exist
    """
    # Initialize config if not provided
    if config is None:
        config = KGConfig.default()

    # Check if collection exists
    if not _chromadb_exists(config=config):
        raise RuntimeError(
            "ChromaDB collection does not exist. Please call get_entities_collection() first "
            "to create the collection before adding or updating entities."
        )

    # Load collection using config
    collection = _load_chromadb(config=config)

    logger.info(f"Processing {len(entity_uris)} entities...")

    # Generate descriptions in batches
    all_descriptions = {}
    for i in range(0, len(entity_uris), batch_size):
        batch = entity_uris[i:i + batch_size]
        logger.info(f"Generating descriptions for batch {i // batch_size + 1}/{(len(entity_uris) - 1) // batch_size + 1}...")
        batch_descriptions = config.describe_entities_batch(batch)
        all_descriptions.update(batch_descriptions)

    # Filter out entities with empty descriptions
    valid_entities = [(uri, desc) for uri, desc in all_descriptions.items()
                      if desc and desc.strip()]

    if not valid_entities:
        logger.info("No valid descriptions generated. Nothing to add or update.")
        return 0, 0, collection

    logger.info(f"\nChecking which entities exist in collection...")

    # Separate into add and update lists
    to_add = []
    to_update = []

    # Query for each entity URI to check if it exists
    for entity_uri, description in valid_entities:
        # Query by metadata to find if entity exists
        results = collection.get(
            where={"entity_uri": entity_uri}
        )

        if results['ids']:
            # Entity exists - prepare for update
            to_update.append({
                'id': results['ids'][0],
                'uri': entity_uri,
                'description': description
            })
        else:
            # Entity doesn't exist - prepare for addition
            to_add.append({
                'uri': entity_uri,
                'description': description
            })

    logger.info(f"Found {len(to_update)} entities to update and {len(to_add)} entities to add")

    # Update existing entities in batches
    num_updated = 0
    if to_update:
        logger.info(f"\nUpdating {len(to_update)} existing entities...")
        for i in range(0, len(to_update), batch_size):
            batch = to_update[i:i + batch_size]

            batch_ids = [item['id'] for item in batch]
            batch_documents = [item['description'] for item in batch]
            batch_metadatas = [{"entity_uri": item['uri']} for item in batch]

            collection.update(
                ids=batch_ids,
                documents=batch_documents,
                metadatas=batch_metadatas
            )

            num_updated += len(batch)
            logger.info(f"Updated batch {i // batch_size + 1}/{(len(to_update) - 1) // batch_size + 1}")

    # Add new entities in batches
    num_added = 0
    if to_add:
        logger.info(f"\nAdding {len(to_add)} new entities...")
        next_id = _get_next_entity_id(collection)

        for i in range(0, len(to_add), batch_size):
            batch = to_add[i:i + batch_size]

            batch_ids = [f"entity_{next_id + j}" for j in range(len(batch))]
            batch_documents = [item['description'] for item in batch]
            batch_metadatas = [{"entity_uri": item['uri']} for item in batch]

            collection.add(
                ids=batch_ids,
                documents=batch_documents,
                metadatas=batch_metadatas
            )

            num_added += len(batch)
            next_id += len(batch)
            logger.info(f"Added batch {i // batch_size + 1}/{(len(to_add) - 1) // batch_size + 1}")

    logger.info(f"\n✓ Complete: Added {num_added}, Updated {num_updated}")
    logger.info(f"✓ Total entities in collection: {collection.count()}")

    # Save to CSV if requested
    if save_csv and (to_add or to_update):
        records = []
        for item in to_add:
            records.append({
                'entity_uri': item['uri'],
                'description': item['description'],
                'status': 'added'
            })
        for item in to_update:
            records.append({
                'entity_uri': item['uri'],
                'description': item['description'],
                'status': 'updated'
            })

        df = pd.DataFrame(records)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        csv_filename = f"entities_{timestamp}.csv"
        csv_path = os.path.join(config.data_dir, csv_filename)

        os.makedirs(config.data_dir, exist_ok=True)
        df.to_csv(csv_path, index=False)
        logger.info(f"✓ Saved entity records to {csv_path}")

    return num_added, num_updated, collection


def delete_entities(
        entity_uris: List[str],
        config: Optional[KGConfig] = None
) -> int:
    """
    Delete entities from ChromaDB by their URIs.

    Args:
        entity_uris: List of entity URIs to delete
        config: Optional KGConfig instance for loading collection.
            If None, uses default KGConfig with environment variables or built-in defaults.

    Returns:
        Number of entities successfully deleted

    Raises:
        RuntimeError: If ChromaDB collection does not exist
    """
    # Initialize config if not provided
    if config is None:
        config = KGConfig.default()

    # Check if collection exists
    if not _chromadb_exists(config=config):
        raise RuntimeError(
            "ChromaDB collection does not exist. Please call get_entities_collection() first "
            "to create the collection before deleting entities."
        )

    # Load collection using config
    collection = _load_chromadb(config=config)

    logger.info(f"Searching for {len(entity_uris)} entities to delete...")

    # Find ChromaDB IDs for each entity URI
    ids_to_delete = []
    for entity_uri in entity_uris:
        results = collection.get(
            where={"entity_uri": entity_uri}
        )

        if results['ids']:
            ids_to_delete.extend(results['ids'])

    # Delete all found IDs in one operation
    if ids_to_delete:
        logger.info(f"Deleting {len(ids_to_delete)} entities...")
        collection.delete(ids=ids_to_delete)
        logger.info(f"✓ Successfully deleted {len(ids_to_delete)} entities")
        logger.info(f"✓ Remaining entities in collection: {collection.count()}")
    else:
        logger.info("No matching entities found to delete")

    return len(ids_to_delete)
