"""Relation-specific ChromaDB operations."""

import os
import re
from typing import Dict, List, Optional

import chromadb
import pandas as pd

from kgnode._chromadb_shared import _chromadb_exists, _load_chromadb, uri_to_label
from kgnode.core.kg_config import KGConfig
from kgnode.core.logging_config import get_logger


logger = get_logger(__name__)


def compile_relations_chromadb(
        csv_path: Optional[str] = None,
        config: Optional[KGConfig] = None,
        batch_size: int = 100,
        force_recreate: bool = False,
        relation_limit: Optional[int] = None
) -> chromadb.Collection:
    """
    Complete pipeline to compile a ChromaDB collection for KG relations/predicates.

    This function orchestrates the entire process:
    1. Retrieves all unique relations from KG
    2. Adds rich descriptions to relations using domain/range/type info
    3. Compiles ChromaDB relations collection

    Args:
        csv_path: Path for intermediate CSV file. If None, defaults to
            ~/.kgnode/data/kg_relations.csv
        config: Optional KGConfig instance. If None, uses default.
        batch_size: Number of documents to insert per batch into ChromaDB (default: 100)
        force_recreate: If True, delete existing collection and create new one
        relation_limit: Optional limit on number of relations to process (top N by usage count).
                       If None, processes all relations.

    Returns:
        ChromaDB collection object for relations
    """
    # Initialize config if not provided
    if config is None:
        config = KGConfig.default()

    # Set default csv_path if not provided
    if csv_path is None:
        data_dir = config.data_dir
        csv_path = os.path.join(data_dir, "kg_relations.csv")

    logger.info("=" * 60)
    logger.info("STEP 1: Retrieving all unique relations from KG")
    logger.info("=" * 60)

    # Step 1: Get all relations from KG
    from kgnode._relation_ranker import get_all_relations

    relations = get_all_relations(output_file=csv_path, config=config, limit=relation_limit)

    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Creating descriptions for relations")
    logger.info("=" * 60)

    # Step 2: Add descriptions to relations
    logger.info(f"Processing {len(relations)} relations...")

    # Read CSV and add descriptions
    df = pd.read_csv(csv_path)

    if 'relation' not in df.columns:
        raise ValueError("CSV must have a 'relation' column")

    relation_uris = df['relation'].unique().tolist()

    # Create descriptions using relation descriptor
    from kgnode._relation_descriptor import create_relation_description

    descriptions = []
    for i, relation_uri in enumerate(relation_uris):
        if (i + 1) % 50 == 0:
            logger.info(f"  Processed {i + 1}/{len(relation_uris)} relations...")

        desc = create_relation_description(relation_uri, config=config)
        descriptions.append(desc)

    df['description'] = descriptions
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved relation descriptions to {csv_path}")

    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Compiling ChromaDB relations collection")
    logger.info("=" * 60)

    # Step 3: Compile ChromaDB from the CSV with descriptions
    collection_name = config.RELATION_COLLECTION_NAME
    persist_directory = config.chroma_persist_dir
    embedding_model = config.embedding_model

    # Remove rows with empty descriptions
    df = df.dropna(subset=['description'])
    df = df[df['description'].str.strip() != '']

    logger.info(f"Found {len(df)} relations with descriptions")

    # Initialize ChromaDB client
    logger.info(f"Initializing ChromaDB at {persist_directory}...")
    client = chromadb.PersistentClient(path=persist_directory)

    # Delete existing collection if force_recreate
    if force_recreate:
        try:
            client.delete_collection(name=collection_name)
            logger.info(f"Deleted existing collection '{collection_name}'")
        except:
            pass

    # Create collection
    logger.info(f"Creating collection '{collection_name}' with model '{embedding_model}'...")
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={
            "hnsw:space": "cosine",
            "description": "Relation/predicate descriptions from knowledge graph"
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
    relations = df['relation'].tolist()
    descriptions = df['description'].tolist()

    # Insert in batches
    total_batches = (len(relations) + batch_size - 1) // batch_size
    logger.info(f"\nInserting {len(relations)} relations in {total_batches} batches...")

    for i in range(0, len(relations), batch_size):
        batch_relations = relations[i:i + batch_size]
        batch_descriptions = descriptions[i:i + batch_size]

        # Create IDs for this batch (relation_0, relation_1, ...)
        batch_ids = [f"relation_{j}" for j in range(i, i + len(batch_relations))]

        # Create metadatas with relation URIs
        batch_metadatas = [
            {
                "relation_uri": rel,
                "label": rel.split('#')[-1].split('/')[-1]  # Extract label from URI
            }
            for rel in batch_relations
        ]

        # Insert batch
        collection.add(
            ids=batch_ids,
            documents=batch_descriptions,
            metadatas=batch_metadatas
        )

        batch_num = (i // batch_size) + 1
        logger.info(f"Batch {batch_num}/{total_batches} completed ({len(batch_relations)} relations)")

    final_count = collection.count()
    logger.info(f"\n✓ Relations ChromaDB compilation complete!")
    logger.info(f"✓ Total relations indexed: {final_count}")
    logger.info(f"✓ Collection: {collection_name}")
    logger.info(f"✓ Persisted at: {persist_directory}")

    logger.info("\n" + "=" * 60)
    logger.info("✓ PIPELINE COMPLETE")
    logger.info("=" * 60)

    return collection


def _enhanced_query_relation(relation_name: str, predicted_relations: List[str]) -> str:
    """Build type-enhanced query string for relation semantic search.

    Uses multi-term approach: includes both original predicted relation names and camelCase-split words
    for better semantic matching (12.5% improvement over basic type enhancement).

    Args:
        relation_name: Original relation verb/action from query (e.g., "publish", "in year")
        predicted_relations: Predicted canonical relation names (e.g., ["authoredBy"], ["yearOfPublication"])

    Returns:
        Enhanced query string with multi-term enrichment

    Examples:
        relation_name="publish", predicted_relations=["authoredBy"]
        -> "authoredBy authored by publish"

        relation_name="in year", predicted_relations=["yearOfPublication"]
        -> "yearOfPublication year of publication in year"

        relation_name="affiliated with", predicted_relations=[]
        -> "affiliated with"
    """
    if not predicted_relations:
        return relation_name

    # Multi-term approach for better semantic matching
    terms = []

    for predicted_relation in predicted_relations:
        # Add original camelCase type
        terms.append(predicted_relation)

        # Split camelCase: "authoredBy" -> "authored by"
        relation_words = re.sub('([A-Z])', r' \1', predicted_relation).strip().lower()
        terms.append(relation_words)

    # Add the extracted relation name
    terms.append(relation_name)

    # Combine all terms
    enhanced_query = " ".join(terms)

    return enhanced_query


def semantic_search_relations(
    collection: chromadb.Collection,
    query: str,
    n_results: int = 5,
    predicted_relations: Optional[List[str]] = None
) -> List[Dict[str, any]]:
    """Search for similar relations based on query with optional predicted relation names.

    Args:
        collection: Relations ChromaDB collection
        query: Query text to search for (relation verb/action, e.g., "published", "authored by")
        n_results: Number of results to return
        predicted_relations: Optional list of predicted canonical relation names from schema
            (e.g., ["authoredBy", "publishedIn"]). If provided, these are incorporated into
            the query for better semantic matching.

    Returns:
        List of dictionaries with relation URIs, labels, descriptions, and similarity scores.
        Each dict has keys: relation_uri, label, distance, description
    """
    # Enhance query with predicted relation names if provided
    if predicted_relations:
        enhanced_query = _enhanced_query_relation(query, predicted_relations)
        logger.debug(f"  Relation search enhanced: '{query}' -> '{enhanced_query}'")
    else:
        enhanced_query = query

    results = collection.query(
        query_texts=[enhanced_query],
        n_results=n_results
    )

    # Format results
    formatted_results = []
    for i in range(len(results['ids'][0])):
        metadata = results['metadatas'][0][i] if results['metadatas'] else {}
        formatted_results.append({
            'relation_uri': metadata.get('relation_uri', ''),
            'label': metadata.get('label', ''),
            'distance': results['distances'][0][i] if 'distances' in results else None,
            'description': results['documents'][0][i]
        })

    return formatted_results


def get_relations_collection(config: Optional[KGConfig] = None) -> chromadb.Collection:
    """Get or create ChromaDB collection for relations.

    Creates a separate collection for KG relations/predicates using the
    RELATION_COLLECTION_NAME from KGConfig.

    Args:
        config: Optional KGConfig instance. If None, uses default.

    Returns:
        ChromaDB collection for relations

    Note:
        This collection must be populated separately using compile_relations_chromadb().
        Relations should be added via the compilation pipeline.
    """
    if config is None:
        config = KGConfig.default()

    persist_directory = config.chroma_persist_dir
    embedding_model = config.embedding_model

    client = chromadb.PersistentClient(path=persist_directory)

    collection = client.get_or_create_collection(
        name=config.RELATION_COLLECTION_NAME,
        embedding_function=chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        ),
        metadata={"hnsw:space": "cosine"}
    )

    return collection


def compile_relations_chromadb_from_csv(
        csv_path: str,
        config: Optional[KGConfig] = None,
        batch_size: int = 100,
        force_recreate: bool = False
) -> chromadb.Collection:
    """
    Compile ChromaDB collection for relations directly from a CSV file with descriptions.

    Expected CSV format:
    - Required columns: 'relation', 'description'
    - Optional column: 'count' (usage count)

    Args:
        csv_path: Path to CSV file with relation URIs and descriptions
        config: Optional KGConfig instance for collection settings.
            If None, uses default KGConfig with environment variables or built-in defaults.
        batch_size: Number of documents to insert per batch (larger = faster but more memory)
        force_recreate: If True, delete existing collection and create new one

    Returns:
        ChromaDB collection object for relations
    """
    # Initialize config if not provided
    if config is None:
        config = KGConfig.default()

    # Extract config values
    collection_name = config.RELATION_COLLECTION_NAME
    persist_directory = config.chroma_persist_dir
    embedding_model = config.embedding_model

    # Read the CSV file
    logger.info(f"Reading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Validate required columns
    if 'relation' not in df.columns or 'description' not in df.columns:
        raise ValueError("CSV must have 'relation' and 'description' columns")

    # Remove rows with empty descriptions
    df = df.dropna(subset=['description'])
    df = df[df['description'].str.strip() != '']

    logger.info(f"Found {len(df)} relations with descriptions")

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
            "description": "Relation/predicate descriptions from knowledge graph"
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
    relations = df['relation'].tolist()
    descriptions = df['description'].tolist()

    # Insert in batches for performance
    total_batches = (len(relations) + batch_size - 1) // batch_size
    logger.info(f"\nInserting {len(relations)} relations in {total_batches} batches...")

    for i in range(0, len(relations), batch_size):
        batch_relations = relations[i:i + batch_size]
        batch_descriptions = descriptions[i:i + batch_size]

        # Create IDs (relation_0, relation_1, ...)
        batch_ids = [f"relation_{j}" for j in range(i, min(i + batch_size, len(relations)))]

        # Store original relation URI in metadata
        batch_metadatas = [
            {
                "relation_uri": rel,
                "label": rel.split('#')[-1].split('/')[-1]  # Extract label from URI
            }
            for rel in batch_relations
        ]

        collection.add(
            ids=batch_ids,
            documents=batch_descriptions,
            metadatas=batch_metadatas
        )

        batch_num = (i // batch_size) + 1
        logger.info(f"Batch {batch_num}/{total_batches} completed ({len(batch_relations)} relations)")

    final_count = collection.count()
    logger.info(f"\n✓ ChromaDB compilation complete!")
    logger.info(f"✓ Total relations indexed: {final_count}")
    logger.info(f"✓ Collection: {collection_name}")
    logger.info(f"✓ Persisted at: {persist_directory}")

    return collection


def add_or_update_relations(
        relation_uris: List[str],
        config: Optional[KGConfig] = None,
        batch_size: int = 50,
        save_csv: bool = False
) -> tuple[int, int, chromadb.Collection]:
    """
    Add new relations to ChromaDB or update existing ones with new descriptions.

    This function generates descriptions for the provided relation URIs using
    create_relation_description(), then either adds them as new entries or updates existing ones.

    Args:
        relation_uris: List of relation URIs to add or update
        config: Optional KGConfig instance for generating descriptions and loading collection.
            If None, uses default KGConfig with environment variables or built-in defaults.
        batch_size: Batch size for description generation and DB operations
        save_csv: If True, saves the processed relations to a timestamped CSV file
            in config.chroma_persist_dir with columns: relation_uri, description, status

    Returns:
        Tuple of (num_added, num_updated, collection)

    Raises:
        RuntimeError: If ChromaDB collection does not exist
    """
    # Initialize config if not provided
    if config is None:
        config = KGConfig.default()

    # Check if collection exists
    collection_name = config.RELATION_COLLECTION_NAME
    if not _chromadb_exists(config=config, collection_name=collection_name):
        raise RuntimeError(
            "Relations ChromaDB collection does not exist. Please call get_relations_collection() first "
            "or run compile_relations_chromadb() to create the collection."
        )

    # Load collection
    collection = _load_chromadb(config=config, collection_name=collection_name)

    logger.info(f"Processing {len(relation_uris)} relations...")

    # Generate descriptions in batches
    from kgnode._relation_descriptor import create_relation_description

    all_descriptions = {}
    for i in range(0, len(relation_uris), batch_size):
        batch = relation_uris[i:i + batch_size]
        logger.info(f"Generating descriptions for batch {i // batch_size + 1}/{(len(relation_uris) - 1) // batch_size + 1}...")

        for relation_uri in batch:
            desc = create_relation_description(relation_uri, config=config)
            all_descriptions[relation_uri] = desc

    # Filter out relations with empty descriptions
    valid_relations = [(uri, desc) for uri, desc in all_descriptions.items()
                       if desc and desc.strip()]

    if not valid_relations:
        logger.info("No valid descriptions generated. Nothing to add or update.")
        return 0, 0, collection

    logger.info(f"\nChecking which relations exist in collection...")

    # Separate into add and update lists
    to_add = []
    to_update = []

    # Query for each relation URI to check if it exists
    for relation_uri, description in valid_relations:
        # Query by metadata to find if relation exists
        results = collection.get(
            where={"relation_uri": relation_uri}
        )

        if results['ids']:
            # Relation exists - prepare for update
            to_update.append({
                'id': results['ids'][0],
                'uri': relation_uri,
                'description': description
            })
        else:
            # Relation doesn't exist - prepare for addition
            to_add.append({
                'uri': relation_uri,
                'description': description
            })

    logger.info(f"Found {len(to_update)} relations to update and {len(to_add)} relations to add")

    # Update existing relations in batches
    num_updated = 0
    if to_update:
        logger.info(f"\nUpdating {len(to_update)} existing relations...")
        for i in range(0, len(to_update), batch_size):
            batch = to_update[i:i + batch_size]

            batch_ids = [item['id'] for item in batch]
            batch_documents = [item['description'] for item in batch]
            batch_metadatas = [
                {
                    "relation_uri": item['uri'],
                    "label": item['uri'].split('#')[-1].split('/')[-1]
                }
                for item in batch
            ]

            collection.update(
                ids=batch_ids,
                documents=batch_documents,
                metadatas=batch_metadatas
            )

            num_updated += len(batch)
            logger.info(f"Updated batch {i // batch_size + 1}/{(len(to_update) - 1) // batch_size + 1}")

    # Add new relations in batches
    num_added = 0
    if to_add:
        logger.info(f"\nAdding {len(to_add)} new relations...")

        # Get next available ID
        all_ids = collection.get()['ids']
        if all_ids:
            # Extract numeric parts from relation_N format
            id_numbers = []
            for id_str in all_ids:
                if id_str.startswith('relation_'):
                    try:
                        num = int(id_str.split('_')[1])
                        id_numbers.append(num)
                    except (ValueError, IndexError):
                        continue
            next_id = max(id_numbers) + 1 if id_numbers else 0
        else:
            next_id = 0

        for i in range(0, len(to_add), batch_size):
            batch = to_add[i:i + batch_size]

            batch_ids = [f"relation_{next_id + j}" for j in range(len(batch))]
            batch_documents = [item['description'] for item in batch]
            batch_metadatas = [
                {
                    "relation_uri": item['uri'],
                    "label": item['uri'].split('#')[-1].split('/')[-1]
                }
                for item in batch
            ]

            collection.add(
                ids=batch_ids,
                documents=batch_documents,
                metadatas=batch_metadatas
            )

            num_added += len(batch)
            next_id += len(batch)
            logger.info(f"Added batch {i // batch_size + 1}/{(len(to_add) - 1) // batch_size + 1}")

    logger.info(f"\n✓ Complete: Added {num_added}, Updated {num_updated}")
    logger.info(f"✓ Total relations in collection: {collection.count()}")

    # Save to CSV if requested
    if save_csv and (to_add or to_update):
        records = []
        for item in to_add:
            records.append({
                'relation_uri': item['uri'],
                'description': item['description'],
                'status': 'added'
            })
        for item in to_update:
            records.append({
                'relation_uri': item['uri'],
                'description': item['description'],
                'status': 'updated'
            })

        # Save to timestamped CSV
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        csv_filename = f"relations_{timestamp}.csv"
        csv_path = os.path.join(config.data_dir, csv_filename)

        os.makedirs(config.data_dir, exist_ok=True)
        df = pd.DataFrame(records)
        df.to_csv(csv_path, index=False)
        logger.info(f"✓ Saved update log to {csv_path}")

    return num_added, num_updated, collection


def delete_relations(
        relation_uris: List[str],
        config: Optional[KGConfig] = None
) -> int:
    """
    Delete relations from ChromaDB by their URIs.

    Args:
        relation_uris: List of relation URIs to delete
        config: Optional KGConfig instance for loading collection.
            If None, uses default KGConfig with environment variables or built-in defaults.

    Returns:
        Number of relations successfully deleted

    Raises:
        RuntimeError: If ChromaDB collection does not exist
    """
    # Initialize config if not provided
    if config is None:
        config = KGConfig.default()

    # Check if collection exists
    collection_name = config.RELATION_COLLECTION_NAME
    if not _chromadb_exists(config=config, collection_name=collection_name):
        raise RuntimeError(
            "Relations ChromaDB collection does not exist. Please call get_relations_collection() first "
            "or run compile_relations_chromadb() to create the collection."
        )

    # Load collection
    collection = _load_chromadb(config=config, collection_name=collection_name)

    logger.info(f"Searching for {len(relation_uris)} relations to delete...")

    # Find ChromaDB IDs for each relation URI
    ids_to_delete = []
    for relation_uri in relation_uris:
        results = collection.get(
            where={"relation_uri": relation_uri}
        )

        if results['ids']:
            ids_to_delete.extend(results['ids'])

    # Delete all found IDs in one operation
    if ids_to_delete:
        logger.info(f"Deleting {len(ids_to_delete)} relations...")
        collection.delete(ids=ids_to_delete)
        logger.info(f"✓ Successfully deleted {len(ids_to_delete)} relations")
        logger.info(f"✓ Remaining relations in collection: {collection.count()}")
    else:
        logger.info("No matching relations found to delete")

    return len(ids_to_delete)
