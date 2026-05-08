"""Entity ranking operations - retrieve top entities from knowledge graph by degree."""

import csv
import os
import threading
import time
from functools import lru_cache
from typing import Dict, List, Optional

from kgnode.core.kg_config import KGConfig
from kgnode.core.logging_config import get_logger
from kgnode.core.sparql_query import execute_sparql_query


logger = get_logger(__name__)


@lru_cache(maxsize=10)
def get_top_entities_by_degree(
    limit: int = 1_000_000,
    output_file: Optional[str] = None,
    config: Optional[KGConfig] = None
) -> List[Dict[str, str]]:
    """
    Get top N entities from knowledge graph sorted by degree (number of connections).
    Saves results to CSV file.

    Args:
        limit (int): Number of top entities to retrieve. Default is 1,000,000.
        output_file (str): Path to output CSV file. If None, defaults to ~/.kgnode/data/top_entities.csv.
            Can be overridden via KGNODE_DATA_DIR environment variable.
        config: Optional KGConfig instance for configuration.
            If None, uses default KGConfig with environment variables or built-in defaults.

    Returns:
        List[Dict]: List of entities with their URIs and degrees.
                   Each dict has keys: 'entity', 'degree'
    """
    # Initialize config if not provided
    if config is None:
        config = KGConfig.default()

    # Use default path if not provided
    if output_file is None:
        output_file = config.csv_path

    # Ensure parent directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    logger.debug(f"Querying top {limit:,} entities by degree, For 1 million nodes KG it takes 7 seconds to run.")

    sparql_query = f"""
    SELECT ?entity (COUNT(?o) as ?degree)
        WHERE {{
          ?entity ?p ?o .
        }}
        GROUP BY ?entity
        ORDER BY DESC(?degree)
        LIMIT {limit}
    """

    # in+out
    # SELECT ?entity (COUNT(?connection) as ?degree)
    #     WHERE {{
    #       {{ ?entity ?p ?o }}
    #       UNION
    #       {{ ?s ?p ?entity }}
    #     }}
    #     GROUP BY ?entity
    #     ORDER BY DESC(?degree)
    #     LIMIT {limit}

    # indegree
    # SELECT ?entity (COUNT(?s) as ?degree)
    # WHERE {{
    #   ?s ?p ?entity .
    # }}
    # GROUP BY ?entity
    # ORDER BY DESC(?degree)
    # LIMIT {limit}

    # outdegree
    # SELECT ?entity (COUNT(?o) as ?degree)
    #     WHERE {{
    #       ?entity ?p ?o .
    #     }}
    #     GROUP BY ?entity
    #     ORDER BY DESC(?degree)
    #     LIMIT {limit}

    # Spinner setup
    spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    spinner_running = True

    def spin():
        i = 0
        start = time.time()
        while spinner_running:
            elapsed = int(time.time() - start)
            mins, secs = divmod(elapsed, 60)
            print(f'\r{spinner_chars[i % len(spinner_chars)]} Querying... {mins:02d}:{secs:02d}', end='', flush=True)
            i += 1
            time.sleep(0.1)

    # Start spinner
    spinner_thread = threading.Thread(target=spin)
    spinner_thread.start()

    start_time = time.time()
    results = execute_sparql_query(sparql_query, config=config)
    query_time = time.time() - start_time

    # Stop spinner
    spinner_running = False
    spinner_thread.join()

    logger.info(f"\r✓ Query completed in {query_time:.1f} seconds ({query_time / 60:.1f} minutes)")
    logger.info(f"✓ Retrieved {len(results):,} entities")
    logger.info(f"Saving to {output_file}...")

    # Save to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['entity', 'degree']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for idx, row in enumerate(results):
            writer.writerow({
                'entity': row['entity'],
                'degree': row['degree']
            })

            if (idx + 1) % 100_000 == 0:
                logger.info(f"  Written {idx + 1:,} rows...")

    total_time = time.time() - start_time
    logger.info(f"✓ Done! Saved {len(results):,} entities to {output_file}")
    logger.info(f"Total time: {total_time:.1f} seconds")

    return results


if __name__ == "__main__":
    entities = get_top_entities_by_degree(limit=10000)
    print(entities)
