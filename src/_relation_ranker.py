"""Relation ranking operations - retrieve all unique relations from knowledge graph."""

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
def get_all_relations(
    output_file: Optional[str] = None,
    config: Optional[KGConfig] = None,
    limit: Optional[int] = None
) -> List[Dict[str, str]]:
    """
    Get all unique relations/predicates from knowledge graph with usage counts.
    Saves results to CSV file.

    Args:
        output_file: Path to output CSV file. If None, defaults to ~/.kgnode/data/kg_relations.csv
        config: Optional KGConfig instance. If None, uses default.
        limit: Optional limit on number of relations to return (top N by usage count).
               If None, returns all relations.

    Returns:
        List[Dict]: List of relations with their URIs and usage counts.
                   Each dict has keys: 'relation', 'count'
    """
    # Initialize config if not provided
    if config is None:
        config = KGConfig.default()

    # Use default path if not provided
    if output_file is None:
        data_dir = config.data_dir
        output_file = os.path.join(data_dir, "kg_relations.csv")

    # Ensure parent directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    logger.debug("Querying all unique relations/predicates from KG...")

    # SPARQL query to get all unique predicates with their usage counts
    sparql_query = """
    SELECT ?relation (COUNT(?relation) AS ?count)
    WHERE {
      ?s ?relation ?o .
    }
    GROUP BY ?relation
    ORDER BY DESC(?count)
    """

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

    logger.info(f"\r✓ Query completed in {query_time:.1f} seconds")
    logger.info(f"✓ Retrieved {len(results):,} unique relations")

    # Apply limit if specified
    if limit is not None and limit < len(results):
        results = results[:limit]
        logger.info(f"✓ Limited to top {limit:,} relations by usage count")

    logger.info(f"Saving to {output_file}...")

    # Save to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['relation', 'count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for idx, row in enumerate(results):
            writer.writerow({
                'relation': row['relation'],
                'count': row['count']
            })

    total_time = time.time() - start_time
    logger.info(f"✓ Done! Saved {len(results):,} relations to {output_file}")
    logger.info(f"Total time: {total_time:.1f} seconds")

    return results


if __name__ == "__main__":
    relations = get_all_relations()
    print(f"Retrieved {len(relations)} relations")
