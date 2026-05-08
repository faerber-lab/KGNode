"""Add subgraph extraction information to balanced_200.jsonl.

This script reads filtered_seeds and filtered_relations from the dataset
(added by add_filtered_entity_info.py and add_filtered_relation_info.py),
extracts subgraphs for each filtered seed, and saves the results.

PREREQUISITES:
1. Run add_seed_node_info.py first (adds extracted_nodes and extracted_relation_nodes)
2. Run add_filtered_entity_info.py (adds filtered_entity_info with filtered_seeds)
3. Run add_filtered_relation_info.py (adds filtered_relation_info with filtered_relations)
4. Then run this script
"""

import json
import os
import sys
from typing import Any, Dict, List


# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logging_util import setup_dual_output

from kgnode import KGConfig
from kgnode.core.logging_config import get_logger
from kgnode.subgraph_extraction import get_subgraphs


logger = get_logger(__name__)


# NOTE: Filtering logic removed - now done by add_filtered_entity_info.py and add_filtered_relation_info.py
# This script only does subgraph extraction from pre-filtered nodes


def extract_subgraphs_for_seed(
    seed_node: Dict[str, Any],
    question_text: str,
    config: KGConfig,
    max_hops: int = 6,
    max_k: int = 4,
    all_seed_nodes: List[Dict[str, Any]] = None,
    all_relation_nodes: List[Dict[str, Any]] = None
) -> tuple[Dict[str, Any], str]:
    """Extract subgraphs for a single seed node with metadata.

    Args:
        seed_node: Seed node dict with 'entity_uri', 'label', 'score', 'entity_from'
        question_text: Question string
        config: KGConfig instance
        max_hops: Maximum hops for traversal
        max_k: Maximum neighbors per hop
        all_seed_nodes: Optional list of all filtered seed nodes
        all_relation_nodes: Optional list of all filtered relation nodes

    Returns:
        Tuple of (seed_result_dict, template_text), or (None, None) if extraction fails
    """
    try:
        subgraphs, template_text = get_subgraphs(
            seed_node=seed_node['entity_uri'],
            query=question_text,
            config=config,
            max_hops=max_hops,
            max_k=max_k,
            seed_nodes=all_seed_nodes,
            relation_nodes=all_relation_nodes
        )

        result = {
            'seed_uri': seed_node['entity_uri'],
            'seed_label': seed_node['label'],
            'seed_score': seed_node['score'],
            'entity_from': seed_node['entity_from'],
            'subgraph_count': len(subgraphs),
            'subgraphs': subgraphs  # Already contains 'probability' and 'path_depth'
        }

        return result, template_text
    except Exception as e:
        logger.error(
            f"Failed to extract subgraphs for seed {seed_node['entity_uri']}: {e}"
        )
        return None, None


def _process_single_question(
    question: Dict[str, Any],
    config: KGConfig,
    question_idx: int,
    total: int,
) -> bool:
    """Process a single question and add subgraph extraction info.

    Reads filtered_seeds and filtered_relations from the question
    (added by previous scripts) and extracts subgraphs.

    Args:
        question: Question dict
        config: KGConfig instance
        question_idx: Current question index (1-based)
        total: Total number of questions

    Returns:
        True if successful, False otherwise
    """
    question_id = question.get('id', f'Q{question_idx:04d}')
    print(f"  [{question_idx}/{total}] {question_id}")

    # Get question text
    question_text = question.get('question', {}).get('1', '')
    if not question_text:
        print("    Warning: No question text found, skipping")
        return False

    # Read filtered_seeds from filtered_entity_info (added by add_filtered_entity_info.py)
    if 'filtered_entity_info' not in question:
        print("    Warning: No filtered_entity_info found (run add_filtered_entity_info.py first), skipping")
        return False

    filtered_seeds = question['filtered_entity_info'].get('filtered_seeds', [])

    if not filtered_seeds:
        print("    Warning: No filtered_seeds found, skipping")
        return False

    print(f"    Using {len(filtered_seeds)} pre-filtered seeds")

    # Read filtered_relations from filtered_relation_info (added by add_filtered_relation_info.py)
    filtered_relations = []
    if 'filtered_relation_info' in question:
        filtered_relations = question['filtered_relation_info'].get('filtered_relations', [])
        if filtered_relations:
            print(f"    Using {len(filtered_relations)} pre-filtered relations")

    # Extract subgraphs for each filtered seed
    seed_results = []
    total_subgraphs = 0
    template_text = None

    for j, seed in enumerate(filtered_seeds, 1):
        print(
            f"    Seed {j}/{len(filtered_seeds)}: {seed['label']} "
            f"(score: {seed['score']:.4f})"
        )

        result, seed_template = extract_subgraphs_for_seed(
            seed_node=seed,
            question_text=question_text,
            config=config,
            max_hops=MAX_HOPS,
            max_k=MAX_K,
            all_seed_nodes=filtered_seeds,
            all_relation_nodes=filtered_relations if filtered_relations else None
        )

        if result:
            # Capture template from first successful extraction
            if template_text is None:
                template_text = seed_template
                print(f"    Template: {template_text}")

            seed_results.append(result)
            total_subgraphs += result['subgraph_count']
            print(f"      Extracted {result['subgraph_count']} subgraphs")
        else:
            print("      Failed to extract subgraphs (skipping seed)")

    # If no seeds succeeded, set template to ERROR
    if template_text is None:
        template_text = "ERROR"

    # Add subgraph_extraction field to question
    question['subgraph_extraction'] = {
        'template_text': template_text,
        'filtered_seeds': seed_results
    }

    print(f"    Total: {total_subgraphs} subgraphs from {len(seed_results)} seeds\n")
    return True


def _process_all_questions(
    questions: List[Dict[str, Any]],
    config: KGConfig,
    file_path: str,
    force_regenerate: bool = False,
):
    """Process all questions in one batch."""
    total = len(questions)
    processed = 0
    skipped = 0

    for i, question in enumerate(questions, 1):
        question_id = question.get('id', f'Q{i:04d}')

        # Skip if already processed (unless force_regenerate is True)
        if 'subgraph_extraction' in question and not force_regenerate:
            print(f"[{i}/{total}] {question_id}: SKIPPED (already processed)")
            skipped += 1
            continue

        success = _process_single_question(question, config, i, total)
        if success:
            processed += 1

        # Save incrementally after each question
        with open(file_path, 'w', encoding='utf-8') as f:
            for q in questions:
                f.write(json.dumps(q, ensure_ascii=False) + '\n')

    print(f"\nAll results saved to {file_path}\n")
    _print_summary(questions, processed, skipped)


def _process_by_query_type(
    questions: List[Dict[str, Any]],
    config: KGConfig,
    file_path: str,
    query_types: List[str] = None,
    force_regenerate: bool = False,
):
    """Process questions grouped by query type."""
    # Group questions by query_type
    type_groups = {}
    for q in questions:
        qtype = q.get("query_type", "UNKNOWN")
        if qtype not in type_groups:
            type_groups[qtype] = []
        type_groups[qtype].append(q)

    # Filter by requested types
    if query_types:
        type_groups = {k: v for k, v in type_groups.items() if k in query_types}

    type_stats = {}
    total_processed = 0
    total_skipped = 0

    print(f"Processing {len(type_groups)} query types\n")

    for qtype_idx, (qtype, qtype_questions) in enumerate(type_groups.items(), 1):
        print("=" * 70)
        print(f"[{qtype_idx}/{len(type_groups)}] Query Type: {qtype}")
        print(f"Questions: {len(qtype_questions)}")
        print("=" * 70 + "\n")

        processed = 0
        skipped = 0

        for i, question in enumerate(qtype_questions, 1):
            question_id = question.get('id', f'Q{i:04d}')

            # Skip if already processed (unless force_regenerate is True)
            if 'subgraph_extraction' in question and not force_regenerate:
                print(f"  [{i}/{len(qtype_questions)}] {question_id}: SKIPPED (already processed)")
                skipped += 1
                total_skipped += 1
                continue

            success = _process_single_question(question, config, i, len(qtype_questions))
            if success:
                processed += 1
                total_processed += 1

            # Save incrementally after each question
            with open(file_path, 'w', encoding='utf-8') as f:
                for q in questions:
                    f.write(json.dumps(q, ensure_ascii=False) + '\n')

        type_stats[qtype] = {
            'total': len(qtype_questions),
            'processed': processed,
            'skipped': skipped,
        }

        print(f"\n--- {qtype} Summary ---")
        print(f"  Processed: {processed}/{len(qtype_questions)}")
        print(f"  Skipped:   {skipped}/{len(qtype_questions)}\n")

    # Overall summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)

    total_questions = sum(s['total'] for s in type_stats.values())
    print(f"\nTotal questions: {total_questions}")
    print(f"Processed: {total_processed}")
    print(f"Skipped:   {total_skipped}")

    print("\nPer-type breakdown:")
    for qtype, stats in type_stats.items():
        print(f"  {qtype:<25} Processed: {stats['processed']}/{stats['total']}")

    print("=" * 70)


def _print_summary(questions: List[Dict[str, Any]], processed: int, skipped: int):
    """Print summary statistics."""
    questions_with_subgraphs = sum(
        1 for q in questions
        if 'subgraph_extraction' in q and q['subgraph_extraction']['filtered_seeds']
    )
    total_seeds = sum(
        len(q.get('subgraph_extraction', {}).get('filtered_seeds', []))
        for q in questions
    )
    total_subgraphs = sum(
        sum(
            seed['subgraph_count']
            for seed in q.get('subgraph_extraction', {}).get('filtered_seeds', [])
        )
        for q in questions
    )

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total questions: {len(questions)}")
    print(f"Processed this run: {processed}")
    print(f"Skipped (already done): {skipped}")
    print(f"Questions with subgraphs: {questions_with_subgraphs}/{len(questions)}")
    print(f"Total filtered seeds: {total_seeds}")
    print(f"Total subgraphs extracted: {total_subgraphs}")
    if total_seeds > 0:
        print(f"Average subgraphs per seed: {total_subgraphs/total_seeds:.1f}")
    if questions_with_subgraphs > 0:
        print(f"Average seeds per question: {total_seeds/questions_with_subgraphs:.1f}")
    print("="*60)


def add_subgraph_info(
    file_path: str,
    max_questions: int = None,
    question_type_based_generation: bool = False,
    query_types: List[str] = None,
    force_regenerate: bool = False,
):
    """Process questions and add subgraph extraction information.

    Args:
        file_path: Path to the JSONL file containing questions
        max_questions: Optional cap for quick testing
        question_type_based_generation: If True, process by query type. If False, process all
        query_types: Optional list of query types to process. If None, process all types
        force_regenerate: If True, regenerate subgraphs even if already processed
    """
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return

    # Load questions
    print(f"Loading questions from {file_path}...")
    questions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    if max_questions:
        questions = questions[:max_questions]

    print(f"Loaded {len(questions)} questions\n")

    # Initialize config
    config = KGConfig.default()

    # --- Process by query type or all at once ---
    if question_type_based_generation:
        _process_by_query_type(
            questions, config, file_path, query_types, force_regenerate
        )
    else:
        _process_all_questions(
            questions, config, file_path, force_regenerate
        )



if __name__ == "__main__":
    setup_dual_output(__file__)

    # ============================================================================
    # CONFIGURATION - Edit these variables as needed
    # ============================================================================

    # Subgraph extraction parameters
    MAX_HOPS = 6
    MAX_K = 2

    # File path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "./data/balanced_400.jsonl")

    # Set to True to process by query type, False to process all questions
    question_type_based_generation = False

    # Specify query types to process (only used if question_type_based_generation=True)
    # Example: ["DISAMBIGUATION", "DOUBLE_INTENT"]
    # None = process all types
    query_types = None #["DISAMBIGUATION", "DOUBLE_INTENT"]

    # Maximum number of questions to process (None = all)
    max_questions = None

    # Force regeneration even if already processed
    force_regenerate = True

    # ============================================================================

    print("=" * 70)
    print("SUBGRAPH EXTRACTION FOR EVAL DATASET")
    print("=" * 70)
    print(f"File: {data_path}")
    print(f"Max hops: {MAX_HOPS}, Max k: {MAX_K}")
    if question_type_based_generation:
        print(f"Mode: BY QUERY TYPE")
        if query_types:
            print(f"Types: {', '.join(query_types)}")
    else:
        print(f"Mode: ALL QUESTIONS")
    if max_questions:
        print(f"Max questions: {max_questions}")
    if force_regenerate:
        print(f"Force regenerate: ENABLED (will override existing results)")
    print("=" * 70 + "\n")

    # Run the processing
    add_subgraph_info(
        data_path,
        max_questions=max_questions,
        question_type_based_generation=question_type_based_generation,
        query_types=query_types,
        force_regenerate=force_regenerate,
    )
