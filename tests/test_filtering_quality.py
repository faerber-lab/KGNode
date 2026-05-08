"""
Test script to evaluate if top-K filtering removes valuable subgraphs.

This script analyzes:
1. How many subgraphs with golden entities/relations are filtered out
2. Whether probability is a reliable indicator of subgraph value
3. Alternative filtering strategies that preserve golden content

Usage:
    python tests/test_filtering_quality.py --sample-size 20 --top-k 3
"""

import json
import argparse
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict
import numpy as np


def normalize_uri(uri: str) -> str:
    """Remove angle brackets from URI."""
    return uri.strip("<>").strip()


def extract_uris_from_subgraph(subgraph: Dict) -> Tuple[Set[str], Set[str]]:
    """Extract all entity and relation URIs from a subgraph."""
    entities = set()
    relations = set()

    for triplet in subgraph.get('triplet_uris', []):
        if len(triplet) >= 3:
            subject_uri, predicate_uri, object_uri = triplet[0], triplet[1], triplet[2]
            entities.add(normalize_uri(subject_uri))
            entities.add(normalize_uri(object_uri))
            relations.add(normalize_uri(predicate_uri))

    return entities, relations


def contains_golden_content(
    subgraph: Dict,
    golden_entities: Set[str],
    golden_relations: Set[str]
) -> Tuple[bool, int, int]:
    """Check if subgraph contains golden entities or relations."""
    entities, relations = extract_uris_from_subgraph(subgraph)

    entity_matches = len(entities & golden_entities)
    relation_matches = len(relations & golden_relations)

    is_valuable = entity_matches > 0 or relation_matches > 0

    return is_valuable, entity_matches, relation_matches


def analyze_filtering_per_question(
    question: Dict,
    top_k: int = 3
) -> Dict[str, Any]:
    """Analyze filtering impact for a single question."""

    # Extract golden entities and relations
    golden_entities = set(
        normalize_uri(e) for e in question.get('golden_entities', [])
    )
    golden_relations = set(
        normalize_uri(r) for r in question.get('golden_relations', [])
    )

    # Get all subgraphs from all seeds
    subgraph_extraction = question.get('subgraph_extraction', {})
    filtered_seeds = subgraph_extraction.get('filtered_seeds', [])

    all_subgraphs = []
    seed_to_subgraphs = {}

    for seed_info in filtered_seeds:
        seed_uri = seed_info['seed_uri']
        subgraphs = seed_info.get('subgraphs', [])

        # Analyze each subgraph
        for sg in subgraphs:
            is_valuable, entity_count, relation_count = contains_golden_content(
                sg, golden_entities, golden_relations
            )

            sg_analysis = {
                'subgraph': sg,
                'seed_uri': seed_uri,
                'probability': sg['probability'],
                'path_depth': sg['path_depth'],
                'is_valuable': is_valuable,
                'golden_entity_count': entity_count,
                'golden_relation_count': relation_count,
            }
            all_subgraphs.append(sg_analysis)

        seed_to_subgraphs[seed_uri] = [
            sg_analysis for sg_analysis in all_subgraphs if sg_analysis['seed_uri'] == seed_uri
        ]

    # Simulate top-K filtering per seed
    kept_subgraphs = []
    filtered_out_subgraphs = []

    for seed_uri, seed_subgraphs in seed_to_subgraphs.items():
        # Sort by probability (descending)
        sorted_subgraphs = sorted(
            seed_subgraphs,
            key=lambda x: x['probability'],
            reverse=True
        )

        # Keep top-K
        kept = sorted_subgraphs[:top_k]
        filtered = sorted_subgraphs[top_k:]

        kept_subgraphs.extend(kept)
        filtered_out_subgraphs.extend(filtered)

    # Analyze what was kept vs filtered
    total_subgraphs = len(all_subgraphs)
    total_kept = len(kept_subgraphs)
    total_filtered = len(filtered_out_subgraphs)

    valuable_kept = sum(1 for sg in kept_subgraphs if sg['is_valuable'])
    valuable_filtered = sum(1 for sg in filtered_out_subgraphs if sg['is_valuable'])

    # Count golden entities/relations before and after
    golden_entities_before = set()
    golden_relations_before = set()

    for sg_analysis in all_subgraphs:
        if sg_analysis['is_valuable']:
            entities, relations = extract_uris_from_subgraph(sg_analysis['subgraph'])
            golden_entities_before.update(entities & golden_entities)
            golden_relations_before.update(relations & golden_relations)

    golden_entities_after = set()
    golden_relations_after = set()

    for sg_analysis in kept_subgraphs:
        if sg_analysis['is_valuable']:
            entities, relations = extract_uris_from_subgraph(sg_analysis['subgraph'])
            golden_entities_after.update(entities & golden_entities)
            golden_relations_after.update(relations & golden_relations)

    # Check if any valuable subgraphs were filtered out
    lost_golden_entities = golden_entities_before - golden_entities_after
    lost_golden_relations = golden_relations_before - golden_relations_after

    result = {
        'question_id': question['id'],
        'total_subgraphs': total_subgraphs,
        'kept_subgraphs': total_kept,
        'filtered_subgraphs': total_filtered,
        'valuable_kept': valuable_kept,
        'valuable_filtered': valuable_filtered,
        'golden_entities_before': len(golden_entities_before),
        'golden_entities_after': len(golden_entities_after),
        'golden_entities_lost': len(lost_golden_entities),
        'golden_relations_before': len(golden_relations_before),
        'golden_relations_after': len(golden_relations_after),
        'golden_relations_lost': len(lost_golden_relations),
        'status': 'FAIL' if (lost_golden_entities or lost_golden_relations) else 'PASS',
        'filtered_out_details': filtered_out_subgraphs,
    }

    return result


def main():
    parser = argparse.ArgumentParser(description='Test Top-K Filtering Quality')
    parser.add_argument(
        '--dataset',
        type=str,
        default='eval/data/balanced_200.jsonl',
        help='Path to dataset file'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=20,
        help='Number of questions to test (default: 20)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='Number of subgraphs to keep per seed (default: 3)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print(f"TOP-K FILTERING QUALITY ANALYSIS (K={args.top_k})")
    print("=" * 80)
    print(f"\nDataset: {args.dataset}")
    print(f"Sample size: {args.sample_size}")

    # Load dataset
    print("\nLoading dataset...")
    questions = []
    with open(args.dataset, 'r') as f:
        for i, line in enumerate(f):
            if args.sample_size and i >= args.sample_size:
                break
            questions.append(json.loads(line))
    print(f"✓ Loaded {len(questions)} questions")

    # Analyze each question
    results = []
    failures = []

    print("\n" + "=" * 80)
    print("ANALYZING QUESTIONS")
    print("=" * 80)

    for i, question in enumerate(questions, 1):
        # Skip if no subgraph extraction
        if 'subgraph_extraction' not in question:
            print(f"[{i}/{len(questions)}] {question['id']}: ✗ No subgraph extraction")
            continue

        result = analyze_filtering_per_question(question, args.top_k)
        results.append(result)

        # Print result
        if result['status'] == 'FAIL':
            failures.append(result)
            print(f"\n[{i}/{len(questions)}] {result['question_id']}: ❌ FAIL")
            print(f"  Total subgraphs: {result['total_subgraphs']}")
            print(f"  Kept: {result['kept_subgraphs']} ({result['valuable_kept']} valuable)")
            print(f"  Filtered: {result['filtered_subgraphs']} ({result['valuable_filtered']} valuable)")
            print(f"  Golden entities lost: {result['golden_entities_lost']}/{result['golden_entities_before']}")
            print(f"  Golden relations lost: {result['golden_relations_lost']}/{result['golden_relations_before']}")

            # Show which filtered-out subgraphs contained golden content
            if result['valuable_filtered'] > 0:
                print(f"\n  Valuable subgraphs that were filtered out:")
                for sg in result['filtered_out_details']:
                    if sg['is_valuable']:
                        print(f"    - Prob={sg['probability']:.3f}, Entities={sg['golden_entity_count']}, Relations={sg['golden_relation_count']}")
        else:
            print(f"[{i}/{len(questions)}] {result['question_id']}: ✓ PASS "
                  f"({result['kept_subgraphs']}/{result['total_subgraphs']} kept)")

    # Aggregate statistics
    print("\n" + "=" * 80)
    print("AGGREGATE STATISTICS")
    print("=" * 80)

    if results:
        total_questions = len(results)
        total_subgraphs = sum(r['total_subgraphs'] for r in results)
        total_kept = sum(r['kept_subgraphs'] for r in results)
        total_filtered = sum(r['filtered_subgraphs'] for r in results)
        total_valuable_kept = sum(r['valuable_kept'] for r in results)
        total_valuable_filtered = sum(r['valuable_filtered'] for r in results)

        questions_losing_golden = len(failures)

        total_golden_entities_lost = sum(r['golden_entities_lost'] for r in results)
        total_golden_relations_lost = sum(r['golden_relations_lost'] for r in results)

        print(f"\nOverall:")
        print(f"  Questions analyzed: {total_questions}")
        print(f"  Questions losing golden content: {questions_losing_golden} ({questions_losing_golden/total_questions*100:.1f}%)")

        print(f"\nSubgraph Statistics:")
        print(f"  Total subgraphs: {total_subgraphs}")
        print(f"  Kept: {total_kept} ({total_kept/total_subgraphs*100:.1f}%)")
        print(f"  Filtered out: {total_filtered} ({total_filtered/total_subgraphs*100:.1f}%)")

        print(f"\nValuable Subgraph Distribution:")
        print(f"  Valuable kept: {total_valuable_kept}")
        print(f"  Valuable filtered out: {total_valuable_filtered}")

        if total_valuable_filtered > 0:
            print(f"  ⚠️ WARNING: {total_valuable_filtered} valuable subgraphs were filtered out!")

        print(f"\nGolden Content Loss:")
        print(f"  Golden entities lost: {total_golden_entities_lost}")
        print(f"  Golden relations lost: {total_golden_relations_lost}")

        # Analyze probability distribution of filtered valuable subgraphs
        if failures:
            print(f"\n📊 Probability Analysis of Filtered Valuable Subgraphs:")

            filtered_valuable_probs = []
            for result in failures:
                for sg in result['filtered_out_details']:
                    if sg['is_valuable']:
                        filtered_valuable_probs.append(sg['probability'])

            if filtered_valuable_probs:
                print(f"  Count: {len(filtered_valuable_probs)}")
                print(f"  Mean probability: {np.mean(filtered_valuable_probs):.3f}")
                print(f"  Median probability: {np.median(filtered_valuable_probs):.3f}")
                print(f"  Min probability: {np.min(filtered_valuable_probs):.3f}")
                print(f"  Max probability: {np.max(filtered_valuable_probs):.3f}")

        print("\n" + "=" * 80)
        print("VERDICT")
        print("=" * 80)

        if questions_losing_golden > 0:
            print(f"""
⚠️ CRITICAL FINDING: Top-K filtering REMOVES VALUABLE SUBGRAPHS!

- {questions_losing_golden}/{total_questions} questions ({questions_losing_golden/total_questions*100:.1f}%) lose golden content
- {total_valuable_filtered} valuable subgraphs are filtered out
- {total_golden_entities_lost} golden entities lost
- {total_golden_relations_lost} golden relations lost

PROBLEM: Probability is NOT a reliable indicator of subgraph value!
  - Low-probability subgraphs can contain critical golden entities/relations
  - Pure probability-based filtering removes needed information

RECOMMENDATIONS:
1. Use golden-content-aware filtering:
   - ALWAYS keep subgraphs with golden entities/relations
   - Apply top-K only to remaining subgraphs
2. Increase K to 5-7 to reduce loss
3. Add probability threshold instead of hard top-K cutoff
""")
        else:
            print(f"""
✓ GOOD NEWS: Top-K filtering preserves golden content!

- All {total_questions} questions retain their golden entities/relations
- Filtering is safe with K={args.top_k}

Current approach is acceptable.
""")


if __name__ == "__main__":
    import sys

    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
