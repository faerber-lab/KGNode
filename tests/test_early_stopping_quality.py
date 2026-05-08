"""
Test script to evaluate QUALITY of subgraphs that would be lost with early stopping.

This script checks:
1. Do low-probability subgraphs contain golden entities/relations?
2. Are they actually useful for answering the query?
3. Would early stopping filter out noise or valuable information?

Usage:
    python tests/test_early_stopping_quality.py --sample-size 20
"""

import json
import time
import argparse
from typing import Dict, List, Any, Tuple, Set
from collections import defaultdict
import numpy as np

from kgnode.core.kg_config import KGConfig
from kgnode.subgraph_extraction import get_subgraphs


def load_dataset(file_path: str, sample_size: int = None) -> List[Dict]:
    """Load dataset from JSONL file."""
    questions = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if sample_size and i >= sample_size:
                break
            questions.append(json.loads(line))
    return questions


def extract_uris_from_subgraph(subgraph: Dict) -> Tuple[Set[str], Set[str]]:
    """Extract all entity and relation URIs from a subgraph."""
    entities = set()
    relations = set()

    for triplet in subgraph.get('triplet_uris', []):
        if len(triplet) >= 3:
            subject_uri, predicate_uri, object_uri = triplet[0], triplet[1], triplet[2]
            entities.add(subject_uri)
            entities.add(object_uri)
            relations.add(predicate_uri)

    return entities, relations


def contains_golden_content(
    subgraph: Dict,
    golden_entities: Set[str],
    golden_relations: Set[str]
) -> Tuple[bool, bool, int, int]:
    """Check if subgraph contains golden entities or relations."""
    entities, relations = extract_uris_from_subgraph(subgraph)

    entity_matches = entities & golden_entities
    relation_matches = relations & golden_relations

    has_golden_entity = len(entity_matches) > 0
    has_golden_relation = len(relation_matches) > 0

    return has_golden_entity, has_golden_relation, len(entity_matches), len(relation_matches)


def analyze_subgraph_quality(
    question: Dict,
    config: KGConfig,
    absolute_threshold: float = 1.5,
    relative_threshold_ratio: float = 0.65
) -> Dict[str, Any]:
    """Analyze quality of subgraphs and simulate early stopping impact."""

    seed_info = question.get('filtered_seeds_info', {})
    filtered_seeds = seed_info.get('filtered_seeds', [])

    if not filtered_seeds:
        return None

    # Extract golden entities and relations
    golden_entities = set()
    for entity_str in question.get('golden_entities', []):
        # Remove < and > brackets
        entity_uri = entity_str.strip('<>').strip()
        golden_entities.add(entity_uri)

    golden_relations = set()
    for relation_str in question.get('golden_relations', []):
        # Remove < and > brackets
        relation_uri = relation_str.strip('<>').strip()
        golden_relations.add(relation_uri)

    # Run subgraph extraction
    seed_node = filtered_seeds[0]['entity_uri']
    query_text = question['question']['1']

    try:
        subgraphs, template = get_subgraphs(
            seed_node=seed_node,
            query=query_text,
            config=config,
            max_hops=10,
            max_k=2,
            verbose=False
        )
    except Exception as e:
        return None

    if not subgraphs:
        return None

    # Analyze each subgraph
    subgraph_analysis = []
    for i, sg in enumerate(subgraphs):
        has_entity, has_relation, entity_count, relation_count = contains_golden_content(
            sg, golden_entities, golden_relations
        )

        analysis = {
            'index': i,
            'probability': sg['probability'],
            'path_depth': sg['path_depth'],
            'has_golden_entity': has_entity,
            'has_golden_relation': has_relation,
            'golden_entity_count': entity_count,
            'golden_relation_count': relation_count,
            'is_valuable': has_entity or has_relation,
        }
        subgraph_analysis.append(analysis)

    # Sort by probability (same as actual implementation)
    subgraph_analysis.sort(key=lambda x: x['probability'], reverse=True)

    # Calculate median probability
    probs = [sg['probability'] for sg in subgraph_analysis]
    median_prob = np.median(probs)

    # Simulate early stopping with absolute threshold
    # Early stopping would check: next_priority < absolute_threshold
    # We simulate by finding where we would have stopped

    would_keep_absolute = []
    would_lose_absolute = []

    for sg in subgraph_analysis:
        # In early stopping, we check queue before popping
        # If current sg has prob < threshold, it would never be explored
        if sg['probability'] >= absolute_threshold:
            would_keep_absolute.append(sg)
        else:
            would_lose_absolute.append(sg)

    # Simulate early stopping with relative threshold
    relative_threshold = median_prob * relative_threshold_ratio

    would_keep_relative = []
    would_lose_relative = []

    for sg in subgraph_analysis:
        if sg['probability'] >= relative_threshold:
            would_keep_relative.append(sg)
        else:
            would_lose_relative.append(sg)

    # Analyze quality of lost subgraphs
    result = {
        'question_id': question['id'],
        'query_type': question['query_type'],
        'total_subgraphs': len(subgraphs),
        'median_prob': median_prob,
        'golden_entity_count': len(golden_entities),
        'golden_relation_count': len(golden_relations),

        # Overall statistics
        'valuable_subgraphs': sum(1 for sg in subgraph_analysis if sg['is_valuable']),
        'valuable_ratio': sum(1 for sg in subgraph_analysis if sg['is_valuable']) / len(subgraph_analysis),

        # Absolute threshold analysis
        'absolute_threshold': absolute_threshold,
        'keep_count_absolute': len(would_keep_absolute),
        'lose_count_absolute': len(would_lose_absolute),
        'lose_valuable_absolute': sum(1 for sg in would_lose_absolute if sg['is_valuable']),
        'lose_valuable_ratio_absolute': (
            sum(1 for sg in would_lose_absolute if sg['is_valuable']) / len(would_lose_absolute)
            if would_lose_absolute else 0
        ),

        # Relative threshold analysis
        'relative_threshold': relative_threshold,
        'keep_count_relative': len(would_keep_relative),
        'lose_count_relative': len(would_lose_relative),
        'lose_valuable_relative': sum(1 for sg in would_lose_relative if sg['is_valuable']),
        'lose_valuable_ratio_relative': (
            sum(1 for sg in would_lose_relative if sg['is_valuable']) / len(would_lose_relative)
            if would_lose_relative else 0
        ),

        # Detailed breakdowns
        'would_lose_absolute': would_lose_absolute,
        'would_lose_relative': would_lose_relative,
    }

    return result


def print_detailed_analysis(result: Dict):
    """Print detailed analysis for a single question."""
    print(f"\n{'='*80}")
    print(f"Question: {result['question_id']} ({result['query_type']})")
    print(f"{'='*80}")

    print(f"\n📊 Overall Statistics:")
    print(f"  Total subgraphs: {result['total_subgraphs']}")
    print(f"  Median probability: {result['median_prob']:.3f}")
    print(f"  Valuable subgraphs: {result['valuable_subgraphs']}/{result['total_subgraphs']} ({result['valuable_ratio']*100:.1f}%)")
    print(f"  Golden entities in query: {result['golden_entity_count']}")
    print(f"  Golden relations in query: {result['golden_relation_count']}")

    # Absolute threshold analysis
    print(f"\n🔴 Absolute Threshold Analysis (threshold = {result['absolute_threshold']}):")
    if result['lose_count_absolute'] > 0:
        print(f"  Would KEEP: {result['keep_count_absolute']} subgraphs")
        print(f"  Would LOSE: {result['lose_count_absolute']} subgraphs")
        print(f"  Lost valuable subgraphs: {result['lose_valuable_absolute']}/{result['lose_count_absolute']} ({result['lose_valuable_ratio_absolute']*100:.1f}%)")

        if result['lose_valuable_absolute'] > 0:
            print(f"  ⚠️ WARNING: Would lose {result['lose_valuable_absolute']} subgraphs with golden content!")
            print(f"  Details of lost valuable subgraphs:")
            for sg in result['would_lose_absolute']:
                if sg['is_valuable']:
                    print(f"    - Prob={sg['probability']:.3f}, Depth={sg['path_depth']}, "
                          f"Entities={sg['golden_entity_count']}, Relations={sg['golden_relation_count']}")
        else:
            print(f"  ✓ SAFE: All lost subgraphs are noise (no golden content)")
    else:
        print(f"  ✓ No subgraphs would be lost")

    # Relative threshold analysis
    print(f"\n🟡 Relative Threshold Analysis (threshold = {result['relative_threshold']:.3f}):")
    if result['lose_count_relative'] > 0:
        print(f"  Would KEEP: {result['keep_count_relative']} subgraphs")
        print(f"  Would LOSE: {result['lose_count_relative']} subgraphs")
        print(f"  Lost valuable subgraphs: {result['lose_valuable_relative']}/{result['lose_count_relative']} ({result['lose_valuable_ratio_relative']*100:.1f}%)")

        if result['lose_valuable_relative'] > 0:
            print(f"  ⚠️ WARNING: Would lose {result['lose_valuable_relative']} subgraphs with golden content!")
            print(f"  Details of lost valuable subgraphs:")
            for sg in result['would_lose_relative']:
                if sg['is_valuable']:
                    print(f"    - Prob={sg['probability']:.3f}, Depth={sg['path_depth']}, "
                          f"Entities={sg['golden_entity_count']}, Relations={sg['golden_relation_count']}")
        else:
            print(f"  ✓ SAFE: All lost subgraphs are noise (no golden content)")
    else:
        print(f"  ✓ No subgraphs would be lost")


def main():
    parser = argparse.ArgumentParser(description='Test Early Stopping Quality Impact')
    parser.add_argument(
        '--sample-size',
        type=int,
        default=20,
        help='Number of questions to test (default: 20)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='eval/data/balanced_200.jsonl',
        help='Path to dataset file'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("OPTIMIZATION #1 QUALITY ANALYSIS: Early Stopping Impact on Relevance")
    print("=" * 80)
    print(f"\nDataset: {args.dataset}")
    print(f"Sample size: {args.sample_size}")

    # Load dataset
    print("\nLoading dataset...")
    questions = load_dataset(args.dataset, args.sample_size)
    print(f"✓ Loaded {len(questions)} questions")

    # Initialize config
    config = KGConfig.default()

    # Analyze each question
    results = []

    for i, question in enumerate(questions, 1):
        print(f"\n{'='*80}")
        print(f"Processing {i}/{len(questions)}: {question['id']}")
        print(f"{'='*80}")

        result = analyze_subgraph_quality(question, config)

        if result:
            results.append(result)
            print_detailed_analysis(result)
        else:
            print("  ✗ No subgraphs found or error occurred")

    # Aggregate analysis
    print("\n" + "=" * 80)
    print("AGGREGATE ANALYSIS")
    print("=" * 80)

    if results:
        # Overall statistics
        total_questions = len(results)

        # Absolute threshold impact
        questions_losing_subgraphs_abs = sum(1 for r in results if r['lose_count_absolute'] > 0)
        questions_losing_valuable_abs = sum(1 for r in results if r['lose_valuable_absolute'] > 0)
        total_lost_abs = sum(r['lose_count_absolute'] for r in results)
        total_lost_valuable_abs = sum(r['lose_valuable_absolute'] for r in results)

        print(f"\n🔴 ABSOLUTE THRESHOLD (1.5) IMPACT:")
        print(f"  Questions losing subgraphs: {questions_losing_subgraphs_abs}/{total_questions} ({questions_losing_subgraphs_abs/total_questions*100:.1f}%)")
        print(f"  Questions losing VALUABLE subgraphs: {questions_losing_valuable_abs}/{total_questions} ({questions_losing_valuable_abs/total_questions*100:.1f}%)")
        print(f"  Total subgraphs lost: {total_lost_abs}")
        print(f"  Total VALUABLE subgraphs lost: {total_lost_valuable_abs}")

        if total_lost_abs > 0:
            print(f"  Valuable ratio of lost: {total_lost_valuable_abs/total_lost_abs*100:.1f}%")

        # Relative threshold impact
        questions_losing_subgraphs_rel = sum(1 for r in results if r['lose_count_relative'] > 0)
        questions_losing_valuable_rel = sum(1 for r in results if r['lose_valuable_relative'] > 0)
        total_lost_rel = sum(r['lose_count_relative'] for r in results)
        total_lost_valuable_rel = sum(r['lose_valuable_relative'] for r in results)

        print(f"\n🟡 RELATIVE THRESHOLD (0.65 * median) IMPACT:")
        print(f"  Questions losing subgraphs: {questions_losing_subgraphs_rel}/{total_questions} ({questions_losing_subgraphs_rel/total_questions*100:.1f}%)")
        print(f"  Questions losing VALUABLE subgraphs: {questions_losing_valuable_rel}/{total_questions} ({questions_losing_valuable_rel/total_questions*100:.1f}%)")
        print(f"  Total subgraphs lost: {total_lost_rel}")
        print(f"  Total VALUABLE subgraphs lost: {total_lost_valuable_rel}")

        if total_lost_rel > 0:
            print(f"  Valuable ratio of lost: {total_lost_valuable_rel/total_lost_rel*100:.1f}%")

        # Verdict
        print("\n" + "=" * 80)
        print("VERDICT")
        print("=" * 80)

        if questions_losing_valuable_abs > 0 or questions_losing_valuable_rel > 0:
            print(f"""
⚠️ CRITICAL FINDING: Early stopping WOULD LOSE VALUABLE SUBGRAPHS!

- {questions_losing_valuable_abs} questions would lose subgraphs with golden content (absolute)
- {questions_losing_valuable_rel} questions would lose subgraphs with golden content (relative)
- Total valuable subgraphs at risk: {total_lost_valuable_abs} (absolute), {total_lost_valuable_rel} (relative)

RECOMMENDATION: DO NOT IMPLEMENT Optimization #1
  ✗ Would remove subgraphs containing golden entities/relations
  ✗ Could reduce SPARQL generation accuracy
  ✗ Risk outweighs potential speedup benefit
  ✓ Keep current reactive stopping approach
  ✓ Use Optimization #2 (LRU cache) for safe speedup
""")
        else:
            print(f"""
✓ INTERESTING FINDING: Lost subgraphs appear to be noise!

- 0 questions would lose valuable subgraphs
- All lost subgraphs lack golden entities/relations
- Early stopping might actually IMPROVE quality by filtering noise

RECOMMENDATION: Consider implementing with conservative thresholds
  ✓ Could improve precision by removing noise
  ✓ Would speed up execution
  ⚠️ Test on larger sample first (50-100 questions)
  ⚠️ Monitor impact on end-to-end accuracy
""")


if __name__ == "__main__":
    import sys
    import logging

    logging.basicConfig(
        level=logging.WARNING,
        format='%(levelname)s - %(message)s'
    )

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
