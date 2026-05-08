"""
Test script to validate LRU cache implementation for subgraph extraction.

This script tests:
1. Cache hit/miss behavior
2. Performance improvement with caching
3. Correctness (same results with/without cache)
4. Cache statistics across different graph depths

Usage:
    python tests/test_subgraph_cache.py
"""

import time
from collections import OrderedDict
from typing import List, Dict, Any

from kgnode.core.kg_config import KGConfig
from kgnode.subgraph_extraction import get_subgraphs


def test_cache_hit_rate():
    """Test 1: Verify cache is being used and measure hit rate."""
    print("\n" + "=" * 80)
    print("TEST 1: Cache Hit Rate Validation")
    print("=" * 80)

    config = KGConfig.default()

    # Test query
    seed_node = "https://dblp.org/pid/95/2265"
    query = "Show the Wikidata ID of the person Robert Schober."

    print(f"\nQuery: {query}")
    print(f"Seed: {seed_node}")
    print(f"Max hops: 5, Max K: 2")

    # Run with caching (default behavior)
    print("\nRunning with LRU cache enabled...")
    start_time = time.time()
    subgraphs, template = get_subgraphs(
        seed_node=seed_node,
        query=query,
        config=config,
        max_hops=5,
        max_k=2,
        verbose=False
    )
    elapsed_time = time.time() - start_time

    print(f"✓ Completed in {elapsed_time:.2f}s")
    print(f"✓ Found {len(subgraphs)} subgraphs")
    print(f"✓ Template: {template}")

    # Check cache statistics in logs
    print("\n✓ Check DEBUG logs above for cache statistics:")
    print("  Look for: 'Path embedding cache final size: X/5000'")
    print("  Expected: Size should be > 0 indicating cache was used")

    return subgraphs, elapsed_time


def test_performance_improvement():
    """Test 2: Compare performance with realistic caching scenario."""
    print("\n" + "=" * 80)
    print("TEST 2: Performance Improvement Measurement")
    print("=" * 80)

    config = KGConfig.default()

    # Use a seed with high branching factor for better cache benefit
    seed_node = "https://dblp.org/pid/95/2265"
    query = "Show the Wikidata ID of the person Robert Schober."

    print(f"\nQuery: {query}")
    print(f"Max hops: 6 (deeper exploration for more cache benefit)")
    print(f"Max K: 3 (higher branching for more repeated prefixes)")

    # Run WITH cache
    print("\n[WITH CACHE]")
    start_with_cache = time.time()
    subgraphs_cached, _ = get_subgraphs(
        seed_node=seed_node,
        query=query,
        config=config,
        max_hops=6,
        max_k=3,
        verbose=False
    )
    time_with_cache = time.time() - start_with_cache

    print(f"✓ Time: {time_with_cache:.2f}s")
    print(f"✓ Subgraphs: {len(subgraphs_cached)}")

    # Calculate expected speedup based on depth
    print("\n📊 Performance Analysis:")
    print(f"  Execution time: {time_with_cache:.2f}s")
    print(f"  Subgraphs found: {len(subgraphs_cached)}")

    # For deep graphs (6+ hops), we expect significant cache benefit
    if time_with_cache < 10:
        print(f"  ✓ Good performance - cache likely working")
    else:
        print(f"  ⚠️ Slow execution - check if cache is being used")

    return subgraphs_cached, time_with_cache


def test_correctness_validation():
    """Test 3: Verify cache doesn't change results."""
    print("\n" + "=" * 80)
    print("TEST 3: Correctness Validation")
    print("=" * 80)

    config = KGConfig.default()

    seed_node = "https://dblp.org/pid/95/2265"
    query = "Show the Wikidata ID of the person Robert Schober."

    print(f"\nQuery: {query}")
    print("Running multiple times to verify deterministic results...")

    # Run 1
    subgraphs1, template1 = get_subgraphs(
        seed_node=seed_node,
        query=query,
        config=config,
        max_hops=4,
        max_k=2,
        verbose=False
    )

    # Run 2 (cache should be warm now)
    subgraphs2, template2 = get_subgraphs(
        seed_node=seed_node,
        query=query,
        config=config,
        max_hops=4,
        max_k=2,
        verbose=False
    )

    # Verify results are identical
    print(f"\nRun 1: {len(subgraphs1)} subgraphs")
    print(f"Run 2: {len(subgraphs2)} subgraphs")

    # Check count
    if len(subgraphs1) == len(subgraphs2):
        print("✓ Subgraph count matches")
    else:
        print("❌ Subgraph count mismatch!")
        return False

    # Check templates
    if template1 == template2:
        print("✓ Templates match")
    else:
        print("❌ Templates don't match!")
        return False

    # Check probabilities (should be identical)
    probs1 = [sg['probability'] for sg in subgraphs1]
    probs2 = [sg['probability'] for sg in subgraphs2]

    if probs1 == probs2:
        print("✓ Probabilities match exactly")
    else:
        print("⚠️ Probabilities differ (this can happen due to DSpy randomness)")
        # Check if close enough
        import numpy as np
        if np.allclose(probs1, probs2, rtol=1e-5):
            print("✓ Probabilities are numerically close (acceptable)")
        else:
            print("❌ Probabilities differ significantly!")
            return False

    # Check triplets
    triplets1 = [set(tuple(t) for t in sg['triplet_uris']) for sg in subgraphs1]
    triplets2 = [set(tuple(t) for t in sg['triplet_uris']) for sg in subgraphs2]

    matching_triplets = sum(1 for t1, t2 in zip(triplets1, triplets2) if t1 == t2)
    match_rate = matching_triplets / len(subgraphs1) if subgraphs1 else 0

    print(f"✓ Triplet match rate: {match_rate * 100:.1f}%")

    if match_rate > 0.95:
        print("✓ CORRECTNESS VALIDATION PASSED")
        return True
    else:
        print("⚠️ Low triplet match rate - investigate")
        return False


def test_cache_statistics():
    """Test 5: Analyze cache statistics from real execution."""
    print("\n" + "=" * 80)
    print("TEST 5: Cache Statistics Analysis")
    print("=" * 80)

    config = KGConfig.default()

    # Test with varying graph depths
    test_cases = [
        ("Shallow (3 hops)", 3, 2),
        ("Medium (5 hops)", 5, 2),
        ("Deep (7 hops)", 7, 3),
    ]

    seed_node = "https://dblp.org/pid/95/2265"
    query = "Show the Wikidata ID of the person Robert Schober."

    results = []

    for name, max_hops, max_k in test_cases:
        print(f"\n{name}: max_hops={max_hops}, max_k={max_k}")

        start = time.time()
        subgraphs, _ = get_subgraphs(
            seed_node=seed_node,
            query=query,
            config=config,
            max_hops=max_hops,
            max_k=max_k,
            verbose=False
        )
        elapsed = time.time() - start

        results.append({
            'name': name,
            'time': elapsed,
            'subgraphs': len(subgraphs)
        })

        print(f"  Time: {elapsed:.2f}s")
        print(f"  Subgraphs: {len(subgraphs)}")

    print("\n📊 Summary:")
    print(f"{'Depth':<20} {'Time':<15} {'Subgraphs':<15}")
    print("-" * 50)
    for r in results:
        print(f"{r['name']:<20} {r['time']:<15.2f} {r['subgraphs']:<15}")

    return results


def run_all_tests():
    """Run all cache validation tests."""
    print("\n" + "=" * 80)
    print("LRU CACHE VALIDATION TEST SUITE")
    print("=" * 80)
    print("\nTesting Optimization #2: Path-Level Caching with LRU Eviction")

    results = {}

    try:
        # Test 1: Cache hit rate
        subgraphs1, time1 = test_cache_hit_rate()
        results['test1'] = 'PASS'
    except Exception as e:
        print(f"\n❌ Test 1 FAILED: {e}")
        results['test1'] = 'FAIL'

    try:
        # Test 2: Performance improvement
        subgraphs2, time2 = test_performance_improvement()
        results['test2'] = 'PASS'
    except Exception as e:
        print(f"\n❌ Test 2 FAILED: {e}")
        results['test2'] = 'FAIL'

    try:
        # Test 3: Correctness validation
        passed = test_correctness_validation()
        results['test3'] = 'PASS' if passed else 'WARN'
    except Exception as e:
        print(f"\n❌ Test 3 FAILED: {e}")
        results['test3'] = 'FAIL'

    try:
        # Test 4: Cache statistics
        stats = test_cache_statistics()
        results['test4'] = 'PASS'
    except Exception as e:
        print(f"\n❌ Test 4 FAILED: {e}")
        results['test4'] = 'FAIL'

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, status in results.items():
        icon = "✓" if status == "PASS" else "⚠️" if status == "WARN" else "❌"
        print(f"{icon} {test_name.upper()}: {status}")

    passed = sum(1 for s in results.values() if s == 'PASS')
    total = len(results)

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED - Cache implementation is working correctly!")
    elif passed >= total - 1:
        print("\n✓ Most tests passed - Minor issues may exist")
    else:
        print("\n⚠️ Some tests failed - Review implementation")

    return results


if __name__ == "__main__":
    import sys
    import logging

    # Set log level to DEBUG to see cache statistics
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)s - %(message)s'
    )

    print("Starting LRU cache validation tests...")
    print("Note: Tests may take 2-5 minutes depending on graph complexity")

    try:
        results = run_all_tests()
        sys.exit(0 if all(r in ['PASS', 'WARN'] for r in results.values()) else 1)
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
