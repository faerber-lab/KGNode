"""Add generated SPARQL to filtered subgraph JSONL.

This script:
1. Reads questions with pre-computed filtered_seeds and merged_graphs
2. Generates SPARQL for each question using generate_sparql()
3. Adds generated_sparql_info to each question record
4. Saves updated records to output JSONL
5. Reports statistics and failure breakdown
"""

import json
import os
import signal
import sys
import time
from typing import Any, Dict, Optional


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logging_util import setup_dual_output

from dotenv import load_dotenv

from kgnode.core.kg_config import KGConfig
from kgnode.core.sparql_query import execute_sparql_query
from kgnode.exceptions import SPARQLGenerationError
from kgnode.sparql_answer_generator import generate_sparql


# ============================================================================
# FAILURE CODE REGISTRY
# ============================================================================

FAILURE_CODES = {
    "NO_SUBGRAPH_EXTRACTION": "subgraph_extraction field missing from question",
    "NO_FILTERED_SEEDS":      "filtered_seeds is empty or missing",
    "NO_MERGED_GRAPHS":       "merged_graphs is empty or missing in filtered_subgraphs_info",
    "LM_API_KEY_MISSING":     "LM API key not configured (check OPENAI_API_KEY env var)",
    "SPARQL_GENERATION_FAILED": "SPARQLGenerationError: all retries exhausted",
    "TIMEOUT":                "Processing exceeded 15 second timeout",
    "UNEXPECTED_ERROR":       "Unexpected exception during generation",
}


# ============================================================================
# HELPERS
# ============================================================================

class TimeoutException(Exception):
    """Raised when processing times out."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutException("Processing timed out after 15 seconds")


def get_question_text(question: Dict[str, Any]) -> str:
    """Return first available question phrasing."""
    q = question.get("question", {})
    if isinstance(q, dict):
        return q.get("1") or q.get("2") or ""
    return str(q)


def normalize_sparql(sparql: str) -> str:
    """Minimal normalization for loose string comparison."""
    return " ".join(sparql.lower().split())


# ============================================================================
# CORE PROCESSING
# ============================================================================

def process_question(
    question: Dict[str, Any],
    config: KGConfig,
    kg_connection,
    timeout_seconds: int = 15,
) -> Dict[str, Any]:
    """Generate SPARQL for one question and return a result dict.

    Args:
        question: Question dict with subgraph_extraction data.
        config: KG configuration.
        kg_connection: SPARQL execution function.
        timeout_seconds: Max seconds to process (default 15).

    Returns:
        Dict with keys: success, generated_sparql, failure_code,
        failure_reason, elapsed_seconds, query_used.
    """
    result: Dict[str, Any] = {
        "success": False,
        "generated_sparql": None,
        "sparql_results": None,
        "failure_code": None,
        "failure_reason": None,
        "elapsed_seconds": 0.0,
        "query_used": None,
    }

    t0 = time.time()

    # --- Check required fields ---
    if "subgraph_extraction" not in question:
        result["failure_code"] = "NO_SUBGRAPH_EXTRACTION"
        result["failure_reason"] = FAILURE_CODES["NO_SUBGRAPH_EXTRACTION"]
        result["elapsed_seconds"] = round(time.time() - t0, 2)
        return result

    raw_seeds = question["subgraph_extraction"].get("filtered_seeds", [])
    if not raw_seeds:
        result["failure_code"] = "NO_FILTERED_SEEDS"
        result["failure_reason"] = FAILURE_CODES["NO_FILTERED_SEEDS"]
        result["elapsed_seconds"] = round(time.time() - t0, 2)
        return result

    merged_graphs = (
        question.get("filtered_subgraphs_info", {}).get("merged_graphs", [])
    )
    if not merged_graphs:
        result["failure_code"] = "NO_MERGED_GRAPHS"
        result["failure_reason"] = FAILURE_CODES["NO_MERGED_GRAPHS"]
        result["elapsed_seconds"] = round(time.time() - t0, 2)
        return result

    # --- Prepare inputs ---
    query = get_question_text(question)
    result["query_used"] = query

    filtered_seeds = [
        {"seed_uri": s["seed_uri"], "seed_label": s["seed_label"]}
        for s in raw_seeds
    ]

    # Get relation nodes if available (from extracted_relation_nodes)
    relation_nodes = question.get("extracted_relation_nodes", [])

    # --- Generate with timeout ---
    try:
        # Set alarm signal for timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)

        try:
            sparql, sparql_results = generate_sparql(
                query=query,
                filtered_seeds=filtered_seeds,
                merged_graphs=merged_graphs,
                config=config,
                kg_connection=kg_connection,
                max_retries=3,
                relation_nodes=relation_nodes if relation_nodes else None
            )
            result["success"] = True
            result["generated_sparql"] = sparql
            result["sparql_results"] = sparql_results
        finally:
            # Cancel alarm
            signal.alarm(0)

    except TimeoutException as e:
        result["failure_code"] = "TIMEOUT"
        result["failure_reason"] = FAILURE_CODES["TIMEOUT"]

    except SPARQLGenerationError as e:
        result["failure_code"] = "SPARQL_GENERATION_FAILED"
        result["failure_reason"] = str(e)

    except ValueError as e:
        result["failure_code"] = "LM_API_KEY_MISSING"
        result["failure_reason"] = str(e)

    except Exception as e:
        result["failure_code"] = "UNEXPECTED_ERROR"
        result["failure_reason"] = f"{type(e).__name__}: {str(e)}"

    result["elapsed_seconds"] = round(time.time() - t0, 2)
    return result


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def add_generated_sparql(
    file_path: str,
    output_path: Optional[str] = None,
    max_questions: Optional[int] = None,
    question_type_based_generation: bool = False,
    query_types: Optional[list] = None,
    force_regenerate: bool = False,
) -> Dict[str, Any]:
    """Process all questions, adding generated_sparql_info to each record.

    Args:
        file_path: Input JSONL path (must contain filtered_subgraphs_info).
        output_path: Output JSONL path. Defaults to in-place update.
        max_questions: Optional cap for quick testing.
        question_type_based_generation: If True, process by query type. If False, process all.
        query_types: Optional list of query types to process. If None, process all types.
        force_regenerate: If True, regenerate SPARQL even if already processed.

    Returns:
        Stats dictionary.
    """
    load_dotenv()

    if output_path is None:
        output_path = file_path  # Same file - in-place update

    config = KGConfig.default()
    def kg_connection(q):
        return execute_sparql_query(q, config=config)

    # --- Load ---
    print(f"Loading questions from {file_path}...")
    questions = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    if max_questions:
        questions = questions[:max_questions]

    total = len(questions)
    print(f"Loaded {total} questions\n")

    # --- Process by query type or all at once ---
    if question_type_based_generation:
        return _process_by_query_type(
            questions, config, kg_connection, output_path, query_types, force_regenerate
        )
    else:
        return _process_all_questions(
            questions, config, kg_connection, output_path, force_regenerate
        )


def _process_all_questions(
    questions: list,
    config: KGConfig,
    kg_connection,
    output_path: str,
    force_regenerate: bool = False,
) -> Dict[str, Any]:
    """Process all questions in one batch."""
    total = len(questions)

    # --- Stats ---
    stats: Dict[str, Any] = {
        "total": total,
        "success": 0,
        "failed": 0,
        "failure_codes": {},
        "total_elapsed": 0.0,
        "exact_match": 0,
        "failures": [],
    }

    # --- Process ---
    for i, question in enumerate(questions, 1):
        qid = question.get("id", f"Q{i:04d}")

        # Skip if already processed (unless force_regenerate is True)
        if "generated_sparql_info" in question and not force_regenerate:
            print(f"[{i}/{total}] {qid}: SKIPPED (already processed)")
            # Update stats for already processed questions
            if question["generated_sparql_info"].get("success"):
                stats["success"] += 1
            else:
                stats["failed"] += 1
                code = question["generated_sparql_info"].get("failure_code")
                if code:
                    stats["failure_codes"][code] = stats["failure_codes"].get(code, 0) + 1
            continue

        result = process_question(question, config, kg_connection)

        question["generated_sparql_info"] = {
            "success":           result["success"],
            "generated_sparql":  result["generated_sparql"],
            "sparql_results":    result.get("sparql_results"),
            "failure_code":      result["failure_code"],
            "failure_reason":    result["failure_reason"],
            "elapsed_seconds":   result["elapsed_seconds"],
            "query_used":        result["query_used"],
        }

        stats["total_elapsed"] += result["elapsed_seconds"]

        if result["success"]:
            stats["success"] += 1

            # Loose match against golden
            golden = question.get("golden_sparql", "")
            if golden and normalize_sparql(result["generated_sparql"]) == normalize_sparql(golden):
                stats["exact_match"] += 1

            print(f"[{i}/{total}] {qid}: ✓  ({result['elapsed_seconds']:.1f}s)")

        else:
            stats["failed"] += 1
            code = result["failure_code"]
            stats["failure_codes"][code] = stats["failure_codes"].get(code, 0) + 1

            stats["failures"].append({
                "id":             qid,
                "query":          (result["query_used"] or "")[:70],
                "failure_code":   code,
                "failure_reason": result["failure_reason"],
            })

            print(f"[{i}/{total}] {qid}: ✗ {code}  ({result['elapsed_seconds']:.1f}s)")

        # --- Save incrementally after each question ---
        with open(output_path, "w", encoding="utf-8") as f:
            for q in questions:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"\nAll results saved to {output_path}\n")

    # --- Summary ---
    _print_summary(stats)

    return stats


def _process_by_query_type(
    questions: list,
    config: KGConfig,
    kg_connection,
    output_path: str,
    query_types: Optional[list] = None,
    force_regenerate: bool = False,
) -> Dict[str, Any]:
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

    all_stats = {}
    total_processed = 0

    print(f"Processing {len(type_groups)} query types\n")

    for qtype_idx, (qtype, qtype_questions) in enumerate(type_groups.items(), 1):
        print("=" * 70)
        print(f"[{qtype_idx}/{len(type_groups)}] Query Type: {qtype}")
        print(f"Questions: {len(qtype_questions)}")
        print("=" * 70 + "\n")

        # Process this type's questions
        type_stats = {
            "total": len(qtype_questions),
            "success": 0,
            "failed": 0,
            "failure_codes": {},
            "total_elapsed": 0.0,
            "exact_match": 0,
            "failures": [],
        }

        for i, question in enumerate(qtype_questions, 1):
            qid = question.get("id", f"Q{i:04d}")
            total_processed += 1

            # Skip if already processed (unless force_regenerate is True)
            if "generated_sparql_info" in question and not force_regenerate:
                print(f"  [{i}/{len(qtype_questions)}] {qid}: SKIPPED (already processed)")
                if question["generated_sparql_info"].get("success"):
                    type_stats["success"] += 1
                else:
                    type_stats["failed"] += 1
                    code = question["generated_sparql_info"].get("failure_code")
                    if code:
                        type_stats["failure_codes"][code] = type_stats["failure_codes"].get(code, 0) + 1
                continue

            result = process_question(question, config, kg_connection)

            question["generated_sparql_info"] = {
                "success":           result["success"],
                "generated_sparql":  result["generated_sparql"],
                "sparql_results":    result.get("sparql_results"),
                "failure_code":      result["failure_code"],
                "failure_reason":    result["failure_reason"],
                "elapsed_seconds":   result["elapsed_seconds"],
                "query_used":        result["query_used"],
            }

            type_stats["total_elapsed"] += result["elapsed_seconds"]

            if result["success"]:
                type_stats["success"] += 1

                golden = question.get("golden_sparql", "")
                if golden and normalize_sparql(result["generated_sparql"]) == normalize_sparql(golden):
                    type_stats["exact_match"] += 1

                print(f"  [{i}/{len(qtype_questions)}] {qid}: ✓  ({result['elapsed_seconds']:.1f}s)")

            else:
                type_stats["failed"] += 1
                code = result["failure_code"]
                type_stats["failure_codes"][code] = type_stats["failure_codes"].get(code, 0) + 1

                type_stats["failures"].append({
                    "id":             qid,
                    "query":          (result["query_used"] or "")[:70],
                    "failure_code":   code,
                    "failure_reason": result["failure_reason"],
                })

                print(f"  [{i}/{len(qtype_questions)}] {qid}: ✗ {code}  ({result['elapsed_seconds']:.1f}s)")

            # --- Save incrementally after each question ---
            with open(output_path, "w", encoding="utf-8") as f:
                for q in questions:
                    f.write(json.dumps(q, ensure_ascii=False) + "\n")

        all_stats[qtype] = type_stats

        # Print summary for this type
        print(f"\n--- {qtype} Summary ---")
        print(f"  Success: {type_stats['success']}/{type_stats['total']} ({type_stats['success']/type_stats['total']*100:.1f}%)")
        print(f"  Failed:  {type_stats['failed']}/{type_stats['total']} ({type_stats['failed']/type_stats['total']*100:.1f}%)")
        if type_stats['success'] > 0:
            print(f"  Exact match: {type_stats['exact_match']}/{type_stats['success']} ({type_stats['exact_match']/type_stats['success']*100:.1f}%)")
        print(f"  Time: {type_stats['total_elapsed']:.1f}s\n")

    # Overall summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)

    total_success = sum(s["success"] for s in all_stats.values())
    total_failed = sum(s["failed"] for s in all_stats.values())
    total_questions = sum(s["total"] for s in all_stats.values())
    total_time = sum(s["total_elapsed"] for s in all_stats.values())

    print(f"\nTotal questions processed: {total_questions}")
    print(f"Overall success: {total_success}/{total_questions} ({total_success/total_questions*100:.1f}%)")
    print(f"Overall failed:  {total_failed}/{total_questions} ({total_failed/total_questions*100:.1f}%)")
    print(f"Total time: {total_time:.1f}s")

    print("\nPer-type breakdown:")
    for qtype, stats in all_stats.items():
        print(f"  {qtype:<25} {stats['success']}/{stats['total']} ({stats['success']/stats['total']*100:.0f}%)")

    print("=" * 70)

    return all_stats


def _print_summary(stats: Dict[str, Any]) -> None:
    total = stats["total"]
    success = stats["success"]
    failed = stats["failed"]
    elapsed = stats["total_elapsed"]

    print("=" * 70)
    print("SPARQL GENERATION RESULTS")
    print("=" * 70)

    print("\nOverall:")
    print(f"  Total questions : {total}")
    print(f"  Generated OK    : {success}  ({success/total*100:.1f}%)")
    print(f"  Failed          : {failed}   ({failed/total*100:.1f}%)")
    if success > 0:
        print(
            f"  Exact match (golden): {stats['exact_match']}  "
            f"({stats['exact_match']/success*100:.1f}% of successes)"
        )
    print(f"  Avg time/question: {elapsed/total:.1f}s")
    print(f"  Total time       : {elapsed:.0f}s")

    if stats["failure_codes"]:
        print("\nFailure Breakdown:")
        for code, count in sorted(
            stats["failure_codes"].items(), key=lambda x: -x[1]
        ):
            pct = count / total * 100
            desc = FAILURE_CODES.get(code, "")
            print(f"  {code:<30} {count:>3}  ({pct:.1f}%)")
            print(f"    -> {desc}")

    failures = stats.get("failures", [])
    if failures:
        print(f"\nFailed Cases ({len(failures)}):")
        for fail in failures[:20]:
            print(f"  [{fail['id']}] {fail['failure_code']}")
            print(f"    Q: {fail['query']}...")
            reason = fail.get("failure_reason") or ""
            if reason:
                # Truncate long error messages
                short = reason if len(reason) <= 120 else reason[:117] + "..."
                print(f"    E: {short}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")

    print("=" * 70)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    setup_dual_output(__file__)

    # ============================================================================
    # CONFIGURATION - Edit these variables as needed
    # ============================================================================

    input_path = "./data/balanced_400.jsonl"
    output_path = None  # None = same file (in-place update)

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
    print("SPARQL GENERATION FOR EVAL DATASET")
    print("=" * 70)
    print(f"File: {input_path} (in-place update)")
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

    add_generated_sparql(
        input_path,
        output_path,
        max_questions=max_questions,
        question_type_based_generation=question_type_based_generation,
        query_types=query_types,
        force_regenerate=force_regenerate,
    )
