"""
Four-State System
EVAL_WRONG_0 = "wrong_0"       # Definitely incorrect, execution error
EVAL_WRONG_1 = "wrong_1"       # Definitely incorrect, Need manual verification
EVAL_CORRECT_0 = "correct_0"   # Auto-correct (no check needed)
EVAL_CORRECT_1 = "correct_1"   # Needs manual verification

9-Case Logic Implementation

| #   | Golden            | Generated         | State     | Reason                          |
|-----|-------------------|-------------------|-----------|---------------------------------|
| 1   | Error/Timeout     | Error/Timeout     | wrong_0   | Both failed                     |
| 2   | Error/Timeout     | Success + Results | correct_1 | Golden failed, verify generated |
| 3   | Error/Timeout     | Success + Empty   | correct_1 | Golden failed, verify generated |
| 4   | Success + Results | Error/Timeout     | wrong_0   | Generated failed                |
| 5   | Success + Results | Success + Results | correct_1 | Compare via F1                  |
| 6   | Success + Results | Success + Empty   | wrong_1   | Under-retrieval                 |
| 7   | Success + Empty   | Error/Timeout     | wrong_0   | Generated failed                |
| 8   | Success + Empty   | Success + Results | correct_0 | Golden may be wrong ✓           |
| 9   | Success + Empty   | Success + Empty   | correct_0 | Both empty ✓                    |

Evaluate SPARQL queries by query type, executing both golden and generated queries.

Key differences from standard evaluation:
1. Executes BOTH golden_sparql and generated_sparql against the live KG
2. Compares actual execution results instead of trusting stored answers
3. Handles empty result sets correctly (both empty = correct)
4. Multi-variable SELECT support (e.g., ?firstanswer ?secondanswer)
5. Filters by query_type (DOUBLE_INTENT, BOOLEAN, COUNT, etc.)

This addresses cases where generated SPARQL is semantically equivalent but
stored golden answers don't reflect current KG state.
"""

import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from multiprocessing import Process, Queue
from typing import Any, Dict, List, Optional, Set, Tuple

import dspy
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logging_util import setup_dual_output

from kgnode.core.kg_config import KGConfig
from kgnode.core.sparql_query import execute_sparql_query


# ============================================================================
# SPARQL EXECUTION
# ============================================================================

def _execute_sparql_worker(sparql: str, config: KGConfig, result_queue: Queue):
    """Worker function to execute SPARQL in a separate process."""
    try:
        results = execute_sparql_query(sparql, config=config)
        result_queue.put(("success", results, None))
    except Exception as e:
        result_queue.put(("error", [], str(e)))


def _execute_sparql_safe(
    sparql: str,
    config: KGConfig,
    timeout_seconds: int = 15,
) -> Tuple[bool, List[Dict[str, Any]], Optional[str]]:
    """Execute SPARQL with timeout and return (success, results, error_msg).

    Uses multiprocessing to ensure the query can be forcefully terminated.

    Args:
        sparql: SPARQL query string
        config: KGConfig instance
        timeout_seconds: Maximum execution time (default: 15 seconds)

    Returns:
        Tuple of (success, results, error_message)
        - success: True if executed without error
        - results: List of result dicts (empty if failed or timeout)
        - error_message: None if success, error string if failed/timeout
    """
    result_queue = Queue()
    process = Process(target=_execute_sparql_worker, args=(sparql, config, result_queue))

    try:
        process.start()
        process.join(timeout=timeout_seconds)

        if process.is_alive():
            # Process is still running - terminate it
            process.terminate()
            process.join(timeout=2)  # Wait up to 2 seconds for termination
            if process.is_alive():
                process.kill()  # Force kill if still alive
            return False, [], f"Execution timeout ({timeout_seconds}s exceeded)"

        # Process completed - check result
        if not result_queue.empty():
            status, results, error = result_queue.get()
            if status == "success":
                return True, results, None
            else:
                return False, [], f"Execution failed: {error}"
        else:
            return False, [], "Execution failed: No result returned"

    except Exception as e:
        if process.is_alive():
            process.terminate()
            process.join(timeout=1)
        return False, [], f"Execution failed: {str(e)}"


# ============================================================================
# EVALUATION STATES
# ============================================================================

# Four-state evaluation system:
# - "wrong_0": Execution error/timeout (no manual check needed)
# - "wrong_1": Empty retrieval when golden has results (needs LLM judge)
# - "correct_0": Definitely correct (no manual check needed)
# - "correct_1": Need to check answer manually to verify correctness

EVAL_WRONG_0 = "wrong_0"      # Execution error
EVAL_WRONG_1 = "wrong_1"      # Empty retrieval - needs verification
EVAL_CORRECT_0 = "correct_0"  # Auto-correct
EVAL_CORRECT_1 = "correct_1"  # Needs verification


# ============================================================================
# RESULT NORMALIZATION
# ============================================================================

def _normalize_sparql_results(results: List[Dict[str, Any]]) -> Set[Tuple[str, ...]]:
    """Convert SPARQL result rows to normalized tuples for comparison.

    Handles:
    - Multi-variable SELECT: {var1: val1, var2: val2} -> ("val1", "val2")
    - Empty results: [] -> set()
    - Variable ordering: sorts keys alphabetically for consistency

    Returns:
        Set of tuples representing result rows
    """
    if not results:
        return set()

    normalized = set()
    for row in results:
        # Sort keys to ensure consistent ordering
        sorted_keys = sorted(row.keys())
        values = tuple(str(row.get(k, "")).strip() for k in sorted_keys)
        normalized.add(values)

    return normalized


# ============================================================================
# COMPARISON METRICS
# ============================================================================

def _evaluate_execution_state(
    golden_success: bool,
    golden_results: List[Dict[str, Any]],
    generated_success: bool,
    generated_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate SPARQL execution using four-state logic.

    Four states:
    - wrong_0: Execution error/timeout (no manual check needed)
    - wrong_1: Empty retrieval when golden has results (needs LLM judge)
    - correct_0: Definitely correct (no manual check needed)
    - correct_1: Need to check answer manually

    9 cases based on (golden_state, generated_state):
    1. (Error, Error) → wrong_0
    2. (Error, Success+Results) → correct_1
    3. (Error, Success+Empty) → correct_1
    4. (Results, Error) → wrong_0
    5. (Results, Results) → correct_1 (compare via F1)
    6. (Results, Empty) → wrong_1 (needs LLM judge)
    7. (Empty, Error) → wrong_0
    8. (Empty, Results) → correct_0
    9. (Empty, Empty) → correct_0
    """
    golden_set = _normalize_sparql_results(golden_results)
    generated_set = _normalize_sparql_results(generated_results)

    has_golden_results = bool(golden_set)
    has_generated_results = bool(generated_set)

    # Case 1: Both failed → wrong_0
    if not golden_success and not generated_success:
        return {
            "evaluation_state": EVAL_WRONG_0,
            "reason": "Both queries failed to execute",
            "golden_count": 0,
            "generated_count": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    # Case 2 & 3: Golden failed, generated succeeded
    if not golden_success and generated_success:
        return {
            "evaluation_state": EVAL_CORRECT_1,
            "reason": "Golden failed, generated succeeded - needs manual verification",
            "golden_count": 0,
            "generated_count": len(generated_set),
            "precision": None,
            "recall": None,
            "f1": None,
        }

    # Case 4: Golden succeeded, generated failed → wrong_0
    if golden_success and not generated_success:
        return {
            "evaluation_state": EVAL_WRONG_0,
            "reason": "Generated query failed to execute",
            "golden_count": len(golden_set),
            "generated_count": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    # Both succeeded from here on

    # Case 9: Both empty → correct_0
    if not has_golden_results and not has_generated_results:
        return {
            "evaluation_state": EVAL_CORRECT_0,
            "reason": "Both queries returned empty results",
            "golden_count": 0,
            "generated_count": 0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        }

    # Case 8: Golden empty, generated has results → correct_0
    if not has_golden_results and has_generated_results:
        return {
            "evaluation_state": EVAL_CORRECT_0,
            "reason": "Golden empty but generated found results (golden may be wrong)",
            "golden_count": 0,
            "generated_count": len(generated_set),
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        }

    # Case 6: Golden has results, generated empty → wrong_1 (needs LLM judge)
    if has_golden_results and not has_generated_results:
        return {
            "evaluation_state": EVAL_WRONG_1,
            "reason": "Generated returned empty when golden has results - needs analysis",
            "golden_count": len(golden_set),
            "generated_count": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    # Case 5: Both have results → correct_1 (compare)
    tp = len(golden_set & generated_set)
    precision = tp / len(generated_set) if generated_set else 0.0
    recall = tp / len(golden_set) if golden_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    exact_match = golden_set == generated_set

    return {
        "evaluation_state": EVAL_CORRECT_1,
        "reason": f"Both returned results - F1={f1:.3f} (exact_match={exact_match})",
        "golden_count": len(golden_set),
        "generated_count": len(generated_set),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "exact_match": exact_match,
    }


# ============================================================================
# LLM JUDGE
# ============================================================================

class SPARQLJudgeSignature(dspy.Signature):
    """Compare two SPARQL queries for semantic equivalence."""

    question: str = dspy.InputField(desc="Natural language question")
    query_type: str = dspy.InputField(desc="Query type (DOUBLE_INTENT, BOOLEAN, COUNT, etc.)")
    golden_sparql: str = dspy.InputField(desc="Ground-truth SPARQL")
    generated_sparql: str = dspy.InputField(desc="Generated SPARQL")
    golden_results_preview: str = dspy.InputField(desc="First 5 rows from golden execution")
    generated_results_preview: str = dspy.InputField(desc="First 5 rows from generated execution")

    is_equivalent: bool = dspy.OutputField(desc="True if semantically equivalent")
    failure_code: str = dspy.OutputField(
        desc="One of: WRONG_QUERY_TYPE, MISSING_UNION, WRONG_ENTITY_URI, WRONG_RELATION, "
             "MISSING_CONSTRAINT, EXTRA_CONSTRAINT, WRONG_VARIABLE, CORRECT_EQUIVALENT"
    )
    explanation: str = dspy.OutputField(desc="2-4 sentence explanation of differences")


JUDGE_INSTRUCTION = """
You are evaluating SPARQL queries by comparing execution results.

Rules:
1. Both queries have been EXECUTED - compare actual results, not just syntax
2. If both returned the same results, mark is_equivalent=True
3. Variable names (?x vs ?paper, ?firstanswer vs ?title) don't matter if results match
4. Focus on: predicates used, constraints, filters, entity URIs

Failure codes:
- WRONG_RELATION: Different predicates used
- WRONG_ENTITY_URI: Different entity URIs
- MISSING_CONSTRAINT: Missing filter/condition
- EXTRA_CONSTRAINT: Over-constrained query
- WRONG_VARIABLE: Incorrect variable selection
- MISSING_UNION: Missing UNION clause
- MISSING_NEGATION: Missing NOT EXISTS/FILTER
- CORRECT_EQUIVALENT: Results match despite different syntax
"""


def _llm_judge(
    question: str,
    query_type: str,
    golden_sparql: str,
    generated_sparql: str,
    golden_results: List[Dict],
    generated_results: List[Dict],
    judge_module: dspy.Module,
) -> Dict[str, Any]:
    """Call LLM judge to explain failure."""
    try:
        golden_preview = json.dumps(golden_results[:5], ensure_ascii=False, indent=2)
        generated_preview = json.dumps(generated_results[:5], ensure_ascii=False, indent=2)

        result = judge_module(
            question=question,
            query_type=query_type,
            golden_sparql=golden_sparql,
            generated_sparql=generated_sparql,
            golden_results_preview=golden_preview,
            generated_results_preview=generated_preview,
        )

        is_eq = result.is_equivalent
        if isinstance(is_eq, str):
            is_eq = is_eq.strip().lower() in ("true", "yes", "1")

        return {
            "is_equivalent": bool(is_eq),
            "failure_code": str(result.failure_code).strip(),
            "explanation": str(result.explanation).strip(),
        }

    except Exception as e:
        return {
            "is_equivalent": False,
            "failure_code": "JUDGE_ERROR",
            "explanation": f"Judge failed: {type(e).__name__}: {e}",
        }


# ============================================================================
# EVALUATION ORCHESTRATOR
# ============================================================================

def _evaluate_question(
    question: Dict[str, Any],
    config: KGConfig,
    judge_module: dspy.Module,
) -> Dict[str, Any]:
    """Evaluate one question by executing both queries."""
    golden_sparql = question.get("golden_sparql", "")
    generated_info = question.get("generated_sparql_info", {})
    generated_sparql = generated_info.get("generated_sparql", "")
    query_type = question.get("query_type", "UNKNOWN")

    q_text = question.get("question", {})
    if isinstance(q_text, dict):
        q_text = q_text.get("1", "")

    # Execute both queries
    g_success, golden_results, g_error = _execute_sparql_safe(golden_sparql, config)
    gen_success, generated_results, gen_error = _execute_sparql_safe(generated_sparql, config)

    # Evaluate using four-state logic
    metrics = _evaluate_execution_state(
        g_success, golden_results,
        gen_success, generated_results
    )

    eval_state = metrics["evaluation_state"]

    # LLM judge for correct_1 and wrong_1 (needs verification/explanation)
    if eval_state == EVAL_CORRECT_1 or eval_state == EVAL_WRONG_1:
        verdict = _llm_judge(
            question=q_text,
            query_type=query_type,
            golden_sparql=golden_sparql,
            generated_sparql=generated_sparql,
            generated_results=generated_results,
            golden_results=golden_results,
            judge_module=judge_module,
        )
    elif eval_state == EVAL_CORRECT_0:
        verdict = {
            "is_equivalent": True,
            "failure_code": "CORRECT_AUTO",
            "explanation": metrics["reason"],
        }
    else:  # EVAL_WRONG_0
        verdict = {
            "is_equivalent": False,
            "failure_code": "EXECUTION_FAILURE",
            "explanation": metrics["reason"],
        }

    return {
        "golden_execution": {
            "success": g_success,
            "result_count": len(golden_results) if g_success else 0,
            "error": g_error,
        },
        "generated_execution": {
            "success": gen_success,
            "result_count": len(generated_results) if gen_success else 0,
            "error": gen_error,
        },
        "comparison_metrics": metrics,
        "llm_judge": verdict,
    }


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def evaluate_sparql_by_query_type(
    file_path: str,
    query_type: str,
    summary_path: Optional[str] = None,
    max_questions: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate SPARQL queries by query type, executing both golden and generated SPARQL.

    Args:
        file_path: JSONL with generated_sparql_info (results saved back to this file)
        query_type: Query type to filter (e.g., "DOUBLE_INTENT", "BOOLEAN", "COUNT")
        summary_path: Output .md path (defaults to eval/report/)
        max_questions: Limit for testing

    Returns:
        Stats dict with accuracy metrics
    """
    load_dotenv()

    # Setup output paths in eval/report/
    eval_dir = os.path.dirname(os.path.abspath(__file__))
    report_dir = os.path.join(eval_dir, "report")
    os.makedirs(report_dir, exist_ok=True)

    input_basename = os.path.splitext(os.path.basename(file_path))[0]
    query_type_lower = query_type.lower()

    if summary_path is None:
        summary_path = os.path.join(report_dir, f"{input_basename}_{query_type_lower}_report.md")

    # Setup
    config = KGConfig.default()
    api_key = config.lm_api_key
    if not api_key:
        raise ValueError("LM API key not found - set OPENAI_API_KEY")

    lm = dspy.LM(model=config.openai_model, api_key=api_key, cache=False)
    dspy.configure(lm=lm)

    sig = SPARQLJudgeSignature.with_instructions(JUDGE_INSTRUCTION)
    judge_module = dspy.ChainOfThought(sig)

    # Load ALL questions from input file
    print(f"Loading from {file_path} ...")
    print(f"Filtering for query_type: {query_type}\n")
    all_questions = []  # All questions from input file
    filtered_questions = []  # Only questions matching query_type
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                q = json.loads(line)
                all_questions.append(q)
                if q.get("query_type") == query_type:
                    filtered_questions.append(q)

    if max_questions:
        filtered_questions = filtered_questions[:max_questions]

    total = len(filtered_questions)
    print(f"Loaded {len(all_questions)} total questions")
    print(f"Found {total} {query_type} questions to evaluate\n")

    # Evaluate filtered questions
    t0 = time.time()
    correct_0_count = 0  # Auto-correct
    correct_1_count = 0  # Needs verification
    wrong_0_count = 0    # Execution error
    wrong_1_count = 0    # Empty retrieval - needs explanation
    judge_calls = 0
    failure_codes = defaultdict(int)

    for i, question in enumerate(filtered_questions, 1):
        qid = question.get("id", f"Q{i:04d}")

        ev = _evaluate_question(question, config, judge_module)
        question["execution_evaluation"] = ev

        metrics = ev["comparison_metrics"]
        verdict = ev["llm_judge"]
        eval_state = metrics["evaluation_state"]

        # Count by state
        if eval_state == EVAL_CORRECT_0:
            correct_0_count += 1
        elif eval_state == EVAL_CORRECT_1:
            correct_1_count += 1
            judge_calls += 1
        elif eval_state == EVAL_WRONG_0:
            wrong_0_count += 1
        else:  # EVAL_WRONG_1
            wrong_1_count += 1
            judge_calls += 1

        failure_codes[verdict["failure_code"]] += 1

        # Progress
        if eval_state == EVAL_CORRECT_0:
            status = "✓ C0"
        elif eval_state == EVAL_CORRECT_1:
            status = "? C1"
        elif eval_state == EVAL_WRONG_0:
            status = "✗ W0"
        else:
            status = "✗ W1"

        f1_str = f"F1={metrics.get('f1', 0):.3f}" if metrics.get('f1') is not None else ""
        fc = verdict.get("failure_code", "")

        # Check for timeout
        gen_exec = ev.get("generated_execution", {})
        error_msg = gen_exec.get("error") or ""
        timeout_marker = " ⏱" if "timeout" in error_msg.lower() else ""

        print(f"[{i:3d}/{total}] {qid:<10} {status}  {f1_str:<12} {fc}{timeout_marker}")

    elapsed = time.time() - t0

    # Save updated results back to input file
    print(f"\nSaving updated results -> {file_path}")
    with open(file_path, "w", encoding="utf-8") as f:
        for q in all_questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    # Write summary (only for filtered questions)
    _write_summary_md(filtered_questions, query_type, summary_path, elapsed)

    # Print summary
    print("\n" + "=" * 70)
    print(f"{query_type} EVALUATION COMPLETE")
    print("=" * 70)
    print(f"  Total questions      : {total}")
    print(f"  Correct_0 (auto)     : {correct_0_count} ({correct_0_count/total*100:.1f}%)")
    print(f"  Correct_1 (verify)   : {correct_1_count} ({correct_1_count/total*100:.1f}%)")
    print(f"  Wrong_0 (exec error) : {wrong_0_count} ({wrong_0_count/total*100:.1f}%)")
    print(f"  Wrong_1 (empty ret)  : {wrong_1_count} ({wrong_1_count/total*100:.1f}%)")
    print(f"  LLM judge calls      : {judge_calls}")
    print(f"  Total time           : {elapsed:.1f}s")
    print(f"\n  Input file updated   : {file_path}")
    print(f"  Markdown report      : {summary_path}")
    print("=" * 70)

    return {
        "total": total,
        "correct_0": correct_0_count,
        "correct_1": correct_1_count,
        "wrong_0": wrong_0_count,
        "wrong_1": wrong_1_count,
        "judge_calls": judge_calls,
        "failure_codes": dict(failure_codes),
        "elapsed": elapsed,
    }


def _write_summary_md(
    questions: List[Dict],
    query_type: str,
    output_path: str,
    elapsed: float,
) -> None:
    """Write markdown summary report."""
    total = len(questions)

    # Count by evaluation state
    correct_0 = sum(1 for q in questions
                    if q.get("execution_evaluation", {})
                     .get("comparison_metrics", {})
                     .get("evaluation_state") == EVAL_CORRECT_0)

    correct_1 = sum(1 for q in questions
                    if q.get("execution_evaluation", {})
                     .get("comparison_metrics", {})
                     .get("evaluation_state") == EVAL_CORRECT_1)

    wrong_0 = sum(1 for q in questions
                  if q.get("execution_evaluation", {})
                   .get("comparison_metrics", {})
                   .get("evaluation_state") == EVAL_WRONG_0)

    wrong_1 = sum(1 for q in questions
                  if q.get("execution_evaluation", {})
                   .get("comparison_metrics", {})
                   .get("evaluation_state") == EVAL_WRONG_1)

    # Count timeouts
    golden_timeouts = sum(1 for q in questions
                          if "timeout" in (q.get("execution_evaluation", {})
                           .get("golden_execution", {})
                           .get("error") or "").lower())

    generated_timeouts = sum(1 for q in questions
                             if "timeout" in (q.get("execution_evaluation", {})
                              .get("generated_execution", {})
                              .get("error") or "").lower())

    # Collect all failure codes
    failure_codes = defaultdict(int)
    llm_judge_codes = defaultdict(int)  # Only from correct_1 and wrong_1 cases

    for q in questions:
        ev = q.get("execution_evaluation", {})
        metrics = ev.get("comparison_metrics", {})
        eval_state = metrics.get("evaluation_state")
        code = ev.get("llm_judge", {}).get("failure_code", "UNKNOWN")

        failure_codes[code] += 1

        # Track LLM judge codes only for correct_1 and wrong_1 (where LLM judge was actually used)
        if eval_state in (EVAL_CORRECT_1, EVAL_WRONG_1):
            llm_judge_codes[code] += 1

    judge_count = correct_1 + wrong_1

    lines = [
        f"# {query_type} SPARQL Evaluation Report",
        "",
        "**Evaluation method**: Four-state evaluation (correct_0, correct_1, wrong_0, wrong_1)",
        f"**Query type**: {query_type}",
        f"**Questions evaluated**: {total}",
        "",
        "## Evaluation States",
        "",
        "| State | Count | % | Description |",
        "|-------|-------|---|-------------|",
        f"| **Correct_0** | {correct_0} | {correct_0/total*100:.1f}% | Auto-correct (no verification needed) |",
        f"| **Correct_1** | {correct_1} | {correct_1/total*100:.1f}% | Needs manual verification (LLM judge) |",
        f"| **Wrong_0** | {wrong_0} | {wrong_0/total*100:.1f}% | Execution error/timeout (no LLM judge) |",
        f"| **Wrong_1** | {wrong_1} | {wrong_1/total*100:.1f}% | Empty retrieval - needs explanation (LLM judge) |",
        "",
        f"**Golden timeouts**: {golden_timeouts}",
        f"**Generated timeouts**: {generated_timeouts}",
        f"**Total time**: {elapsed:.1f}s",
        "",
        "## Failure Code Distribution (All)",
        "",
        "| Code | Count | % |",
        "|------|-------|---|",
    ]

    for code, count in sorted(failure_codes.items(), key=lambda x: -x[1]):
        pct = count / total * 100 if total else 0
        lines.append(f"| `{code}` | {count} | {pct:.1f}% |")

    # Add LLM Judge Code Distribution table
    lines.extend(["", "## LLM Judge Code Distribution (Correct_1 + Wrong_1)", ""])
    lines.append("| Code | Count | % of LLM Judge Calls |")
    lines.append("|------|-------|----------------------|")

    for code, count in sorted(llm_judge_codes.items(), key=lambda x: -x[1]):
        pct = count / judge_count * 100 if judge_count else 0
        lines.append(f"| `{code}` | {count} | {pct:.1f}% |")

    lines.extend(["", "## Question Details", ""])

    for q in questions:
        ev = q.get("execution_evaluation", {})
        metrics = ev.get("comparison_metrics", {})
        eval_state = metrics.get("evaluation_state")

        qid = q.get("id", "?")
        q_text = q.get("question", {})
        if isinstance(q_text, dict):
            q_text = q_text.get("1", "")

        verdict = ev.get("llm_judge", {})
        gen_exec = ev.get("generated_execution", {})
        gold_exec = ev.get("golden_execution", {})

        if eval_state == EVAL_CORRECT_0:
            state_icon = "✓"
        elif eval_state == EVAL_CORRECT_1:
            state_icon = "?"
        elif eval_state == EVAL_WRONG_0:
            state_icon = "✗"
        else:  # EVAL_WRONG_1
            state_icon = "✗?"

        lines.append(f"### {state_icon} {qid} — {eval_state}")
        lines.append(f"**Q**: {q_text}")

        if metrics.get('f1') is not None:
            lines.append(f"**F1**: {metrics.get('f1', 0):.3f} | "
                        f"P: {metrics.get('precision', 0):.3f} | "
                        f"R: {metrics.get('recall', 0):.3f}")

        lines.append(f"**Golden**: {gold_exec.get('result_count', 0)} results "
                    f"({'✓' if gold_exec.get('success') else '✗ ' + gold_exec.get('error', '')})")
        lines.append(f"**Generated**: {gen_exec.get('result_count', 0)} results "
                    f"({'✓' if gen_exec.get('success') else '✗ ' + gen_exec.get('error', '')})")

        lines.append(f"**Reason**: {metrics.get('reason', '')}")

        if eval_state in (EVAL_CORRECT_1, EVAL_WRONG_1):
            lines.append(f"**LLM Judge**: `{verdict.get('failure_code')}` - {verdict.get('explanation')}")

        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Summary written -> {output_path}")


def _write_combined_report(
    all_results: Dict[str, Dict[str, Any]],
    input_path: str,
    total_elapsed: float,
) -> str:
    """Write combined markdown report aggregating all query types.

    Args:
        all_results: Dict mapping query_type -> stats dict
        input_path: Input JSONL path
        total_elapsed: Total elapsed time across all query types

    Returns:
        Path to combined report file
    """
    # Create combined report in eval/logs/
    eval_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(eval_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    combined_path = os.path.join(logs_dir, f"combined_report_{timestamp}.md")

    # Calculate aggregate statistics
    total_questions = sum(r.get("total", 0) for r in all_results.values() if "error" not in r)
    total_c0 = sum(r.get("correct_0", 0) for r in all_results.values() if "error" not in r)
    total_c1 = sum(r.get("correct_1", 0) for r in all_results.values() if "error" not in r)
    total_w0 = sum(r.get("wrong_0", 0) for r in all_results.values() if "error" not in r)
    total_w1 = sum(r.get("wrong_1", 0) for r in all_results.values() if "error" not in r)
    total_judge_calls = sum(r.get("judge_calls", 0) for r in all_results.values() if "error" not in r)

    # Aggregate failure codes across all types
    combined_failure_codes = defaultdict(int)
    for result in all_results.values():
        if "error" not in result:
            for code, count in result.get("failure_codes", {}).items():
                combined_failure_codes[code] += count

    lines = [
        "# Combined SPARQL Evaluation Report",
        "",
        f"**Generated**: {timestamp.replace('_', ' ')}",
        f"**Input file**: {input_path}",
        f"**Total time**: {total_elapsed:.1f}s",
        "",
        "## Overall Statistics",
        "",
        f"**Total questions evaluated**: {total_questions}",
        f"**Correct_0 (auto-correct)**: {total_c0} ({total_c0/total_questions*100:.1f}%)" if total_questions > 0 else "**Correct_0**: 0",
        f"**Correct_1 (needs verification)**: {total_c1} ({total_c1/total_questions*100:.1f}%)" if total_questions > 0 else "**Correct_1**: 0",
        f"**Wrong_0 (execution error)**: {total_w0} ({total_w0/total_questions*100:.1f}%)" if total_questions > 0 else "**Wrong_0**: 0",
        f"**Wrong_1 (empty retrieval)**: {total_w1} ({total_w1/total_questions*100:.1f}%)" if total_questions > 0 else "**Wrong_1**: 0",
        f"**LLM judge calls**: {total_judge_calls}",
        "",
        "## Per Query Type Breakdown",
        "",
        "| Query Type | Total | C0 | C1 | W0 | W1 | C0% | Judge Calls |",
        "|------------|-------|----|----|----|----|-----|-------------|",
    ]

    # Add per-type rows
    for query_type, result in sorted(all_results.items()):
        if "error" in result:
            lines.append(f"| {query_type} | ERROR | - | - | - | - | - | - |")
        else:
            total = result["total"]
            c0 = result["correct_0"]
            c1 = result["correct_1"]
            w0 = result["wrong_0"]
            w1 = result["wrong_1"]
            judge = result["judge_calls"]
            c0_pct = (c0/total*100) if total > 0 else 0
            lines.append(f"| {query_type} | {total} | {c0} | {c1} | {w0} | {w1} | {c0_pct:.1f}% | {judge} |")

    lines.extend([
        "",
        "## Combined Failure Code Distribution",
        "",
        "| Code | Count | % of Total |",
        "|------|-------|------------|",
    ])

    for code, count in sorted(combined_failure_codes.items(), key=lambda x: -x[1]):
        pct = (count / total_questions * 100) if total_questions > 0 else 0
        lines.append(f"| `{code}` | {count} | {pct:.1f}% |")

    lines.extend([
        "",
        "## Evaluation States Legend",
        "",
        "- **Correct_0** (Definitely Correct): Auto-correct (no verification needed)",
        "- **Correct_1** (Likely Correct): Needs manual verification (LLM judge used)",
        "- **Wrong_0** (Definitely Wrong): Execution error/timeout (no LLM judge)",
        "- **Wrong_1** (Likely Wrong): Empty retrieval - needs explanation (LLM judge used)",
        "",
    ])

    # ========================================================================
    # PAPER TABLES DATA (DBLP-QuAD, gpt-4o-mini)
    # ========================================================================
    lines.extend([
        "",
        "---",
        "",
        "# Data for Paper Tables (DBLP-QuAD, gpt-4o-mini)",
        "",
        "## Table 3: Nine-Case SPARQL Evaluation Logic",
        "",
        "State mapping (already documented in code):",
        "- Case 1-4, 7: **Definitely Wrong** (W0)",
        "- Case 8-9: **Definitely Correct** (C0)",
        "- Case 2-3, 5: **Likely Correct** (C1)",
        "- Case 6: **Likely Wrong** (W1)",
        "",
        "## Table 4: Error Distribution (%)",
        "",
        "**Failure code mapping to paper categories:**",
        "",
    ])

    # Map failure codes to paper categories
    paper_categories = {
        "CORRECT_EQUIVALENT": "Correct Equivalent",
        "EXECUTION_FAILURE": "Execution Failed",
        "EXTRA_CONSTRAINT": "Over-Constrained",
        "MISSING_CONSTRAINT": "Under-Constrained",
        "MISSING_NEGATION": "Missing Negation",
        "MISSING_UNION": "Missing OR Logic",
        "WRONG_ENTITY_URI": "Wrong Entity URI",
        "WRONG_RELATION": "Wrong Predicate",
        # Map other codes if present
        "WRONG_VARIABLE": "Wrong Variable",
        "WRONG_QUERY_TYPE": "Wrong Query Type",
    }

    category_counts = defaultdict(int)
    for code, count in combined_failure_codes.items():
        # Try to find matching paper category
        matched = False
        for key, category in paper_categories.items():
            if key in code or code in key:
                category_counts[category] += count
                matched = True
                break
        if not matched:
            # Add unmapped codes to "Other" category
            category_counts["Other"] += count

    lines.append("| Category | Count | % |")
    lines.append("|----------|-------|---|")
    for category in ["Correct Equivalent", "Execution Failed", "Over-Constrained", "Under-Constrained",
                     "Missing Negation", "Missing OR Logic", "Wrong Entity URI", "Wrong Predicate", "Other"]:
        count = category_counts.get(category, 0)
        pct = (count / total_questions * 100) if total_questions > 0 else 0
        lines.append(f"| {category} | {count} | {pct:.1f}% |")

    lines.extend([
        "",
        "## Table 6: SPARQL Evaluation Metrics (%)",
        "",
    ])

    # Calculate metrics
    executable_pct = ((total_questions - total_w0) / total_questions * 100) if total_questions > 0 else 0
    definitely_wrong_pct = (total_w0 / total_questions * 100) if total_questions > 0 else 0
    definitely_correct_pct = (total_c0 / total_questions * 100) if total_questions > 0 else 0
    likely_wrong_pct = (total_w1 / total_questions * 100) if total_questions > 0 else 0
    likely_correct_pct = (total_c1 / total_questions * 100) if total_questions > 0 else 0

    lines.extend([
        "| Metric | Value |",
        "|--------|-------|",
        f"| Executable | {executable_pct:.1f}% |",
        f"| Exact Match | N/A (requires per-question analysis) |",
        f"| F1 Score | N/A (requires per-question analysis) |",
        "",
        "**State Distribution:**",
        "",
        f"| Definitely Wrong (W0) | {definitely_wrong_pct:.1f}% |",
        f"| Definitely Correct (C0) | {definitely_correct_pct:.1f}% |",
        f"| Likely Wrong (W1) | {likely_wrong_pct:.1f}% |",
        f"| Likely Correct (C1) | {likely_correct_pct:.1f}% |",
        "",
        "## LaTeX Table Format",
        "",
        "**For Table 4 (Error Distribution):**",
        "```latex",
    ])

    # Generate LaTeX snippet for Table 4
    lines.append("% DBLP-QuAD column for gpt-4o-mini")
    for category in ["Correct Equivalent", "Execution Failed", "Over-Constrained", "Under-Constrained",
                     "Missing Negation", "Missing OR Logic", "Wrong Entity URI", "Wrong Predicate"]:
        count = category_counts.get(category, 0)
        pct = (count / total_questions * 100) if total_questions > 0 else 0
        lines.append(f"{category} & {pct:.1f} & - & - & - \\\\")

    lines.extend([
        "```",
        "",
        "**For Table 6 (SPARQL Metrics):**",
        "```latex",
        "% DBLP-QuAD column for gpt-4o-mini",
        f"Executable & {executable_pct:.1f} & - & - & - \\\\",
        "Exact Match & N/A & - & - & - \\\\",
        "F1 Score & N/A & - & - & - \\\\",
        "\\cline{2-5}",
        f"Definitely Wrong & {definitely_wrong_pct:.1f} & - & - & - \\\\",
        f"Definitely Correct & {definitely_correct_pct:.1f} & - & - & - \\\\",
        "\\cline{2-5}",
        f"Likely Wrong & {likely_wrong_pct:.1f} & - & - & - \\\\",
        f"Likely Correct & {likely_correct_pct:.1f} & - & - & - \\\\",
        "```",
        "",
    ])

    with open(combined_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nCombined report written -> {combined_path}")
    return combined_path


def evaluate_all_questions(
    file_path: str,
    max_questions: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate all questions regardless of query type.

    Args:
        file_path: JSONL with generated_sparql_info (results saved back to this file)
        max_questions: Limit for testing

    Returns:
        Stats dict with accuracy metrics
    """
    load_dotenv()

    # Setup output paths
    eval_dir = os.path.dirname(os.path.abspath(__file__))
    report_dir = os.path.join(eval_dir, "report")
    os.makedirs(report_dir, exist_ok=True)

    input_basename = os.path.splitext(os.path.basename(file_path))[0]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    summary_path = os.path.join(report_dir, f"{input_basename}_all_report_{timestamp}.md")

    # Setup
    config = KGConfig.default()
    api_key = config.lm_api_key
    if not api_key:
        raise ValueError("LM API key not found - set OPENAI_API_KEY")

    lm = dspy.LM(model=config.openai_model, api_key=api_key, cache=False)
    dspy.configure(lm=lm)

    sig = SPARQLJudgeSignature.with_instructions(JUDGE_INSTRUCTION)
    judge_module = dspy.ChainOfThought(sig)

    # Load all questions
    print(f"Loading from {file_path} ...")
    print("Processing ALL questions (ignoring query types)\n")
    all_questions = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_questions.append(json.loads(line))

    if max_questions:
        all_questions = all_questions[:max_questions]

    total = len(all_questions)
    print(f"Loaded {total} questions\n")

    # Evaluate
    t0 = time.time()
    correct_0_count = 0
    correct_1_count = 0
    wrong_0_count = 0
    wrong_1_count = 0
    judge_calls = 0
    failure_codes = defaultdict(int)

    for i, question in enumerate(all_questions, 1):
        qid = question.get("id", f"Q{i:04d}")

        ev = _evaluate_question(question, config, judge_module)
        question["execution_evaluation"] = ev

        metrics = ev["comparison_metrics"]
        verdict = ev["llm_judge"]
        eval_state = metrics["evaluation_state"]

        # Count by state
        if eval_state == EVAL_CORRECT_0:
            correct_0_count += 1
        elif eval_state == EVAL_CORRECT_1:
            correct_1_count += 1
            judge_calls += 1
        elif eval_state == EVAL_WRONG_0:
            wrong_0_count += 1
        else:  # EVAL_WRONG_1
            wrong_1_count += 1
            judge_calls += 1

        failure_codes[verdict["failure_code"]] += 1

        # Progress
        if eval_state == EVAL_CORRECT_0:
            status = "✓ C0"
        elif eval_state == EVAL_CORRECT_1:
            status = "? C1"
        elif eval_state == EVAL_WRONG_0:
            status = "✗ W0"
        else:
            status = "✗ W1"

        f1_str = f"F1={metrics.get('f1', 0):.3f}" if metrics.get('f1') is not None else ""
        fc = verdict.get("failure_code", "")

        # Check for timeout
        gen_exec = ev.get("generated_execution", {})
        error_msg = gen_exec.get("error") or ""
        timeout_marker = " ⏱" if "timeout" in error_msg.lower() else ""

        print(f"[{i:3d}/{total}] {qid:<10} {status}  {f1_str:<12} {fc}{timeout_marker}")

    elapsed = time.time() - t0

    # Save updated results back to input file
    print(f"\nSaving updated results -> {file_path}")
    with open(file_path, "w", encoding="utf-8") as f:
        for q in all_questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    # Write summary
    _write_summary_md(all_questions, "ALL_QUESTIONS", summary_path, elapsed)

    # Print summary
    print("\n" + "=" * 70)
    print("ALL QUESTIONS EVALUATION COMPLETE")
    print("=" * 70)
    print(f"  Total questions      : {total}")
    print(f"  Correct_0 (auto)     : {correct_0_count} ({correct_0_count/total*100:.1f}%)")
    print(f"  Correct_1 (verify)   : {correct_1_count} ({correct_1_count/total*100:.1f}%)")
    print(f"  Wrong_0 (exec error) : {wrong_0_count} ({wrong_0_count/total*100:.1f}%)")
    print(f"  Wrong_1 (empty ret)  : {wrong_1_count} ({wrong_1_count/total*100:.1f}%)")
    print(f"  LLM judge calls      : {judge_calls}")
    print(f"  Total time           : {elapsed:.1f}s")
    print(f"\n  Input file updated   : {file_path}")
    print(f"  Markdown report      : {summary_path}")
    print("=" * 70)

    return {
        "total": total,
        "correct_0": correct_0_count,
        "correct_1": correct_1_count,
        "wrong_0": wrong_0_count,
        "wrong_1": wrong_1_count,
        "judge_calls": judge_calls,
        "failure_codes": dict(failure_codes),
        "elapsed": elapsed,
    }


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    setup_dual_output(__file__)

    # ============================================================================
    # CONFIGURATION
    # ============================================================================

    input_path = "./data/balanced_400.jsonl"

    # Set to True to process by query type, False to process all questions
    question_type_based_generation = False

    # Query types to process (only used if question_type_based_generation=True)
    # Can be a single string or a list of strings
    QUERY_TYPES = [
        "DISAMBIGUATION", "DOUBLE_INTENT"
    ]
    # Or use a single type: QUERY_TYPES = "DISAMBIGUATION", "SUPERLATIVE+COMPARATIVE", BOOLEAN", "COUNT", "NEGATION", "DOUBLE_NEGATION", "UNION", "MULTI_FACT", "SINGLE_FACT",

    # Maximum number of questions to process (None = all)
    max_questions = None

    # ============================================================================

    # Command line overrides
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    if len(sys.argv) > 2:
        # Command line can override with comma-separated types
        QUERY_TYPES = sys.argv[2].split(",")

    # Convert single string to list
    if isinstance(QUERY_TYPES, str):
        QUERY_TYPES = [QUERY_TYPES]

    # Track total time
    total_start = time.time()

    if question_type_based_generation:
        # Evaluate by query type
        all_results = {}
        print("=" * 70)
        print(f"BATCH EVALUATION: {len(QUERY_TYPES)} query types")
        print("=" * 70)

        for i, query_type in enumerate(QUERY_TYPES, 1):
            print(f"\n[{i}/{len(QUERY_TYPES)}] Starting evaluation for: {query_type}")
            print("-" * 70)

            try:
                result = evaluate_sparql_by_query_type(input_path, query_type, max_questions=max_questions)
                all_results[query_type] = result
            except Exception as e:
                print(f"ERROR evaluating {query_type}: {e}")
                all_results[query_type] = {"error": str(e)}

        # Print overall summary
        print("\n" + "=" * 70)
        print("BATCH EVALUATION SUMMARY")
        print("=" * 70)

        for query_type, result in all_results.items():
            if "error" in result:
                print(f"{query_type:<25} ERROR: {result['error']}")
            else:
                total = result["total"]
                c0 = result["correct_0"]
                c1 = result["correct_1"]
                w0 = result["wrong_0"]
                w1 = result["wrong_1"]
                print(f"{query_type:<25} Total: {total:3d} | C0: {c0:2d} | C1: {c1:2d} | W0: {w0:2d} | W1: {w1:2d}")

        print("=" * 70)

        # Generate combined report
        total_elapsed = time.time() - total_start
        _write_combined_report(all_results, input_path, total_elapsed)

    else:
        # Evaluate all questions at once
        print("=" * 70)
        print("EVALUATING ALL QUESTIONS (IGNORING QUERY TYPES)")
        print("=" * 70)

        try:
            result = evaluate_all_questions(input_path, max_questions=max_questions)
            all_results = {"ALL_QUESTIONS": result}
        except Exception as e:
            print(f"ERROR evaluating: {e}")
            all_results = {"ALL_QUESTIONS": {"error": str(e)}}

        # Generate combined report
        total_elapsed = time.time() - total_start
        _write_combined_report(all_results, input_path, total_elapsed)