"""Run the full evaluation pipeline for all questions in a JSONL file.

PREREQUISITES (must be run once before this script):
  1. Build the evaluation JSONL dataset:
       python eval/load_questions_util.py
         → creates eval/data/balanced_200.jsonl (or balanced_N.jsonl)

  2. Populate ChromaDB with target entities:
       python eval/add_target_entities_to_chromadb.py
         → seeds the entity collection that step-1 searches against

  3. Populate ChromaDB with target relations:
       python eval/add_target_relations_to_chromadb.py
         → seeds the relation collection that step-1 searches against
         → skipping this causes 0 relations to be found (step1: 0 relations)

Chains all 5 pipeline steps in order:
  1. Seed + relation node extraction  (add_seed_node_info.py logic)
  2. Entity seed filtering            (add_filtered_entity_info.py logic)
  3. Relation filtering               (add_filtered_relation_info.py logic)
  4. Subgraph extraction              (add_subgraph_info.py logic)
  5. Subgraph top-K merging           (add_filter_subgraph_info.py logic)
  6. SPARQL generation + evaluation   (add_sparql_generated_answer + evaluate_question_type_sparql)

The output file can then be fed directly to evaluate_sparql_scenarios.py, giving
a real-case evaluation where every pipeline stage runs fresh (no stored/golden-derived data).

Usage:
    # 0. Prerequisites (run once to build the dataset and populate ChromaDB)
    python eval/load_questions_util.py
    python eval/add_target_entities_to_chromadb.py
    python eval/add_target_relations_to_chromadb.py

    # 1. Run full pipeline (produces pipeline_fresh.jsonl + optional stats report)
        python eval/full_evaluation_pipeline.py --input eval/data/balanced_200.jsonl --output eval/data/full-pipeline.jsonl --report --report-out eval/report/full-pipeline-stats.md --sparql-report-dir eval/report/

    # (optional) Compare SPARQL generation scenarios (baseline vs GEPA artifacts)
    python eval/optimizers/evaluate_sparql_scenarios.py --data eval/data/full-pipeline.jsonl --scenario "Baseline:" --scenario "GEPA-200:src/kgnode/optimization/artifacts/sparql/sparql_gepa_light_200.json" --out eval/optimizers/report/all_sparql_scenarios_eval.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from dotenv import load_dotenv

from kgnode import KGConfig, SearchMode
from kgnode.seed_finder import get_seed_and_relation_nodes

# Import filter/extraction helpers from evaluation scripts
from add_filtered_entity_info import filter_seeds
from add_filtered_relation_info import deduplicate_relation_nodes_by_types, filter_relations
from add_subgraph_info import extract_subgraphs_for_seed
from add_filter_subgraph_info import apply_topk_and_merge
from add_sparql_generated_answer import add_generated_sparql
from evaluate_question_type_sparql import evaluate_sparql_by_query_type



# CONSTANTS  (same defaults as the individual pipeline scripts)

N_RESULTS_ENTITY = 10
N_RESULTS_RELATION = 3
MAX_HOPS = 6
MAX_K = 4
TOP_K_MERGE = 3


# PER-QUESTION PIPELINE


def run_pipeline_for_question(
    question: Dict[str, Any],
    config: KGConfig,
) -> Dict[str, Any]:
    """Run all 5 pipeline steps for a single question.

    Runs fresh: seed extraction → entity filter → relation filter →
    subgraph extraction → subgraph top-K merge.

    Args:
        question: Question dict with at minimum 'question' and 'id' fields.
        config: KGConfig instance (shared across questions for efficiency).

    Returns:
        The question dict mutated in-place with pipeline output fields:
          - extracted_nodes, extracted_relation_nodes, extracted_entities
          - filtered_entity_info.filtered_seeds
          - filtered_relation_info.filtered_relations
          - subgraph_extraction.filtered_seeds, .template_text
          - filtered_subgraphs_info.merged_graphs
    """
    question_id = question.get("id", "?")
    question_text = _get_question_text(question)

    if not question_text:
        print(f"  [SKIP] {question_id}: no question text")
        return question

    # Step 1: Seed + relation node extraction 
    t0 = time.time()
    seed_nodes, extracted_entities, relation_nodes, extracted_relations = (
        get_seed_and_relation_nodes(
            query=question_text,
            n_results_entity=N_RESULTS_ENTITY,
            n_results_relation=N_RESULTS_RELATION,
            config=config,
            search_mode=SearchMode.semantic,
        )
    )
    question["extracted_nodes"] = seed_nodes
    question["extracted_relation_nodes"] = relation_nodes
    question["extracted_entities"] = [
        {"name": e["name"], "types": e.get("types", [])}
        for e in extracted_entities
    ]
    print(
        f"  [step1] {question_id}: {len(seed_nodes)} seeds, "
        f"{len(relation_nodes)} relations  ({time.time()-t0:.1f}s)"
    )

    # Step 2: Entity (seed) filtering 
    t0 = time.time()
    filtered_seeds = filter_seeds(seed_nodes)
    question["filtered_entity_info"] = {
        "filtered_seeds": filtered_seeds,
        "stats": {
            "original_count": len(seed_nodes),
            "filtered_count": len(filtered_seeds),
        },
    }
    print(
        f"  [step2] {question_id}: {len(seed_nodes)}→{len(filtered_seeds)} seeds "
        f"({time.time()-t0:.1f}s)"
    )

    if not filtered_seeds:
        print(f"  [WARN] {question_id}: no filtered seeds — skipping steps 3-5")
        question["filtered_relation_info"] = {"filtered_relations": []}
        question["subgraph_extraction"] = {"template_text": "NO_SEEDS", "filtered_seeds": []}
        question["filtered_subgraphs_info"] = {"merged_graphs": []}
        return question

    # Step 3: Relation filtering 
    t0 = time.time()
    deduped_relations = deduplicate_relation_nodes_by_types(relation_nodes)
    filtered_relations = filter_relations(deduped_relations, use_fuzzy_matching=False)
    question["filtered_relation_info"] = {
        "filtered_relations": filtered_relations,
        "stats": {
            "original_count": len(deduped_relations),
            "filtered_count": len(filtered_relations),
        },
    }
    print(
        f"  [step3] {question_id}: {len(relation_nodes)}→{len(filtered_relations)} relations "
        f"({time.time()-t0:.1f}s)"
    )

    # Step 4: Subgraph extraction 
    t0 = time.time()
    seed_results: List[Dict[str, Any]] = []
    template_text: Optional[str] = None

    for j, seed in enumerate(filtered_seeds, 1):
        result, seed_template = extract_subgraphs_for_seed(
            seed_node=seed,
            question_text=question_text,
            config=config,
            max_hops=MAX_HOPS,
            max_k=MAX_K,
            all_seed_nodes=filtered_seeds,
            all_relation_nodes=filtered_relations if filtered_relations else None,
        )
        if result:
            if template_text is None:
                template_text = seed_template
            seed_results.append(result)

    question["subgraph_extraction"] = {
        "template_text": template_text or "ERROR",
        "filtered_seeds": seed_results,
    }
    total_subgraphs = sum(s["subgraph_count"] for s in seed_results)
    print(
        f"  [step4] {question_id}: {total_subgraphs} subgraphs from "
        f"{len(seed_results)}/{len(filtered_seeds)} seeds  ({time.time()-t0:.1f}s)"
    )

    #  Step 5: Top-K merge 
    t0 = time.time()
    merged_graphs = apply_topk_and_merge(seed_results, k=TOP_K_MERGE)
    question["filtered_subgraphs_info"] = {
        "top_k_per_seed": TOP_K_MERGE,
        "filtering_method": "top_k_then_merge",
        "original_subgraph_count": total_subgraphs,
        "merged_graph_count": len(merged_graphs),
        "merged_graphs": merged_graphs,
    }
    print(
        f"  [step5] {question_id}: {total_subgraphs}→{len(merged_graphs)} merged graphs "
        f"({time.time()-t0:.1f}s)"
    )

    return question


# FULL PIPELINE RUNNER

def run_full_pipeline(
    input_path: str,
    output_path: str,
    max_questions: Optional[int] = None,
    query_types: Optional[List[str]] = None,
    report_path: Optional[str] = None,
    sparql_report_dir: Optional[str] = None,
) -> None:
    """Run the full 6-step pipeline for all questions and write results.

    Always re-runs all pipeline steps for every question (no caching).

    Steps 1-5: seed extraction → entity filter → relation filter →
    subgraph extraction → top-K merge (per question).
    Step 6: SPARQL generation + evaluation by query type.

    Args:
        input_path: Path to input JSONL file.
        output_path: Path to write results.  Use the same path as input for
            in-place update, or a different path for side-by-side comparison.
        max_questions: Optional cap for quick testing.
        query_types: Optional list of query types to restrict processing.
        report_path: If set, write a pipeline stats markdown report here.
        sparql_report_dir: Directory for per-type SPARQL evaluation reports.
            Defaults to eval/report/.
    """
    load_dotenv()
    config = KGConfig.default()

    # Prerequisites reminder
    print("PREREQUISITES — must be run once before this script:")
    print("  python eval/load_questions_util.py")
    print("  python eval/add_target_entities_to_chromadb.py")
    print("  python eval/add_target_relations_to_chromadb.py")
    print("  (0 relations in step1 = add_target_relations_to_chromadb.py not run)")
    print()

    # Load questions
    print(f"Loading questions from {input_path}…")
    questions = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    if max_questions:
        questions = questions[:max_questions]

    if query_types:
        questions = [q for q in questions if q.get("query_type") in query_types]

    total = len(questions)
    print(f"Loaded {total} questions\n")

    processed = 0
    failed = 0
    t_wall = time.time()

    for i, question in enumerate(questions, 1):
        question_id = question.get("id", f"Q{i:04d}")

        print(f"\n[{i}/{total}] {question_id}")
        try:
            run_pipeline_for_question(question, config)
            processed += 1
        except Exception as exc:
            print(f"  [ERROR] {question_id}: {type(exc).__name__}: {exc}")
            failed += 1

        # Save after every question (incremental safety)
        with open(output_path, "w", encoding="utf-8") as f:
            for q in questions:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")

    wall = time.time() - t_wall
    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"  Input:     {input_path}")
    print(f"  Output:    {output_path}")
    print(f"  Total:     {total}")
    print(f"  Processed: {processed}")
    print(f"  Failed:    {failed}")
    print(f"  Time:      {wall:.0f}s")
    print(f"{'='*70}")

    if report_path:
        _generate_pipeline_report(questions, input_path, output_path, report_path)

    # Step 6: SPARQL generation + evaluation by query type
    _run_sparql_evaluation(output_path, questions, max_questions, sparql_report_dir)


# SPARQL EVALUATION STEP

def _run_sparql_evaluation(
    output_path: str,
    questions: List[Dict[str, Any]],
    max_questions: Optional[int],
    sparql_report_dir: Optional[str],
) -> None:
    """Step 6: generate SPARQL + evaluate by query type."""
    report_dir = sparql_report_dir or os.path.join(os.path.dirname(output_path), "..", "report")
    report_dir = os.path.normpath(report_dir)
    os.makedirs(report_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print("STEP 6: SPARQL GENERATION + EVALUATION")
    print(f"{'='*70}")

    # 6a: generate SPARQL (writes generated_sparql_info into output_path in-place)
    print("\n[step6a] Generating SPARQL for all questions…")
    add_generated_sparql(
        file_path=output_path,
        output_path=output_path,
        max_questions=max_questions,
        force_regenerate=True,
    )

    # 6b: evaluate by each distinct query type present in the output
    present_types = sorted({q.get("query_type", "") for q in questions if q.get("query_type")})
    print(f"\n[step6b] Evaluating {len(present_types)} query type(s): {present_types}")

    all_stats: List[Dict[str, Any]] = []
    for qt in present_types:
        print(f"  Evaluating {qt}…")
        basename = os.path.splitext(os.path.basename(output_path))[0]
        stats = evaluate_sparql_by_query_type(
            file_path=output_path,
            query_type=qt,
            output_jsonl_path=os.path.join(report_dir, f"{basename}_{qt.lower()}_results.jsonl"),
            summary_path=os.path.join(report_dir, f"{basename}_{qt.lower()}_report.md"),
            max_questions=max_questions,
        )
        all_stats.append({"query_type": qt, **stats})

    # Combined summary
    _write_sparql_summary(all_stats, report_dir, output_path)


def _write_sparql_summary(
    all_stats: List[Dict[str, Any]],
    report_dir: str,
    output_path: str,
) -> None:
    """Write a combined markdown table of SPARQL eval stats across all query types."""
    summary_path = os.path.join(
        report_dir,
        f"{os.path.splitext(os.path.basename(output_path))[0]}_sparql_summary.md",
    )
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# SPARQL Evaluation Summary",
        "",
        f"Generated: {now}  ",
        f"Source: `{output_path}`",
        "",
        "| Query Type | Total | Correct (auto) | Correct (verified) | Exec error | Empty |",
        "|------------|-------|----------------|--------------------|------------|-------|",
    ]
    for s in all_stats:
        qt = s.get("query_type", "?")
        total = s.get("total", 0)
        c0 = s.get("correct_0_count", 0)
        c1 = s.get("correct_1_count", 0)
        w0 = s.get("wrong_0_count", 0)
        w1 = s.get("wrong_1_count", 0)
        lines.append(f"| {qt} | {total} | {c0} | {c1} | {w0} | {w1} |")
    lines.append("")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nSPARQL summary written to {summary_path}")


# REPORT GENERATION

def _generate_pipeline_report(
    questions: List[Dict[str, Any]],
    input_path: str,
    output_path: str,
    report_path: str,
) -> None:
    """Write a markdown pipeline stats report to report_path."""
    os.makedirs(os.path.dirname(report_path), exist_ok=True) if os.path.dirname(report_path) else None

    total = len(questions)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Aggregate counts
    seed_totals: Dict[str, List[int]] = {}  # qtype → [raw_seeds, filtered_seeds]
    rel_totals: Dict[str, List[int]] = {}   # qtype → [raw_rels, filtered_rels]
    subgraph_totals: Dict[str, List[int]] = {}  # qtype → [subgraphs, merged]

    # Golden coverage: how many target entities/relations survive each filter stage
    golden_entity_in = 0
    golden_entity_filtered = 0
    golden_relation_in = 0
    golden_relation_filtered = 0

    rows = []  # per-question detail

    for q in questions:
        qtype = q.get("query_type", "?")
        qid = q.get("id", "?")

        raw_seeds = len(q.get("extracted_nodes", []))
        fei = q.get("filtered_entity_info", {})
        filt_seeds = len(fei.get("filtered_seeds", []))

        raw_rels = len(q.get("extracted_relation_nodes", []))
        fri = q.get("filtered_relation_info", {})
        filt_rels = len(fri.get("filtered_relations", []))

        se = q.get("subgraph_extraction", {})
        seed_results = se.get("filtered_seeds", [])
        subgraphs = sum(s.get("subgraph_count", 0) for s in seed_results)

        fsi = q.get("filtered_subgraphs_info", {})
        merged = len(fsi.get("merged_graphs", []))

        # Golden entity coverage
        target_entities: List[str] = []
        for field in ("answer_entities", "target_entities"):
            val = q.get(field)
            if isinstance(val, list):
                target_entities = val
                break
        if target_entities:
            golden_entity_in += len(target_entities)
            extracted_uris = {n.get("uri", "") for n in q.get("extracted_nodes", [])}
            filtered_uris = {
                n.get("uri", "") for n in fei.get("filtered_seeds", [])
            }
            golden_entity_filtered += sum(1 for e in target_entities if e in filtered_uris)

        # Golden relation coverage
        target_relations: List[str] = []
        for field in ("answer_relations", "target_relations"):
            val = q.get(field)
            if isinstance(val, list):
                target_relations = val
                break
        if target_relations:
            golden_relation_in += len(target_relations)
            filtered_rel_uris = {
                r.get("uri", "") for r in fri.get("filtered_relations", [])
            }
            golden_relation_filtered += sum(1 for r in target_relations if r in filtered_rel_uris)

        # Accumulate per-type
        for d, raw, filt in [
            (seed_totals, raw_seeds, filt_seeds),
            (rel_totals, raw_rels, filt_rels),
            (subgraph_totals, subgraphs, merged),
        ]:
            if qtype not in seed_totals:
                seed_totals[qtype] = [0, 0]
            if qtype not in rel_totals:
                rel_totals[qtype] = [0, 0]
            if qtype not in subgraph_totals:
                subgraph_totals[qtype] = [0, 0]

        seed_totals[qtype][0] += raw_seeds
        seed_totals[qtype][1] += filt_seeds
        rel_totals[qtype][0] += raw_rels
        rel_totals[qtype][1] += filt_rels
        subgraph_totals[qtype][0] += subgraphs
        subgraph_totals[qtype][1] += merged

        rows.append((qid, qtype, raw_seeds, filt_seeds, raw_rels, filt_rels, subgraphs, merged))

    # Build markdown
    lines = [
        f"# Pipeline Stats Report",
        f"",
        f"Generated: {now}  ",
        f"Input: `{input_path}`  ",
        f"Output: `{output_path}`  ",
        f"Questions: **{total}**",
        f"",
        f"---",
        f"",
        f"## Summary",
        f"",
        f"| Metric | Count |",
        f"|--------|-------|",
        f"| Total questions | {total} |",
    ]

    total_raw_seeds = sum(v[0] for v in seed_totals.values())
    total_filt_seeds = sum(v[1] for v in seed_totals.values())
    total_raw_rels = sum(v[0] for v in rel_totals.values())
    total_filt_rels = sum(v[1] for v in rel_totals.values())
    total_subgraphs = sum(v[0] for v in subgraph_totals.values())
    total_merged = sum(v[1] for v in subgraph_totals.values())

    lines += [
        f"| Seeds extracted (raw) | {total_raw_seeds} |",
        f"| Seeds after filter | {total_filt_seeds} |",
        f"| Relations extracted (raw) | {total_raw_rels} |",
        f"| Relations after filter | {total_filt_rels} |",
        f"| Subgraphs total | {total_subgraphs} |",
        f"| Merged graphs total | {total_merged} |",
        f"",
    ]

    if golden_entity_in > 0:
        pct = 100 * golden_entity_filtered / golden_entity_in
        lines += [
            f"## Golden Entity Coverage",
            f"",
            f"Target entities surviving seed filter: **{golden_entity_filtered}/{golden_entity_in}** ({pct:.1f}%)",
            f"",
        ]

    if golden_relation_in > 0:
        pct = 100 * golden_relation_filtered / golden_relation_in
        lines += [
            f"## Golden Relation Coverage",
            f"",
            f"Target relations surviving relation filter: **{golden_relation_filtered}/{golden_relation_in}** ({pct:.1f}%)",
            f"",
        ]

    # Per-type breakdown
    lines += [
        f"## Per Query-Type Breakdown",
        f"",
        f"| Type | Questions | Seeds raw→filt | Rels raw→filt | Subgraphs→Merged |",
        f"|------|-----------|----------------|---------------|-----------------|",
    ]
    type_counts: Dict[str, int] = {}
    for q in questions:
        qt = q.get("query_type", "?")
        type_counts[qt] = type_counts.get(qt, 0) + 1

    for qtype in sorted(type_counts):
        n = type_counts[qtype]
        s = seed_totals.get(qtype, [0, 0])
        r = rel_totals.get(qtype, [0, 0])
        sg = subgraph_totals.get(qtype, [0, 0])
        lines.append(
            f"| {qtype} | {n} | {s[0]}→{s[1]} | {r[0]}→{r[1]} | {sg[0]}→{sg[1]} |"
        )

    # Per-question table
    lines += [
        f"",
        f"## Per-Question Detail",
        f"",
        f"| ID | Type | Seeds raw→filt | Rels raw→filt | Subgraphs→Merged |",
        f"|----|------|----------------|---------------|-----------------|",
    ]
    for qid, qtype, rs, fs, rr, fr, sg, mg in rows:
        lines.append(
            f"| {qid} | {qtype} | {rs}→{fs} | {rr}→{fr} | {sg}→{mg} |"
        )

    lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nReport written to {report_path}")


def _get_question_text(question: Dict[str, Any]) -> str:
    q = question.get("question", {})
    if isinstance(q, str):
        return q
    if isinstance(q, dict):
        return q.get("1") or q.get("2") or next(iter(q.values()), "")
    return str(q)


def _parse_args() -> argparse.Namespace:
    default_input  = os.path.join(os.path.dirname(__file__), "data", "balanced_200.jsonl")
    default_output = os.path.join(os.path.dirname(__file__), "data", "pipeline_fresh.jsonl")

    parser = argparse.ArgumentParser(
        description=(
            "Run the full evaluation pipeline (seed→filter→subgraph→merge) "
            "for all questions in a JSONL file.  Output can then be used with "
            "evaluate_sparql_scenarios.py for real-case SPARQL evaluation."
        )
    )
    parser.add_argument(
        "--input", "-i",
        default=default_input,
        help=f"Input JSONL file  (default: {default_input})",
    )
    parser.add_argument(
        "--output", "-o",
        default=default_output,
        help=f"Output JSONL file  (default: {default_output})",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Write a pipeline stats markdown report after completion",
    )
    parser.add_argument(
        "--report-out",
        default=None,
        metavar="PATH",
        help="Path for the report markdown. Defaults to eval/report/pipeline_stats_<timestamp>.md",
    )
    parser.add_argument(
        "--max-questions", "-n",
        type=int,
        default=None,
        help="Cap number of questions processed (useful for quick smoke tests)",
    )
    parser.add_argument(
        "--query-types", "-t",
        nargs="+",
        default=None,
        metavar="TYPE",
        help="Only process questions of these query types (e.g. BOOLEAN COUNT)",
    )
    parser.add_argument(
        "--sparql-report-dir",
        default=None,
        metavar="DIR",
        help="Directory for per-type SPARQL evaluation reports (default: eval/report/)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    print("=" * 70)
    print("FULL EVALUATION PIPELINE")
    print("=" * 70)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    if args.max_questions:
        print(f"Cap:    {args.max_questions} questions")
    if args.query_types:
        print(f"Types:  {args.query_types}")
    print("=" * 70 + "\n")

    report_path = None
    if args.report or args.report_out:
        if args.report_out:
            report_path = args.report_out
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(
                os.path.dirname(__file__), "report", f"pipeline_stats_{ts}.md"
            )

    run_full_pipeline(
        input_path=args.input,
        output_path=args.output,
        max_questions=args.max_questions,
        query_types=args.query_types,
        report_path=report_path,
        sparql_report_dir=args.sparql_report_dir,
    )
