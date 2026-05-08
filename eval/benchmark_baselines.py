"""Multi-level SPARQL generation baselines for DBLP-QuAD.

Levels:
  0 — Raw LLM          : question + schema only (no entities)
  1 — Oracle Entities  : question + golden entity URIs, no schema (tests memorized KG knowledge)
  2 — Oracle + Schema  : question + golden entity URIs + schema (pure SPARQL reasoning)
  3 — Pipeline + Schema: question + pipeline-extracted URIs + schema (real-world bridge)

Usage:
  python bb.py --level 0 1 2 3 --questions data/test.jsonl --schema ../_data/schema.nt --out-prefix data/baseline
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
import math
from pathlib import Path
import time
from typing import Any, Dict, Dict, List, Optional, Tuple, Tuple

import matplotlib.pyplot as plt
import numpy as np

import dspy
from SPARQLWrapper import JSON, SPARQLWrapper

from evaluate_sparql_and_answer_quality import (
    evaluate_sparql_and_answer_quality,
    _evaluate_question,
    _write_summary_md,
    SPARQLJudgeSignature,
    JUDGE_INSTRUCTION,
)
from kgnode.core.kg_config import KGConfig
from kgnode.seed_finder import SearchMode, get_seed_nodes

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DSPy Signatures — one per level
# ─────────────────────────────────────────────────────────────────────────────

class L0_Signature(dspy.Signature):
    """You are an expert SPARQL engineer for knowledge graph.

    Given the natural language question, its query type, and the full RDF schema
    (N-Triples format), infer the relevant entity URIs, construct valid SPARQL 1.1,
    and return the raw query.

    DBLP entity URIs follow these patterns:
      • Authors  : https://dblp.org/pid/<path>   (e.g., https://dblp.org/pid/h/AlainHertz)
      • Papers   : https://dblp.org/rec/<path>   (e.g., https://dblp.org/rec/journals/dm/Hertz86)

    ══ QUERY TYPE → SPARQL FORM (MANDATORY) ══
      BOOLEAN  → MUST use ASK { … }              ← yes/no existence check, returns true/false
      COUNT    → MUST use SELECT (COUNT(?x) AS ?answer) WHERE { … }
      otherwise→ MUST use SELECT DISTINCT ?answer WHERE { … }

    If query_type is BOOLEAN you MUST start the query with ASK — never SELECT.

    Additional rules:
      • Do NOT hallucinate predicates — use only predicates found in schema_context.
      • Do NOT hallucinate entity URIs — infer them from the question and anchor them literally in SPARQL.
      • Do NOT put placeholders, the sparql must be executable as-is with no post-processing.
      • No markdown code fences — raw SPARQL only.
      • All URIs in full angle-bracket form <https://…> (no PREFIX shortcuts).
      • Time-relative filters: FILTER(?y > YEAR(NOW())-N)

    EXAMPLES
    --------
    query_type: SELECT
    Question: What papers did Wazir Muhammad author?
    SPARQL:
    SELECT DISTINCT ?answer WHERE {
      ?answer <https://dblp.org/rdf/schema#authoredBy> <https://dblp.org/pid/211/3355>
    }

    query_type: BOOLEAN
    Question: Did Wanlei Zhou and Morshed U. Chowdhury co-author 'MVGL Analyser…'?
    SPARQL:
    ASK {
      <https://dblp.org/rec/conf/ACISicis/IslamZC09> <https://dblp.org/rdf/schema#authoredBy> <https://dblp.org/pid/92/2939> .
      <https://dblp.org/rec/conf/ACISicis/IslamZC09> <https://dblp.org/rdf/schema#authoredBy> <https://dblp.org/pid/74/4874>
    }
    """

    question: str = dspy.InputField(desc="Natural language question")
    query_type: str = dspy.InputField(
        desc="Expected SPARQL form: BOOLEAN → must use ASK, COUNT → must use COUNT aggregate, otherwise SELECT DISTINCT"
    )
    schema_context: str = dspy.InputField(desc="schema in N-Triples format — full URIs only")
    sparql: str = dspy.OutputField(desc="Valid SPARQL 1.1 query, no markdown fences, raw text")


class L1_Signature(dspy.Signature):
    """You are an expert SPARQL engineer. The knowledge graph is DBLP (https://dblp.org).

    You are given a question, its query type, and the exact KG entity URIs.
    Rely on your knowledge of the DBLP RDF schema — you do NOT receive the schema file.


    ══ QUERY TYPE → SPARQL FORM (MANDATORY) ══
      BOOLEAN  → MUST use ASK { … }              ← returns true/false, never SELECT
      COUNT    → MUST use SELECT (COUNT(?x) AS ?answer) WHERE { … }
      otherwise→ MUST use SELECT DISTINCT ?answer WHERE { … }

    Additional rules:
      • Anchor every entity_uri literally.
      • No markdown fences — raw SPARQL only. All URIs in full angle-bracket form.
      • Time-relative filters: FILTER(?y > YEAR(NOW())-N)

    EXAMPLES
    --------
    query_type: SELECT
    Question: What papers did Wazir Muhammad author?
    Entity URIs: <https://dblp.org/pid/211/3355>
    SPARQL:
    SELECT DISTINCT ?answer WHERE {
      ?answer <https://dblp.org/rdf/schema#authoredBy> <https://dblp.org/pid/211/3355>
    }

    query_type: BOOLEAN
    Question: Did Wanlei Zhou and Morshed U. Chowdhury co-author 'MVGL Analyser…'?
    Entity URIs: <https://dblp.org/pid/92/2939>, <https://dblp.org/pid/74/4874>, <https://dblp.org/rec/conf/ACISicis/IslamZC09>
    SPARQL:
    ASK {
      <https://dblp.org/rec/conf/ACISicis/IslamZC09> <https://dblp.org/rdf/schema#authoredBy> <https://dblp.org/pid/92/2939> .
      <https://dblp.org/rec/conf/ACISicis/IslamZC09> <https://dblp.org/rdf/schema#authoredBy> <https://dblp.org/pid/74/4874>
    }
    """

    question: str = dspy.InputField(desc="Natural language question about DBLP")
    query_type: str = dspy.InputField(
        desc="Expected SPARQL form: BOOLEAN → must use ASK, COUNT → must use COUNT aggregate, otherwise SELECT DISTINCT"
    )
    entity_uris: str = dspy.InputField(desc="Comma-separated list of relevant KG entity URIs")
    sparql: str = dspy.OutputField(desc="Valid SPARQL 1.1 query, no markdown fences, raw text")


class L2_Signature(dspy.Signature):
    """You are an expert SPARQL engineer. Entity linking is already solved — your only job is
    to write correct SPARQL logic.

    You receive the question, its query type, the exact golden entity URIs, and the DBLP
    RDF schema for predicate lookup.

    ══ QUERY TYPE → SPARQL FORM (MANDATORY) ══
      BOOLEAN  → MUST use ASK { … }              ← returns true/false, never SELECT
      COUNT    → MUST use SELECT (COUNT(?x) AS ?answer) WHERE { … }
      otherwise→ MUST use SELECT DISTINCT ?answer WHERE { … }

    Additional rules:
      • Use ALL provided entity URIs as explicit anchors.
      • Do NOT invent predicates — use only what appears in schema_context.
      • No markdown fences — raw SPARQL only. All URIs in full angle-bracket form.
      • Time-relative filters: FILTER(?y > YEAR(NOW())-N)

    EXAMPLES
    --------
    query_type: SELECT
    Question: Which papers did Dan O. Popa publish in the last 9 years?
    Entity URIs: <https://dblp.org/pid/61/5165>
    SPARQL:
    SELECT DISTINCT ?answer WHERE {
      ?answer <https://dblp.org/rdf/schema#authoredBy> <https://dblp.org/pid/61/5165> .
      ?answer <https://dblp.org/rdf/schema#yearOfPublication> ?y .
      FILTER(?y > YEAR(NOW())-9)
    }

    query_type: BOOLEAN
    Question: Did Wanlei Zhou and Morshed U. Chowdhury co-author 'MVGL Analyser…'?
    Entity URIs: <https://dblp.org/pid/92/2939>, <https://dblp.org/pid/74/4874>, <https://dblp.org/rec/conf/ACISicis/IslamZC09>
    SPARQL:
    ASK {
      <https://dblp.org/rec/conf/ACISicis/IslamZC09> <https://dblp.org/rdf/schema#authoredBy> <https://dblp.org/pid/92/2939> .
      <https://dblp.org/rec/conf/ACISicis/IslamZC09> <https://dblp.org/rdf/schema#authoredBy> <https://dblp.org/pid/74/4874>
    }

    """

    question: str = dspy.InputField(desc="Natural language question")
    query_type: str = dspy.InputField(
        desc="Expected SPARQL form: BOOLEAN → must use ASK, COUNT → must use COUNT aggregate, otherwise SELECT DISTINCT"
    )
    entity_uris: str = dspy.InputField(desc="Golden entity URIs — use these as exact anchors")
    schema_context: str = dspy.InputField(desc="schema in N-Triples format — predicate reference")
    sparql: str = dspy.OutputField(desc="Valid SPARQL 1.1 query, no markdown fences, raw text")


class L3_Signature(dspy.Signature):
    """You are an expert SPARQL engineer for knowledge graph.

    Entity linking was performed automatically by a retrieval pipeline — URIs may be
    imperfect (some correct, some wrong or missing). Your job is to write the best
    possible SPARQL using the provided URIs, schema, and the declared query type.

    ══ QUERY TYPE → SPARQL FORM (MANDATORY) ══
      BOOLEAN  → MUST use ASK { … }              ← returns true/false, never SELECT
      COUNT    → MUST use SELECT (COUNT(?x) AS ?answer) WHERE { … }
      otherwise→ MUST use SELECT DISTINCT ?answer WHERE { … }

    Strategy for URIs:
      • Trust URIs that clearly match named entities in the question.
      • If no URI matches a required entity, use a variable with
        FILTER(CONTAINS(LCASE(?label), "name")) on the label predicate.
      • Use only predicates found in schema_context.
      • No markdown fences — raw SPARQL only. All URIs in full angle-bracket form.
      • Time-relative filters: FILTER(?y > YEAR(NOW())-N)

    EXAMPLES
    --------
    query_type: SELECT
    Question: What papers did Wazir Muhammad author?
    Pipeline URIs: <https://dblp.org/pid/211/3355>  [Wazir Muhammad]
    SPARQL:
    SELECT DISTINCT ?answer WHERE {
      ?answer <https://dblp.org/rdf/schema#authoredBy> <https://dblp.org/pid/211/3355>
    }

    query_type: SELECT
    Question: Which papers did Dan O. Popa publish in the last 9 years?
    Entity URIs: <https://dblp.org/pid/61/5165>
    SPARQL:
    SELECT DISTINCT ?answer WHERE {
      ?answer <https://dblp.org/rdf/schema#authoredBy> <https://dblp.org/pid/61/5165> .
      ?answer <https://dblp.org/rdf/schema#yearOfPublication> ?y .
      FILTER(?y > YEAR(NOW())-9)
    }
    """

    question: str = dspy.InputField(desc="Natural language question")
    query_type: str = dspy.InputField(
        desc="Expected SPARQL form: BOOLEAN → must use ASK, COUNT → must use COUNT aggregate, otherwise SELECT DISTINCT"
    )
    pipeline_uris: str = dspy.InputField(
        desc="Pipeline-extracted entity URIs with labels: '<uri>  [label]' one per line"
    )
    schema_context: str = dspy.InputField(desc="schema in N-Triples format — predicate reference")
    sparql: str = dspy.OutputField(desc="Valid SPARQL 1.1 query, no markdown fences, raw text")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return items
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, list) else [obj]
    except json.JSONDecodeError:
        pass
    for line in raw.splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def load_schema(path: str, max_chars: int = 60_000) -> str:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    if len(text) > max_chars:
        text = text[:max_chars] + "\n# [TRUNCATED]\n"
    return text


def get_question_text(item: dict[str, Any]) -> str:
    q = item.get("question", "")
    if isinstance(q, dict):
        return q.get("1") or q.get("2") or next(iter(q.values()), "")
    return str(q)


def strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        end = -1 if lines[-1].strip() == "```" else len(lines)
        text = "\n".join(lines[1:end]).strip()
    return text


_sparql_endpoint: str | None = None

def _get_endpoint() -> str:
    global _sparql_endpoint
    if _sparql_endpoint is None:
        _sparql_endpoint = KGConfig.default().sparql_endpoint
    return _sparql_endpoint


def run_sparql(sparql: str) -> tuple[str, Any, str | None]:

    try:
        wrapper = SPARQLWrapper(_get_endpoint())
        wrapper.setQuery(sparql)
        wrapper.setReturnFormat(JSON)
        raw = wrapper.query().convert()
        if "boolean" in raw:
            results = [{"boolean": raw["boolean"]}]
        elif "results" in raw:
            results = [
                {var: row[var]["value"] for var in row}
                for row in raw["results"]["bindings"]
            ]
        else:
            results = []
        return "ok", results, None
    except Exception as e:
        return "error", None, f"{type(e).__name__}: {e}"


def record(item: dict, question: str, sparql: str, level: int) -> dict:
    status, answer, error = run_sparql(sparql)
    # Flat fields for human readability
    out: dict[str, Any] = {
        "id": item.get("id"),
        "level": level,
        "query_type": item.get("query_type"),
        "question": question,
        "golden_sparql": item.get("golden_sparql"),
        "answer": item.get("answer"),          # golden answer — evaluator reads this key
        "golden_entities": item.get("golden_entities", []),
        "golden_relations": item.get("golden_relations", []),
        "execution_status": status,
        # Nested block that evaluate_sparql_and_answer_quality expects
        "generated_sparql_info": {
            "generated_sparql": sparql if status == "ok" else None,
            "sparql_results": answer,
            "execution_error": error,
        },
    }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Per-level runners
# ─────────────────────────────────────────────────────────────────────────────

def run_level0(
    items: list[dict],
    schema: str,
    out_path: str = "",
    eval_path: str = "",
    judge_module: Any = None,
) -> list[dict]:
    """Level 0 — Raw LLM: question + full schema, no entities."""
    gen = dspy.Predict(L0_Signature)
    done_ids = _load_done_ids(out_path) if out_path else set()
    if done_ids:
        log.info("L0 resuming — skipping %d already-done items", len(done_ids))
    accumulated = _load_jsonl(out_path) if (eval_path and out_path and Path(out_path).exists()) else []
    t0 = time.time()
    results = []
    for i, item in enumerate(items):
        if str(item.get("id")) in done_ids:
            continue
        q = get_question_text(item)
        qt = item.get("query_type", "SELECT")
        log.info("L0 [%d/%d] %s (%s)", i + 1, len(items), item.get("id"), qt)
        item_t0 = time.time()
        out = gen(question=q, query_type=qt, schema_context=schema)
        sparql = strip_fences(out.sparql)
        r = record(item, q, sparql, level=0)
        r["time_s"] = round(time.time() - item_t0, 3)
        if eval_path and judge_module is not None:
            r["evaluation_info"] = _evaluate_question(r, judge_module)
        results.append(r)
        if out_path:
            _append_jsonl(out_path, r)
        if eval_path and judge_module is not None:
            accumulated.append(r)
            try:
                _write_summary_md(accumulated, eval_path, time.time() - t0)
            except Exception as exc:
                log.warning("Incremental summary failed: %s", exc)
    return results


def run_level1(
    items: list[dict],
    out_path: str = "",
    eval_path: str = "",
    judge_module: Any = None,
) -> list[dict]:
    """Level 1 — Oracle entities, no schema: tests memorized KG knowledge."""
    gen = dspy.Predict(L1_Signature)
    done_ids = _load_done_ids(out_path) if out_path else set()
    if done_ids:
        log.info("L1 resuming — skipping %d already-done items", len(done_ids))
    accumulated = _load_jsonl(out_path) if (eval_path and out_path and Path(out_path).exists()) else []
    t0 = time.time()
    results = []
    for i, item in enumerate(items):
        if str(item.get("id")) in done_ids:
            continue
        q = get_question_text(item)
        qt = item.get("query_type", "SELECT")
        entity_uris = ", ".join(item.get("golden_entities", []))
        log.info("L1 [%d/%d] %s (%s)", i + 1, len(items), item.get("id"), qt)
        item_t0 = time.time()
        out = gen(question=q, query_type=qt, entity_uris=entity_uris)
        sparql = strip_fences(out.sparql)
        r = record(item, q, sparql, level=1)
        r["time_s"] = round(time.time() - item_t0, 3)
        if eval_path and judge_module is not None:
            r["evaluation_info"] = _evaluate_question(r, judge_module)
        results.append(r)
        if out_path:
            _append_jsonl(out_path, r)
        if eval_path and judge_module is not None:
            accumulated.append(r)
            try:
                _write_summary_md(accumulated, eval_path, time.time() - t0)
            except Exception as exc:
                log.warning("Incremental summary failed: %s", exc)
    return results


def run_level2(
    items: list[dict],
    schema: str,
    out_path: str = "",
    eval_path: str = "",
    judge_module: Any = None,
) -> list[dict]:
    """Level 2 — Oracle entities + schema: pure SPARQL reasoning test."""
    gen = dspy.Predict(L2_Signature)
    done_ids = _load_done_ids(out_path) if out_path else set()
    if done_ids:
        log.info("L2 resuming — skipping %d already-done items", len(done_ids))
    accumulated = _load_jsonl(out_path) if (eval_path and out_path and Path(out_path).exists()) else []
    t0 = time.time()
    results = []
    for i, item in enumerate(items):
        if str(item.get("id")) in done_ids:
            continue
        q = get_question_text(item)
        qt = item.get("query_type", "SELECT")
        entity_uris = ", ".join(item.get("golden_entities", []))
        log.info("L2 [%d/%d] %s (%s)", i + 1, len(items), item.get("id"), qt)
        item_t0 = time.time()
        out = gen(question=q, query_type=qt, entity_uris=entity_uris, schema_context=schema)
        sparql = strip_fences(out.sparql)
        r = record(item, q, sparql, level=2)
        r["time_s"] = round(time.time() - item_t0, 3)
        if eval_path and judge_module is not None:
            r["evaluation_info"] = _evaluate_question(r, judge_module)
        results.append(r)
        if out_path:
            _append_jsonl(out_path, r)
        if eval_path and judge_module is not None:
            accumulated.append(r)
            try:
                _write_summary_md(accumulated, eval_path, time.time() - t0)
            except Exception as exc:
                log.warning("Incremental summary failed: %s", exc)
    return results


def run_level3(
    items: list[dict],
    schema: str,
    config: KGConfig,
    out_path: str = "",
    eval_path: str = "",
    judge_module: Any = None,
) -> list[dict]:
    """Level 3 — Pipeline URIs + schema: bridges oracle and real pipeline."""
    gen = dspy.Predict(L3_Signature)
    done_ids = _load_done_ids(out_path) if out_path else set()
    if done_ids:
        log.info("L3 resuming — skipping %d already-done items", len(done_ids))
    accumulated = _load_jsonl(out_path) if (eval_path and out_path and Path(out_path).exists()) else []
    t0 = time.time()
    results = []
    for i, item in enumerate(items):
        if str(item.get("id")) in done_ids:
            continue
        q = get_question_text(item)
        qt = item.get("query_type", "SELECT")
        log.info("L3 [%d/%d] %s (%s) — running seed finder…", i + 1, len(items), item.get("id"), qt)

        # Run the real pipeline seed finder
        try:
            seed_nodes, _ = get_seed_nodes(
                query=q,
                n_results=3,
                config=config,
                search_mode=SearchMode.semantic,
            )
            pipeline_uris_str = "\n".join(
                f"<{s['entity_uri']}>  [{s.get('label', '')}]"
                for s in seed_nodes
            ) or "(no entities found)"
        except Exception as e:
            log.warning("Seed finder failed for %s: %s", item.get("id"), e)
            pipeline_uris_str = "(seed finder error)"

        item_t0 = time.time()
        out = gen(question=q, query_type=qt, pipeline_uris=pipeline_uris_str, schema_context=schema)
        sparql = strip_fences(out.sparql)
        r = record(item, q, sparql, level=3)
        r["time_s"] = round(time.time() - item_t0, 3)
        r["pipeline_uris"] = pipeline_uris_str
        if eval_path and judge_module is not None:
            r["evaluation_info"] = _evaluate_question(r, judge_module)
        results.append(r)
        if out_path:
            _append_jsonl(out_path, r)
        if eval_path and judge_module is not None:
            accumulated.append(r)
            try:
                _write_summary_md(accumulated, eval_path, time.time() - t0)
            except Exception as exc:
                log.warning("Incremental summary failed: %s", exc)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Summary printer
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(level: int, results: list[dict]) -> None:
    total = len(results)
    syntax_errors = sum(1 for r in results if r["execution_status"] == "error")
    ok = total - syntax_errors
    sparql_results = [r["generated_sparql_info"].get("sparql_results") for r in results if r["execution_status"] == "ok"]
    has_results = sum(1 for sr in sparql_results if sr)
    empty_logic = ok - has_results  # valid SPARQL but no matching rows = logic/entity failure
    log.info(
        "Level %d | total=%d  syntax_ok=%d  syntax_error=%d  "
        "has_results=%d  logic_empty=%d (valid SPARQL but no rows returned)",
        level, total, ok, syntax_errors, has_results, empty_logic,
    )


def write_jsonl(path: str, records: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("Wrote %d records → %s", len(records), path)


def _append_jsonl(path: str, record: dict) -> None:

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_done_ids(path: str) -> set[str]:

    done: set[str] = set()
    p = Path(path)
    if not p.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("id") is not None:
                    done.add(str(obj["id"]))
            except json.JSONDecodeError:
                pass
    return done
    

def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _get_is_correct(item: Dict[str, Any]) -> Optional[bool]:
    return (
        item.get("evaluation_info", {})
            .get("answer_metrics", {})
            .get("is_correct", None)
    )


def _get_time_seconds(item: Dict[str, Any]) -> float:

    for k in ("time_s", "time_sec", "elapsed_s", "elapsed_sec", "duration_s", "z_time"):
        v = item.get(k, None)
        if isinstance(v, (int, float)):
            return float(v)

    for path in (
        ("generated_sparql_info", "time_s"),
        ("generated_sparql_info", "elapsed_s"),
        ("evaluation_info", "time_s"),
        ("evaluation_info", "elapsed_s"),
    ):
        cur = item
        ok = True
        for p in path:
            if not isinstance(cur, dict) or p not in cur:
                ok = False
                break
            cur = cur[p]
        if ok and isinstance(cur, (int, float)):
            return float(cur)

    return 0.0


def plot_visualizations(jsonl_path: str, img_path: str) -> None:
    items = _load_jsonl(jsonl_path)
    if not items:
        raise ValueError(f"No items found in {jsonl_path}")

    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for it in items:
        qtype = str(it.get("query_type", "UNKNOWN"))
        by_type[qtype].append(it)

    total = len(items)
    total_success = 0
    total_fail = 0
    total_time_s = 0.0

    stats: List[Tuple[str, int, int, int]] = []  # (qtype, n, success, fail)
    for qtype in sorted(by_type.keys()):
        group = by_type[qtype]
        n = len(group)
        succ = 0
        fail = 0
        for it in group:
            is_corr = _get_is_correct(it)
            if is_corr is True:
                succ += 1
            else:
                fail += 1
            total_time_s += _get_time_seconds(it)
        stats.append((qtype, n, succ, fail))
        total_success += succ
        total_fail += fail

    # Layout
    n_types = len(stats)
    cols = min(4, n_types)  
    rows = math.ceil(n_types / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows))
    if n_types == 1:
        axes = [axes]
    else:
        axes = list(axes.ravel())

    level = items[0].get("level", None)
    title = "Visualization of success and failure per question type"
    if level is not None:
        title += f" — Level {level}"

    total_success_pct = (total_success / total * 100.0) if total else 0.0
    total_fail_pct = (total_fail / total * 100.0) if total else 0.0
    subtitle = (
        f"Total: {total} | "
        f"{total_success_pct:.1f}% success ({total_success}) | "
        f"{total_fail_pct:.1f}% failure ({total_fail}) | "
        f"Total time: {total_time_s:.2f}s"
    )

    fig.suptitle(title + "\n" + subtitle, fontsize=12, y=0.98)

    for ax_i, (qtype, n, succ, fail) in enumerate(stats):
        ax = axes[ax_i]

        # Percentages for labels
        succ_pct = (succ / n * 100.0) if n else 0.0
        fail_pct = (fail / n * 100.0) if n else 0.0

        values = [succ, fail]
        labels = [f"{succ_pct:.0f}% success", f"{fail_pct:.0f}% failure"]

        ax.pie(values, labels=labels, startangle=90, textprops={"fontsize": 10})
        ax.set_title(f"Question Type: {qtype}\n(n={n})", fontsize=11)

    for j in range(n_types, len(axes)):
        axes[j].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(img_path, dpi=200)
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SPARQL baseline evaluations (L0–L3)")
    p.add_argument("--level", nargs="+", type=int, default=[0],
                   choices=[0, 1, 2, 3], metavar="N",
                   help="Baseline levels to run (default: all)")
    p.add_argument("--questions", default="data/test.jsonl", help="Path to test JSONL")
    p.add_argument("--schema", default="../_data/schema.nt", help="Path to schema.nt")
    p.add_argument("--out-prefix", default="data/baseline",
                   help="Output path prefix; files are <prefix>_L{N}.jsonl")
    p.add_argument("--model", default=None,
                   help="Override LLM model (default: from KGConfig)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    config = KGConfig.default()

    if not config.lm_api_key:
        raise ValueError("No LLM API key — set OPENAI_API_KEY or KGNODE_LM_API_KEY")

    model = args.model or config.openai_model
    dspy.configure(lm=dspy.LM(model=model, api_key=config.lm_api_key))

    sig = SPARQLJudgeSignature.with_instructions(JUDGE_INSTRUCTION)
    judge_module = dspy.ChainOfThought(sig)

    levels = sorted(set(args.level))
    items = load_jsonl(args.questions)
    log.info("Loaded %d questions from %s", len(items), args.questions)

    # Only load schema when needed
    needs_schema = any(lv in levels for lv in (0, 2, 3))
    schema = load_schema(args.schema) if needs_schema else ""
    if schema:
        log.info("Schema loaded: %d chars", len(schema))

    for lv in levels:
        log.info("═══ Running Level %d ═══", lv)
        out_path = f"{args.out_prefix}_L{lv}.jsonl"
        eval_md = f"{args.out_prefix}_L{lv}_eval.md"
        if lv == 0:
            _ = run_level0(items, schema, out_path, eval_md, judge_module)
        elif lv == 1:
            _ = run_level1(items, out_path, eval_md, judge_module)
        elif lv == 2:
            _ = run_level2(items, schema, out_path, eval_md, judge_module)
        else:
            _ = run_level3(items, schema, config, out_path, eval_md, judge_module)

        all_records = _load_jsonl(out_path)
        log.info("Level %d complete — %d total records in %s", lv, len(all_records), out_path)
        print_summary(lv, all_records)

        # Plot visualization for the resulting JSONs
        plot_visualizations(out_path, f"{args.out_prefix}_L{lv}_vis.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
    # plot_visualizations(f"data/baseline_L2.jsonl", f"data/output/baseline_L2_vis.png")



# python benchmark_baselines.py   --questions data/balanced_100_initial.jsonl   --schema ../_data/schema.nt   --out-prefix data/outputs/baseline

