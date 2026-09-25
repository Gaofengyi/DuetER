"""Budget-aware lexical candidate policy over the existing HMAC-FTS5 indexes.

This experiment isolates the policy and cap changes before constructing a
specialized Block-Max WAND file format.  FTS5 computes exact BM25 over the
selected opaque HMAC terms; cumulative document frequency replaces the fixed
"two rarest terms" rule.  A fixed primary operating point is separated from
diagnostic test-set sweeps.  Stored semantic rankings are reused to recompute
dual-path fusion without rerunning document encoding or semantic DPE.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import sqlite3
import time
from pathlib import Path

import numpy as np

from benchmark_million_semantic_hybrid import (
    LEXICAL_KEY,
    atomic_json,
    evaluate,
    load_queries_qrels,
    read_ids,
)
from run_experiment import tokenize


ROOT = Path(__file__).resolve().parent


def keyed_term(value: str) -> str:
    return hmac.new(LEXICAL_KEY, value.encode("utf-8"), hashlib.sha256).hexdigest()[:32]


def select_terms_under_budget(
    frequencies: dict[str, int],
    budget: int,
    alpha: float,
    document_count: int | None = None,
) -> list[str]:
    """Greedy benefit/cost selection with at least one matched term."""
    if budget < 1:
        raise ValueError("budget must be positive")
    if not frequencies:
        return []
    documents = max(document_count or max(frequencies.values()), 1)
    ranked = sorted(
        frequencies,
        key=lambda term: (
            -math.log1p(documents / max(frequencies[term], 1))
            / (frequencies[term] ** alpha),
            frequencies[term],
            term,
        ),
    )
    selected: list[str] = []
    spent = 0
    for term in ranked:
        cost = frequencies[term]
        if not selected or spent + cost <= budget:
            selected.append(term)
            spent += cost
    return selected


def weighted_rrf_pair(
    semantic: np.ndarray,
    lexical: np.ndarray,
    depth: int,
    semantic_weight: float,
    constant: int = 60,
) -> np.ndarray:
    scores: dict[int, float] = {}
    for weight, ranking in (
        (semantic_weight, semantic),
        (1.0 - semantic_weight, lexical),
    ):
        if weight <= 0.0:
            continue
        for position, value in enumerate(ranking):
            doc = int(value)
            if doc < 0:
                continue
            scores[doc] = scores.get(doc, 0.0) + weight / (constant + position + 1)
    ordered = sorted(scores, key=lambda doc: (-scores[doc], doc))[:depth]
    output = np.full(depth, -1, dtype=np.int32)
    output[: len(ordered)] = ordered
    return output


def fuse_all(
    semantic: np.ndarray,
    lexical: np.ndarray,
    depth: int,
    semantic_weight: float,
) -> np.ndarray:
    output = np.empty((len(semantic), depth), dtype=np.int32)
    for row in range(len(semantic)):
        output[row] = weighted_rrf_pair(
            semantic[row], lexical[row], depth, semantic_weight
        )
    return output


def candidate_coverage(
    rankings: np.ndarray,
    cap: int,
    doc_ids: list[str],
    query_ids: list[str],
    qrels: dict[str, dict[str, int]],
) -> float:
    values = []
    for row, qid in zip(rankings, query_ids):
        relevant = {doc_id for doc_id, gain in qrels[qid].items() if gain > 0}
        returned = {doc_ids[int(idx)] for idx in row[:cap] if int(idx) >= 0}
        values.append(len(relevant.intersection(returned)) / len(relevant))
    return float(np.mean(values))


def load_semantic_rankings(dataset: str, scale: str) -> tuple[np.ndarray, Path]:
    if scale == "million":
        path = ROOT / "results" / "million_semantic_hybrid" / f"{dataset}_1000000" / "rankings.npz"
    else:
        full_counts = {"msmarco": 8841823, "nq": 2681468, "hotpotqa": 5233329}
        path = ROOT / "results" / "full_semantic_hybrid" / f"{dataset}_{full_counts[dataset]}" / "rankings.npz"
    payload = np.load(path)
    return np.asarray(payload["semantic"], dtype=np.int32), path


def run(args: argparse.Namespace) -> dict[str, object]:
    suffix = "1000000" if args.scale == "million" else "full"
    max_docs = 1_000_000 if args.scale == "million" else {
        "msmarco": 8_841_823,
        "nq": 2_681_468,
        "hotpotqa": 5_233_329,
    }[args.dataset]
    cache_root = ROOT / "cache" / (
        "million_semantic" if args.scale == "million" else "full_semantic"
    )
    cache_dir = cache_root / f"{args.dataset}_{max_docs}"
    db_path = ROOT / "results" / "keyed_fts5" / f"{args.dataset}_{suffix}.sqlite3"
    query_ids, query_texts, qrels, _ = load_queries_qrels(
        ROOT / "data" / args.dataset, args.split
    )
    doc_ids = read_ids(cache_dir / "doc_ids.txt")
    if args.semantic_rankings is None:
        semantic_repeats, semantic_path = load_semantic_rankings(args.dataset, args.scale)
    else:
        semantic_path = args.semantic_rankings
        semantic_repeats = np.asarray(np.load(semantic_path)["semantic"], dtype=np.int32)
    if semantic_repeats.shape[1] != len(query_ids):
        raise RuntimeError("semantic ranking/query mismatch")

    budgets = sorted(set(args.posting_budgets))
    raw_path = args.output.with_suffix(".raw.npz")
    trace_path = args.output.with_suffix(".trace.json")
    if raw_path.exists() and trace_path.exists():
        raw = np.load(raw_path)
        rankings = {budget: np.asarray(raw[f"budget_{budget}"], dtype=np.int32) for budget in budgets}
        traces_raw = json.loads(trace_path.read_text(encoding="utf-8"))
        traces = {budget: traces_raw[str(budget)] for budget in budgets}
        if any(values.shape != (len(query_ids), args.max_cap) for values in rankings.values()):
            raise RuntimeError("raw lexical ranking cache shape mismatch")
        print(f"reused raw lexical rankings from {raw_path}", flush=True)
    else:
        rankings = {
            budget: np.full((len(query_ids), args.max_cap), -1, dtype=np.int32)
            for budget in budgets
        }
        traces = {
            budget: {"latency_ms": [], "terms": [], "estimated_postings": []}
            for budget in budgets
        }
        connection = sqlite3.connect(db_path)
        connection.execute("PRAGMA cache_size=-262144")
        connection.execute("CREATE VIRTUAL TABLE IF NOT EXISTS vocab USING fts5vocab(postings, 'row')")
        connection.commit()
        for qi, query in enumerate(query_texts):
            terms = list(dict.fromkeys(keyed_term(term) for term in tokenize(query)))
            if terms:
                placeholders = ",".join("?" for _ in terms)
                frequencies = {
                    str(term): int(df)
                    for term, df in connection.execute(
                        f"SELECT term, doc FROM vocab WHERE term IN ({placeholders})", terms
                    ).fetchall()
                }
            else:
                frequencies = {}
            expression_cache: dict[tuple[str, ...], tuple[np.ndarray, float]] = {}
            for budget in budgets:
                selected = select_terms_under_budget(
                    frequencies, budget, args.alpha, document_count=len(doc_ids)
                )
                key = tuple(sorted(selected))
                if key not in expression_cache:
                    if key:
                        expression = " OR ".join(key)
                        started = time.perf_counter()
                        rows = connection.execute(
                            "SELECT rowid FROM postings WHERE postings MATCH ? "
                            "ORDER BY bm25(postings) LIMIT ?",
                            (expression, args.max_cap),
                        ).fetchall()
                        elapsed = (time.perf_counter() - started) * 1000.0
                        values = np.asarray([int(row[0]) - 1 for row in rows], dtype=np.int32)
                    else:
                        values = np.empty(0, dtype=np.int32)
                        elapsed = 0.0
                    expression_cache[key] = values, elapsed
                values, elapsed = expression_cache[key]
                rankings[budget][qi, : len(values)] = values
                traces[budget]["latency_ms"].append(elapsed)
                traces[budget]["terms"].append(len(selected))
                traces[budget]["estimated_postings"].append(
                    sum(frequencies[term] for term in selected)
                )
            if qi == 0 or qi + 1 == len(query_ids) or (qi + 1) % 250 == 0:
                print(f"[{args.dataset}/{args.scale}] {qi + 1:,}/{len(query_ids):,}", flush=True)
        connection.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(raw_path, **{f"budget_{key}": value for key, value in rankings.items()})
        trace_path.write_text(
            json.dumps({str(key): value for key, value in traces.items()}), encoding="utf-8"
        )
        print(f"cached raw lexical rankings at {raw_path}", flush=True)

    semantic_metrics = [
        evaluate(repeat, doc_ids, query_ids, qrels) for repeat in semantic_repeats
    ]
    rows: list[dict[str, object]] = []
    saved_rankings: dict[str, np.ndarray] = {}
    for budget in budgets:
        lexical_top100 = rankings[budget][:, :100]
        fusion_by_weight = {}
        for weight in args.semantic_weights:
            per_repeat = [
                evaluate(
                    fuse_all(repeat, lexical_top100, 100, weight),
                    doc_ids,
                    query_ids,
                    qrels,
                )
                for repeat in semantic_repeats
            ]
            fusion_by_weight[str(weight)] = {
                metric: float(np.mean([item[metric] for item in per_repeat]))
                for metric in per_repeat[0]
            }
        for cap in args.caps:
            lexical = rankings[budget][:, :cap]
            lexical_metrics = evaluate(lexical[:, :100], doc_ids, query_ids, qrels)
            coverage = candidate_coverage(lexical, cap, doc_ids, query_ids, qrels)
            trace = traces[budget]
            row = {
                "posting_budget": budget,
                "candidate_cap": cap,
                "lexical_metrics": lexical_metrics,
                "relevant_candidate_coverage": coverage,
                "mean_selected_terms": float(np.mean(trace["terms"])),
                "mean_estimated_postings": float(np.mean(trace["estimated_postings"])),
                "latency_mean_ms": float(np.mean(trace["latency_ms"])),
                "latency_p95_ms": float(np.percentile(trace["latency_ms"], 95)),
                "hybrid_by_semantic_weight": fusion_by_weight,
            }
            rows.append(row)
            saved_rankings[f"budget_{budget}_cap_{cap}"] = lexical

    mean_semantic = {
        metric: float(np.mean([item[metric] for item in semantic_metrics]))
        for metric in semantic_metrics[0]
    }
    eligible = [
        row
        for row in rows
        if row["latency_p95_ms"] <= args.latency_gate_ms
        and row["relevant_candidate_coverage"] >= args.coverage_gate
        and max(
            values["nDCG@10"]
            for weight, values in row["hybrid_by_semantic_weight"].items()
            if float(weight) < 1.0
        ) >= 0.99 * mean_semantic["nDCG@10"]
    ]
    recommended = max(
        eligible if eligible else rows,
        key=lambda row: (
            max(v["nDCG@10"] for v in row["hybrid_by_semantic_weight"].values()),
            row["relevant_candidate_coverage"],
            -row["latency_p95_ms"],
        ),
    )
    best_weight = max(
        recommended["hybrid_by_semantic_weight"],
        key=lambda weight: recommended["hybrid_by_semantic_weight"][weight]["nDCG@10"],
    )
    primary = next(
        row
        for row in rows
        if row["posting_budget"] == args.primary_budget
        and row["candidate_cap"] == args.primary_cap
    )
    primary_weight = str(args.primary_semantic_weight)
    primary_hybrid = primary["hybrid_by_semantic_weight"][primary_weight]
    primary_gate_passed = (
        primary["latency_p95_ms"] <= args.latency_gate_ms
        and primary["relevant_candidate_coverage"] >= args.coverage_gate
        and primary_hybrid["nDCG@10"] >= mean_semantic["nDCG@10"]
    )
    report = {
        "dataset": args.dataset,
        "split": args.split,
        "scale": args.scale,
        "documents": len(doc_ids),
        "queries": len(query_ids),
        "variant": "budget-aware cumulative-DF HMAC-FTS5 exact-BM25 policy; policy oracle before specialized Block-Max WAND storage",
        "configuration": {
            "posting_budgets": budgets,
            "candidate_caps": args.caps,
            "semantic_weights": args.semantic_weights,
            "priority_alpha": args.alpha,
            "latency_gate_ms": args.latency_gate_ms,
            "coverage_gate": args.coverage_gate,
            "primary_posting_budget": args.primary_budget,
            "primary_candidate_cap": args.primary_cap,
            "primary_semantic_weight": args.primary_semantic_weight,
        },
        "semantic_ranking_source": str(semantic_path),
        "semantic_dpe_mean": mean_semantic,
        "selection_warning": "The recommended/best-weight fields are diagnostic test-set selection. The success claim uses only the fixed primary operating point.",
        "gate_passed": bool(primary_gate_passed),
        "primary": primary
        | {
            "semantic_weight": args.primary_semantic_weight,
            "hybrid_metrics": primary_hybrid,
        },
        "recommended": recommended | {"diagnostic_best_semantic_weight": best_weight},
        "sweep": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)
    np.savez_compressed(args.output.with_suffix(".rankings.npz"), **saved_rankings)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("msmarco", "nq", "hotpotqa"), required=True)
    parser.add_argument("--scale", choices=("million", "full"), default="million")
    parser.add_argument("--split", default="test")
    parser.add_argument("--semantic-rankings", type=Path, default=None)
    parser.add_argument("--posting-budgets", nargs="+", type=int, default=[10000, 50000, 250000])
    parser.add_argument("--caps", nargs="+", type=int, default=[200, 500, 1000])
    parser.add_argument("--max-cap", type=int, default=1000)
    parser.add_argument("--semantic-weights", nargs="+", type=float, default=[0.9, 0.95, 0.97, 0.98, 1.0])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--latency-gate-ms", type=float, default=100.0)
    parser.add_argument("--coverage-gate", type=float, default=0.75)
    parser.add_argument("--primary-budget", type=int, default=50000)
    parser.add_argument("--primary-cap", type=int, default=1000)
    parser.add_argument("--primary-semantic-weight", type=float, default=0.98)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if max(args.caps) > args.max_cap:
        raise ValueError("max-cap must cover every requested cap")
    if args.primary_budget not in args.posting_budgets:
        raise ValueError("primary-budget must be included in posting-budgets")
    if args.primary_cap not in args.caps:
        raise ValueError("primary-cap must be included in caps")
    if args.primary_semantic_weight not in args.semantic_weights:
        raise ValueError("primary-semantic-weight must be included in semantic-weights")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    result = run(parsed)
    print(json.dumps({"gate_passed": result["gate_passed"], "recommended": result["recommended"]}, indent=2))
