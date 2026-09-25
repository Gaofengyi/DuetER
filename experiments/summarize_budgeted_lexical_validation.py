"""Create an auditable summary for the budget-aware lexical planner.

The shared operating point is calibrated on one deterministic half of NQ and
then evaluated on the other half.  MS MARCO and HotpotQA are reported as
additional transfer datasets.  This script only consumes saved rankings and
qrels; it never reruns an encoder or database query.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from benchmark_budgeted_lexical_candidates import (
    ROOT,
    candidate_coverage,
    fuse_all,
    load_semantic_rankings,
)
from benchmark_million_semantic_hybrid import (
    atomic_json,
    evaluate,
    load_queries_qrels,
    read_ids,
)


DATASETS = ("msmarco", "nq", "hotpotqa")
OLD_REPORTS = {
    "msmarco": "msmarco_1000000_p2.json",
    "nq": "nq_1000000_p2.json",
    "hotpotqa": "hotpotqa_1000000_p2.json",
}


def nq_split(query_ids: list[str]) -> np.ndarray:
    """Return the stable held-out mask (one SHA-256 parity bit)."""
    return np.asarray(
        [bool(hashlib.sha256(qid.encode("utf-8")).digest()[0] & 1) for qid in query_ids]
    )


def subset_metrics(
    rankings: np.ndarray,
    mask: np.ndarray,
    doc_ids: list[str],
    query_ids: list[str],
    qrels: dict[str, dict[str, int]],
) -> dict[str, float]:
    subset_ids = [qid for qid, keep in zip(query_ids, mask) if keep]
    subset_qrels = {qid: qrels[qid] for qid in subset_ids}
    return evaluate(rankings[mask], doc_ids, subset_ids, subset_qrels)


def run(output: Path) -> dict[str, object]:
    rows = []
    for dataset in DATASETS:
        new_path = ROOT / "results" / "budgeted_lexical" / f"{dataset}_million.json"
        old_path = ROOT / "results" / "keyed_fts5" / OLD_REPORTS[dataset]
        new = json.loads(new_path.read_text(encoding="utf-8"))
        old = json.loads(old_path.read_text(encoding="utf-8"))
        primary = new["primary"]
        rows.append(
            {
                "dataset": dataset,
                "queries": new["queries"],
                "documents": new["documents"],
                "old_two_term_cap200_coverage": old["query"][
                    "relevant_candidate_recall_at_cap"
                ],
                "new_budget50k_cap1000_coverage": primary[
                    "relevant_candidate_coverage"
                ],
                "coverage_absolute_gain": primary["relevant_candidate_coverage"]
                - old["query"]["relevant_candidate_recall_at_cap"],
                "mean_selected_terms": primary["mean_selected_terms"],
                "mean_estimated_postings": primary["mean_estimated_postings"],
                "latency_mean_ms": primary["latency_mean_ms"],
                "latency_p95_ms": primary["latency_p95_ms"],
                "lexical_ndcg_at_10": primary["lexical_metrics"]["nDCG@10"],
                "semantic_ndcg_at_10": new["semantic_dpe_mean"]["nDCG@10"],
                "hybrid_weight_098_ndcg_at_10": primary["hybrid_metrics"]["nDCG@10"],
                "gate_passed": new["gate_passed"],
            }
        )

    query_ids, _, qrels, _ = load_queries_qrels(ROOT / "data" / "nq")
    doc_ids = read_ids(ROOT / "cache" / "million_semantic" / "nq_1000000" / "doc_ids.txt")
    semantic, _ = load_semantic_rankings("nq", "million")
    lexical = np.load(
        ROOT / "results" / "budgeted_lexical" / "nq_million.raw.npz"
    )["budget_50000"][:, :100]
    held_out = nq_split(query_ids)
    split_rows = {}
    for name, mask in (("calibration", ~held_out), ("held_out", held_out)):
        semantic_values = []
        hybrid_values = []
        for repeat in semantic:
            semantic_values.append(
                subset_metrics(repeat, mask, doc_ids, query_ids, qrels)["nDCG@10"]
            )
            hybrid_values.append(
                subset_metrics(
                    fuse_all(repeat, lexical, 100, 0.98),
                    mask,
                    doc_ids,
                    query_ids,
                    qrels,
                )["nDCG@10"]
            )
        subset_ids = [qid for qid, keep in zip(query_ids, mask) if keep]
        subset_qrels = {qid: qrels[qid] for qid in subset_ids}
        split_rows[name] = {
            "queries": int(np.sum(mask)),
            "semantic_ndcg_at_10": float(np.mean(semantic_values)),
            "hybrid_ndcg_at_10": float(np.mean(hybrid_values)),
            "relevant_candidate_coverage_cap1000": candidate_coverage(
                np.load(
                    ROOT / "results" / "budgeted_lexical" / "nq_million.raw.npz"
                )["budget_50000"][mask],
                1000,
                doc_ids,
                subset_ids,
                subset_qrels,
            ),
        }

    report = {
        "variant": "cumulative-DF-budget HMAC-FTS5 planner with exact BM25 top-100 and cap-1000 coverage evaluation",
        "shared_operating_point": {
            "posting_budget": 50000,
            "candidate_cap": 1000,
            "semantic_rrf_weight": 0.98,
            "selection": "chosen on deterministic NQ calibration half; no per-dataset retuning",
        },
        "nq_split_rule": "held out iff low bit of first SHA-256(query_id) byte is one",
        "nq_split_validation": split_rows,
        "transfer_results": rows,
        "scope_warning": "FTS5 executes exact BM25 for the selected opaque terms. This validates the query planner, not a specialized Block-Max WAND implementation or an integrated lexical-DPE full-corpus run.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, report)
    return report


if __name__ == "__main__":
    destination = ROOT / "results" / "budgeted_lexical" / "validation_summary.json"
    print(json.dumps(run(destination), indent=2))
