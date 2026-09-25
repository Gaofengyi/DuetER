"""Consolidate final held-out DuetRank and budgeted lexical-DPE results."""

from __future__ import annotations

import json

import numpy as np

from benchmark_budgeted_lexical_candidates import ROOT, fuse_all, load_semantic_rankings
from benchmark_million_semantic_hybrid import atomic_json, load_queries_qrels, read_ids
from experiment_adaptive_fusion import ndcg_rows
from experiment_candidate_rank_fusion import paired_bootstrap, split_mask


def learned(dataset: str) -> dict[str, object]:
    payload = json.loads(
        (
            ROOT
            / "results"
            / "budgeted_lexical_dpe"
            / f"learned_fusion_{dataset}.json"
        ).read_text(encoding="utf-8")
    )["datasets"][0]
    dpe = json.loads(
        (
            ROOT
            / "results"
            / "budgeted_lexical_dpe"
            / f"{dataset}_d200.json"
        ).read_text(encoding="utf-8")
    )
    held = payload["held_out"]
    return {
        "dataset": dataset,
        "queries_calibration": payload["calibration"]["queries"],
        "queries_held_out": held["queries"],
        "semantic_ndcg_at_10": held["semantic_ndcg_at_10"],
        "selected_fixed_rrf_semantic_weight": payload["selection"][
            "selected_fixed_rrf_semantic_weight"
        ],
        "calibrated_fixed_rrf_ndcg_at_10": held[
            "calibrated_fixed_rrf_ndcg_at_10"
        ],
        "duetrank_ndcg_at_10": held["semantic_backoff_ndcg_at_10"],
        "duetrank_vs_semantic": held["backoff_vs_semantic"],
        "duetrank_vs_calibrated_fixed_rrf": held[
            "backoff_vs_calibrated_fixed_rrf"
        ],
        "model_bytes": payload["selection"]["model_bytes"],
        "client_inference_ms": payload["selection"][
            "client_inference_mean_ms_per_query"
        ],
        "lexical_plaintext_ndcg_at_10": dpe["lexical_plaintext_metrics"]["nDCG@10"],
        "lexical_dpe_ndcg_at_10": dpe["lexical_dpe_mean"]["nDCG@10"],
        "lexical_dpe_online_mean_ms_warm": dpe["online_mean_ms"],
        "lexical_dpe_online_p95_ms_warm": dpe["online_p95_ms"],
    }


def msmarco_fixed() -> dict[str, object]:
    dataset = "msmarco"
    query_ids, _, qrels, _ = load_queries_qrels(ROOT / "data" / dataset)
    doc_ids = read_ids(
        ROOT / "cache" / "million_semantic" / "msmarco_1000000" / "doc_ids.txt"
    )
    doc_to_index = {doc_id: index for index, doc_id in enumerate(doc_ids)}
    semantic, _ = load_semantic_rankings(dataset, "million")
    lexical = np.load(
        ROOT
        / "results"
        / "budgeted_lexical_dpe"
        / "msmarco_d200.rankings.npz"
    )["lexical_dpe"]
    held_out = split_mask(query_ids, "outer")
    weights = (0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 1.0)
    semantic_rows = np.asarray(
        [ndcg_rows(row, doc_to_index, query_ids, qrels) for row in semantic]
    )
    by_weight = {
        weight: np.asarray(
            [
                ndcg_rows(
                    fuse_all(semantic[repeat], lexical[repeat], 100, weight),
                    doc_to_index,
                    query_ids,
                    qrels,
                )
                for repeat in range(len(semantic))
            ]
        )
        for weight in weights
    }
    selected = max(
        weights, key=lambda weight: float(np.mean(by_weight[weight][:, ~held_out]))
    )
    system = by_weight[selected]
    return {
        "dataset": dataset,
        "queries_calibration": int(np.sum(~held_out)),
        "queries_held_out": int(np.sum(held_out)),
        "selected_fixed_semantic_weight": selected,
        "semantic_ndcg_at_10": float(np.mean(semantic_rows[:, held_out])),
        "fixed_fusion_ndcg_at_10": float(np.mean(system[:, held_out])),
        "fixed_fusion_vs_semantic": paired_bootstrap(
            semantic_rows, system, held_out
        ),
        "warning": "Only 19 held-out queries; this is a small-sample diagnostic, not primary DuetRank evidence.",
    }


def run() -> dict[str, object]:
    return {
        "method": {
            "name": "DuetRank",
            "server": "budget-50K/cap-1000 candidates; stored lexical DPE top-200; semantic DPE top-100",
            "client": "exact BM25 lexical top-100, query/rank features, target-domain calibrated HistGradientBoosting ranker, optional semantic RRF backoff",
            "privacy": "model and fusion remain client-side; no additional server-visible feature or access leakage",
        },
        "primary_held_out": [learned("nq"), learned("hotpotqa")],
        "small_sample": msmarco_fixed(),
        "negative_controls": {
            "query_router_nq_held_out_ndcg_at_10": 0.5572721874334298,
            "query_router_nq_semantic_ndcg_at_10": 0.5647430462065047,
            "cross_dataset_ms_learned_ndcg_at_10": 0.578332374846548,
            "cross_dataset_ms_semantic_ndcg_at_10": 0.604462014045137,
            "conclusion": "simple routing and zero-shot cross-domain learned fusion fail; target-domain calibration is required",
        },
        "limitations": [
            "NQ gain is statistically positive but modest and affects a minority of queries.",
            "HotpotQA provides the large gain; cross-domain transfer to MS MARCO fails.",
            "The million-document subsets are relevance-preserving scale tests, not canonical full-corpus effectiveness runs.",
            "No full-corpus budgeted lexical-DPE/DuetRank execution has been completed.",
        ],
    }


if __name__ == "__main__":
    report = run()
    destination = ROOT / "results" / "budgeted_lexical_dpe" / "FINAL_SUMMARY.json"
    atomic_json(destination, report)
    print(json.dumps(report, indent=2))
