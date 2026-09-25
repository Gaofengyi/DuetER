"""Outer-holdout Recall@10/@20 audit for complete-corpus DuetRank outputs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT_FOR_DEPS = Path(__file__).resolve().parent
if (ROOT_FOR_DEPS / ".deps").exists():
    sys.path.insert(0, str(ROOT_FOR_DEPS / ".deps"))

import joblib
import numpy as np

from benchmark_million_semantic_hybrid import atomic_json, load_queries_qrels, read_ids
from experiment_adaptive_fusion import query_features
from experiment_candidate_rank_fusion import (
    learned_rankings,
    mix_rankings,
    paired_bootstrap,
    split_mask,
)
from experiment_fixed_rrf_weight_sweep import CONFIGS, ROOT
from experiment_pisces_union_fusion import metric_rows as pisces_union_rows


FUSION_RESULTS = {
    "nq": ROOT / "results/budgeted_lexical_dpe/learned_fusion_nq_full_d500_o300.json",
    "hotpotqa": ROOT / "results/budgeted_lexical_dpe/learned_fusion_hotpotqa_full.json",
    "msmarco": ROOT / "results/budgeted_lexical_dpe/learned_fusion_msmarco_dev_full_b250k_d500_o300.json",
}
TRACES = {
    "nq": (ROOT / "results/budgeted_lexical/nq_full.trace.json", "50000"),
    "hotpotqa": (ROOT / "results/budgeted_lexical/hotpotqa_full.trace.json", "50000"),
    "msmarco": (ROOT / "results/budgeted_lexical/msmarco_dev_full.trace.json", "250000"),
}


def metric_rows(
    rankings: np.ndarray,
    doc_ids: list[str],
    query_ids: list[str],
    qrels: dict[str, dict[str, int]],
    depth: int = 10,
) -> dict[str, np.ndarray]:
    recall = np.zeros(len(query_ids), dtype=np.float64)
    hit = np.zeros(len(query_ids), dtype=np.float64)
    ndcg = np.zeros(len(query_ids), dtype=np.float64)
    for index, (row, query_id) in enumerate(zip(rankings, query_ids)):
        truth = qrels[query_id]
        relevant = {doc_id for doc_id, gain in truth.items() if int(gain) > 0}
        ranked = [doc_ids[int(value)] for value in row[:depth] if int(value) >= 0]
        found = relevant.intersection(ranked)
        recall[index] = len(found) / max(len(relevant), 1)
        hit[index] = bool(found)
        observed = [int(truth.get(doc_id, 0)) for doc_id in ranked]
        ideal = sorted((int(gain) for gain in truth.values()), reverse=True)[:depth]
        dcg = sum(
            (2.0**gain - 1.0) / math.log2(position + 2.0)
            for position, gain in enumerate(observed)
        )
        idcg = sum(
            (2.0**gain - 1.0) / math.log2(position + 2.0)
            for position, gain in enumerate(ideal)
        )
        ndcg[index] = dcg / idcg if idcg else 0.0
    return {"recall": recall, "hit": hit, "ndcg": ndcg}


def variable_depth_recall_rows(
    rankings: np.ndarray,
    depths: np.ndarray,
    doc_ids: list[str],
    query_ids: list[str],
    qrels: dict[str, dict[str, int]],
) -> np.ndarray:
    """Recall using the same per-query result count as a Pisces union."""
    recall = np.zeros(len(query_ids), dtype=np.float64)
    for index, (row, raw_depth, query_id) in enumerate(zip(rankings, depths, query_ids)):
        relevant = {
            doc_id for doc_id, gain in qrels[query_id].items() if int(gain) > 0
        }
        depth = int(raw_depth)
        ranked = {
            doc_ids[int(value)] for value in row[:depth] if int(value) >= 0
        }
        recall[index] = len(relevant.intersection(ranked)) / max(len(relevant), 1)
    return recall


def load_selection(dataset: str) -> dict[str, object]:
    payload = json.loads(FUSION_RESULTS[dataset].read_text(encoding="utf-8"))
    return payload["datasets"][0]


def run_dataset(dataset: str) -> dict[str, object]:
    config = CONFIGS[dataset]
    selection_payload = load_selection(dataset)
    selection = selection_payload["selection"]
    query_ids, query_texts, qrels, _ = load_queries_qrels(
        ROOT / "data" / dataset, str(config["split"])
    )
    doc_ids = read_ids(
        ROOT / "cache/full_semantic" / f"{dataset}_{config['documents']}" / "doc_ids.txt"
    )
    semantic = np.asarray(np.load(config["semantic"])["semantic"], dtype=np.int32)
    lexical = np.asarray(np.load(config["lexical"])["lexical_dpe"], dtype=np.int32)
    holdout = split_mask(query_ids, "outer")
    semantic_weight = float(selection["semantic_backoff_weight"])

    ranking_cache = (
        ROOT / "results/budgeted_lexical_dpe" / f"duetrank_{dataset}_outer.rankings.npz"
    )
    if ranking_cache.exists():
        cached = np.load(ranking_cache)
        duet_rankings = np.asarray(cached["duetrank"], dtype=np.int32)
        if duet_rankings.shape != semantic.shape:
            raise ValueError(f"stale DuetRank cache for {dataset}: {duet_rankings.shape}")
    else:
        trace_path, trace_key = TRACES[dataset]
        trace = json.loads(trace_path.read_text(encoding="utf-8"))[trace_key]
        features = query_features(query_texts, semantic, lexical[0, :, :100], trace)
        model = joblib.load(selection["model_path"])
        generated = []
        for repeat in range(len(semantic)):
            learned = learned_rankings(
                model,
                semantic[repeat],
                lexical[repeat],
                features,
                mask=holdout,
            )
            generated.append(
                learned
                if semantic_weight == 0.0
                else mix_rankings(semantic[repeat], learned, semantic_weight)
            )
        duet_rankings = np.stack(generated)
        np.savez_compressed(
            ranking_cache,
            duetrank=duet_rankings,
            outer_holdout=holdout,
        )

    semantic_by_depth: dict[int, dict[str, np.ndarray]] = {}
    duet_by_depth: dict[int, dict[str, np.ndarray]] = {}
    for depth in (10, 20):
        semantic_repeats = [
            metric_rows(row, doc_ids, query_ids, qrels, depth) for row in semantic
        ]
        duet_repeats = [
            metric_rows(row, doc_ids, query_ids, qrels, depth) for row in duet_rankings
        ]
        semantic_by_depth[depth] = {
            name: np.stack([repeat[name] for repeat in semantic_repeats])
            for name in semantic_repeats[0]
        }
        duet_by_depth[depth] = {
            name: np.stack([repeat[name] for repeat in duet_repeats])
            for name in duet_repeats[0]
        }

    pisces_repeats = [
        pisces_union_rows(
            semantic[repeat],
            lexical[repeat],
            10,
            query_ids,
            qrels,
            doc_ids,
        )
        for repeat in range(len(semantic))
    ]
    pisces_recall = np.stack([repeat["union_recall"] for repeat in pisces_repeats])
    pisces_sizes = np.stack([repeat["union_size"] for repeat in pisces_repeats])
    duet_matched_recall = np.stack(
        [
            variable_depth_recall_rows(
                duet_rankings[repeat],
                pisces_sizes[repeat],
                doc_ids,
                query_ids,
                qrels,
            )
            for repeat in range(len(semantic))
        ]
    )

    stored_ndcg = float(selection_payload["held_out"]["semantic_backoff_ndcg_at_10"])
    recomputed_ndcg = float(np.mean(duet_by_depth[10]["ndcg"][:, holdout]))
    if not np.isclose(stored_ndcg, recomputed_ndcg, atol=1e-10):
        raise AssertionError(
            f"{dataset} DuetRank nDCG audit failed: {stored_ndcg} != {recomputed_ndcg}"
        )
    result = {
        "dataset": dataset,
        "documents": int(config["documents"]),
        "outer_holdout_queries": int(np.sum(holdout)),
        "repeats": len(semantic),
        "semantic_recall_at_10": float(np.mean(semantic_by_depth[10]["recall"][:, holdout])),
        "duetrank_recall_at_10": float(np.mean(duet_by_depth[10]["recall"][:, holdout])),
        "semantic_hit_at_10": float(np.mean(semantic_by_depth[10]["hit"][:, holdout])),
        "duetrank_hit_at_10": float(np.mean(duet_by_depth[10]["hit"][:, holdout])),
        "semantic_ndcg_at_10": float(np.mean(semantic_by_depth[10]["ndcg"][:, holdout])),
        "duetrank_ndcg_at_10": recomputed_ndcg,
        "recall_gain_vs_semantic": paired_bootstrap(
            semantic_by_depth[10]["recall"], duet_by_depth[10]["recall"], holdout
        ),
        "hit_gain_vs_semantic": paired_bootstrap(
            semantic_by_depth[10]["hit"], duet_by_depth[10]["hit"], holdout
        ),
        "ndcg_audit_reference": stored_ndcg,
        "ranking_cache": str(ranking_cache),
        "matched_maximum_budget": {
            "pisces_union_at_10_recall": float(np.mean(pisces_recall[:, holdout])),
            "pisces_union_mean_output_documents": float(np.mean(pisces_sizes[:, holdout])),
            "semantic_recall_at_20": float(
                np.mean(semantic_by_depth[20]["recall"][:, holdout])
            ),
            "duetrank_recall_at_20": float(
                np.mean(duet_by_depth[20]["recall"][:, holdout])
            ),
            "duetrank_20_vs_semantic_20": paired_bootstrap(
                semantic_by_depth[20]["recall"],
                duet_by_depth[20]["recall"],
                holdout,
            ),
            "duetrank_20_vs_pisces_union_10": paired_bootstrap(
                pisces_recall,
                duet_by_depth[20]["recall"],
                holdout,
            ),
        },
        "matched_actual_cardinality": {
            "duetrank_recall_at_per_query_pisces_union_size": float(
                np.mean(duet_matched_recall[:, holdout])
            ),
            "duetrank_matched_vs_pisces_union_10": paired_bootstrap(
                pisces_recall,
                duet_matched_recall,
                holdout,
            ),
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/budgeted_lexical_dpe/duetrank_recall_audit.json",
    )
    args = parser.parse_args()
    output = {
        "metric": "mean per-query judged-relevant-document Recall@10 and Recall@20",
        "comparison": "DuetRank@20 and Pisces semantic@10 union lexical@10 have the same maximum output budget. A second control matches the Pisces union's actual per-query cardinality.",
        "datasets": [run_dataset(dataset) for dataset in args.datasets],
    }
    atomic_json(args.output, output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
