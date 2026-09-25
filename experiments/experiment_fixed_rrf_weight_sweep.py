"""Leakage-neutral fixed-RRF semantic-weight sweep on complete corpora.

The semantic weight is selected on the deterministic calibration half only.
The untouched outer half is used once for reporting.  Besides effectiveness,
the experiment measures whether the lexical path actually changes Top-10.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmark_million_semantic_hybrid import atomic_json, load_queries_qrels, read_ids
from experiment_candidate_rank_fusion import paired_bootstrap, split_mask


ROOT = Path(__file__).resolve().parent
DISCOUNTS = 1.0 / np.log2(np.arange(2, 12, dtype=np.float64))

CONFIGS = {
    "nq": {
        "split": "test",
        "documents": 2_681_468,
        "semantic": ROOT / "results/full_semantic_hybrid/nq_2681468/rankings.npz",
        "lexical": ROOT / "results/budgeted_lexical_dpe/nq_full_d500_o300.rankings.npz",
    },
    "hotpotqa": {
        "split": "test",
        "documents": 5_233_329,
        "semantic": ROOT / "results/full_semantic_hybrid/hotpotqa_5233329/rankings.npz",
        "lexical": ROOT / "results/budgeted_lexical_dpe/hotpotqa_full_d200.rankings.npz",
    },
    "msmarco": {
        "split": "dev",
        "documents": 8_841_823,
        "semantic": ROOT / "results/msmarco_dev_full_semantic/rankings.npz",
        "lexical": ROOT / "results/budgeted_lexical_dpe/msmarco_dev_full_b250k_d500_o300.rankings.npz",
    },
}


def ideal_dcg(gains: list[int]) -> float:
    ordered = np.asarray(sorted(gains, reverse=True)[:10], dtype=np.float64)
    return float(np.sum((np.power(2.0, ordered) - 1.0) * DISCOUNTS[: len(ordered)]))


def candidate_matrices(
    semantic: np.ndarray,
    lexical: np.ndarray,
    query_ids: list[str],
    qrels: dict[str, dict[str, int]],
    doc_ids: list[str],
) -> tuple[np.ndarray, ...]:
    """Construct an ascending-document-id union for deterministic RRF ties."""
    queries = len(semantic)
    width = 200
    documents = np.full((queries, width), -1, dtype=np.int32)
    semantic_rrf = np.zeros((queries, width), dtype=np.float32)
    lexical_rrf = np.zeros((queries, width), dtype=np.float32)
    gains = np.zeros((queries, width), dtype=np.float32)
    in_semantic_top10 = np.zeros((queries, width), dtype=bool)
    in_semantic_top100 = np.zeros((queries, width), dtype=bool)
    in_lexical_top100 = np.zeros((queries, width), dtype=bool)
    ideals = np.zeros(queries, dtype=np.float64)
    relevant_counts = np.zeros(queries, dtype=np.int32)

    for query_index, query_id in enumerate(query_ids):
        semantic_row = [int(value) for value in semantic[query_index, :100] if int(value) >= 0]
        lexical_row = [int(value) for value in lexical[query_index, :100] if int(value) >= 0]
        semantic_positions = {document: rank for rank, document in enumerate(semantic_row)}
        lexical_positions = {document: rank for rank, document in enumerate(lexical_row)}
        union = sorted(set(semantic_positions).union(lexical_positions))
        documents[query_index, : len(union)] = union
        gain_by_index = {
            index: int(qrels[query_id].get(doc_ids[index], 0)) for index in union
        }
        qrel_gains = [int(value) for value in qrels[query_id].values() if int(value) > 0]
        ideals[query_index] = ideal_dcg(qrel_gains)
        relevant_counts[query_index] = len(qrel_gains)
        for column, document in enumerate(union):
            semantic_rank = semantic_positions.get(document)
            lexical_rank = lexical_positions.get(document)
            if semantic_rank is not None:
                semantic_rrf[query_index, column] = 1.0 / (61.0 + semantic_rank)
                in_semantic_top100[query_index, column] = True
                in_semantic_top10[query_index, column] = semantic_rank < 10
            if lexical_rank is not None:
                lexical_rrf[query_index, column] = 1.0 / (61.0 + lexical_rank)
                in_lexical_top100[query_index, column] = True
            gains[query_index, column] = gain_by_index[document]
    return (
        documents,
        semantic_rrf,
        lexical_rrf,
        gains,
        in_semantic_top10,
        in_semantic_top100,
        in_lexical_top100,
        ideals,
        relevant_counts,
    )


def score_repeat(
    semantic: np.ndarray,
    lexical: np.ndarray,
    weights: np.ndarray,
    query_ids: list[str],
    qrels: dict[str, dict[str, int]],
    doc_ids: list[str],
    block_size: int = 128,
) -> dict[str, np.ndarray]:
    (
        documents,
        semantic_rrf,
        lexical_rrf,
        gains,
        in_semantic_top10,
        in_semantic_top100,
        in_lexical_top100,
        ideals,
        relevant_counts,
    ) = candidate_matrices(semantic, lexical, query_ids, qrels, doc_ids)
    queries = len(query_ids)
    count = len(weights)
    ndcg = np.zeros((queries, count), dtype=np.float32)
    mrr = np.zeros((queries, count), dtype=np.float32)
    recall = np.zeros((queries, count), dtype=np.float32)
    changed = np.zeros((queries, count), dtype=np.uint8)
    promoted_count = np.zeros((queries, count), dtype=np.uint8)
    exclusive_count = np.zeros((queries, count), dtype=np.uint8)
    relevant_rescue = np.zeros((queries, count), dtype=np.uint8)

    for first in range(0, queries, block_size):
        last = min(first + block_size, queries)
        valid = documents[first:last] >= 0
        scores = (
            weights[None, :, None] * semantic_rrf[first:last, None, :]
            + (1.0 - weights[None, :, None]) * lexical_rrf[first:last, None, :]
        )
        scores = np.where(valid[:, None, :] & (scores > 0.0), scores, -np.inf)
        # Candidate ids are ascending, so stable sorting implements the same
        # document-id tie break as weighted_rrf_pair().
        order = np.argsort(-scores, axis=2, kind="stable")[:, :, :10]
        top_documents = np.take_along_axis(documents[first:last, None, :], order, axis=2)
        top_scores = np.take_along_axis(scores, order, axis=2)
        selected_valid = np.isfinite(top_scores)
        top_documents = np.where(selected_valid, top_documents, -1)
        zero_columns = np.flatnonzero(np.isclose(weights, 0.0))
        one_columns = np.flatnonzero(np.isclose(weights, 1.0))
        if len(zero_columns) and not np.array_equal(
            top_documents[:, int(zero_columns[0]), :], lexical[first:last, :10]
        ):
            raise AssertionError("RRF weight 0 must reproduce lexical Top-10")
        if len(one_columns) and not np.array_equal(
            top_documents[:, int(one_columns[0]), :], semantic[first:last, :10]
        ):
            raise AssertionError("RRF weight 1 must reproduce semantic Top-10")
        top_gains = np.where(
            selected_valid,
            np.take_along_axis(gains[first:last, None, :], order, axis=2),
            0.0,
        )
        positive = top_gains > 0
        dcg = np.sum((np.power(2.0, top_gains) - 1.0) * DISCOUNTS, axis=2)
        ndcg[first:last] = np.divide(
            dcg,
            ideals[first:last, None],
            out=np.zeros_like(dcg),
            where=ideals[first:last, None] > 0,
        )
        first_positive = np.argmax(positive, axis=2)
        has_positive = np.any(positive, axis=2)
        mrr[first:last] = np.where(has_positive, 1.0 / (first_positive + 1.0), 0.0)
        recall[first:last] = np.divide(
            np.sum(positive, axis=2),
            relevant_counts[first:last, None],
            out=np.zeros((last - first, count), dtype=np.float64),
            where=relevant_counts[first:last, None] > 0,
        )
        semantic_top10 = semantic[first:last, None, :10]
        changed[first:last] = np.any(top_documents != semantic_top10, axis=2)
        promoted = np.take_along_axis(
            (in_lexical_top100 & ~in_semantic_top10)[first:last, None, :], order, axis=2
        ) & selected_valid
        exclusive = np.take_along_axis(
            (in_lexical_top100 & ~in_semantic_top100)[first:last, None, :], order, axis=2
        ) & selected_valid
        promoted_count[first:last] = np.sum(promoted, axis=2)
        exclusive_count[first:last] = np.sum(exclusive, axis=2)
        relevant_rescue[first:last] = np.any(exclusive & positive, axis=2)
    return {
        "ndcg": ndcg,
        "mrr": mrr,
        "recall": recall,
        "changed": changed,
        "promoted_count": promoted_count,
        "exclusive_count": exclusive_count,
        "relevant_rescue": relevant_rescue,
    }


def mean_metric(values: np.ndarray, mask: np.ndarray, column: int) -> float:
    return float(np.mean(values[:, mask, column]))


def run_dataset(dataset: str, weights: np.ndarray) -> dict[str, object]:
    config = CONFIGS[dataset]
    split = str(config["split"])
    query_ids, _, qrels, _ = load_queries_qrels(ROOT / "data" / dataset, split)
    doc_ids = read_ids(
        ROOT / "cache/full_semantic" / f"{dataset}_{config['documents']}" / "doc_ids.txt"
    )
    semantic = np.asarray(np.load(config["semantic"])["semantic"], dtype=np.int32)
    lexical = np.asarray(np.load(config["lexical"])["lexical_dpe"], dtype=np.int32)
    if semantic.shape[0] != lexical.shape[0] or semantic.shape[1] != len(query_ids):
        raise ValueError(f"ranking shape mismatch for {dataset}: {semantic.shape}, {lexical.shape}")
    repeats = []
    for repeat_index in range(len(semantic)):
        repeats.append(
            score_repeat(
                semantic[repeat_index],
                lexical[repeat_index],
                weights,
                query_ids,
                qrels,
                doc_ids,
            )
        )
    stacked = {
        key: np.stack([repeat[key] for repeat in repeats]) for key in repeats[0]
    }
    holdout = split_mask(query_ids, "outer")
    calibration = ~holdout
    calibration_scores = np.mean(stacked["ndcg"][:, calibration, :], axis=(0, 1))
    selected_index = int(np.argmax(calibration_scores))
    semantic_index = int(np.flatnonzero(np.isclose(weights, 1.0))[0])

    curve = []
    for column, weight in enumerate(weights):
        curve.append(
            {
                "semantic_weight": float(weight),
                "lexical_weight": float(1.0 - weight),
                "calibration_ndcg_at_10": mean_metric(stacked["ndcg"], calibration, column),
                "heldout_ndcg_at_10": mean_metric(stacked["ndcg"], holdout, column),
                "heldout_mrr_at_10": mean_metric(stacked["mrr"], holdout, column),
                "heldout_recall_at_10": mean_metric(stacked["recall"], holdout, column),
                "heldout_changed_query_fraction": mean_metric(stacked["changed"], holdout, column),
                "heldout_lexical_promoted_docs_per_top10": mean_metric(
                    stacked["promoted_count"], holdout, column
                ),
                "heldout_lexical_exclusive_docs_per_top10": mean_metric(
                    stacked["exclusive_count"], holdout, column
                ),
                "heldout_relevant_lexical_rescue_query_fraction": mean_metric(
                    stacked["relevant_rescue"], holdout, column
                ),
            }
        )
    baseline = np.transpose(stacked["ndcg"][:, :, semantic_index], (0, 1))
    selected = np.transpose(stacked["ndcg"][:, :, selected_index], (0, 1))
    selected_row = curve[selected_index]
    return {
        "dataset": dataset,
        "split": split,
        "documents": int(config["documents"]),
        "queries": len(query_ids),
        "calibration_queries": int(np.sum(calibration)),
        "heldout_queries": int(np.sum(holdout)),
        "repeats": len(semantic),
        "rrf_constant": 60,
        "selection_rule": "maximize mean calibration-half nDCG@10; outer holdout untouched",
        "selected": selected_row,
        "heldout_gain_vs_semantic": paired_bootstrap(baseline, selected, holdout),
        "curve": curve,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/budgeted_lexical_dpe/fixed_rrf_weight_sweep.json",
    )
    args = parser.parse_args()
    weights = np.round(np.linspace(0.0, 1.0, 101), 2)
    result = {
        "weights": weights.tolist(),
        "protocol": "All 101 weights are evaluated, but each dataset weight is selected only on its calibration half.",
        "datasets": [run_dataset(dataset, weights) for dataset in args.datasets],
    }
    atomic_json(args.output, result)
    print(json.dumps({d["dataset"]: d["selected"] for d in result["datasets"]}, indent=2))


if __name__ == "__main__":
    main()
