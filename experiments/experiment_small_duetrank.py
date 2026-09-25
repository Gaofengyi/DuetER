"""Nested held-out DuetRank evaluation for BEIR SciFact and NFCorpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEPS = ROOT / ".deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from benchmark_million_semantic_hybrid import atomic_json
from experiment_adaptive_fusion import ndcg_rows, query_features
from experiment_candidate_rank_fusion import (
    document_feature_matrix,
    learned_rankings,
    mix_rankings,
    paired_bootstrap,
    rank_maps,
    split_mask,
    training_matrix,
)
from dueter_common import load_beir


def recall_rows(
    rankings: np.ndarray,
    doc_to_index: dict[str, int],
    query_ids: list[str],
    qrels: dict[str, dict[str, int]],
    depth: int = 10,
) -> np.ndarray:
    values = np.zeros(len(query_ids), dtype=np.float64)
    for row, query_id in enumerate(query_ids):
        relevant = {
            doc_to_index[doc_id]
            for doc_id, gain in qrels[query_id].items()
            if gain > 0 and doc_id in doc_to_index
        }
        if relevant:
            returned = {int(value) for value in rankings[row, :depth] if int(value) >= 0}
            values[row] = len(returned.intersection(relevant)) / len(relevant)
    return values


def fit_model(
    semantic: np.ndarray,
    lexical: np.ndarray,
    features: np.ndarray,
    mask: np.ndarray,
    query_ids: list[str],
    qrels: dict[str, dict[str, int]],
    doc_to_index: dict[str, int],
    leaves: int,
) -> tuple[HistGradientBoostingClassifier, int, int]:
    matrix, labels, weights = training_matrix(
        semantic, lexical, features, mask, query_ids, qrels, doc_to_index
    )
    model = HistGradientBoostingClassifier(
        max_iter=120,
        learning_rate=0.06,
        max_leaf_nodes=leaves,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=20260826,
    )
    model.fit(matrix, labels, sample_weight=weights)
    return model, len(labels), int(np.sum(labels))


def evaluate_dataset(dataset: str) -> dict[str, object]:
    result_dir = ROOT / "results" / f"{dataset}_granite_duetrank_rerun"
    archive = np.load(result_dir / "rankings.npz")
    exported_query_ids = [str(value) for value in archive["query_ids"]]
    exported_query_texts = [str(value) for value in archive["query_texts"]]
    doc_ids = [str(value) for value in archive["doc_ids"]]
    semantic = np.asarray(archive["semantic_dpe"], dtype=np.int32)
    lexical = np.asarray(archive["lexical_dpe"], dtype=np.int32)

    loaded_doc_ids, _, query_ids, query_texts, qrels = load_beir(
        ROOT / "data" / dataset, "test"
    )
    if query_ids != exported_query_ids or query_texts != exported_query_texts:
        raise ValueError(f"{dataset}: exported query order does not match BEIR input")
    if loaded_doc_ids != doc_ids:
        raise ValueError(f"{dataset}: exported document order does not match BEIR input")
    doc_to_index = {doc_id: index for index, doc_id in enumerate(doc_ids)}

    traces = {
        "terms": archive["tail_query_tokens"].astype(float).tolist(),
        "estimated_postings": archive["tail_posting_reads"].astype(float).tolist(),
    }
    features = query_features(query_texts, semantic[None, :, :], lexical, traces)
    outer_test = split_mask(query_ids, "outer")
    calibration = ~outer_test
    inner_validation = split_mask(query_ids, "inner") & calibration
    inner_train = calibration & ~inner_validation

    # All structural and backoff choices are made on the nested inner split.
    candidates: list[tuple[float, int, float]] = []
    for leaves in (3, 7, 15, 31):
        model, _, _ = fit_model(
            semantic,
            lexical,
            features,
            inner_train,
            query_ids,
            qrels,
            doc_to_index,
            leaves,
        )
        learned = learned_rankings(
            model, semantic, lexical, features, mask=inner_validation
        )
        for semantic_weight in (0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 1.0):
            ranking = (
                learned
                if semantic_weight == 0.0
                else mix_rankings(semantic, learned, semantic_weight)
            )
            score = float(
                np.mean(ndcg_rows(ranking, doc_to_index, query_ids, qrels)[inner_validation])
            )
            candidates.append((score, leaves, semantic_weight))
    inner_score, leaves, semantic_weight = max(candidates)

    model, training_rows, training_positives = fit_model(
        semantic,
        lexical,
        features,
        calibration,
        query_ids,
        qrels,
        doc_to_index,
        leaves,
    )
    learned = learned_rankings(model, semantic, lexical, features, mask=outer_test)
    dual = (
        learned
        if semantic_weight == 0.0
        else mix_rankings(semantic, learned, semantic_weight)
    )
    model_path = result_dir / "duetrank_nested_holdout.joblib"
    joblib.dump(model, model_path)
    np.savez_compressed(
        result_dir / "duetrank_nested_holdout.rankings.npz",
        query_ids=np.asarray(query_ids, dtype=str),
        outer_test=outer_test,
        semantic=semantic,
        lexical=lexical,
        dual=dual,
    )

    # Reverse the outer split and repeat the full nested calibration.  Combining
    # the two untouched folds yields an out-of-fold prediction for every query.
    reverse_test = calibration
    reverse_calibration = outer_test
    reverse_inner_validation = split_mask(query_ids, "inner-reverse") & reverse_calibration
    reverse_inner_train = reverse_calibration & ~reverse_inner_validation
    reverse_candidates: list[tuple[float, int, float]] = []
    for reverse_leaves in (3, 7, 15, 31):
        reverse_model, _, _ = fit_model(
            semantic,
            lexical,
            features,
            reverse_inner_train,
            query_ids,
            qrels,
            doc_to_index,
            reverse_leaves,
        )
        reverse_learned = learned_rankings(
            reverse_model,
            semantic,
            lexical,
            features,
            mask=reverse_inner_validation,
        )
        for reverse_weight in (0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 1.0):
            reverse_ranking = (
                reverse_learned
                if reverse_weight == 0.0
                else mix_rankings(semantic, reverse_learned, reverse_weight)
            )
            reverse_score = float(
                np.mean(
                    ndcg_rows(reverse_ranking, doc_to_index, query_ids, qrels)[
                        reverse_inner_validation
                    ]
                )
            )
            reverse_candidates.append(
                (reverse_score, reverse_leaves, reverse_weight)
            )
    reverse_score, reverse_leaves, reverse_weight = max(reverse_candidates)
    reverse_model, reverse_rows, reverse_positives = fit_model(
        semantic,
        lexical,
        features,
        reverse_calibration,
        query_ids,
        qrels,
        doc_to_index,
        reverse_leaves,
    )
    reverse_learned = learned_rankings(
        reverse_model, semantic, lexical, features, mask=reverse_test
    )
    reverse_dual = (
        reverse_learned
        if reverse_weight == 0.0
        else mix_rankings(semantic, reverse_learned, reverse_weight)
    )
    joblib.dump(reverse_model, result_dir / "duetrank_nested_holdout_fold2.joblib")
    dual_oof = dual.copy()
    dual_oof[reverse_test] = reverse_dual[reverse_test]

    ndcg = {
        "semantic": ndcg_rows(semantic, doc_to_index, query_ids, qrels),
        "lexical": ndcg_rows(lexical, doc_to_index, query_ids, qrels),
        "dual": ndcg_rows(dual, doc_to_index, query_ids, qrels),
    }
    recall = {
        "semantic": recall_rows(semantic, doc_to_index, query_ids, qrels),
        "lexical": recall_rows(lexical, doc_to_index, query_ids, qrels),
        "dual": recall_rows(dual, doc_to_index, query_ids, qrels),
    }
    oof_ndcg = ndcg_rows(dual_oof, doc_to_index, query_ids, qrels)
    oof_recall = recall_rows(dual_oof, doc_to_index, query_ids, qrels)
    all_queries = np.ones(len(query_ids), dtype=bool)

    def means(rows: dict[str, np.ndarray]) -> dict[str, float]:
        return {key: float(np.mean(value[outer_test])) for key, value in rows.items()}

    return {
        "dataset": dataset,
        "documents": len(doc_ids),
        "queries": len(query_ids),
        "protocol": {
            "calibration_queries": int(np.sum(calibration)),
            "inner_train_queries": int(np.sum(inner_train)),
            "inner_validation_queries": int(np.sum(inner_validation)),
            "outer_test_queries": int(np.sum(outer_test)),
            "split": "SHA-256 deterministic query split",
        },
        "selection": {
            "max_leaf_nodes": leaves,
            "semantic_backoff_weight": semantic_weight,
            "inner_validation_ndcg_at_10": inner_score,
            "training_rows": training_rows,
            "training_positives": training_positives,
            "model_path": str(model_path),
        },
        "held_out": {
            "ndcg_at_10": means(ndcg),
            "recall_at_10": means(recall),
            "ndcg_dual_vs_semantic": paired_bootstrap(
                ndcg["semantic"][None, :], ndcg["dual"][None, :], outer_test
            ),
            "ndcg_dual_vs_lexical": paired_bootstrap(
                ndcg["lexical"][None, :], ndcg["dual"][None, :], outer_test
            ),
            "recall_dual_vs_semantic": paired_bootstrap(
                recall["semantic"][None, :], recall["dual"][None, :], outer_test
            ),
            "recall_dual_vs_lexical": paired_bootstrap(
                recall["lexical"][None, :], recall["dual"][None, :], outer_test
            ),
        },
        "two_fold_out_of_fold": {
            "fold2_selection": {
                "max_leaf_nodes": reverse_leaves,
                "semantic_backoff_weight": reverse_weight,
                "inner_validation_ndcg_at_10": reverse_score,
                "training_rows": reverse_rows,
                "training_positives": reverse_positives,
            },
            "ndcg_at_10": {
                "semantic": float(np.mean(ndcg["semantic"])),
                "lexical": float(np.mean(ndcg["lexical"])),
                "dual": float(np.mean(oof_ndcg)),
            },
            "recall_at_10": {
                "semantic": float(np.mean(recall["semantic"])),
                "lexical": float(np.mean(recall["lexical"])),
                "dual": float(np.mean(oof_recall)),
            },
            "ndcg_dual_vs_semantic": paired_bootstrap(
                ndcg["semantic"][None, :], oof_ndcg[None, :], all_queries
            ),
            "ndcg_dual_vs_lexical": paired_bootstrap(
                ndcg["lexical"][None, :], oof_ndcg[None, :], all_queries
            ),
            "recall_dual_vs_semantic": paired_bootstrap(
                recall["semantic"][None, :], oof_recall[None, :], all_queries
            ),
            "recall_dual_vs_lexical": paired_bootstrap(
                recall["lexical"][None, :], oof_recall[None, :], all_queries
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["scifact", "nfcorpus"])
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "small_duetrank_nested_holdout.json",
    )
    args = parser.parse_args()
    report = {"datasets": [evaluate_dataset(dataset) for dataset in args.datasets]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
