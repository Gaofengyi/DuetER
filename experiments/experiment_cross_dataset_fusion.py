"""Train candidate fusion on NQ/HotpotQA calibration queries and transfer to MS MARCO."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from benchmark_budgeted_lexical_candidates import ROOT, fuse_all, load_semantic_rankings
from benchmark_million_semantic_hybrid import atomic_json, load_queries_qrels, read_ids
from experiment_adaptive_fusion import ndcg_rows, query_features
from experiment_candidate_rank_fusion import (
    learned_rankings,
    split_mask,
    training_matrix,
)


def load_dataset(dataset: str, lexical_source: str = "budgeted-dpe") -> dict[str, object]:
    query_ids, query_texts, qrels, _ = load_queries_qrels(ROOT / "data" / dataset)
    doc_ids = read_ids(
        ROOT / "cache" / "million_semantic" / f"{dataset}_1000000" / "doc_ids.txt"
    )
    semantic, _ = load_semantic_rankings(dataset, "million")
    if lexical_source == "budgeted-dpe":
        ranking_path = (
            ROOT
            / "results"
            / "budgeted_lexical_dpe"
            / (f"{dataset}_d200.rankings.npz" if dataset == "msmarco" else f"{dataset}_d200_r1.rankings.npz")
        )
        lexical = np.load(ranking_path)["lexical_dpe"][0]
    else:
        lexical = np.load(
            ROOT / "results" / "budgeted_lexical" / f"{dataset}_million.raw.npz"
        )["budget_50000"]
    trace = json.loads(
        (
            ROOT
            / "results"
            / "budgeted_lexical"
            / f"{dataset}_million.trace.json"
        ).read_text(encoding="utf-8")
    )["50000"]
    features = query_features(query_texts, semantic, lexical[:, :100], trace)
    return {
        "query_ids": query_ids,
        "qrels": qrels,
        "doc_ids": doc_ids,
        "doc_to_index": {doc_id: index for index, doc_id in enumerate(doc_ids)},
        "semantic": semantic,
        "lexical": lexical,
        "features": features,
    }


def run() -> dict[str, object]:
    training_parts = []
    training_labels = []
    training_weights = []
    training_counts = {}
    for dataset in ("nq", "hotpotqa"):
        data = load_dataset(dataset)
        calibration = ~split_mask(data["query_ids"], "outer")
        matrix, labels, weights = training_matrix(
            data["semantic"][0],
            data["lexical"],
            data["features"],
            calibration,
            data["query_ids"],
            data["qrels"],
            data["doc_to_index"],
        )
        training_parts.append(matrix)
        training_labels.append(labels)
        # Equalize the two source datasets after their per-query normalization.
        training_weights.append(weights / np.sum(weights))
        training_counts[dataset] = {
            "queries": int(np.sum(calibration)),
            "rows": len(matrix),
            "positives": int(np.sum(labels)),
        }
    matrix = np.concatenate(training_parts)
    labels = np.concatenate(training_labels)
    weights = np.concatenate(training_weights)
    model = HistGradientBoostingClassifier(
        max_iter=120,
        learning_rate=0.06,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=20260826,
    )
    model.fit(matrix, labels, sample_weight=weights)
    model_path = ROOT / "results" / "budgeted_lexical" / "cross_dataset_fusion.joblib"
    joblib.dump(model, model_path)

    target = load_dataset("msmarco")
    semantic_values = []
    fixed_values = []
    learned_values = []
    for semantic in target["semantic"]:
        learned = learned_rankings(
            model, semantic, target["lexical"], target["features"]
        )
        semantic_values.append(
            ndcg_rows(
                semantic,
                target["doc_to_index"],
                target["query_ids"],
                target["qrels"],
            )
        )
        fixed_values.append(
            ndcg_rows(
                fuse_all(semantic, target["lexical"][:, :100], 100, 0.98),
                target["doc_to_index"],
                target["query_ids"],
                target["qrels"],
            )
        )
        learned_values.append(
            ndcg_rows(
                learned,
                target["doc_to_index"],
                target["query_ids"],
                target["qrels"],
            )
        )
    semantic_values = np.asarray(semantic_values)
    fixed_values = np.asarray(fixed_values)
    learned_values = np.asarray(learned_values)
    return {
        "training": training_counts,
        "model": {
            "max_leaf_nodes": 15,
            "source": "capacity frozen from the larger HotpotQA inner calibration split; actual budgeted-DPE lexical rankings",
            "path": str(model_path),
        },
        "target": {
            "dataset": "msmarco",
            "queries": len(target["query_ids"]),
            "semantic_ndcg_at_10": float(np.mean(semantic_values)),
            "fixed_rrf_098_ndcg_at_10": float(np.mean(fixed_values)),
            "learned_ranker_ndcg_at_10": float(np.mean(learned_values)),
        },
        "warning": "MS MARCO labels are used only for final evaluation, but its 43-query test split is too small for a strong generalization claim.",
    }


if __name__ == "__main__":
    report = run()
    destination = ROOT / "results" / "budgeted_lexical" / "cross_dataset_fusion.json"
    atomic_json(destination, report)
    print(json.dumps(report, indent=2))
