"""Candidate-level learned fusion with deterministic query holdout.

The model is trained only on one SHA-256 parity half of each dataset.  It uses
rank positions and query-level confidence features, never document text or
identifiers.  Lexical candidates down to rank 1000 are exposed to the learner
so the experiment tests whether the enlarged candidate pool contains useful
evidence that top-100 RRF discards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from lexical_planner import (
    ROOT,
    fuse_all,
    load_semantic_rankings,
    weighted_rrf_pair,
)
from semantic_backend import atomic_json, load_queries_qrels, read_ids
from duetrank_features import ndcg_rows, query_features


def split_mask(query_ids: list[str], salt: str = "outer") -> np.ndarray:
    return np.asarray(
        [
            bool(
                hashlib.sha256(f"{salt}:{query_id}".encode("utf-8")).digest()[0]
                & 1
            )
            for query_id in query_ids
        ]
    )


def rank_maps(row: np.ndarray, depth: int) -> dict[int, int]:
    return {
        int(document): rank
        for rank, document in enumerate(row[:depth])
        if int(document) >= 0
    }


def document_feature_matrix(
    documents: list[int],
    semantic_positions: dict[int, int],
    lexical_positions: dict[int, int],
    query_row: np.ndarray,
) -> np.ndarray:
    semantic_rank = np.asarray(
        [semantic_positions.get(document, -1) for document in documents], dtype=np.float64
    )
    lexical_rank = np.asarray(
        [lexical_positions.get(document, -1) for document in documents], dtype=np.float64
    )
    semantic_present = semantic_rank >= 0
    lexical_present = lexical_rank >= 0
    semantic_reciprocal = np.zeros(len(documents), dtype=np.float64)
    lexical_reciprocal = np.zeros(len(documents), dtype=np.float64)
    np.divide(
        1.0,
        semantic_rank + 1.0,
        out=semantic_reciprocal,
        where=semantic_present,
    )
    np.divide(
        1.0,
        lexical_rank + 1.0,
        out=lexical_reciprocal,
        where=lexical_present,
    )
    semantic_rrf = np.where(semantic_present, 1.0 / (semantic_rank + 61.0), 0.0)
    lexical_rrf = np.where(lexical_present, 1.0 / (lexical_rank + 61.0), 0.0)
    base = np.column_stack(
        [
            semantic_reciprocal,
            lexical_reciprocal,
            semantic_rrf,
            lexical_rrf,
            semantic_present,
            lexical_present,
            semantic_present & lexical_present,
            np.log1p(np.where(semantic_present, semantic_rank + 1.0, 101.0))
            / math.log(102),
            np.log1p(np.where(lexical_present, lexical_rank + 1.0, 1001.0))
            / math.log(1002),
            semantic_reciprocal * lexical_reciprocal,
            np.maximum(semantic_rrf, lexical_rrf),
        ]
    )
    return np.concatenate(
        [base, np.broadcast_to(query_row, (len(documents), len(query_row)))], axis=1
    )


def training_matrix(
    semantic: np.ndarray,
    lexical: np.ndarray,
    query_feature_rows: np.ndarray,
    train_mask: np.ndarray,
    query_ids: list[str],
    qrels: dict[str, dict[str, int]],
    doc_to_index: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = []
    labels = []
    weights = []
    rng = np.random.default_rng(20260826)
    for query_index in np.flatnonzero(train_mask):
        semantic_positions = rank_maps(semantic[query_index], 100)
        lexical_positions = rank_maps(lexical[query_index], 1000)
        relevant = {
            doc_to_index[doc_id]
            for doc_id, gain in qrels[query_ids[query_index]].items()
            if gain > 0 and doc_id in doc_to_index
        }
        candidates = set(semantic_positions).union(lexical_positions)
        positives = candidates.intersection(relevant)
        hard_negatives = set(list(semantic_positions)[:50]).union(
            list(lexical_positions)[:100]
        ) - relevant
        remaining = np.asarray(list(candidates - positives - hard_negatives), dtype=np.int64)
        if len(remaining) > 100:
            remaining = rng.choice(remaining, size=100, replace=False)
        negatives = hard_negatives.union(int(value) for value in remaining)
        selected = list(positives) + list(negatives)
        positive_weight = 0.5 / max(len(positives), 1)
        negative_weight = 0.5 / max(len(negatives), 1)
        selected_labels = np.asarray(
            [int(document in positives) for document in selected], dtype=np.int8
        )
        features.append(
            document_feature_matrix(
                selected,
                semantic_positions,
                lexical_positions,
                query_feature_rows[query_index],
            )
        )
        labels.append(selected_labels)
        weights.append(
            np.where(selected_labels > 0, positive_weight, negative_weight)
        )
    return (
        np.concatenate(features).astype(np.float32),
        np.concatenate(labels),
        np.concatenate(weights),
    )


def learned_rankings(
    model: HistGradientBoostingClassifier,
    semantic: np.ndarray,
    lexical: np.ndarray,
    query_feature_rows: np.ndarray,
    depth: int = 100,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    output = np.asarray(semantic[:, :depth], dtype=np.int32).copy()
    selected_queries = range(len(semantic)) if mask is None else np.flatnonzero(mask)
    for query_index in selected_queries:
        semantic_positions = rank_maps(semantic[query_index], 100)
        lexical_positions = rank_maps(lexical[query_index], 1000)
        candidates = list(set(semantic_positions).union(lexical_positions))
        matrix = document_feature_matrix(
            candidates,
            semantic_positions,
            lexical_positions,
            query_feature_rows[query_index],
        ).astype(np.float32)
        probabilities = model.predict_proba(matrix)[:, 1]
        ordered = sorted(
            zip(candidates, probabilities), key=lambda item: (-item[1], item[0])
        )[:depth]
        output[query_index, : len(ordered)] = [document for document, _ in ordered]
    return output


def mix_rankings(
    semantic: np.ndarray, learned: np.ndarray, semantic_weight: float
) -> np.ndarray:
    output = np.empty_like(semantic)
    for row in range(len(semantic)):
        output[row] = weighted_rrf_pair(
            semantic[row], learned[row], 100, semantic_weight
        )
    return output


def paired_bootstrap(
    baseline: np.ndarray,
    system: np.ndarray,
    mask: np.ndarray,
    samples: int = 5000,
) -> dict[str, float]:
    differences = np.mean(system[:, mask] - baseline[:, mask], axis=0)
    rng = np.random.default_rng(20260826)
    bootstrap = np.empty(samples, dtype=np.float64)
    chunk = 250
    for first in range(0, samples, chunk):
        last = min(first + chunk, samples)
        indices = rng.integers(
            0, len(differences), size=(last - first, len(differences))
        )
        bootstrap[first:last] = np.mean(differences[indices], axis=1)
    return {
        "mean_absolute_gain": float(np.mean(differences)),
        "ci95_low": float(np.percentile(bootstrap, 2.5)),
        "ci95_high": float(np.percentile(bootstrap, 97.5)),
        "one_sided_bootstrap_p": float(
            (1 + np.sum(bootstrap <= 0.0)) / (samples + 1)
        ),
        "query_win_fraction": float(np.mean(differences > 1e-12)),
        "query_tie_fraction": float(np.mean(np.abs(differences) <= 1e-12)),
        "query_loss_fraction": float(np.mean(differences < -1e-12)),
    }


FULL_DOCUMENTS = {
    "msmarco": 8_841_823,
    "nq": 2_681_468,
    "hotpotqa": 5_233_329,
}


def run_dataset(
    dataset: str,
    lexical_source: str = "raw",
    scale: str = "million",
    lexical_rankings_path: Path | None = None,
    split: str = "test",
    semantic_rankings_path: Path | None = None,
    trace_path: Path | None = None,
    budget_key: int = 50_000,
    model_tag: str | None = None,
) -> dict[str, object]:
    query_ids, query_texts, qrels, _ = load_queries_qrels(
        ROOT / "data" / dataset, split
    )
    documents = 1_000_000 if scale == "million" else FULL_DOCUMENTS[dataset]
    semantic_cache_root = "million_semantic" if scale == "million" else "full_semantic"
    doc_ids = read_ids(
        ROOT / "cache" / semantic_cache_root / f"{dataset}_{documents}" / "doc_ids.txt"
    )
    doc_to_index = {doc_id: index for index, doc_id in enumerate(doc_ids)}
    semantic_repeats = (
        load_semantic_rankings(dataset, scale)[0]
        if semantic_rankings_path is None
        else np.asarray(np.load(semantic_rankings_path)["semantic"], dtype=np.int32)
    )
    if lexical_source == "raw":
        raw_path = lexical_rankings_path or (
            ROOT / "results" / "budgeted_lexical" / f"{dataset}_{scale}.raw.npz"
        )
        lexical = np.load(raw_path)[f"budget_{budget_key}"]
        lexical_repeats = lexical[None, :, :]
    else:
        scale_label = "" if scale == "million" else "_full"
        final_name = f"{dataset}{scale_label}_d200.rankings.npz"
        fallback_name = f"{dataset}{scale_label}_d200_r1.rankings.npz"
        final_path = ROOT / "results" / "budgeted_lexical_dpe" / final_name
        ranking_path = lexical_rankings_path or (
            final_path
            if final_path.exists()
            else ROOT / "results" / "budgeted_lexical_dpe" / fallback_name
        )
        lexical_repeats = np.load(ranking_path)["lexical_dpe"]
        lexical = lexical_repeats[0]
    selected_trace_path = trace_path or (
        ROOT / "results" / "budgeted_lexical" / f"{dataset}_{scale}.trace.json"
    )
    trace = json.loads(selected_trace_path.read_text(encoding="utf-8"))[str(budget_key)]
    q_features = query_features(query_texts, semantic_repeats, lexical[:, :100], trace)
    outer_holdout = split_mask(query_ids, "outer")
    calibration = ~outer_holdout
    inner_validation = split_mask(query_ids, "inner") & calibration
    inner_train = calibration & ~inner_validation

    # Select model capacity and the conservative semantic-backoff weight without
    # touching the outer holdout.
    candidates = []
    inner_matrix, inner_labels, inner_sample_weights = training_matrix(
        semantic_repeats[0],
        lexical,
        q_features,
        inner_train,
        query_ids,
        qrels,
        doc_to_index,
    )
    for leaves in (7, 15, 31):
        model = HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.06,
            max_leaf_nodes=leaves,
            min_samples_leaf=40,
            l2_regularization=1.0,
            random_state=20260826,
        )
        model.fit(inner_matrix, inner_labels, sample_weight=inner_sample_weights)
        learned = learned_rankings(
            model, semantic_repeats[0], lexical, q_features, mask=inner_validation
        )
        for semantic_weight in (0.0, 0.5, 0.8, 0.9, 0.95):
            ranking = (
                learned
                if semantic_weight == 0.0
                else mix_rankings(semantic_repeats[0], learned, semantic_weight)
            )
            score = float(
                np.mean(
                    ndcg_rows(ranking, doc_to_index, query_ids, qrels)[inner_validation]
                )
            )
            candidates.append((score, leaves, semantic_weight))
    selection_score, best_leaves, best_semantic_weight = max(candidates)

    matrix, labels, sample_weights = training_matrix(
        semantic_repeats[0],
        lexical,
        q_features,
        calibration,
        query_ids,
        qrels,
        doc_to_index,
    )
    model = HistGradientBoostingClassifier(
        max_iter=120,
        learning_rate=0.06,
        max_leaf_nodes=best_leaves,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=20260826,
    )
    model.fit(matrix, labels, sample_weight=sample_weights)
    model_suffix = "" if model_tag is None else f"_{model_tag}"
    model_path = (
        ROOT
        / "results"
        / "budgeted_lexical_dpe"
        / f"{dataset}_{split}_{scale}_b{budget_key}_duetrank_{lexical_source}{model_suffix}.joblib"
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    semantic_values = []
    fixed_weights = (0.8, 0.9, 0.95, 0.98, 1.0)
    fixed_by_weight: dict[float, list[np.ndarray]] = {
        weight: [] for weight in fixed_weights
    }
    learned_values = []
    mixed_values = []
    inference_ms_per_query = []
    for repeat_index, semantic in enumerate(semantic_repeats):
        lexical_repeat = lexical_repeats[repeat_index % len(lexical_repeats)]
        inference_started = time.perf_counter()
        learned = learned_rankings(
            model, semantic, lexical_repeat, q_features, mask=outer_holdout
        )
        inference_ms_per_query.append(
            (time.perf_counter() - inference_started)
            * 1000.0
            / max(int(np.sum(outer_holdout)), 1)
        )
        mixed = (
            learned
            if best_semantic_weight == 0.0
            else mix_rankings(semantic, learned, best_semantic_weight)
        )
        semantic_values.append(ndcg_rows(semantic, doc_to_index, query_ids, qrels))
        for fixed_weight in fixed_weights:
            fixed_by_weight[fixed_weight].append(
                ndcg_rows(
                    fuse_all(
                        semantic,
                        lexical_repeat[:, :100],
                        100,
                        fixed_weight,
                    ),
                    doc_to_index,
                    query_ids,
                    qrels,
                )
            )
        learned_values.append(ndcg_rows(learned, doc_to_index, query_ids, qrels))
        mixed_values.append(ndcg_rows(mixed, doc_to_index, query_ids, qrels))
    semantic_values = np.asarray(semantic_values)
    fixed_by_weight_arrays = {
        weight: np.asarray(values) for weight, values in fixed_by_weight.items()
    }
    selected_fixed_weight = max(
        fixed_weights,
        key=lambda weight: float(
            np.mean(fixed_by_weight_arrays[weight][:, calibration])
        ),
    )
    fixed_values = fixed_by_weight_arrays[selected_fixed_weight]
    learned_values = np.asarray(learned_values)
    mixed_values = np.asarray(mixed_values)

    def report(mask: np.ndarray) -> dict[str, float | int]:
        return {
            "queries": int(np.sum(mask)),
            "semantic_ndcg_at_10": float(np.mean(semantic_values[:, mask])),
            "calibrated_fixed_rrf_ndcg_at_10": float(
                np.mean(fixed_values[:, mask])
            ),
            "learned_ranker_ndcg_at_10": float(np.mean(learned_values[:, mask])),
            "semantic_backoff_ndcg_at_10": float(np.mean(mixed_values[:, mask])),
            "backoff_vs_semantic": paired_bootstrap(
                semantic_values, mixed_values, mask
            ),
            "backoff_vs_calibrated_fixed_rrf": paired_bootstrap(
                fixed_values, mixed_values, mask
            ),
        }

    return {
        "dataset": dataset,
        "split": split,
        "scale": scale,
        "documents": documents,
        "posting_budget": budget_key,
        "lexical_source": lexical_source,
        "lexical_depth": int(lexical.shape[1]),
        "selection": {
            "inner_validation_queries": int(np.sum(inner_validation)),
            "max_leaf_nodes": best_leaves,
            "semantic_backoff_weight": best_semantic_weight,
            "inner_validation_ndcg_at_10": selection_score,
            "training_rows": len(matrix),
            "training_positives": int(np.sum(labels)),
            "model_path": str(model_path),
            "model_bytes": model_path.stat().st_size,
            "client_inference_mean_ms_per_query": float(
                np.mean(inference_ms_per_query)
            ),
            "selected_fixed_rrf_semantic_weight": selected_fixed_weight,
        },
        "calibration": {
            "queries": int(np.sum(calibration)),
            "semantic_ndcg_at_10": float(np.mean(semantic_values[:, calibration])),
            "calibrated_fixed_rrf_ndcg_at_10": float(
                np.mean(fixed_values[:, calibration])
            ),
            "learned_ranker_ndcg_at_10": None,
            "semantic_backoff_ndcg_at_10": None,
            "backoff_vs_semantic": None,
            "backoff_vs_calibrated_fixed_rrf": None,
        },
        "held_out": report(outer_holdout),
        "warning": "Hyperparameters use only an inner split of the calibration half. The outer SHA-256 split is untouched until final evaluation.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["nq"])
    parser.add_argument(
        "--lexical-source", choices=("raw", "budgeted-dpe"), default="raw"
    )
    parser.add_argument("--scale", choices=("million", "full"), default="million")
    parser.add_argument("--lexical-rankings", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--semantic-rankings", type=Path, default=None)
    parser.add_argument("--trace", type=Path, default=None)
    parser.add_argument("--budget-key", type=int, default=50_000)
    parser.add_argument("--model-tag", default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "budgeted_lexical" / "candidate_rank_fusion.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    report = {
        "datasets": [
            run_dataset(
                dataset,
                lexical_source=args.lexical_source,
                scale=args.scale,
                lexical_rankings_path=args.lexical_rankings,
                split=args.split,
                semantic_rankings_path=args.semantic_rankings,
                trace_path=args.trace,
                budget_key=args.budget_key,
                model_tag=args.model_tag,
            )
            for dataset in args.datasets
        ]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))
