"""Held-out experiments for query-adaptive dual-path rank fusion.

The router observes only client-available query features and the two returned
rank lists.  Labels are used on a deterministic calibration half to learn a
query-level semantic weight; the other half is never used for fitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from lexical_planner import (
    ROOT,
    fuse_all,
    load_semantic_rankings,
)
from semantic_backend import (
    atomic_json,
    load_queries_qrels,
    read_ids,
)
from common import tokenize


WEIGHTS = np.asarray([0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 1.0])


def held_out_mask(query_ids: list[str]) -> np.ndarray:
    return np.asarray(
        [bool(hashlib.sha256(qid.encode("utf-8")).digest()[0] & 1) for qid in query_ids]
    )


def dcg_at_10(ranking: np.ndarray, relevant: dict[int, int]) -> float:
    value = 0.0
    for position, raw in enumerate(ranking[:10]):
        gain = relevant.get(int(raw), 0)
        if gain > 0:
            value += (2.0**gain - 1.0) / math.log2(position + 2.0)
    return value


def ndcg_rows(
    rankings: np.ndarray,
    doc_to_index: dict[str, int],
    query_ids: list[str],
    qrels: dict[str, dict[str, int]],
) -> np.ndarray:
    output = np.zeros(len(query_ids), dtype=np.float64)
    for row, qid in enumerate(query_ids):
        relevant = {
            doc_to_index[doc_id]: gain
            for doc_id, gain in qrels[qid].items()
            if gain > 0 and doc_id in doc_to_index
        }
        ideal = sorted(relevant.values(), reverse=True)[:10]
        denominator = sum(
            (2.0**gain - 1.0) / math.log2(position + 2.0)
            for position, gain in enumerate(ideal)
        )
        if denominator:
            output[row] = dcg_at_10(rankings[row], relevant) / denominator
    return output


def overlap_features(semantic: np.ndarray, lexical: np.ndarray) -> np.ndarray:
    """Features for one semantic repeat and one fixed lexical ranking."""
    rows = []
    for s_row, l_row in zip(semantic, lexical):
        s_valid = [int(value) for value in s_row if int(value) >= 0]
        l_valid = [int(value) for value in l_row if int(value) >= 0]
        features = []
        for depth in (5, 10, 20, 50, 100):
            left = set(s_valid[:depth])
            right = set(l_valid[:depth])
            features.append(len(left.intersection(right)) / max(depth, 1))
        lexical_positions = {doc: rank for rank, doc in enumerate(l_valid[:100])}
        reciprocal_agreement = sum(
            1.0 / ((rank + 1.0) * (lexical_positions[doc] + 1.0))
            for rank, doc in enumerate(s_valid[:100])
            if doc in lexical_positions
        )
        features.append(math.log1p(reciprocal_agreement))
        rows.append(features)
    return np.asarray(rows, dtype=np.float64)


def query_features(
    query_texts: list[str],
    semantic_repeats: np.ndarray,
    lexical: np.ndarray,
    trace: dict[str, list[float]],
) -> np.ndarray:
    overlap = np.mean(
        [overlap_features(repeat, lexical) for repeat in semantic_repeats], axis=0
    )
    static = []
    terms_selected = trace["terms"]
    postings = trace["estimated_postings"]
    for query, selected, posting_count in zip(
        query_texts, terms_selected, postings
    ):
        tokens = tokenize(query)
        static.append(
            [
                len(tokens),
                len(set(tokens)),
                sum(character.isdigit() for character in query) > 0,
                sum(character.isupper() for character in query),
                selected,
                math.log1p(posting_count),
                posting_count / 50000.0,
            ]
        )
    return np.concatenate([np.asarray(static, dtype=np.float64), overlap], axis=1)


def fused_ndcg_cube(
    semantic_repeats: np.ndarray,
    lexical: np.ndarray,
    doc_to_index: dict[str, int],
    query_ids: list[str],
    qrels: dict[str, dict[str, int]],
) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    semantic_values = np.asarray(
        [ndcg_rows(repeat, doc_to_index, query_ids, qrels) for repeat in semantic_repeats]
    )
    values = {}
    for weight in WEIGHTS:
        values[float(weight)] = np.asarray(
            [
                ndcg_rows(
                    fuse_all(repeat, lexical, 100, float(weight)),
                    doc_to_index,
                    query_ids,
                    qrels,
                )
                for repeat in semantic_repeats
            ]
        )
    return semantic_values, values


def choose_targets(values: dict[float, np.ndarray]) -> np.ndarray:
    """Choose the largest weight within an exact per-query maximum tie."""
    means = np.stack([np.mean(values[float(weight)], axis=0) for weight in WEIGHTS], axis=1)
    best = np.max(means, axis=1, keepdims=True)
    tied = means >= best - 1e-12
    return np.asarray([np.flatnonzero(row)[-1] for row in tied], dtype=np.int32)


def train_router(
    features: np.ndarray, targets: np.ndarray, calibration: np.ndarray
) -> tuple[RandomForestClassifier, dict[str, object]]:
    candidates = []
    counts = np.bincount(targets[calibration], minlength=len(WEIGHTS))
    nonempty = int(np.sum(counts > 0))
    folds = max(2, min(5, int(np.min(counts[counts > 0])) if nonempty > 1 else 2))
    for depth in (3, 5, 8, None):
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=depth,
            min_samples_leaf=15,
            class_weight="balanced_subsample",
            random_state=20260826,
            n_jobs=-1,
        )
        if nonempty > 1 and folds >= 2:
            splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=20260826)
            score = float(
                np.mean(
                    cross_val_score(
                        model,
                        features[calibration],
                        targets[calibration],
                        cv=splitter,
                        scoring="balanced_accuracy",
                        n_jobs=-1,
                    )
                )
            )
        else:
            score = 0.0
        candidates.append((score, depth, model))
    score, depth, model = max(candidates, key=lambda item: item[0])
    model.fit(features[calibration], targets[calibration])
    return model, {
        "cv_balanced_accuracy": score,
        "max_depth": depth,
        "class_counts": {
            str(float(WEIGHTS[index])): int(value)
            for index, value in enumerate(counts)
            if value
        },
    }


def routed_rankings(
    semantic_repeats: np.ndarray,
    lexical: np.ndarray,
    predicted: np.ndarray,
) -> np.ndarray:
    output = np.empty_like(semantic_repeats)
    for repeat_index, semantic in enumerate(semantic_repeats):
        by_weight = {
            index: fuse_all(semantic, lexical, 100, float(weight))
            for index, weight in enumerate(WEIGHTS)
        }
        for query_index, weight_index in enumerate(predicted):
            output[repeat_index, query_index] = by_weight[int(weight_index)][query_index]
    return output


def run_dataset(dataset: str) -> dict[str, object]:
    query_ids, query_texts, qrels, _ = load_queries_qrels(ROOT / "data" / dataset)
    doc_ids = read_ids(
        ROOT / "cache" / "million_semantic" / f"{dataset}_1000000" / "doc_ids.txt"
    )
    doc_to_index = {doc_id: index for index, doc_id in enumerate(doc_ids)}
    semantic, _ = load_semantic_rankings(dataset, "million")
    lexical = np.load(
        ROOT / "results" / "budgeted_lexical" / f"{dataset}_million.raw.npz"
    )["budget_50000"][:, :100]
    trace = json.loads(
        (
            ROOT
            / "results"
            / "budgeted_lexical"
            / f"{dataset}_million.trace.json"
        ).read_text(encoding="utf-8")
    )["50000"]
    features = query_features(query_texts, semantic, lexical, trace)
    semantic_values, fused_values = fused_ndcg_cube(
        semantic, lexical, doc_to_index, query_ids, qrels
    )
    targets = choose_targets(fused_values)
    holdout = held_out_mask(query_ids)
    calibration = ~holdout
    model, training = train_router(features, targets, calibration)
    predicted = model.predict(features).astype(np.int32)
    routed = routed_rankings(semantic, lexical, predicted)
    routed_values = np.asarray(
        [ndcg_rows(repeat, doc_to_index, query_ids, qrels) for repeat in routed]
    )
    oracle = np.max(
        np.stack([np.mean(value, axis=0) for value in fused_values.values()], axis=1),
        axis=1,
    )

    def split_report(mask: np.ndarray) -> dict[str, object]:
        return {
            "queries": int(np.sum(mask)),
            "semantic_ndcg_at_10": float(np.mean(semantic_values[:, mask])),
            "fixed_by_weight": {
                str(weight): float(np.mean(fused_values[float(weight)][:, mask]))
                for weight in WEIGHTS
            },
            "router_ndcg_at_10": float(np.mean(routed_values[:, mask])),
            "oracle_weight_ndcg_at_10": float(np.mean(oracle[mask])),
            "predicted_weight_counts": {
                str(float(WEIGHTS[index])): int(np.sum(predicted[mask] == index))
                for index in range(len(WEIGHTS))
                if np.any(predicted[mask] == index)
            },
        }

    return {
        "dataset": dataset,
        "features": [
            "token_count",
            "unique_token_count",
            "has_digit",
            "uppercase_count",
            "selected_term_count",
            "log_estimated_postings",
            "budget_fraction",
            "overlap_at_5",
            "overlap_at_10",
            "overlap_at_20",
            "overlap_at_50",
            "overlap_at_100",
            "reciprocal_rank_agreement",
        ],
        "training": training,
        "calibration": split_report(calibration),
        "held_out": split_report(holdout),
        "warning": "Oracle selects a weight with qrels and is only a headroom diagnostic. Router results alone are label-free on held-out queries.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["nq", "hotpotqa"])
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "budgeted_lexical" / "adaptive_fusion.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    report = {"datasets": [run_dataset(dataset) for dataset in args.datasets]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))
