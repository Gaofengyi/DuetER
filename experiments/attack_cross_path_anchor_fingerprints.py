"""Known-anchor cross-path linkage attack against DuetDPE's concrete view.

The cloud is given independently pseudonymized semantic and lexical DPE
coordinates plus a small set of known cross-path document correspondences.
It never receives either DPE key, either plaintext representation, or the
remaining semantic--lexical mapping.  For every unknown document, the attack
builds a fingerprint from its ciphertext distances to the known anchors and
matches fingerprints across paths after randomly permuting lexical handles.

The complete-corpus experiments materialized lexical ciphertexts only for the
union of rows touched by the evaluated query workload.  Consequently this
script evaluates linkage over a uniformly sampled pool from that materialized
server view; it records both the full corpus size and the materialized scope so
the result is not mistaken for corpus-wide exhaustive matching.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEPS = ROOT / ".deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))

import numpy as np


DATASETS = {
    "nq": {
        "documents": 2_681_468,
        "semantic": ROOT / "cache/full_semantic/nq_2681468/semantic_cipher_fp16.npy",
        "lexical": ROOT / "cache/full_candidate_lexical_dpe/nq_full_h1024/lexical_cipher_fp16.npy",
        "rows": ROOT / "cache/full_candidate_lexical_dpe/nq_full_h1024/candidate_rows.npy",
    },
    "hotpotqa": {
        "documents": 5_233_329,
        "semantic": ROOT / "cache/full_semantic/hotpotqa_5233329/semantic_cipher_fp16.npy",
        "lexical": ROOT / "cache/full_candidate_lexical_dpe/hotpotqa_full_h1024/lexical_cipher_fp16.npy",
        "rows": ROOT / "cache/full_candidate_lexical_dpe/hotpotqa_full_h1024/candidate_rows.npy",
    },
    "msmarco": {
        "documents": 8_841_823,
        "semantic": ROOT / "cache/full_semantic/msmarco_8841823/semantic_cipher_fp16.npy",
        "lexical": ROOT / "cache/full_candidate_lexical_dpe/msmarco_dev_b250k_full_h1024/lexical_cipher_fp16.npy",
        "rows": ROOT / "cache/full_candidate_lexical_dpe/msmarco_dev_b250k_full_h1024/candidate_rows.npy",
    },
}


def gather_rows(array: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Load random memmap rows in sorted disk order, restoring requested order."""
    rows = np.asarray(rows, dtype=np.int64)
    order = np.argsort(rows)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    loaded = np.asarray(array[rows[order]], dtype=np.float32)
    return loaded[inverse]


def distances_to_anchors(points: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Euclidean distances using the exact quantized coordinates seen by cloud."""
    point_norm = np.sum(points * points, axis=1, keepdims=True)
    anchor_norm = np.sum(anchors * anchors, axis=1, keepdims=True).T
    squared = point_norm + anchor_norm - 2.0 * (points @ anchors.T)
    np.maximum(squared, 0.0, out=squared)
    np.sqrt(squared, out=squared)
    return squared


def row_standardize(values: np.ndarray) -> np.ndarray:
    centered = values - np.mean(values, axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    return centered / np.maximum(norms, 1e-12)


def column_percentiles(values: np.ndarray) -> np.ndarray:
    """Calibrate each anchor separately without knowing unknown correspondences."""
    count = len(values)
    if count <= 1:
        return np.zeros_like(values, dtype=np.float32)
    order = np.argsort(values, axis=0, kind="stable")
    ranks = np.empty_like(order, dtype=np.float32)
    rank_values = np.arange(count, dtype=np.float32)[:, None]
    np.put_along_axis(ranks, order, np.broadcast_to(rank_values, order.shape), axis=0)
    ranks /= float(count - 1)
    return ranks


def row_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=np.float32)
    rank_values = np.arange(values.shape[1], dtype=np.float32)[None, :]
    np.put_along_axis(ranks, order, np.broadcast_to(rank_values, order.shape), axis=1)
    return ranks


def reference_percentiles(values: np.ndarray, anchor_pairwise: np.ndarray) -> np.ndarray:
    """Percentile of each target distance in the same-view anchor distribution."""
    count = values.shape[1]
    output = np.empty_like(values, dtype=np.float32)
    for column in range(count):
        reference = np.delete(anchor_pairwise[:, column], column)
        reference = np.sort(reference)
        output[:, column] = np.searchsorted(
            reference, values[:, column], side="right"
        ) / max(len(reference), 1)
    return row_standardize(output)


def linearly_calibrated_fingerprints(
    semantic_distances: np.ndarray,
    lexical_distances: np.ndarray,
    semantic_anchor_pairwise: np.ndarray,
    lexical_anchor_pairwise: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Learn one semantic-to-lexical distance calibration per known anchor."""
    count = semantic_distances.shape[1]
    slope = np.empty(count, dtype=np.float32)
    intercept = np.empty(count, dtype=np.float32)
    lexical_mean = np.empty(count, dtype=np.float32)
    lexical_std = np.empty(count, dtype=np.float32)
    for column in range(count):
        mask = np.arange(count) != column
        source = semantic_anchor_pairwise[mask, column].astype(np.float64)
        target = lexical_anchor_pairwise[mask, column].astype(np.float64)
        source_mean = float(np.mean(source))
        target_mean = float(np.mean(target))
        denominator = float(np.sum((source - source_mean) ** 2))
        slope[column] = (
            float(np.sum((source - source_mean) * (target - target_mean))) / denominator
            if denominator > 1e-12
            else 0.0
        )
        intercept[column] = target_mean - slope[column] * source_mean
        lexical_mean[column] = target_mean
        lexical_std[column] = max(float(np.std(target)), 1e-6)
    predicted_lexical = semantic_distances * slope[None, :] + intercept[None, :]
    semantic_fp = (predicted_lexical - lexical_mean[None, :]) / lexical_std[None, :]
    lexical_fp = (lexical_distances - lexical_mean[None, :]) / lexical_std[None, :]
    return row_standardize(semantic_fp), row_standardize(lexical_fp)


def fingerprints(distances: np.ndarray, method: str) -> np.ndarray:
    if method == "distance_pearson":
        return row_standardize(distances)
    if method == "anchor_percentile":
        return row_standardize(column_percentiles(distances))
    if method == "within_document_rank":
        return row_standardize(row_ranks(distances))
    raise ValueError(f"unknown fingerprint method: {method}")


def matching_metrics(
    semantic_fp: np.ndarray,
    lexical_fp: np.ndarray,
    lexical_permutation: np.ndarray,
) -> dict[str, float]:
    """Rank the true lexical handle for each semantic handle."""
    permuted_lexical = lexical_fp[lexical_permutation]
    true_column = np.empty_like(lexical_permutation)
    true_column[lexical_permutation] = np.arange(len(lexical_permutation))
    similarities = semantic_fp @ permuted_lexical.T
    true_score = similarities[np.arange(len(similarities)), true_column]
    # Stable random handle permutation makes exact ties non-informative.  Half
    # of ties are included in the rank, matching random tie breaking in mean.
    greater = np.sum(similarities > true_score[:, None], axis=1)
    equal = np.sum(similarities == true_score[:, None], axis=1) - 1
    ranks = 1.0 + greater + 0.5 * np.maximum(equal, 0)
    return {
        "top1": float(np.mean(ranks <= 1.0)),
        "top5": float(np.mean(ranks <= 5.0)),
        "top10": float(np.mean(ranks <= 10.0)),
        "mrr": float(np.mean(1.0 / ranks)),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
    }


def random_baseline(pool_size: int) -> dict[str, float]:
    harmonic = float(np.sum(1.0 / np.arange(1, pool_size + 1, dtype=np.float64)))
    return {
        "top1": 1.0 / pool_size,
        "top5": min(5, pool_size) / pool_size,
        "top10": min(10, pool_size) / pool_size,
        "mrr": harmonic / pool_size,
        "median_rank": (pool_size + 1.0) / 2.0,
        "mean_rank": (pool_size + 1.0) / 2.0,
    }


def summarize_trials(trials: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for metric in trials[0]:
        values = np.asarray([trial[metric] for trial in trials], dtype=np.float64)
        half = 0.0
        if len(values) > 1:
            half = 1.96 * float(np.std(values, ddof=1)) / math.sqrt(len(values))
        output[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "normal_ci95_low": float(np.mean(values) - half),
            "normal_ci95_high": float(np.mean(values) + half),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return output


def validate_files(configuration: dict[str, object]) -> None:
    missing = [str(path) for key in ("semantic", "lexical", "rows") if not (path := configuration[key]).exists()]
    if missing:
        raise FileNotFoundError("missing required cache files: " + ", ".join(missing))


def run_dataset(name: str, args: argparse.Namespace) -> dict[str, object]:
    configuration = DATASETS[name]
    validate_files(configuration)
    semantic = np.load(configuration["semantic"], mmap_mode="r")
    lexical = np.load(configuration["lexical"], mmap_mode="r")
    corpus_rows = np.load(configuration["rows"], mmap_mode="r")
    if len(lexical) != len(corpus_rows):
        raise ValueError(f"{name}: lexical coordinates and row map differ in length")
    if int(np.max(corpus_rows)) >= len(semantic):
        raise ValueError(f"{name}: a lexical row lies outside semantic coordinate table")

    max_anchors = max(args.anchor_counts)
    needed = max_anchors + args.pool_size
    if needed > len(corpus_rows):
        raise ValueError(f"{name}: need {needed} distinct rows, have {len(corpus_rows)}")

    methods = [
        "anchor_calibrated_percentile",
        "anchor_linear_calibration",
        "anchor_percentile",
        "distance_pearson",
        "within_document_rank",
    ]
    collected: dict[tuple[int, str], list[dict[str, float]]] = {
        (count, method): [] for count in args.anchor_counts for method in methods
    }
    control_methods = ["anchor_calibrated_percentile", "distance_pearson"]
    controls: dict[tuple[int, str], list[dict[str, float]]] = {
        (count, method): [] for count in args.anchor_counts for method in control_methods
    }
    repeat_seconds: list[float] = []

    for repeat in range(args.repeats):
        start = time.perf_counter()
        rng = np.random.default_rng(args.seed + 1009 * repeat + 17 * len(name))
        selected = rng.choice(len(corpus_rows), size=needed, replace=False)
        anchor_local = selected[:max_anchors]
        target_local = selected[max_anchors:]
        anchor_global = np.asarray(corpus_rows[anchor_local], dtype=np.int64)
        target_global = np.asarray(corpus_rows[target_local], dtype=np.int64)

        semantic_anchors = gather_rows(semantic, anchor_global)
        semantic_targets = gather_rows(semantic, target_global)
        lexical_anchors = gather_rows(lexical, anchor_local)
        lexical_targets = gather_rows(lexical, target_local)
        semantic_distances = distances_to_anchors(semantic_targets, semantic_anchors)
        lexical_distances = distances_to_anchors(lexical_targets, lexical_anchors)
        semantic_anchor_pairwise = distances_to_anchors(semantic_anchors, semantic_anchors)
        lexical_anchor_pairwise = distances_to_anchors(lexical_anchors, lexical_anchors)

        # This destroys the on-disk row-order correspondence.  The inverse is
        # retained only by the evaluator when computing ground-truth ranks.
        lexical_permutation = rng.permutation(args.pool_size)
        for count in args.anchor_counts:
            semantic_slice = semantic_distances[:, :count]
            lexical_slice = lexical_distances[:, :count]
            semantic_pairwise_slice = semantic_anchor_pairwise[:count, :count]
            lexical_pairwise_slice = lexical_anchor_pairwise[:count, :count]
            for method in methods:
                if method == "anchor_calibrated_percentile":
                    semantic_fp = reference_percentiles(
                        semantic_slice, semantic_pairwise_slice
                    )
                    lexical_fp = reference_percentiles(
                        lexical_slice, lexical_pairwise_slice
                    )
                elif method == "anchor_linear_calibration":
                    semantic_fp, lexical_fp = linearly_calibrated_fingerprints(
                        semantic_slice,
                        lexical_slice,
                        semantic_pairwise_slice,
                        lexical_pairwise_slice,
                    )
                else:
                    semantic_fp = fingerprints(semantic_slice, method)
                    lexical_fp = fingerprints(lexical_slice, method)
                collected[(count, method)].append(
                    matching_metrics(semantic_fp, lexical_fp, lexical_permutation)
                )

            # Negative control: break the known anchor correspondence while
            # preserving both marginal geometry distributions.
            shuffled_columns = rng.permutation(count)
            semantic_fp = reference_percentiles(
                semantic_slice, semantic_pairwise_slice
            )
            shuffled_lexical_pairwise = lexical_pairwise_slice[
                shuffled_columns
            ][:, shuffled_columns]
            lexical_fp = reference_percentiles(
                lexical_slice[:, shuffled_columns], shuffled_lexical_pairwise
            )
            controls[(count, "anchor_calibrated_percentile")].append(
                matching_metrics(semantic_fp, lexical_fp, lexical_permutation)
            )
            semantic_fp = fingerprints(semantic_slice, "distance_pearson")
            lexical_fp = fingerprints(
                lexical_slice[:, shuffled_columns], "distance_pearson"
            )
            controls[(count, "distance_pearson")].append(
                matching_metrics(semantic_fp, lexical_fp, lexical_permutation)
            )

        elapsed = time.perf_counter() - start
        repeat_seconds.append(elapsed)
        primary = collected[(max_anchors, "distance_pearson")][-1]
        print(
            f"{name:9s} repeat={repeat + 1}/{args.repeats} "
            f"anchors={max_anchors} top1={primary['top1']:.4f} "
            f"top10={primary['top10']:.4f} mrr={primary['mrr']:.4f} "
            f"seconds={elapsed:.2f}",
            flush=True,
        )

    rows = []
    for count in args.anchor_counts:
        for method in methods:
            rows.append(
                {
                    "known_anchors": count,
                    "method": method,
                    "summary": summarize_trials(collected[(count, method)]),
                    "trials": collected[(count, method)],
                }
            )
        for method in control_methods:
            rows.append(
                {
                    "known_anchors": count,
                    "method": f"{method}_shuffled_control",
                    "summary": summarize_trials(controls[(count, method)]),
                    "trials": controls[(count, method)],
                }
            )

    return {
        "dataset": name,
        "full_corpus_documents": int(configuration["documents"]),
        "semantic_coordinate_shape": list(semantic.shape),
        "lexical_materialized_coordinate_shape": list(lexical.shape),
        "materialized_lexical_fraction": float(len(lexical) / int(configuration["documents"])),
        "unknown_matching_pool": args.pool_size,
        "random_baseline": random_baseline(args.pool_size),
        "mean_seconds_per_repeat": float(np.mean(repeat_seconds)),
        "results": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=list(DATASETS))
    parser.add_argument(
        "--anchor-counts",
        nargs="+",
        type=int,
        default=[2, 4, 8, 16, 32, 64, 128, 256],
    )
    parser.add_argument("--pool-size", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/cross_path_anchor_fingerprint.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(count < 2 for count in args.anchor_counts):
        raise ValueError("at least two anchors are required to form a distance fingerprint")
    report = {
        "attack": "known-anchor cross-path DPE distance fingerprint linkage",
        "attacker_view": (
            "independently permuted semantic and lexical concrete DPE coordinates; "
            "known correspondences only for the specified anchors"
        ),
        "configuration": {
            "datasets": args.datasets,
            "anchor_counts": args.anchor_counts,
            "unknown_matching_pool": args.pool_size,
            "repeats": args.repeats,
            "seed": args.seed,
            "primary_method": "distance_pearson",
        },
        "scope_caveat": (
            "Unknown documents are sampled from lexical ciphertext rows materialized by the "
            "complete-corpus query workload. Reported accuracy is linkage within the stated "
            "pool, not exhaustive one-of-N matching over every corpus document."
        ),
        "datasets": [],
    }
    for dataset in args.datasets:
        report["datasets"].append(run_dataset(dataset, args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
