"""Reproduce the synchronized-query cross-compartment stitching attack.

The attack consumes only the compartment-local fp16 query coordinates that
the cloud receives, plus public timing/co-access information.  Secret selector
coordinates are used only after the attack to score recovery precision/recall.

Outputs:
  results/security/cross_compartment_stitching_nq.json
  results/security/cross_compartment_stitching_lexical.csv
  results/security/cross_compartment_stitching_semantic_pairs.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if (ROOT / ".deps").exists():
    sys.path.insert(0, str(ROOT / ".deps"))
os.environ.setdefault("DUETER_FORCE_CPU_DEPS", "1")

import numpy as np

from benchmark_dual_compartment_full import (
    FULL_DOCUMENTS,
    KEY_LEXICAL,
    KEY_SEMANTIC,
    apply_transform,
    cell_transform,
)
from benchmark_million_lexical_dpe import (
    encrypt_query_matrix,
    keyed_term,
    query_mips,
)
from benchmark_million_semantic_hybrid import (
    IvfFiles,
    encrypt_queries,
    load_queries_qrels,
)
from dueter_common import tokenize


BASE_SEED = 20260917
DATASET = "nq"
DOCUMENTS = FULL_DOCUMENTS[DATASET]
PROJECTION_DIMENSION = 256
LEXICAL_CELLS = 64
SEMANTIC_PROBES = 128
CORRELATION_THRESHOLD = 0.9999


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int32)
        self.rank = np.zeros(size, dtype=np.uint8)

    def find(self, value: int) -> int:
        parent = int(self.parent[value])
        while parent != int(self.parent[parent]):
            parent = int(self.parent[parent])
        while value != parent:
            following = int(self.parent[value])
            self.parent[value] = parent
            value = following
        return parent

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def quantized_local_views(
    global_queries: np.ndarray,
    transforms: list,
) -> np.ndarray:
    """Return exactly the fp16 coordinates visible to the cloud, as fp32."""
    return np.stack(
        [
            apply_transform(global_queries, transform)
            .astype(np.float16)
            .astype(np.float32)
            for transform in transforms
        ],
        axis=0,
    )


def normalized_traces(views: np.ndarray) -> np.ndarray:
    # (cell, query, coordinate) -> (cell * coordinate, query)
    centered = views - np.mean(views, axis=1, keepdims=True)
    traces = centered.transpose(0, 2, 1).reshape(-1, views.shape[1])
    norms = np.linalg.norm(traces, axis=1, keepdims=True)
    return traces / np.maximum(norms, 1e-12)


def true_pair_count(coordinates: np.ndarray) -> int:
    counts = np.bincount(coordinates.reshape(-1), minlength=int(coordinates.max()) + 1)
    return int(np.sum(counts * (counts - 1) // 2))


def lsh_correlation_edges(
    traces: np.ndarray,
    cells: np.ndarray,
    *,
    threshold: float,
    seed: int,
    tables: int = 10,
    bits: int = 12,
) -> tuple[list[tuple[int, int, float]], int]:
    """Find +/- correlated traces without using secret coordinate labels.

    Random-hyperplane LSH is only an acceleration layer. Candidate pairs are
    accepted exclusively by their observed Pearson correlation.
    """
    rng = np.random.default_rng(seed)
    mask = (1 << bits) - 1
    candidates: set[tuple[int, int]] = set()
    weights = (np.uint64(1) << np.arange(bits, dtype=np.uint64))[None, :]
    for _ in range(tables):
        planes = rng.normal(size=(traces.shape[1], bits)).astype(np.float32)
        raw = np.sum((traces @ planes >= 0).astype(np.uint64) * weights, axis=1)
        canonical = np.minimum(raw, np.uint64(mask) ^ raw)
        buckets: dict[int, list[int]] = defaultdict(list)
        for node, signature in enumerate(canonical):
            buckets[int(signature)].append(node)
        for members in buckets.values():
            for offset, left in enumerate(members):
                for right in members[offset + 1 :]:
                    if cells[left] == cells[right]:
                        continue
                    candidates.add((left, right) if left < right else (right, left))

    edges: list[tuple[int, int, float]] = []
    for left, right in candidates:
        correlation = float(np.dot(traces[left], traces[right]))
        if abs(correlation) >= threshold:
            edges.append((left, right, correlation))
    return edges, len(candidates)


def distance_matrix(values: np.ndarray) -> np.ndarray:
    squared = np.sum(values * values, axis=1)
    output = squared[:, None] + squared[None, :] - 2.0 * (values @ values.T)
    return np.maximum(output, 0.0)


def geometry_metrics(
    views: np.ndarray,
    global_queries: np.ndarray,
    edges: list[tuple[int, int, float]],
    multiplier: float,
) -> dict[str, float | int]:
    cells, query_count, projection = views.shape
    node_count = cells * projection
    union = UnionFind(node_count)
    for left, right, _ in edges:
        union.union(left, right)
    representatives: dict[int, int] = {}
    for node in range(node_count):
        representatives.setdefault(union.find(node), node)
    recovered = np.empty((query_count, len(representatives)), dtype=np.float32)
    for column, node in enumerate(representatives.values()):
        cell, coordinate = divmod(node, projection)
        recovered[:, column] = (
            views[cell, :, coordinate] - views[cell, 0, coordinate]
        ) / multiplier
    true_difference = global_queries - global_queries[0]
    true_distances = distance_matrix(true_difference)
    recovered_distances = distance_matrix(recovered)
    upper = np.triu_indices(query_count, k=1)
    truth = true_distances[upper].astype(np.float64)
    estimate = recovered_distances[upper].astype(np.float64)
    pearson = float(np.corrcoef(truth, estimate)[0, 1])
    calibration = float(np.dot(estimate, truth) / max(np.dot(estimate, estimate), 1e-12))
    nrmse = float(
        np.sqrt(np.mean((calibration * estimate - truth) ** 2))
        / max(float(np.mean(truth)), 1e-12)
    )
    k = min(10, query_count - 1)
    recalls = []
    for query in range(query_count):
        true_order = np.argsort(true_distances[query])[1 : k + 1]
        recovered_order = np.argsort(recovered_distances[query])[1 : k + 1]
        recalls.append(len(set(true_order).intersection(map(int, recovered_order))) / k)
    return {
        "predicted_components": len(representatives),
        "distance_pearson": pearson,
        "distance_nrmse_after_scalar_calibration": nrmse,
        "knn_recall_at_10": float(np.mean(recalls)),
    }


def lexical_document_frequencies(query_texts: list[str]) -> dict[str, int]:
    tokens = sorted(
        {
            keyed_term(term)
            for text in query_texts
            for term in dict.fromkeys(tokenize(text))
        }
    )
    database = ROOT / "results" / "keyed_fts5" / f"{DATASET}_full.sqlite3"
    connection = sqlite3.connect(database)
    result: dict[str, int] = {}
    for token in tokens:
        row = connection.execute(
            "SELECT doc FROM vocab WHERE term = ? LIMIT 1", (token,)
        ).fetchone()
        if row is not None:
            result[token] = int(row[0])
    connection.close()
    return result


def lexical_experiment(max_queries: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    source = ROOT / "cache" / "full_candidate_lexical_dpe" / "nq_full_h1024"
    _, query_texts, _, _ = load_queries_qrels(ROOT / "data" / DATASET, "test")
    query_texts = query_texts[:max_queries]
    document_frequency = lexical_document_frequencies(query_texts)
    plain = np.stack(
        [query_mips(text, document_frequency, DOCUMENTS, 1024) for text in query_texts]
    )
    global_queries = encrypt_query_matrix(
        plain,
        source / "lexical_dpe_key.npz",
        0.10,
        3.0,
        BASE_SEED + 9000 + 2027,
    )
    work_dimension = int(global_queries.shape[1])
    transforms = [
        cell_transform(KEY_LEXICAL, cell, work_dimension, PROJECTION_DIMENSION)
        for cell in range(LEXICAL_CELLS)
    ]
    coordinates = np.stack([item.coordinates for item in transforms])
    signs = np.stack([item.signs for item in transforms])
    translations = np.stack([item.translation for item in transforms])
    node_cells = np.repeat(np.arange(LEXICAL_CELLS), PROJECTION_DIMENSION)
    node_coordinates = coordinates.reshape(-1)
    node_signs = signs.reshape(-1)
    node_translations = translations.reshape(-1)
    total_true_pairs = true_pair_count(coordinates)
    multiplier = float(transforms[0].multiplier)
    rows: list[dict[str, object]] = []

    for query_count in (8, 16, 32, 64, 128):
        if query_count > len(global_queries):
            continue
        views = quantized_local_views(global_queries[:query_count], transforms)
        traces = normalized_traces(views)
        started = time.perf_counter()
        edges, candidates = lsh_correlation_edges(
            traces,
            node_cells,
            threshold=CORRELATION_THRESHOLD,
            seed=BASE_SEED + query_count,
        )
        attack_seconds = time.perf_counter() - started
        true_positive = 0
        correct_sign = 0
        intercept_errors: list[float] = []
        for left, right, correlation in edges:
            if node_coordinates[left] != node_coordinates[right]:
                continue
            true_positive += 1
            inferred_sign = 1.0 if correlation >= 0 else -1.0
            true_sign = float(node_signs[left] * node_signs[right])
            correct_sign += int(inferred_sign == true_sign)
            cell_left, coord_left = divmod(left, PROJECTION_DIMENSION)
            cell_right, coord_right = divmod(right, PROJECTION_DIMENSION)
            intercept = float(
                np.mean(
                    views[cell_right, :, coord_right]
                    - inferred_sign * views[cell_left, :, coord_left]
                )
            )
            target = float(
                node_translations[right] - inferred_sign * node_translations[left]
            )
            intercept_errors.append(abs(intercept - target))
        false_positive = len(edges) - true_positive
        precision = true_positive / max(len(edges), 1)
        recall = true_positive / max(total_true_pairs, 1)
        unique_coordinates = int(len(np.unique(coordinates)))
        geometry = geometry_metrics(
            views, global_queries[:query_count], edges, multiplier
        )
        row: dict[str, object] = {
            "queries": query_count,
            "candidate_pairs": candidates,
            "accepted_pairs": len(edges),
            "true_pairs": total_true_pairs,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "precision": precision,
            "recall": recall,
            "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
            "relative_sign_accuracy": correct_sign / max(true_positive, 1),
            "translation_difference_mae": float(np.mean(intercept_errors))
            if intercept_errors
            else None,
            "attack_seconds": attack_seconds,
            **geometry,
        }
        rows.append(row)

    multiplicities = np.bincount(coordinates.reshape(-1), minlength=work_dimension)
    summary = {
        "path": "lexical",
        "dataset": DATASET,
        "cells": LEXICAL_CELLS,
        "work_dimension": work_dimension,
        "projection_dimension": PROJECTION_DIMENSION,
        "pairwise_expected_overlap_r2_over_d": PROJECTION_DIMENSION**2
        / work_dimension,
        "observed_mean_pairwise_overlap": float(
            np.mean(
                [
                    len(np.intersect1d(coordinates[a], coordinates[b]))
                    for a in range(LEXICAL_CELLS)
                    for b in range(a + 1, LEXICAL_CELLS)
                ]
            )
        ),
        "latent_coordinate_coverage": int(np.count_nonzero(multiplicities)),
        "latent_coordinate_coverage_fraction": float(np.count_nonzero(multiplicities))
        / work_dimension,
        "mean_views_per_latent_coordinate": float(np.mean(multiplicities)),
        "max_views_per_latent_coordinate": int(np.max(multiplicities)),
        "server_coordinate_dtype": "float16",
        "correlation_threshold": CORRELATION_THRESHOLD,
    }
    return rows, summary


def semantic_pair_attack(
    global_queries: np.ndarray,
    query_cells: np.ndarray,
    cell_a: int,
    cell_b: int,
) -> dict[str, object]:
    common = np.flatnonzero(
        np.any(query_cells == cell_a, axis=1) & np.any(query_cells == cell_b, axis=1)
    )
    transform_a = cell_transform(
        KEY_SEMANTIC, cell_a, global_queries.shape[1], PROJECTION_DIMENSION
    )
    transform_b = cell_transform(
        KEY_SEMANTIC, cell_b, global_queries.shape[1], PROJECTION_DIMENSION
    )
    view_a = (
        apply_transform(global_queries[common], transform_a)
        .astype(np.float16)
        .astype(np.float32)
    )
    view_b = (
        apply_transform(global_queries[common], transform_b)
        .astype(np.float16)
        .astype(np.float32)
    )
    centered_a = view_a - np.mean(view_a, axis=0, keepdims=True)
    centered_b = view_b - np.mean(view_b, axis=0, keepdims=True)
    normalized_a = centered_a / np.maximum(
        np.linalg.norm(centered_a, axis=0, keepdims=True), 1e-12
    )
    normalized_b = centered_b / np.maximum(
        np.linalg.norm(centered_b, axis=0, keepdims=True), 1e-12
    )
    correlations = normalized_a.T @ normalized_b
    found = np.argwhere(np.abs(correlations) >= CORRELATION_THRESHOLD)
    coordinate_b = {int(value): index for index, value in enumerate(transform_b.coordinates)}
    truth = {
        (index_a, coordinate_b[int(value)])
        for index_a, value in enumerate(transform_a.coordinates)
        if int(value) in coordinate_b
    }
    predicted = {(int(a), int(b)) for a, b in found}
    true_positive = len(predicted.intersection(truth))
    intercepts: list[tuple[int, int, float, float]] = []
    sign_correct = 0
    intercept_errors: list[float] = []
    for a, b in predicted:
        sign = 1.0 if correlations[a, b] >= 0 else -1.0
        intercept = float(np.mean(view_b[:, b] - sign * view_a[:, a]))
        intercepts.append((a, b, sign, intercept))
        if (a, b) in truth:
            expected_sign = float(transform_a.signs[a] * transform_b.signs[b])
            sign_correct += int(sign == expected_sign)
            target = float(transform_b.translation[b] - sign * transform_a.translation[a])
            intercept_errors.append(abs(intercept - target))
    return {
        "cell_a": cell_a,
        "cell_b": cell_b,
        "coaccess_queries": int(len(common)),
        "true_overlap": len(truth),
        "predicted_matches": len(predicted),
        "true_positive": true_positive,
        "false_positive": len(predicted) - true_positive,
        "precision": true_positive / max(len(predicted), 1),
        "recall": true_positive / max(len(truth), 1),
        "relative_sign_accuracy": sign_correct / max(true_positive, 1),
        "translation_difference_mae": float(np.mean(intercept_errors))
        if intercept_errors
        else None,
        "matches": intercepts,
    }


def shared_documents(ivf: IvfFiles, cell_a: int, cell_b: int) -> np.ndarray:
    a0, a1 = int(ivf.offsets[cell_a]), int(ivf.offsets[cell_a + 1])
    b0, b1 = int(ivf.offsets[cell_b]), int(ivf.offsets[cell_b + 1])
    return np.intersect1d(
        np.asarray(ivf.postings[a0:a1], dtype=np.int32),
        np.asarray(ivf.postings[b0:b1], dtype=np.int32),
        assume_unique=True,
    )


def replica_linkage(
    pair_result: dict[str, object],
    ivf: IvfFiles,
    compartment: Path,
    sample_limit: int = 100,
) -> dict[str, object]:
    cell_a, cell_b = int(pair_result["cell_a"]), int(pair_result["cell_b"])
    common = shared_documents(ivf, cell_a, cell_b)
    a0, a1 = int(ivf.offsets[cell_a]), int(ivf.offsets[cell_a + 1])
    b0, b1 = int(ivf.offsets[cell_b]), int(ivf.offsets[cell_b + 1])
    docs_a = np.asarray(ivf.postings[a0:a1], dtype=np.int32)
    docs_b = np.asarray(ivf.postings[b0:b1], dtype=np.int32)
    local = np.load(compartment / "semantic_compartment_fp16.npy", mmap_mode="r")
    rows_a = np.asarray(local[a0:a1], dtype=np.float32)
    rows_b = np.asarray(local[b0:b1], dtype=np.float32)
    matches = [tuple(item) for item in pair_result["matches"]]
    if not len(common) or not matches:
        return {
            "cell_a": cell_a,
            "cell_b": cell_b,
            "shared_replicas": int(len(common)),
            "tested_replicas": 0,
            "top1_linkage_accuracy": None,
            "top10_linkage_accuracy": None,
        }
    index_a = np.asarray([int(item[0]) for item in matches], dtype=np.int32)
    index_b = np.asarray([int(item[1]) for item in matches], dtype=np.int32)
    signs = np.asarray([float(item[2]) for item in matches], dtype=np.float32)
    intercepts = np.asarray([float(item[3]) for item in matches], dtype=np.float32)
    rng = np.random.default_rng(BASE_SEED)
    tested = np.sort(
        rng.choice(common, size=min(sample_limit, len(common)), replace=False)
    )
    top1 = 0
    top10 = 0
    ranks: list[int] = []
    for document in tested:
        position_a = int(np.flatnonzero(docs_a == document)[0])
        position_b = int(np.flatnonzero(docs_b == document)[0])
        expected = signs * rows_a[position_a, index_a] + intercepts
        errors = np.mean((rows_b[:, index_b] - expected[None, :]) ** 2, axis=1)
        rank = 1 + int(np.sum(errors < errors[position_b]))
        ranks.append(rank)
        top1 += int(rank == 1)
        top10 += int(rank <= 10)
    return {
        "cell_a": cell_a,
        "cell_b": cell_b,
        "shared_replicas": int(len(common)),
        "tested_replicas": int(len(tested)),
        "matched_coordinates_used": int(len(matches)),
        "cell_b_candidates": int(len(docs_b)),
        "top1_linkage_accuracy": top1 / len(tested),
        "top10_linkage_accuracy": top10 / len(tested),
        "median_true_replica_rank": float(np.median(ranks)),
    }


def semantic_experiment() -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    source = ROOT / "cache" / "full_semantic" / f"nq_{DOCUMENTS}"
    compartment = ROOT / "cache" / "dual_compartment_full_p256" / f"nq_{DOCUMENTS}"
    queries = np.asarray(np.load(source / "query_embeddings.npy", mmap_mode="r"), dtype=np.float32)
    ivf = IvfFiles.load(source)
    residual = queries - ivf.mean[None, :]
    residual /= np.maximum(np.linalg.norm(residual, axis=1, keepdims=True), 1e-12)
    routing_scores = residual @ np.asarray(ivf.centroids, dtype=np.float32).T
    query_cells = np.argpartition(
        -routing_scores, SEMANTIC_PROBES - 1, axis=1
    )[:, :SEMANTIC_PROBES].astype(np.int32)
    del routing_scores
    global_queries = encrypt_queries(
        queries,
        source / "semantic_dpe_key.npz",
        0.10,
        3.0,
        BASE_SEED + 2027,
    )
    counts = np.bincount(query_cells.reshape(-1), minlength=len(ivf.offsets) - 1)
    top_cells = np.argsort(-counts)[:32].astype(np.int32)
    masks = {int(cell): np.any(query_cells == int(cell), axis=1) for cell in top_cells}
    candidates: list[tuple[int, int, int, int]] = []
    for offset, raw_a in enumerate(top_cells):
        a = int(raw_a)
        for raw_b in top_cells[offset + 1 :]:
            b = int(raw_b)
            coaccess = int(np.count_nonzero(masks[a] & masks[b]))
            shared = int(len(shared_documents(ivf, a, b)))
            if coaccess >= 16:
                candidates.append((coaccess, shared, a, b))
    selected = sorted(candidates, reverse=True)[:12]
    replica_candidates = sorted(candidates, key=lambda row: (row[1], row[0]), reverse=True)
    if replica_candidates and replica_candidates[0] not in selected:
        selected.append(replica_candidates[0])

    rows = [semantic_pair_attack(global_queries, query_cells, a, b) for _, _, a, b in selected]
    aggregate_true = sum(int(row["true_overlap"]) for row in rows)
    aggregate_tp = sum(int(row["true_positive"]) for row in rows)
    aggregate_predicted = sum(int(row["predicted_matches"]) for row in rows)
    best_replica_row = max(
        rows,
        key=lambda row: len(
            shared_documents(ivf, int(row["cell_a"]), int(row["cell_b"]))
        ),
    )
    linkage = replica_linkage(best_replica_row, ivf, compartment)
    summary = {
        "path": "semantic",
        "dataset": DATASET,
        "queries": int(len(queries)),
        "probes_per_query": SEMANTIC_PROBES,
        "work_dimension": int(global_queries.shape[1]),
        "projection_dimension": PROJECTION_DIMENSION,
        "top_cells_examined": int(len(top_cells)),
        "cell_pairs_tested": int(len(rows)),
        "coaccess_query_min": int(min(int(row["coaccess_queries"]) for row in rows)),
        "coaccess_query_median": float(
            np.median([int(row["coaccess_queries"]) for row in rows])
        ),
        "coaccess_query_max": int(max(int(row["coaccess_queries"]) for row in rows)),
        "aggregate_precision": aggregate_tp / max(aggregate_predicted, 1),
        "aggregate_recall": aggregate_tp / max(aggregate_true, 1),
        "correlation_threshold": CORRELATION_THRESHOLD,
    }
    return rows, summary, linkage


def write_csv(path: Path, rows: list[dict[str, object]], excluded: set[str] | None = None) -> None:
    excluded = excluded or set()
    scalar_rows = [
        {key: value for key, value in row.items() if key not in excluded}
        for row in rows
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scalar_rows[0]))
        writer.writeheader()
        writer.writerows(scalar_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lexical-queries", type=int, default=128)
    args = parser.parse_args()
    destination = ROOT / "results" / "security"
    destination.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    lexical_rows, lexical_summary = lexical_experiment(args.lexical_queries)
    semantic_rows, semantic_summary, linkage = semantic_experiment()
    report = {
        "experiment": "synchronized cross-compartment stitching",
        "threat_model": "honest-but-curious cloud; fp16 local query coordinates and Time/co-access leakage only",
        "dataset": DATASET,
        "seed": BASE_SEED,
        "lexical": {"summary": lexical_summary, "query_sweep": lexical_rows},
        "semantic": {
            "summary": semantic_summary,
            "pair_results": [
                {key: value for key, value in row.items() if key != "matches"}
                for row in semantic_rows
            ],
            "replica_linkage": linkage,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    json_path = destination / "cross_compartment_stitching_nq.json"
    lexical_csv = destination / "cross_compartment_stitching_lexical.csv"
    semantic_csv = destination / "cross_compartment_stitching_semantic_pairs.csv"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(lexical_csv, lexical_rows)
    write_csv(semantic_csv, semantic_rows, excluded={"matches"})
    print(json.dumps(report, indent=2))
    print(f"wrote {json_path}")
    print(f"wrote {lexical_csv}")
    print(f"wrote {semantic_csv}")


if __name__ == "__main__":
    main()
