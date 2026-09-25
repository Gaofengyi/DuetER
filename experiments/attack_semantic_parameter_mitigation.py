"""Audit semantic parameter changes against synchronized-query stitching.

The attacker uses only query co-access and the FP16 compartment-local query
coordinates.  Secret selectors/signs/translations are consulted only after the
attack to score recovery.  Besides sampled Pearson-correlation attacks, the
script computes the full compartment graph whose edges have both at least
MIN_COMMON synchronized queries and at least one repeated latent coordinate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from benchmark_million_semantic_hybrid import (
    IvfFiles,
    atomic_json,
    ball_noise,
    dpe_transform,
    encrypt_queries,
    load_queries_qrels,
)


ROOT = Path(__file__).resolve().parent
KEY_SEMANTIC = b"DuetDPE-semantic-compartment-v2"
BASE_SEED = 20260917
GLOBAL_DPE_SEED = 20260823
QUERY_NOISE_SEED = BASE_SEED + 2027
SCALE = 3.0
BETA = 0.10
QUERY_NOISE_RADIUS = SCALE * BETA / 8.0
CELLS = 2048
MIN_COMMON = 8
CORRELATION_THRESHOLD = 0.9999
PROBES = (128, 64, 32, 16, 8)
TRACE_QUERIES = (4, 8, 16, 32)
TRANSCRIPT_SIZES = (64, 128, 256, 512, 1024)
SAMPLED_PAIRS = 96

DATASETS = {
    "nq": {"documents": 2_681_468, "split": "test", "query_file": "query_embeddings.npy"},
    "hotpotqa": {"documents": 5_233_329, "split": "test", "query_file": "query_embeddings.npy"},
    "msmarco": {"documents": 8_841_823, "split": "dev", "query_file": "query_embeddings_dev.npy"},
}

CONFIGS = (
    {"name": "current_d512_rc256", "work_dimension": 512, "projection_dimension": 256},
    {"name": "revised_d8192_rc32", "work_dimension": 8192, "projection_dimension": 32},
    {"name": "revised_d8192_rc64", "work_dimension": 8192, "projection_dimension": 64},
)


@dataclass(frozen=True)
class CellTransform:
    coordinates: np.ndarray
    signs: np.ndarray
    multiplier: float
    translation: np.ndarray


def cell_transform(cell: int, work_dimension: int, projection_dimension: int) -> CellTransform:
    digest = hashlib.sha256(
        KEY_SEMANTIC
        + int(cell).to_bytes(8, "big", signed=False)
        + int(work_dimension).to_bytes(4, "big", signed=False)
        + int(projection_dimension).to_bytes(4, "big", signed=False)
    ).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big", signed=False))
    coordinates = np.sort(
        rng.choice(work_dimension, size=projection_dimension, replace=False)
    ).astype(np.int32)
    signs = rng.choice(
        np.asarray([-1.0, 1.0], dtype=np.float32), projection_dimension
    )
    return CellTransform(
        coordinates=coordinates,
        signs=signs,
        multiplier=math.sqrt(work_dimension / projection_dimension),
        translation=rng.normal(0.0, 1.0, projection_dimension).astype(np.float32),
    )


def transforms_for(work_dimension: int, projection_dimension: int) -> list[CellTransform]:
    return [
        cell_transform(cell, work_dimension, projection_dimension)
        for cell in range(CELLS)
    ]


def selector_overlap_matrix(
    transforms: list[CellTransform], work_dimension: int, device: str
) -> np.ndarray:
    selector = torch.zeros(
        (CELLS, work_dimension), dtype=torch.float32, device=device
    )
    rows = torch.arange(CELLS, device=device)[:, None].expand(
        CELLS, len(transforms[0].coordinates)
    )
    columns = torch.from_numpy(
        np.stack([transform.coordinates for transform in transforms])
    ).to(device)
    selector[rows, columns] = 1.0
    with torch.inference_mode():
        overlap = torch.round(selector @ selector.T).to(torch.int16).cpu().numpy()
    np.fill_diagonal(overlap, 0)
    return overlap


def route_queries(source: Path, queries: np.ndarray, device: str) -> np.ndarray:
    ivf = IvfFiles.load(source)
    residual = queries - ivf.mean[None, :]
    residual /= np.maximum(np.linalg.norm(residual, axis=1, keepdims=True), 1e-12)
    with torch.inference_mode():
        scores = torch.from_numpy(residual).to(device) @ torch.from_numpy(
            ivf.centroids
        ).to(device).T
        cells = torch.topk(scores, k=max(PROBES), dim=1, sorted=True).indices
        return cells.cpu().numpy().astype(np.int32)


def coaccess_matrix(query_cells: np.ndarray, probes: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    incidence = torch.zeros(
        (len(query_cells), CELLS), dtype=torch.float32, device=device
    )
    rows = torch.arange(len(query_cells), device=device)[:, None].expand(
        len(query_cells), probes
    )
    columns = torch.from_numpy(query_cells[:, :probes]).to(device)
    incidence[rows, columns] = 1.0
    with torch.inference_mode():
        coaccess = torch.round(incidence.T @ incidence).to(torch.int16).cpu().numpy()
    active = (incidence.sum(dim=0) > 0).cpu().numpy()
    np.fill_diagonal(coaccess, 0)
    return coaccess, active


def graph_metrics(
    coaccess: np.ndarray,
    active: np.ndarray,
    overlap: np.ndarray,
    *,
    minimum_common: int,
) -> dict[str, float | int]:
    upper = np.triu(np.ones_like(coaccess, dtype=bool), k=1)
    eligible = upper & (coaccess >= minimum_common)
    attackable = eligible & (overlap > 0)
    eligible_pairs = int(np.count_nonzero(eligible))
    attackable_pairs = int(np.count_nonzero(attackable))
    recovered_matches = int(np.sum(overlap[attackable], dtype=np.int64))
    total_repeated_matches = int(np.sum(overlap[upper], dtype=np.int64))
    graph = attackable | attackable.T
    labels = np.full(len(graph), -1, dtype=np.int32)
    component_count = 0
    for start in range(len(graph)):
        if labels[start] >= 0:
            continue
        members = np.zeros(len(graph), dtype=bool)
        frontier = np.zeros(len(graph), dtype=bool)
        members[start] = True
        frontier[start] = True
        while np.any(frontier):
            neighbors = np.any(graph[frontier], axis=0) & ~members
            if not np.any(neighbors):
                break
            members |= neighbors
            frontier = neighbors
        labels[members] = component_count
        component_count += 1
    active_labels = labels[active]
    if len(active_labels):
        counts = np.bincount(active_labels, minlength=component_count)
        largest = int(np.max(counts))
    else:
        largest = 0
    active_count = int(np.count_nonzero(active))
    return {
        "active_compartments": active_count,
        "eligible_coaccess_pairs": eligible_pairs,
        "attackable_overlap_pairs": attackable_pairs,
        "attackable_pair_fraction": attackable_pairs / max(eligible_pairs, 1),
        "recovered_coordinate_pair_matches": recovered_matches,
        "all_selector_coordinate_pair_matches": total_repeated_matches,
        "coordinate_pair_recovery_fraction": recovered_matches
        / max(total_repeated_matches, 1),
        "connected_components_over_all_cells": int(component_count),
        "largest_component_active_cells": largest,
        "largest_component_active_fraction": largest / max(active_count, 1),
    }


def global_queries_8192(queries: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(GLOBAL_DPE_SEED)
    sign1 = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), 8192)
    sign2 = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), 8192)
    permutation = rng.permutation(8192)
    values = SCALE * dpe_transform(queries, sign1, sign2, permutation)
    noise_rng = np.random.default_rng(QUERY_NOISE_SEED)
    values += ball_noise(noise_rng, len(queries), 8192, QUERY_NOISE_RADIUS)
    return values.astype(np.float32)


def local_view(global_queries: np.ndarray, transform: CellTransform, rows: np.ndarray) -> np.ndarray:
    return (
        global_queries[rows][:, transform.coordinates]
        * transform.signs
        * transform.multiplier
        + transform.translation
    ).astype(np.float16).astype(np.float32)


def correlation_validation(
    global_queries: np.ndarray,
    query_cells: np.ndarray,
    probes: int,
    coaccess: np.ndarray,
    transforms: list[CellTransform],
    trace_queries: int,
) -> dict[str, float | int | None]:
    upper_a, upper_b = np.triu_indices(CELLS, k=1)
    counts = coaccess[upper_a, upper_b]
    eligible = np.flatnonzero(counts >= trace_queries)
    if not len(eligible):
        return {
            "trace_queries": trace_queries,
            "pairs_tested": 0,
            "true_matches": 0,
            "predicted_matches": 0,
            "precision": None,
            "recall": None,
            "relative_sign_accuracy": None,
            "translation_difference_mae": None,
        }
    order = eligible[np.argsort(-counts[eligible], kind="stable")[:SAMPLED_PAIRS]]
    true_total = 0
    predicted_total = 0
    true_positive = 0
    sign_correct = 0
    intercept_errors: list[float] = []
    tested = 0
    selected_cells = query_cells[:, :probes]
    for position in order:
        cell_a, cell_b = int(upper_a[position]), int(upper_b[position])
        common = np.flatnonzero(
            np.any(selected_cells == cell_a, axis=1)
            & np.any(selected_cells == cell_b, axis=1)
        )
        if len(common) < trace_queries:
            continue
        common = common[:trace_queries]
        transform_a, transform_b = transforms[cell_a], transforms[cell_b]
        view_a = local_view(global_queries, transform_a, common)
        view_b = local_view(global_queries, transform_b, common)
        centered_a = view_a - np.mean(view_a, axis=0, keepdims=True)
        centered_b = view_b - np.mean(view_b, axis=0, keepdims=True)
        norm_a = centered_a / np.maximum(
            np.linalg.norm(centered_a, axis=0, keepdims=True), 1e-12
        )
        norm_b = centered_b / np.maximum(
            np.linalg.norm(centered_b, axis=0, keepdims=True), 1e-12
        )
        correlations = norm_a.T @ norm_b
        found = np.argwhere(np.abs(correlations) >= CORRELATION_THRESHOLD)
        lookup_b = {
            int(value): index for index, value in enumerate(transform_b.coordinates)
        }
        truth = {
            (index_a, lookup_b[int(value)])
            for index_a, value in enumerate(transform_a.coordinates)
            if int(value) in lookup_b
        }
        predicted = {(int(a), int(b)) for a, b in found}
        intersections = predicted.intersection(truth)
        true_total += len(truth)
        predicted_total += len(predicted)
        true_positive += len(intersections)
        for a, b in intersections:
            inferred_sign = 1.0 if correlations[a, b] >= 0 else -1.0
            expected_sign = float(transform_a.signs[a] * transform_b.signs[b])
            sign_correct += int(inferred_sign == expected_sign)
            intercept = float(np.mean(view_b[:, b] - inferred_sign * view_a[:, a]))
            target = float(
                transform_b.translation[b]
                - inferred_sign * transform_a.translation[a]
            )
            intercept_errors.append(abs(intercept - target))
        tested += 1
    precision = true_positive / max(predicted_total, 1)
    recall = true_positive / max(true_total, 1)
    return {
        "trace_queries": trace_queries,
        "pairs_tested": tested,
        "true_matches": true_total,
        "predicted_matches": predicted_total,
        "true_positive": true_positive,
        "false_positive": predicted_total - true_positive,
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "relative_sign_accuracy": sign_correct / max(true_positive, 1),
        "translation_difference_mae": float(np.mean(intercept_errors))
        if intercept_errors
        else None,
    }


def run_dataset(dataset: str, device: str) -> dict[str, object]:
    cfg = DATASETS[dataset]
    source = ROOT / "cache" / "full_semantic" / f"{dataset}_{cfg['documents']}"
    query_ids, _, _, _ = load_queries_qrels(ROOT / "data" / dataset, str(cfg["split"]))
    queries = np.asarray(
        np.load(source / str(cfg["query_file"]), mmap_mode="r")[: len(query_ids)],
        dtype=np.float32,
    )
    query_cells = route_queries(source, queries, device)
    print(f"[{dataset}] routed {len(queries):,} queries", flush=True)

    transform_sets: dict[str, list[CellTransform]] = {}
    overlaps: dict[str, np.ndarray] = {}
    for config in CONFIGS:
        name = str(config["name"])
        transforms = transforms_for(
            int(config["work_dimension"]), int(config["projection_dimension"])
        )
        transform_sets[name] = transforms
        overlaps[name] = selector_overlap_matrix(
            transforms, int(config["work_dimension"]), device
        )
        print(f"[{dataset}] overlap matrix {name}", flush=True)

    transcript_sizes = sorted(
        set(size for size in TRANSCRIPT_SIZES if size <= len(queries)) | {len(queries)}
    )
    graph_rows: list[dict[str, object]] = []
    full_coaccess: dict[int, np.ndarray] = {}
    for transcript in transcript_sizes:
        for probes in PROBES:
            coaccess, active = coaccess_matrix(
                query_cells[:transcript], probes, device
            )
            if transcript == len(queries):
                full_coaccess[probes] = coaccess
            for config in CONFIGS:
                name = str(config["name"])
                row = {
                    "transcript_queries": transcript,
                    "nprobe": probes,
                    "configuration": name,
                    "work_dimension": int(config["work_dimension"]),
                    "projection_dimension": int(config["projection_dimension"]),
                    "minimum_common_queries": MIN_COMMON,
                    **graph_metrics(
                        coaccess,
                        active,
                        overlaps[name],
                        minimum_common=MIN_COMMON,
                    ),
                }
                graph_rows.append(row)
        print(f"[{dataset}] graph transcript={transcript}", flush=True)

    global_by_dimension = {
        512: encrypt_queries(
            queries,
            source / "semantic_dpe_key.npz",
            BETA,
            SCALE,
            QUERY_NOISE_SEED,
        ),
        8192: global_queries_8192(queries),
    }
    validation_rows: list[dict[str, object]] = []
    for config in CONFIGS:
        name = str(config["name"])
        global_queries = global_by_dimension[int(config["work_dimension"])]
        for probes in PROBES:
            for trace_queries in TRACE_QUERIES:
                row = correlation_validation(
                    global_queries,
                    query_cells,
                    probes,
                    full_coaccess[probes],
                    transform_sets[name],
                    trace_queries,
                )
                validation_rows.append(
                    {
                        "configuration": name,
                        "work_dimension": int(config["work_dimension"]),
                        "projection_dimension": int(config["projection_dimension"]),
                        "nprobe": probes,
                        **row,
                    }
                )
        print(f"[{dataset}] correlation validation {name}", flush=True)

    selector_summaries = []
    upper = np.triu(np.ones((CELLS, CELLS), dtype=bool), k=1)
    for config in CONFIGS:
        name = str(config["name"])
        values = overlaps[name][upper].astype(np.float64)
        selector_summaries.append(
            {
                **config,
                "expected_pair_overlap": int(config["projection_dimension"]) ** 2
                / int(config["work_dimension"]),
                "observed_mean_pair_overlap": float(np.mean(values)),
                "pair_overlap_probability": float(np.mean(values > 0)),
                "total_repeated_coordinate_pairs": int(np.sum(values)),
            }
        )
    result = {
        "experiment": "semantic synchronized-query stitching parameter audit",
        "dataset": dataset,
        "queries": len(queries),
        "cells": CELLS,
        "threat_model": (
            "honest-but-curious cloud observing FP16 local query coordinates, "
            "query ordering, and compartment co-access"
        ),
        "correlation_threshold": CORRELATION_THRESHOLD,
        "minimum_common_queries_for_graph_edge": MIN_COMMON,
        "sampled_pairs_per_correlation_test": SAMPLED_PAIRS,
        "selector_summaries": selector_summaries,
        "graph_rows": graph_rows,
        "correlation_validation": validation_rows,
    }
    destination = ROOT / "results" / "security" / "semantic_parameter_mitigation"
    destination.mkdir(parents=True, exist_ok=True)
    atomic_json(destination / f"{dataset}.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DATASETS) + ("all",), default="all")
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="merge completed per-dataset JSON files without rerunning attacks",
    )
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    selected = tuple(DATASETS) if args.dataset == "all" else (args.dataset,)
    started = time.perf_counter()
    report = {
        "experiment": "semantic synchronized-query stitching parameter audit",
        "device": args.device,
        "datasets": {},
    }
    destination = ROOT / "results" / "security" / "semantic_parameter_mitigation"
    if args.merge_existing:
        for dataset in DATASETS:
            result_path = destination / f"{dataset}.json"
            if result_path.exists():
                report["datasets"][dataset] = json.loads(
                    result_path.read_text(encoding="utf-8")
                )
        atomic_json(destination / "summary.json", report)
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
        return
    for dataset in selected:
        report["datasets"][dataset] = run_dataset(dataset, args.device)
        atomic_json(
            destination / "summary.json",
            report,
        )
    report["elapsed_seconds"] = time.perf_counter() - started
    atomic_json(
        destination / "summary.json",
        report,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
