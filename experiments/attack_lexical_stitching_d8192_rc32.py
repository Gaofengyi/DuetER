"""Reviewer-requested lexical stitching attack at d=8192 and r_c=32."""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if (ROOT / ".deps").exists():
    sys.path.insert(0, str(ROOT / ".deps"))
os.environ.setdefault("DUETER_FORCE_CPU_DEPS", "1")

import numpy as np

from attack_cross_compartment_stitching import (
    CORRELATION_THRESHOLD,
    geometry_metrics,
    lexical_document_frequencies,
    lsh_correlation_edges,
    normalized_traces,
    quantized_local_views,
    true_pair_count,
)
from attack_lexical_stitching_rc32 import compartment_graph
from benchmark_dual_compartment_full import KEY_LEXICAL, cell_transform
from benchmark_million_lexical_dpe import query_mips
from benchmark_million_semantic_hybrid import (
    atomic_json,
    ball_noise,
    dpe_transform,
    load_queries_qrels,
)
from evaluate_lexical_d8192_rc32 import (
    BASE_SEED,
    BETA,
    CELLS,
    DOCUMENTS,
    GLOBAL_PERMUTATION,
    GLOBAL_SIGN1,
    GLOBAL_SIGN2,
    HASH_DIMENSION,
    NOISE_RADIUS,
    PROJECTION_DIMENSION,
    SCALE,
    WORK_DIMENSION,
)


QUERY_COUNTS = (8, 16, 32, 64, 128, 256, 512)


def main() -> None:
    started = time.perf_counter()
    _, query_texts, _, _ = load_queries_qrels(ROOT / "data" / "nq", "test")
    query_texts = query_texts[: max(QUERY_COUNTS)]
    document_frequency = lexical_document_frequencies(query_texts)
    plain = np.stack(
        [
            query_mips(text, document_frequency, DOCUMENTS, HASH_DIMENSION)
            for text in query_texts
        ]
    )
    global_queries = SCALE * dpe_transform(
        plain, GLOBAL_SIGN1, GLOBAL_SIGN2, GLOBAL_PERMUTATION
    )
    global_queries += ball_noise(
        np.random.default_rng(BASE_SEED + 2027),
        len(plain),
        WORK_DIMENSION,
        NOISE_RADIUS,
    )
    transforms = [
        cell_transform(
            KEY_LEXICAL, cell, WORK_DIMENSION, PROJECTION_DIMENSION
        )
        for cell in range(CELLS)
    ]
    coordinates = np.stack([item.coordinates for item in transforms])
    signs = np.stack([item.signs for item in transforms]).reshape(-1)
    translations = np.stack([item.translation for item in transforms]).reshape(-1)
    node_coordinates = coordinates.reshape(-1)
    node_cells = np.repeat(np.arange(CELLS), PROJECTION_DIMENSION)
    total_true_pairs = true_pair_count(coordinates)
    multiplicities = np.bincount(
        node_coordinates, minlength=WORK_DIMENSION
    )
    repeated_coordinates = set(map(int, np.flatnonzero(multiplicities >= 2)))
    graph = compartment_graph(coordinates)
    rows: list[dict[str, object]] = []

    for query_count in QUERY_COUNTS:
        views = quantized_local_views(global_queries[:query_count], transforms)
        traces = normalized_traces(views)
        attack_started = time.perf_counter()
        edges, candidates = lsh_correlation_edges(
            traces,
            node_cells,
            threshold=CORRELATION_THRESHOLD,
            seed=BASE_SEED + WORK_DIMENSION + query_count,
        )
        attack_seconds = time.perf_counter() - attack_started
        true_positive = 0
        correct_sign = 0
        recovered_latent: set[int] = set()
        intercept_errors: list[float] = []
        for left, right, correlation in edges:
            if node_coordinates[left] != node_coordinates[right]:
                continue
            true_positive += 1
            recovered_latent.add(int(node_coordinates[left]))
            inferred_sign = 1.0 if correlation >= 0 else -1.0
            true_sign = float(signs[left] * signs[right])
            correct_sign += int(inferred_sign == true_sign)
            cell_left, coordinate_left = divmod(left, PROJECTION_DIMENSION)
            cell_right, coordinate_right = divmod(right, PROJECTION_DIMENSION)
            intercept = float(
                np.mean(
                    views[cell_right, :, coordinate_right]
                    - inferred_sign * views[cell_left, :, coordinate_left]
                )
            )
            target = float(
                translations[right] - inferred_sign * translations[left]
            )
            intercept_errors.append(abs(intercept - target))
        precision = true_positive / max(len(edges), 1)
        recall = true_positive / max(total_true_pairs, 1)
        row = {
            "queries": query_count,
            "candidate_pairs": candidates,
            "accepted_pairs": len(edges),
            "true_pairs": total_true_pairs,
            "true_positive": true_positive,
            "false_positive": len(edges) - true_positive,
            "precision": precision,
            "recall": recall,
            "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
            "relative_sign_accuracy": correct_sign / max(true_positive, 1),
            "translation_difference_mae": float(np.mean(intercept_errors)),
            "recovered_repeated_latent_coordinates": len(recovered_latent),
            "repeated_latent_coordinate_recall": len(recovered_latent)
            / max(len(repeated_coordinates), 1),
            "attack_seconds": attack_seconds,
            **geometry_metrics(
                views,
                global_queries[:query_count],
                edges,
                float(transforms[0].multiplier),
            ),
        }
        rows.append(row)

    report = {
        "experiment": "lexical synchronized-query stitching at d=8192, r_c=32",
        "dataset": "nq",
        "threat_view": (
            "server-visible fp16 local query coordinates and synchronization only; "
            "secret selectors are used only for post-attack scoring"
        ),
        "work_dimension": WORK_DIMENSION,
        "projection_dimension": PROJECTION_DIMENSION,
        "partitions": CELLS,
        "expected_pairwise_overlap_r2_over_d": (
            PROJECTION_DIMENSION**2 / WORK_DIMENSION
        ),
        "expected_total_cross_partition_pairs": (
            CELLS * (CELLS - 1) // 2
        )
        * PROJECTION_DIMENSION**2
        / WORK_DIMENSION,
        "observed_true_cross_partition_pairs": total_true_pairs,
        "latent_coordinates_observed": int(np.count_nonzero(multiplicities)),
        "latent_coordinate_coverage_fraction": float(
            np.count_nonzero(multiplicities) / WORK_DIMENSION
        ),
        "latent_coordinates_repeated": int(
            np.count_nonzero(multiplicities >= 2)
        ),
        "repeated_latent_coordinate_fraction": float(
            np.count_nonzero(multiplicities >= 2) / WORK_DIMENSION
        ),
        "maximum_views_per_latent_coordinate": int(np.max(multiplicities)),
        "graph": graph,
        "correlation_threshold": CORRELATION_THRESHOLD,
        "query_sweep": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    destination = (
        ROOT / "results" / "security" / "lexical_stitching_d8192_rc32"
    )
    destination.mkdir(parents=True, exist_ok=True)
    atomic_json(destination / "results.json", report)
    with (destination / "query_sweep.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
