"""Evaluate lexical CCADPE quality at work dimension d=8192 and r_c=32.

The experiment preserves the existing 1024-D hashed lexical representation
and changes only the padded DPE work dimension.  To avoid materializing a
25.6 GiB 8192-D intermediate cache, it evaluates the exact selected rows of
the normalized Walsh-Hadamard transform and samples the exact marginal of the
isotropic DPE ball noise on those rows.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if (ROOT / ".deps").exists():
    sys.path.insert(0, str(ROOT / ".deps"))
os.environ.setdefault("DUETER_FORCE_CPU_DEPS", "1")

import numpy as np

from attack_cross_compartment_stitching import lexical_document_frequencies
from benchmark_dual_compartment_full import (
    KEY_LEXICAL,
    cell_transform,
    keyed_lexical_cells,
    metric_mean,
    relevant_coverage,
    top_indices,
)
from benchmark_exact_bm25_client_rerank import (
    build_candidate_term_matrix,
    exact_bm25_rerank,
    load_document_frequencies,
    load_document_lengths,
    query_vocabulary,
    selected_document_rows,
)
from benchmark_million_lexical_dpe import query_mips
from benchmark_million_semantic_hybrid import (
    atomic_json,
    load_queries_qrels,
    read_ids,
)


DATASET = "nq"
DOCUMENTS = 2_681_468
HASH_DIMENSION = 1024
WORK_DIMENSION = 8192
PROJECTION_DIMENSION = 32
CELLS = 64
LOCAL_DEPTH = 16
BASE_SEED = 20260917
BETA = 0.10
SCALE = 3.0
NOISE_RADIUS = 3.0 * SCALE * BETA / 8.0


def selected_dpe_matrix(
    output_coordinates: np.ndarray,
    sign1: np.ndarray,
    sign2: np.ndarray,
    permutation: np.ndarray,
    input_dimension: int,
) -> np.ndarray:
    """Matrix mapping unpadded inputs to selected DPE output coordinates."""
    source_coordinates = permutation[np.asarray(output_coordinates, dtype=np.int64)]
    input_coordinates = np.arange(input_dimension, dtype=np.int64)
    parity_table = np.asarray(
        [int(value).bit_count() & 1 for value in range(WORK_DIMENSION)],
        dtype=np.int8,
    )
    parity = parity_table[
        np.bitwise_and(source_coordinates[:, None], input_coordinates[None, :])
    ]
    hadamard = (1.0 - 2.0 * parity.astype(np.float32)) / math.sqrt(WORK_DIMENSION)
    return (
        hadamard
        * sign1[:input_dimension][None, :]
        * sign2[source_coordinates][:, None]
    ).astype(np.float32)


def selected_ball_noise(
    rng: np.random.Generator,
    rows: int,
    selected_dimension: int,
) -> np.ndarray:
    """Exact marginal of uniform d-ball noise on selected coordinates."""
    selected = rng.normal(size=(rows, selected_dimension)).astype(np.float32)
    omitted_square_norm = rng.chisquare(
        WORK_DIMENSION - selected_dimension, size=rows
    ).astype(np.float32)
    norm = np.sqrt(
        np.sum(selected * selected, axis=1) + omitted_square_norm
    ).astype(np.float32)
    radii = (
        NOISE_RADIUS
        * np.power(rng.random(rows), 1.0 / WORK_DIMENSION)
    ).astype(np.float32)
    return selected * (radii / np.maximum(norm, 1e-12))[:, None]


def build_local_cache(
    source: Path,
    destination: Path,
    transforms: list,
    matrices: list[np.ndarray],
) -> dict[str, object]:
    metadata_path = destination / "metadata.json"
    local_path = destination / "lexical_compartment_fp16.npy"
    norm_path = destination / "lexical_compartment_norms.npy"
    assignment_path = destination / "lexical_compartment_cells.npy"
    if all(path.exists() for path in (metadata_path, local_path, norm_path, assignment_path)):
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    destination.mkdir(parents=True, exist_ok=True)
    rows = np.load(source / "candidate_rows.npy", mmap_mode="r")
    sketches = np.load(source / "lexical_sketch_fp16.npy", mmap_mode="r")
    scope = json.loads((source / "candidate_scope.json").read_text(encoding="utf-8"))
    maximum_norm = float(scope["maximum_sketch_norm"])
    assignments = keyed_lexical_cells(np.asarray(rows), CELLS)
    np.save(assignment_path, assignments)
    local = np.lib.format.open_memmap(
        local_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(rows), PROJECTION_DIMENSION),
    )
    norms = np.lib.format.open_memmap(
        norm_path, mode="w+", dtype=np.float32, shape=(len(rows),)
    )
    noise_rng = np.random.default_rng(BASE_SEED + 2 + 1009)
    started = time.perf_counter()
    block = 4096
    for cell in range(CELLS):
        positions = np.flatnonzero(assignments == cell)
        transform = transforms[cell]
        matrix = matrices[cell]
        for first in range(0, len(positions), block):
            chosen = positions[first : first + block]
            base = np.asarray(sketches[chosen], dtype=np.float32) / maximum_norm
            base_norm = np.sum(base * base, axis=1)
            mips = np.empty((len(chosen), HASH_DIMENSION + 1), dtype=np.float32)
            mips[:, :HASH_DIMENSION] = base
            mips[:, HASH_DIMENSION] = np.sqrt(np.maximum(0.0, 1.0 - base_norm))
            selected = SCALE * (mips @ matrix.T)
            selected += selected_ball_noise(noise_rng, len(chosen), PROJECTION_DIMENSION)
            projected = (
                selected * transform.signs * transform.multiplier
                + transform.translation
            ).astype(np.float32)
            stored = projected.astype(np.float16)
            local[chosen] = stored
            norms[chosen] = np.sum(stored.astype(np.float32) ** 2, axis=1)
        print(
            f"[d8192 r32 cache] cell {cell + 1}/{CELLS} rows={len(positions):,}",
            flush=True,
        )
    local.flush()
    norms.flush()
    metadata = {
        "hash_dimension": HASH_DIMENSION,
        "work_dimension": WORK_DIMENSION,
        "projection_dimension": PROJECTION_DIMENSION,
        "cells": CELLS,
        "rows": int(len(rows)),
        "maximum_sketch_norm": maximum_norm,
        "noise_radius": NOISE_RADIUS,
        "selected_transform_evaluation": True,
        "build_seconds": time.perf_counter() - started,
    }
    atomic_json(metadata_path, metadata)
    return metadata


def build_local_queries(
    query_plain: np.ndarray,
    transforms: list,
    matrices: list[np.ndarray],
) -> np.ndarray:
    union = np.unique(
        np.concatenate([transform.coordinates for transform in transforms])
    ).astype(np.int32)
    union_matrix = selected_dpe_matrix(
        union,
        GLOBAL_SIGN1,
        GLOBAL_SIGN2,
        GLOBAL_PERMUTATION,
        query_plain.shape[1],
    )
    global_selected = SCALE * (query_plain @ union_matrix.T)
    query_rng = np.random.default_rng(BASE_SEED + 2027)
    global_selected += selected_ball_noise(query_rng, len(query_plain), len(union))
    lookup = {int(value): index for index, value in enumerate(union)}
    output = np.empty(
        (len(query_plain), CELLS, PROJECTION_DIMENSION), dtype=np.float32
    )
    for cell, transform in enumerate(transforms):
        positions = np.asarray(
            [lookup[int(value)] for value in transform.coordinates], dtype=np.int32
        )
        local = (
            global_selected[:, positions]
            * transform.signs
            * transform.multiplier
            + transform.translation
        )
        output[:, cell] = local.astype(np.float16).astype(np.float32)
    return output


def retrieve(
    source: Path,
    cache: Path,
    raw_candidates: np.ndarray,
    local_queries: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    candidate_rows = np.load(source / "candidate_rows.npy", mmap_mode="r")
    assignments = np.load(cache / "lexical_compartment_cells.npy", mmap_mode="r")
    local = np.load(cache / "lexical_compartment_fp16.npy", mmap_mode="r")
    norms = np.load(cache / "lexical_compartment_norms.npy", mmap_mode="r")
    output = np.full((len(raw_candidates), 300), -1, dtype=np.int32)
    latencies: list[float] = []
    counts: list[int] = []
    for query_index, raw in enumerate(raw_candidates):
        candidates = np.asarray(raw[raw >= 0], dtype=np.int32)
        positions = np.searchsorted(candidate_rows, candidates)
        if np.any(np.asarray(candidate_rows[positions]) != candidates):
            raise RuntimeError("candidate absent from scoped cache")
        candidate_cells = np.asarray(assignments[positions], dtype=np.int32)
        selected: list[int] = []
        query_started = time.perf_counter()
        for cell in np.unique(candidate_cells):
            ordinals = np.flatnonzero(candidate_cells == cell)
            local_positions = positions[ordinals]
            rows = np.asarray(local[local_positions], dtype=np.float32)
            scores = (
                2.0 * (rows @ local_queries[query_index, int(cell)])
                - np.asarray(norms[local_positions])
            )
            order = top_indices(scores, min(LOCAL_DEPTH, len(scores)))
            selected.extend(map(int, ordinals[order]))
        latencies.append((time.perf_counter() - query_started) * 1000.0)
        ordered = np.unique(selected)
        ordered.sort()
        documents = candidates[ordered]
        depth = min(300, len(documents))
        output[query_index, :depth] = documents[:depth]
        counts.append(len(documents))
        if query_index == 0 or query_index + 1 == len(raw_candidates) or (query_index + 1) % 250 == 0:
            print(f"[d8192 r32 retrieval] {query_index + 1:,}/{len(raw_candidates):,}", flush=True)
    return output, {
        "mean_ms": float(np.mean(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "candidate_mean": float(np.mean(counts)),
    }


# The global key is deterministic and saved for audit by main().
GLOBAL_RNG = np.random.default_rng(BASE_SEED + 2)
GLOBAL_SIGN1 = GLOBAL_RNG.choice(
    np.asarray([-1.0, 1.0], dtype=np.float32), WORK_DIMENSION
)
GLOBAL_SIGN2 = GLOBAL_RNG.choice(
    np.asarray([-1.0, 1.0], dtype=np.float32), WORK_DIMENSION
)
GLOBAL_PERMUTATION = GLOBAL_RNG.permutation(WORK_DIMENSION)


def main() -> None:
    started = time.perf_counter()
    source = ROOT / "cache" / "full_candidate_lexical_dpe" / "nq_full_h1024"
    destination = ROOT / "results" / "security" / "lexical_d8192_rc32"
    cache = ROOT / "cache" / "lexical_d8192_rc32" / f"nq_{DOCUMENTS}"
    destination.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache / "lexical_dpe_key.npz",
        sign1=GLOBAL_SIGN1,
        sign2=GLOBAL_SIGN2,
        permutation=GLOBAL_PERMUTATION,
    )
    transforms = [
        cell_transform(
            KEY_LEXICAL, cell, WORK_DIMENSION, PROJECTION_DIMENSION
        )
        for cell in range(CELLS)
    ]
    matrices = [
        selected_dpe_matrix(
            transform.coordinates,
            GLOBAL_SIGN1,
            GLOBAL_SIGN2,
            GLOBAL_PERMUTATION,
            HASH_DIMENSION + 1,
        )
        for transform in transforms
    ]
    setup = build_local_cache(source, cache, transforms, matrices)

    query_ids, query_texts, qrels, _ = load_queries_qrels(
        ROOT / "data" / DATASET, "test"
    )
    document_frequency = lexical_document_frequencies(query_texts)
    query_plain = np.stack(
        [
            query_mips(text, document_frequency, DOCUMENTS, HASH_DIMENSION)
            for text in query_texts
        ]
    )
    local_queries = build_local_queries(query_plain, transforms, matrices)
    raw_candidates = np.asarray(
        np.load(ROOT / "results" / "budgeted_lexical" / "nq_full.raw.npz")[
            "budget_50000"
        ][:, :1000],
        dtype=np.int32,
    )
    cloud, timing = retrieve(source, cache, raw_candidates, local_queries)
    np.savez_compressed(destination / "cloud_candidates.npz", lexical=cloud)

    rankings = cloud[None, :, :]
    selected_rows = selected_document_rows(rankings)
    terms, term_to_index = query_vocabulary(query_texts)
    database = ROOT / "results" / "keyed_fts5" / "nq_full.sqlite3"
    full_lengths = load_document_lengths(
        database,
        ROOT
        / "cache"
        / "exact_bm25_client_rerank"
        / "nq"
        / "full_doc_lengths_v2.npy",
    )
    document_frequencies = load_document_frequencies(database, terms)
    exact_cache = ROOT / "cache" / "exact_bm25_client_rerank" / "nq_d8192_rc32"
    token_lengths, indptr, indices, counts = build_candidate_term_matrix(
        ROOT / "data" / DATASET / "corpus.jsonl",
        selected_rows,
        term_to_index,
        exact_cache,
    )
    exact, bm25_latency = exact_bm25_rerank(
        rankings,
        selected_rows,
        token_lengths,
        indptr,
        indices,
        counts,
        query_texts,
        term_to_index,
        document_frequencies,
        len(full_lengths),
        float(np.mean(full_lengths)),
    )
    np.savez_compressed(destination / "exact_bm25_rankings.npz", lexical=exact)
    doc_ids = read_ids(
        ROOT / "cache" / "full_semantic" / f"nq_{DOCUMENTS}" / "doc_ids.txt"
    )
    cloud_metrics = metric_mean(rankings, doc_ids, query_ids, qrels)
    exact_metrics = metric_mean(exact, doc_ids, query_ids, qrels)
    coverage = relevant_coverage(cloud, doc_ids, query_ids, qrels)
    baseline_report = json.loads(
        (
            ROOT
            / "results"
            / "security"
            / "lexical_projection_quality"
            / "results.json"
        ).read_text(encoding="utf-8")
    )
    baseline = baseline_report["projection_results"][0]["exact_bm25_metrics"]
    rc32_d2048 = baseline_report["projection_results"][3]["exact_bm25_metrics"]
    report = {
        "experiment": "lexical CCADPE at d=8192, r_c=32",
        "dataset": DATASET,
        "documents": DOCUMENTS,
        "queries": len(query_ids),
        "parameters": {
            "hash_dimension": HASH_DIMENSION,
            "dpe_work_dimension": WORK_DIMENSION,
            "projection_dimension": PROJECTION_DIMENSION,
            "partitions": CELLS,
            "local_depth_per_partition": LOCAL_DEPTH,
            "posting_budget": 50000,
            "returned_depth": 300,
            "beta": BETA,
            "scale": SCALE,
            "seed": BASE_SEED,
        },
        "implementation_note": (
            "Exact selected-row evaluation of the padded 8192-D normalized "
            "Walsh-Hadamard DPE; isotropic ball noise uses its exact joint marginal "
            "on all server-visible selected coordinates."
        ),
        "cache_setup": setup,
        "cloud_candidate_metrics": cloud_metrics,
        "cloud_relevant_candidate_coverage": coverage,
        "exact_bm25_metrics": exact_metrics,
        "retention_vs_d2048_rc256": {
            key: float(exact_metrics[key] / baseline[key]) for key in exact_metrics
        },
        "retention_vs_d2048_rc32": {
            key: float(exact_metrics[key] / rc32_d2048[key]) for key in exact_metrics
        },
        "server_timing": timing,
        "exact_bm25_latency": bm25_latency,
        "candidate_documents_materialized": int(len(selected_rows)),
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(destination / "results.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
