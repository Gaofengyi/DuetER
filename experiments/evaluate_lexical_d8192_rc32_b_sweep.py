"""Retrieval-quality sweep for B=32/16 at d=8192 and r_c=32."""

from __future__ import annotations

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
    ball_noise,
    dpe_transform,
    load_queries_qrels,
    read_ids,
)
from evaluate_lexical_d8192_rc32 import (
    BASE_SEED,
    BETA,
    DOCUMENTS,
    GLOBAL_PERMUTATION,
    GLOBAL_SIGN1,
    GLOBAL_SIGN2,
    HASH_DIMENSION,
    NOISE_RADIUS,
    PROJECTION_DIMENSION,
    SCALE,
    WORK_DIMENSION,
    selected_ball_noise,
    selected_dpe_matrix,
)


CONFIGURATIONS = ((32, 32), (16, 64))


def build_local_cache(
    source: Path,
    destination: Path,
    cells: int,
    transforms: list,
    matrices: list[np.ndarray],
) -> dict[str, object]:
    metadata_path = destination / "metadata.json"
    local_path = destination / "lexical_compartment_fp16.npy"
    norm_path = destination / "lexical_compartment_norms.npy"
    assignment_path = destination / "lexical_compartment_cells.npy"
    required = (metadata_path, local_path, norm_path, assignment_path)
    if all(path.exists() for path in required):
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    destination.mkdir(parents=True, exist_ok=True)
    rows = np.load(source / "candidate_rows.npy", mmap_mode="r")
    sketches = np.load(source / "lexical_sketch_fp16.npy", mmap_mode="r")
    scope = json.loads((source / "candidate_scope.json").read_text(encoding="utf-8"))
    maximum_norm = float(scope["maximum_sketch_norm"])
    assignments = keyed_lexical_cells(np.asarray(rows), cells)
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
    noise_rng = np.random.default_rng(BASE_SEED + 2 + 1009 + cells * 100003)
    started = time.perf_counter()
    block = 4096
    for cell in range(cells):
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
            selected += selected_ball_noise(
                noise_rng, len(chosen), PROJECTION_DIMENSION
            )
            projected = (
                selected * transform.signs * transform.multiplier
                + transform.translation
            ).astype(np.float32)
            stored = projected.astype(np.float16)
            local[chosen] = stored
            norms[chosen] = np.sum(stored.astype(np.float32) ** 2, axis=1)
        print(
            f"[B={cells} cache] cell {cell + 1}/{cells} rows={len(positions):,}",
            flush=True,
        )
    local.flush()
    norms.flush()
    metadata = {
        "work_dimension": WORK_DIMENSION,
        "projection_dimension": PROJECTION_DIMENSION,
        "cells": cells,
        "rows": int(len(rows)),
        "build_seconds": time.perf_counter() - started,
    }
    atomic_json(metadata_path, metadata)
    return metadata


def local_queries(
    global_queries: np.ndarray, transforms: list
) -> np.ndarray:
    output = np.empty(
        (len(global_queries), len(transforms), PROJECTION_DIMENSION),
        dtype=np.float32,
    )
    for cell, transform in enumerate(transforms):
        values = (
            global_queries[:, transform.coordinates]
            * transform.signs
            * transform.multiplier
            + transform.translation
        )
        output[:, cell] = values.astype(np.float16).astype(np.float32)
    return output


def retrieve(
    source: Path,
    cache: Path,
    raw_candidates: np.ndarray,
    queries: np.ndarray,
    local_depth: int,
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
                2.0 * (rows @ queries[query_index, int(cell)])
                - np.asarray(norms[local_positions])
            )
            order = top_indices(scores, min(local_depth, len(scores)))
            selected.extend(map(int, ordinals[order]))
        latencies.append((time.perf_counter() - query_started) * 1000.0)
        ordered = np.unique(selected)
        ordered.sort()
        documents = candidates[ordered]
        depth = min(300, len(documents))
        output[query_index, :depth] = documents[:depth]
        counts.append(len(documents))
        if query_index == 0 or query_index + 1 == len(raw_candidates) or (query_index + 1) % 250 == 0:
            print(
                f"[B={queries.shape[1]} L={local_depth}] "
                f"{query_index + 1:,}/{len(raw_candidates):,}",
                flush=True,
            )
    return output, {
        "mean_ms": float(np.mean(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "candidate_mean": float(np.mean(counts)),
        "candidate_p95": float(np.percentile(counts, 95)),
    }


def main() -> None:
    started = time.perf_counter()
    source = ROOT / "cache" / "full_candidate_lexical_dpe" / "nq_full_h1024"
    destination = ROOT / "results" / "security" / "lexical_d8192_rc32_b_sweep"
    destination.mkdir(parents=True, exist_ok=True)
    query_ids, query_texts, qrels, _ = load_queries_qrels(
        ROOT / "data" / "nq", "test"
    )
    document_frequency = lexical_document_frequencies(query_texts)
    query_plain = np.stack(
        [
            query_mips(text, document_frequency, DOCUMENTS, HASH_DIMENSION)
            for text in query_texts
        ]
    )
    global_queries = SCALE * dpe_transform(
        query_plain, GLOBAL_SIGN1, GLOBAL_SIGN2, GLOBAL_PERMUTATION
    )
    global_queries += ball_noise(
        np.random.default_rng(BASE_SEED + 2027),
        len(query_plain),
        WORK_DIMENSION,
        NOISE_RADIUS,
    )
    raw_candidates = np.asarray(
        np.load(ROOT / "results" / "budgeted_lexical" / "nq_full.raw.npz")[
            "budget_50000"
        ][:, :1000],
        dtype=np.int32,
    )
    cloud_rankings: list[np.ndarray] = []
    partial_rows: list[dict[str, object]] = []
    for cells, local_depth in CONFIGURATIONS:
        transforms = [
            cell_transform(
                KEY_LEXICAL, cell, WORK_DIMENSION, PROJECTION_DIMENSION
            )
            for cell in range(cells)
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
        cache = (
            ROOT
            / "cache"
            / "lexical_d8192_rc32_b_sweep"
            / f"nq_{DOCUMENTS}_b{cells}"
        )
        setup = build_local_cache(
            source, cache, cells, transforms, matrices
        )
        queries = local_queries(global_queries, transforms)
        cloud, timing = retrieve(
            source, cache, raw_candidates, queries, local_depth
        )
        cloud_rankings.append(cloud)
        np.savez_compressed(
            destination / f"b{cells}_l{local_depth}_cloud_candidates.npz",
            lexical=cloud,
        )
        partial_rows.append(
            {
                "partitions": cells,
                "local_depth": local_depth,
                "maximum_return_budget": cells * local_depth,
                "cache_build_seconds": setup["build_seconds"],
                "server_timing": timing,
            }
        )

    stacked = np.stack(cloud_rankings, axis=0)
    selected_rows = selected_document_rows(stacked)
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
    token_lengths, indptr, indices, counts = build_candidate_term_matrix(
        ROOT / "data" / "nq" / "corpus.jsonl",
        selected_rows,
        term_to_index,
        ROOT
        / "cache"
        / "exact_bm25_client_rerank"
        / "nq_d8192_rc32_b_sweep",
    )
    exact, bm25_latency = exact_bm25_rerank(
        stacked,
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
    np.savez_compressed(
        destination / "exact_bm25_rankings.npz",
        configurations=np.asarray(CONFIGURATIONS, dtype=np.int32),
        lexical=exact,
    )
    doc_ids = read_ids(
        ROOT / "cache" / "full_semantic" / f"nq_{DOCUMENTS}" / "doc_ids.txt"
    )
    baseline = json.loads(
        (
            ROOT
            / "results"
            / "security"
            / "lexical_d8192_rc32"
            / "results.json"
        ).read_text(encoding="utf-8")
    )["exact_bm25_metrics"]
    result_rows: list[dict[str, object]] = []
    for index, partial in enumerate(partial_rows):
        cloud_metrics = metric_mean(
            stacked[index : index + 1], doc_ids, query_ids, qrels
        )
        exact_metrics = metric_mean(
            exact[index : index + 1], doc_ids, query_ids, qrels
        )
        coverage = relevant_coverage(stacked[index], doc_ids, query_ids, qrels)
        result_rows.append(
            {
                **partial,
                "cloud_candidate_metrics": cloud_metrics,
                "cloud_relevant_candidate_coverage": coverage,
                "exact_bm25_metrics": exact_metrics,
                "retention_vs_b64_l16": {
                    key: float(exact_metrics[key] / baseline[key])
                    for key in exact_metrics
                },
            }
        )
    report = {
        "experiment": "lexical B sweep at d=8192 and r_c=32",
        "dataset": "nq",
        "documents": DOCUMENTS,
        "queries": len(query_ids),
        "fixed_parameters": {
            "hash_dimension": HASH_DIMENSION,
            "work_dimension": WORK_DIMENSION,
            "projection_dimension": PROJECTION_DIMENSION,
            "posting_budget": 50000,
            "returned_depth": 300,
            "beta": BETA,
            "scale": SCALE,
            "seed": BASE_SEED,
        },
        "comparison_baseline_b64_l16": baseline,
        "configuration_results": result_rows,
        "exact_bm25_latency": bm25_latency,
        "candidate_documents_materialized": int(len(selected_rows)),
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(destination / "results.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
