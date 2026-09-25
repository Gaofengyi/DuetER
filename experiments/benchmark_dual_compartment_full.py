"""Complete-corpus evaluation with Compartment-DPE on both retrieval paths.

The semantic path uses the existing residual-IVF cells.  Every posting copy is
stored in a cell-keyed low-rank coordinate system and is compared only with the
query ciphertext for that cell.  The lexical path first executes the existing
cumulative-posting candidate planner, then partitions candidate rows with a
keyed document partition; every partition has an independent coordinate
subspace and is ranked locally.  The client unions cell-local outputs and
reranks them with exact semantic vectors or the legacy lexical planner order.
The final paper results use ``benchmark_exact_bm25_client_rerank.py`` to
recompute standard BM25 from encrypted token payloads on the returned set.

The prototype derives each cell subspace from a hidden base DPE rotation.  The
base coordinates are never part of the server-visible Compartment view.  This
lets the experiment reuse the audited full-corpus ciphertext caches instead of
re-encoding 16.7 million documents.  It is equivalent to applying a keyed
coordinate-selection projection, sign mask, scale, and translation to each
cell.  It does not claim to hide cell labels or access patterns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Load the CUDA build before legacy experiment modules prepend the CPU-only
# dependency directory.  Importing torch here also makes subsequent local
# imports resolve to the already loaded CUDA module.
DEPENDENCY_ROOT = Path(__file__).resolve().parent
if (DEPENDENCY_ROOT / ".deps").exists():
    sys.path.insert(0, str(DEPENDENCY_ROOT / ".deps"))
if (DEPENDENCY_ROOT / ".gpu_deps").exists() and os.environ.get("DUETER_FORCE_CPU_DEPS") != "1":
    sys.path.insert(0, str(DEPENDENCY_ROOT / ".gpu_deps"))
    os.environ.setdefault("DUETDPE_GPU_DEPS", "1")
try:
    import torch as _torch
except OSError:
    # Lexical-only audit scripts do not require Torch. Semantic routines still
    # import it locally and therefore fail explicitly if their runtime is absent.
    _torch = None

import joblib
import numpy as np

from benchmark_budgeted_lexical_candidates import ROOT
from benchmark_million_lexical_dpe import encrypt_query_matrix, query_mips
from benchmark_million_semantic_hybrid import (
    IvfFiles,
    atomic_json,
    encrypt_queries,
    evaluate,
    load_queries_qrels,
    read_ids,
)
from experiment_adaptive_fusion import query_features
from experiment_candidate_rank_fusion import (
    learned_rankings,
    mix_rankings,
    run_dataset as run_duetrank,
    split_mask,
)


FULL_DOCUMENTS = {
    "msmarco": 8_841_823,
    "nq": 2_681_468,
    "hotpotqa": 5_233_329,
}
SEMANTIC_RESULT_NAMES = {
    "msmarco": "msmarco_8841823",
    "nq": "nq_2681468",
    "hotpotqa": "hotpotqa_5233329",
}
KEY_SEMANTIC = b"DuetDPE-semantic-compartment-v2"
KEY_LEXICAL = b"DuetDPE-lexical-compartment-v2"


@dataclass(frozen=True)
class CellTransform:
    coordinates: np.ndarray
    signs: np.ndarray
    multiplier: float
    translation: np.ndarray


def _seed(key: bytes, cell: int, work_dimension: int, projection_dimension: int) -> int:
    digest = hashlib.sha256(
        key
        + int(cell).to_bytes(8, "big", signed=False)
        + int(work_dimension).to_bytes(4, "big", signed=False)
        + int(projection_dimension).to_bytes(4, "big", signed=False)
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def cell_transform(
    key: bytes, cell: int, work_dimension: int, projection_dimension: int
) -> CellTransform:
    if projection_dimension > work_dimension:
        raise ValueError("projection dimension exceeds base DPE dimension")
    rng = np.random.default_rng(
        _seed(key, cell, work_dimension, projection_dimension)
    )
    coordinates = np.sort(
        rng.choice(work_dimension, size=projection_dimension, replace=False)
    ).astype(np.int32)
    signs = rng.choice(
        np.asarray([-1.0, 1.0], dtype=np.float32), projection_dimension
    )
    multiplier = math.sqrt(work_dimension / projection_dimension)
    translation = rng.normal(0.0, 1.0, projection_dimension).astype(np.float32)
    return CellTransform(coordinates, signs, multiplier, translation)


def apply_transform(values: np.ndarray, transform: CellTransform) -> np.ndarray:
    selected = np.asarray(values[..., transform.coordinates], dtype=np.float32)
    return (
        selected * transform.signs * transform.multiplier
        + transform.translation
    ).astype(np.float32)


def keyed_lexical_cells(rows: np.ndarray, cells: int) -> np.ndarray:
    """HMAC-like keyed partition evaluated with BLAKE2b for build efficiency."""
    output = np.empty(len(rows), dtype=np.uint16 if cells <= 65535 else np.uint32)
    for index, raw in enumerate(np.asarray(rows, dtype=np.int64)):
        digest = hashlib.blake2b(
            int(raw).to_bytes(8, "big", signed=False),
            key=KEY_LEXICAL,
            digest_size=8,
        ).digest()
        output[index] = int.from_bytes(digest, "big", signed=False) % cells
    return output


def percentile(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
    }


def top_indices(scores: np.ndarray, depth: int) -> np.ndarray:
    if depth <= 0:
        return np.empty(0, dtype=np.int64)
    if depth >= len(scores):
        return np.argsort(-scores, kind="stable")
    selected = np.argpartition(-scores, depth - 1)[:depth]
    return selected[np.argsort(-scores[selected], kind="stable")]


def build_semantic_compartment_cache(
    source: Path,
    destination: Path,
    projection_dimension: int,
) -> dict[str, object]:
    destination.mkdir(parents=True, exist_ok=True)
    metadata_path = destination / "semantic_compartment.json"
    cipher_path = destination / "semantic_compartment_fp16.npy"
    norms_path = destination / "semantic_compartment_norms.npy"
    if metadata_path.exists() and cipher_path.exists() and norms_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if int(metadata["projection_dimension"]) == projection_dimension:
            return metadata
        raise RuntimeError("semantic compartment cache has incompatible parameters")

    base = np.load(source / "semantic_cipher_fp16.npy", mmap_mode="r")
    postings = np.load(source / "semantic_postings.npy", mmap_mode="r")
    offsets = np.load(source / "semantic_offsets.npy")
    work_dimension = int(base.shape[1])
    local = np.lib.format.open_memmap(
        cipher_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(postings), projection_dimension),
    )
    norms = np.lib.format.open_memmap(
        norms_path, mode="w+", dtype=np.float32, shape=(len(postings),)
    )
    started = time.perf_counter()
    nonempty = 0
    for cell in range(len(offsets) - 1):
        first, last = int(offsets[cell]), int(offsets[cell + 1])
        if first == last:
            continue
        nonempty += 1
        transform = cell_transform(
            KEY_SEMANTIC, cell, work_dimension, projection_dimension
        )
        rows = np.asarray(base[np.asarray(postings[first:last], dtype=np.int64)])
        transformed = apply_transform(rows, transform)
        stored = transformed.astype(np.float16)
        local[first:last] = stored
        norms[first:last] = np.sum(stored.astype(np.float32) ** 2, axis=1)
        if cell == 0 or cell + 1 == len(offsets) - 1 or (cell + 1) % 128 == 0:
            local.flush()
            norms.flush()
            print(
                f"[semantic compartment build] cell {cell + 1:,}/{len(offsets) - 1:,}",
                flush=True,
            )
    local.flush()
    norms.flush()
    metadata = {
        "variant": "cell-keyed coordinate-selection Compartment-DPE",
        "base_coordinates_server_visible": False,
        "posting_entries": int(len(postings)),
        "cells": int(len(offsets) - 1),
        "nonempty_cells": nonempty,
        "base_work_dimension": work_dimension,
        "projection_dimension": projection_dimension,
        "ciphertext_bytes": int(cipher_path.stat().st_size),
        "norm_bytes": int(norms_path.stat().st_size),
        "build_seconds": time.perf_counter() - started,
    }
    atomic_json(metadata_path, metadata)
    return metadata


def build_lexical_compartment_cache(
    source: Path,
    destination: Path,
    projection_dimension: int,
    cells: int,
) -> dict[str, object]:
    destination.mkdir(parents=True, exist_ok=True)
    metadata_path = destination / "lexical_compartment.json"
    cipher_path = destination / "lexical_compartment_fp16.npy"
    norms_path = destination / "lexical_compartment_norms.npy"
    cells_path = destination / "lexical_compartment_cells.npy"
    required = (metadata_path, cipher_path, norms_path, cells_path)
    if all(path.exists() for path in required):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            int(metadata["projection_dimension"]) == projection_dimension
            and int(metadata["cells"]) == cells
        ):
            return metadata
        raise RuntimeError("lexical compartment cache has incompatible parameters")

    rows = np.load(source / "candidate_rows.npy", mmap_mode="r")
    base = np.load(source / "lexical_cipher_fp16.npy", mmap_mode="r")
    if len(rows) != len(base):
        raise RuntimeError("candidate rows and lexical ciphertexts disagree")
    assignments = keyed_lexical_cells(rows, cells)
    np.save(cells_path, assignments)
    local = np.lib.format.open_memmap(
        cipher_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(rows), projection_dimension),
    )
    norms = np.lib.format.open_memmap(
        norms_path, mode="w+", dtype=np.float32, shape=(len(rows),)
    )
    started = time.perf_counter()
    for cell in range(cells):
        positions = np.flatnonzero(assignments == cell)
        transform = cell_transform(
            KEY_LEXICAL, cell, int(base.shape[1]), projection_dimension
        )
        transformed = apply_transform(np.asarray(base[positions]), transform)
        stored = transformed.astype(np.float16)
        local[positions] = stored
        norms[positions] = np.sum(stored.astype(np.float32) ** 2, axis=1)
        print(
            f"[lexical compartment build] cell {cell + 1:,}/{cells:,} "
            f"rows={len(positions):,}",
            flush=True,
        )
    local.flush()
    norms.flush()
    metadata = {
        "variant": "keyed document-partition Compartment-DPE",
        "base_coordinates_server_visible": False,
        "materialized_candidate_rows": int(len(rows)),
        "cells": cells,
        "projection_dimension": projection_dimension,
        "base_work_dimension": int(base.shape[1]),
        "ciphertext_bytes": int(cipher_path.stat().st_size),
        "norm_bytes": int(norms_path.stat().st_size),
        "cell_bytes": int(cells_path.stat().st_size),
        "build_seconds": time.perf_counter() - started,
        "scope": "full-corpus query workload candidate union; full deployment storage is estimated separately",
    }
    atomic_json(metadata_path, metadata)
    return metadata


def route_semantic_queries(
    ivf: IvfFiles, queries: np.ndarray, probes: int, device: str
) -> tuple[np.ndarray, float]:
    import torch

    started = time.perf_counter()
    residual = np.asarray(queries, dtype=np.float32) - ivf.mean[None, :]
    residual /= np.maximum(
        np.linalg.norm(residual, axis=1, keepdims=True), 1e-12
    )
    with torch.inference_mode():
        query_gpu = torch.from_numpy(residual).to(device)
        centroid_gpu = torch.from_numpy(ivf.centroids).to(device)
        cells = torch.topk(query_gpu @ centroid_gpu.T, k=probes, dim=1).indices
        output = cells.cpu().numpy().astype(np.int32)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
    return output, time.perf_counter() - started


def exact_semantic_rerank(
    emitted: np.ndarray,
    embeddings: np.ndarray,
    queries: np.ndarray,
    exact_dense: np.ndarray,
    top_per_cell_values: list[int],
    output_depth: int,
) -> tuple[dict[int, np.ndarray], dict[int, dict[str, float]], float]:
    repeats, query_count, _, maximum = emitted.shape
    outputs = {
        top: np.full((repeats, query_count, output_depth), -1, dtype=np.int32)
        for top in top_per_cell_values
    }
    diagnostics = {
        top: {"candidate_counts": [], "dense_coverages": []}
        for top in top_per_cell_values
    }
    started = time.perf_counter()
    for repeat in range(repeats):
        for query_index in range(query_count):
            max_candidates = np.unique(
                emitted[repeat, query_index][
                    emitted[repeat, query_index] >= 0
                ]
            )
            if len(max_candidates):
                max_scores = (
                    np.asarray(embeddings[max_candidates], dtype=np.float32)
                    @ queries[query_index]
                )
            else:
                max_scores = np.empty(0, dtype=np.float32)
            for top in top_per_cell_values:
                subset = emitted[repeat, query_index, :, :top].reshape(-1)
                candidates = np.unique(subset[subset >= 0])
                diagnostics[top]["candidate_counts"].append(float(len(candidates)))
                diagnostics[top]["dense_coverages"].append(
                    len(set(map(int, candidates)).intersection(map(int, exact_dense[query_index])))
                    / max(len(exact_dense[query_index]), 1)
                )
                if not len(candidates):
                    continue
                positions = np.searchsorted(max_candidates, candidates)
                scores = max_scores[positions]
                order = top_indices(scores, min(output_depth, len(candidates)))
                outputs[top][repeat, query_index, : len(order)] = candidates[order]
    elapsed = time.perf_counter() - started
    summarized = {
        top: {
            "candidate_mean": float(np.mean(values["candidate_counts"])),
            "candidate_p95": float(np.percentile(values["candidate_counts"], 95)),
            "dense_top100_coverage_mean": float(
                np.mean(values["dense_coverages"])
            ),
        }
        for top, values in diagnostics.items()
    }
    return outputs, summarized, elapsed


def semantic_compartment_retrieval(
    source: Path,
    compartment: Path,
    queries: np.ndarray,
    embeddings: np.ndarray,
    exact_dense: np.ndarray,
    *,
    projection_dimension: int,
    probes: int,
    top_per_cell_values: list[int],
    output_depth: int,
    repeats: int,
    beta: float,
    scale: float,
    seed: int,
    device: str,
    latency_samples: int,
) -> tuple[dict[int, np.ndarray], dict[str, object]]:
    import torch

    ivf = IvfFiles.load(source)
    query_cells, route_seconds = route_semantic_queries(ivf, queries, probes, device)
    global_queries = np.stack(
        [
            encrypt_queries(
                queries,
                source / "semantic_dpe_key.npz",
                beta,
                scale,
                seed + 2027 + repeat * 100003,
            )
            for repeat in range(repeats)
        ]
    )
    local = np.load(compartment / "semantic_compartment_fp16.npy", mmap_mode="r")
    norms = np.load(compartment / "semantic_compartment_norms.npy", mmap_mode="r")
    maximum_top = max(top_per_cell_values)
    emitted = np.full(
        (repeats, len(queries), probes, maximum_top), -1, dtype=np.int32
    )
    by_cell: list[list[tuple[int, int]]] = [[] for _ in range(len(ivf.offsets) - 1)]
    for query_index, cells in enumerate(query_cells):
        for ordinal, cell in enumerate(cells):
            by_cell[int(cell)].append((query_index, ordinal))

    posting_reads = np.zeros(len(queries), dtype=np.int64)
    server_started = time.perf_counter()
    with torch.inference_mode():
        for cell, query_locations in enumerate(by_cell):
            if not query_locations:
                continue
            first, last = int(ivf.offsets[cell]), int(ivf.offsets[cell + 1])
            if first == last:
                continue
            q_indices = np.asarray([item[0] for item in query_locations], dtype=np.int64)
            ordinals = np.asarray([item[1] for item in query_locations], dtype=np.int64)
            posting_reads[q_indices] += last - first
            documents = np.asarray(ivf.postings[first:last], dtype=np.int32)
            rows_gpu = torch.from_numpy(
                np.asarray(local[first:last], dtype=np.float16)
            ).to(device)
            norms_gpu = torch.from_numpy(
                np.asarray(norms[first:last], dtype=np.float32)
            ).to(device)
            transform = cell_transform(
                KEY_SEMANTIC,
                cell,
                int(global_queries.shape[-1]),
                projection_dimension,
            )
            depth = min(maximum_top, len(documents))
            for repeat in range(repeats):
                q_local = apply_transform(global_queries[repeat, q_indices], transform)
                q_gpu = torch.from_numpy(q_local.astype(np.float16)).to(device)
                scores = 2.0 * (q_gpu @ rows_gpu.T).float() - norms_gpu[None, :]
                selected = torch.topk(scores, k=depth, dim=1).indices.cpu().numpy()
                emitted[repeat, q_indices, ordinals, :depth] = documents[selected]
            if cell == 0 or cell + 1 == len(by_cell) or (cell + 1) % 128 == 0:
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                print(
                    f"[semantic compartment query] cell {cell + 1:,}/{len(by_cell):,}",
                    flush=True,
                )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    server_seconds = time.perf_counter() - server_started
    outputs, rerank_diagnostics, rerank_seconds = exact_semantic_rerank(
        emitted,
        embeddings,
        queries,
        exact_dense,
        top_per_cell_values,
        output_depth,
    )

    # Query-major CPU timing exposes interactive latency separately from the
    # cell-major throughput path used above.
    sequential: list[float] = []
    sample_count = min(latency_samples, len(queries))
    for query_index in range(sample_count):
        started = time.perf_counter()
        for cell in query_cells[query_index]:
            cell = int(cell)
            first, last = int(ivf.offsets[cell]), int(ivf.offsets[cell + 1])
            transform = cell_transform(
                KEY_SEMANTIC,
                cell,
                int(global_queries.shape[-1]),
                projection_dimension,
            )
            q_local = apply_transform(global_queries[0, query_index], transform)
            rows = np.asarray(local[first:last], dtype=np.float32)
            scores = 2.0 * (rows @ q_local) - np.asarray(norms[first:last])
            top_indices(scores, min(maximum_top, len(scores)))
        sequential.append((time.perf_counter() - started) * 1000.0)

    return outputs, {
        "query_routing_ms_per_query": route_seconds * 1000.0 / len(queries),
        "posting_entries_read_mean": float(np.mean(posting_reads)),
        "posting_entries_read_p95": float(np.percentile(posting_reads, 95)),
        "batch_server_seconds": server_seconds,
        "batch_server_ms_per_query_repeat": server_seconds * 1000.0
        / (len(queries) * repeats),
        "batch_throughput_queries_per_second": len(queries) * repeats
        / max(server_seconds, 1e-12),
        "interactive_server_latency_ms": percentile(sequential),
        "interactive_latency_sample_queries": sample_count,
        "client_rerank_ms_per_query_repeat": rerank_seconds * 1000.0
        / (len(queries) * repeats),
        "by_top_per_cell": rerank_diagnostics,
    }


def lexical_compartment_retrieval(
    dataset: str,
    source: Path,
    compartment: Path,
    raw_candidates: np.ndarray,
    query_texts: list[str],
    *,
    documents: int,
    projection_dimension: int,
    cells: int,
    top_per_cell_values: list[int],
    output_depth: int,
    repeats: int,
    beta: float,
    scale: float,
    seed: int,
) -> tuple[dict[int, np.ndarray], dict[str, object]]:
    import sqlite3

    connection = sqlite3.connect(
        ROOT / "results" / "keyed_fts5" / f"{dataset}_full.sqlite3"
    )
    document_frequency = {
        str(term): int(df)
        for term, df in connection.execute("SELECT term, doc FROM vocab")
    }
    connection.close()
    plain_queries = np.stack(
        [
            query_mips(text, document_frequency, documents, 1024)
            for text in query_texts
        ]
    )
    global_queries = np.stack(
        [
            encrypt_query_matrix(
                plain_queries,
                source / "lexical_dpe_key.npz",
                beta,
                scale,
                seed + 2027 + repeat * 100003,
            )
            for repeat in range(repeats)
        ]
    )
    candidate_rows = np.load(source / "candidate_rows.npy", mmap_mode="r")
    assignments = np.load(compartment / "lexical_compartment_cells.npy", mmap_mode="r")
    local = np.load(compartment / "lexical_compartment_fp16.npy", mmap_mode="r")
    norms = np.load(compartment / "lexical_compartment_norms.npy", mmap_mode="r")
    maximum_top = max(top_per_cell_values)
    outputs = {
        top: np.full((repeats, len(query_texts), output_depth), -1, dtype=np.int32)
        for top in top_per_cell_values
    }
    candidate_counts = {
        top: [[] for _ in range(repeats)] for top in top_per_cell_values
    }
    server_latencies = [[] for _ in range(repeats)]
    active_cells: list[int] = []

    transforms = [
        cell_transform(
            KEY_LEXICAL,
            cell,
            int(global_queries.shape[-1]),
            projection_dimension,
        )
        for cell in range(cells)
    ]
    local_queries = np.empty(
        (repeats, len(query_texts), cells, projection_dimension), dtype=np.float32
    )
    for cell, transform in enumerate(transforms):
        for repeat in range(repeats):
            local_queries[repeat, :, cell] = apply_transform(
                global_queries[repeat], transform
            )

    for query_index, raw_row in enumerate(raw_candidates):
        candidates = np.asarray(raw_row[raw_row >= 0], dtype=np.int32)
        positions = np.searchsorted(candidate_rows, candidates)
        if np.any(np.asarray(candidate_rows[positions]) != candidates):
            raise RuntimeError("lexical candidate is absent from scoped cache")
        candidate_cells = np.asarray(assignments[positions], dtype=np.int32)
        unique_cells = np.unique(candidate_cells)
        active_cells.append(len(unique_cells))
        selected_ordinals = {
            top: [[] for _ in range(repeats)] for top in top_per_cell_values
        }
        for repeat in range(repeats):
            started = time.perf_counter()
            for cell in unique_cells:
                ordinals = np.flatnonzero(candidate_cells == cell)
                local_positions = positions[ordinals]
                rows = np.asarray(local[local_positions], dtype=np.float32)
                query = local_queries[repeat, query_index, int(cell)]
                scores = 2.0 * (rows @ query) - np.asarray(norms[local_positions])
                order = top_indices(scores, min(maximum_top, len(scores)))
                for top in top_per_cell_values:
                    selected_ordinals[top][repeat].extend(
                        map(int, ordinals[order[:top]])
                    )
            server_latencies[repeat].append(
                (time.perf_counter() - started) * 1000.0
            )
            for top in top_per_cell_values:
                # Preserve the legacy planner order for compatibility.  This
                # routine does not claim exact BM25 client reranking.
                ordered_ordinals = np.unique(selected_ordinals[top][repeat])
                ordered_ordinals.sort()
                selected_docs = candidates[ordered_ordinals]
                depth = min(output_depth, len(selected_docs))
                outputs[top][repeat, query_index, :depth] = selected_docs[:depth]
                candidate_counts[top][repeat].append(float(len(selected_docs)))
        if (
            query_index == 0
            or query_index + 1 == len(query_texts)
            or (query_index + 1) % 250 == 0
        ):
            print(
                f"[{dataset} lexical compartment] {query_index + 1:,}/{len(query_texts):,}",
                flush=True,
            )

    flattened = [value for repeat in server_latencies for value in repeat]
    return outputs, {
        "active_cells_mean": float(np.mean(active_cells)),
        "active_cells_max": int(np.max(active_cells)),
        "server_latency_ms": percentile(flattened),
        "by_top_per_cell": {
            top: {
                "candidate_mean": float(np.mean(candidate_counts[top])),
                "candidate_p95": float(np.percentile(candidate_counts[top], 95)),
            }
            for top in top_per_cell_values
        },
    }


def relevant_coverage(
    rankings: np.ndarray,
    doc_ids: list[str],
    query_ids: list[str],
    qrels: dict[str, dict[str, int]],
) -> float:
    values = []
    for ranking, query_id in zip(rankings, query_ids):
        relevant = {
            doc_id for doc_id, gain in qrels[query_id].items() if int(gain) > 0
        }
        returned = {
            doc_ids[int(value)] for value in ranking if int(value) >= 0
        }
        values.append(len(relevant.intersection(returned)) / max(len(relevant), 1))
    return float(np.mean(values))


def choose_operating_point(
    rows: list[dict[str, object]], ndcg_reference: float, recall_reference: float
) -> int:
    eligible = [
        row
        for row in rows
        if float(row["metrics"]["nDCG@10"]) >= 0.98 * ndcg_reference
        and float(row["metrics"]["Recall@100"]) >= 0.95 * recall_reference
    ]
    if eligible:
        selected = min(eligible, key=lambda row: int(row["top_per_cell"]))
    else:
        selected = max(
            rows,
            key=lambda row: (
                min(
                    float(row["metrics"]["nDCG@10"])
                    / max(ndcg_reference, 1e-12),
                    float(row["metrics"]["Recall@100"])
                    / max(recall_reference, 1e-12),
                ),
                -int(row["top_per_cell"]),
            ),
        )
    return int(selected["top_per_cell"])


def metric_mean(
    rankings: np.ndarray,
    doc_ids: list[str],
    query_ids: list[str],
    qrels: dict[str, dict[str, int]],
) -> dict[str, float]:
    values = [evaluate(row, doc_ids, query_ids, qrels) for row in rankings]
    return {
        metric: float(np.mean([row[metric] for row in values]))
        for metric in values[0]
    }


def communication_bytes(
    semantic_top: int,
    lexical_top: int,
    semantic_probes: int,
    lexical_cells: int,
    projection_dimension: int,
) -> dict[str, float | int]:
    label_bytes = 16
    local_query_bytes = projection_dimension * 2
    semantic_records = semantic_probes * semantic_top
    lexical_records = lexical_cells * lexical_top
    semantic_record_bytes = 16 + 12 + (384 * 2 + 16) + 12 + (20 + 16)
    lexical_record_bytes = 16 + 12 + (1024 * 2 + 16) + 12 + (20 + 16)
    upload = (
        semantic_probes * (label_bytes + local_query_bytes)
        + lexical_cells * (label_bytes + local_query_bytes)
        + 32
    )
    download = (
        16
        + semantic_records * semantic_record_bytes
        + lexical_records * lexical_record_bytes
    )
    total = upload + download
    return {
        "semantic_padded_records": semantic_records,
        "lexical_padded_records": lexical_records,
        "upload_bytes": upload,
        "download_bytes": download,
        "total_bytes": total,
        "total_mib": total / (1024**2),
    }


def recall_at_depth_rows(
    rankings: np.ndarray,
    depth: int,
    doc_ids: list[str],
    query_ids: list[str],
    qrels: dict[str, dict[str, int]],
) -> np.ndarray:
    result = np.zeros(len(query_ids), dtype=np.float64)
    for index, (ranking, query_id) in enumerate(zip(rankings, query_ids)):
        relevant = {
            doc_id for doc_id, gain in qrels[query_id].items() if int(gain) > 0
        }
        returned = {
            doc_ids[int(value)] for value in ranking[:depth] if int(value) >= 0
        }
        result[index] = len(relevant.intersection(returned)) / max(len(relevant), 1)
    return result


def reconstruct_duetrank_and_recall(
    dataset: str,
    semantic: np.ndarray,
    lexical: np.ndarray,
    fusion_report: dict[str, object],
    query_ids: list[str],
    query_texts: list[str],
    qrels: dict[str, dict[str, int]],
    doc_ids: list[str],
    trace_path: Path,
    trace_key: str,
    destination: Path,
) -> dict[str, object]:
    selection = fusion_report["selection"]
    trace = json.loads(trace_path.read_text(encoding="utf-8"))[trace_key]
    features = query_features(query_texts, semantic, lexical[0, :, :100], trace)
    holdout = split_mask(query_ids, "outer")
    model = joblib.load(selection["model_path"])
    semantic_weight = float(selection["semantic_backoff_weight"])
    duet = []
    for repeat in range(len(semantic)):
        learned = learned_rankings(
            model, semantic[repeat], lexical[repeat], features, mask=holdout
        )
        duet.append(
            learned
            if semantic_weight == 0.0
            else mix_rankings(semantic[repeat], learned, semantic_weight)
        )
    duet_array = np.stack(duet)
    np.savez_compressed(destination, duetrank=duet_array, outer_holdout=holdout)

    semantic_recall20 = np.stack(
        [
            recall_at_depth_rows(row, 20, doc_ids, query_ids, qrels)
            for row in semantic
        ]
    )
    duet_recall20 = np.stack(
        [
            recall_at_depth_rows(row, 20, doc_ids, query_ids, qrels)
            for row in duet_array
        ]
    )
    top10_union_recall = []
    top10_union_sizes = []
    duet_matched = []
    for repeat in range(len(semantic)):
        recall = np.zeros(len(query_ids), dtype=np.float64)
        sizes = np.zeros(len(query_ids), dtype=np.int32)
        matched = np.zeros(len(query_ids), dtype=np.float64)
        for query_index, query_id in enumerate(query_ids):
            relevant = {
                doc_id
                for doc_id, gain in qrels[query_id].items()
                if int(gain) > 0
            }
            sem = {
                doc_ids[int(value)]
                for value in semantic[repeat, query_index, :10]
                if int(value) >= 0
            }
            lex = {
                doc_ids[int(value)]
                for value in lexical[repeat, query_index, :10]
                if int(value) >= 0
            }
            union = sem | lex
            recall[query_index] = len(relevant.intersection(union)) / max(
                len(relevant), 1
            )
            sizes[query_index] = len(union)
            returned = {
                doc_ids[int(value)]
                for value in duet_array[
                    repeat, query_index, : int(sizes[query_index])
                ]
                if int(value) >= 0
            }
            matched[query_index] = len(relevant.intersection(returned)) / max(
                len(relevant), 1
            )
        top10_union_recall.append(recall)
        top10_union_sizes.append(sizes)
        duet_matched.append(matched)
    top10_union_recall_array = np.stack(top10_union_recall)
    top10_union_sizes_array = np.stack(top10_union_sizes)
    duet_matched_array = np.stack(duet_matched)
    return {
        "outer_holdout_queries": int(np.sum(holdout)),
        "semantic_recall_at_20": float(
            np.mean(semantic_recall20[:, holdout])
        ),
        "duetrank_recall_at_20": float(np.mean(duet_recall20[:, holdout])),
        "top10_path_union_recall": float(
            np.mean(top10_union_recall_array[:, holdout])
        ),
        "top10_path_union_mean_output_documents": float(
            np.mean(top10_union_sizes_array[:, holdout])
        ),
        "duetrank_recall_at_union_cardinality": float(
            np.mean(duet_matched_array[:, holdout])
        ),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    dataset = args.dataset
    split = args.split
    if split != "test" and not (dataset == "msmarco" and split == "dev"):
        raise ValueError("the full-corpus prototype supports dev only for MS MARCO")
    documents = FULL_DOCUMENTS[dataset]
    semantic_source = ROOT / "cache" / "full_semantic" / f"{dataset}_{documents}"
    lexical_label = (
        "msmarco_dev_b250k_full_h1024"
        if dataset == "msmarco" and split == "dev"
        else f"{dataset}_full_h1024"
    )
    lexical_source = (
        ROOT / "cache" / "full_candidate_lexical_dpe" / lexical_label
    )
    destination_label = dataset if split == "test" else f"{dataset}_{split}"
    destination = args.output_root / destination_label
    semantic_cache_destination = args.cache_root / f"{dataset}_{documents}"
    lexical_cache_destination = (
        semantic_cache_destination
        if split == "test"
        else args.cache_root / f"{dataset}_{documents}_{split}"
    )
    destination.mkdir(parents=True, exist_ok=True)
    query_ids, query_texts, qrels, _ = load_queries_qrels(
        ROOT / "data" / dataset, split
    )
    if args.query_limit is not None:
        query_ids = query_ids[: args.query_limit]
        query_texts = query_texts[: args.query_limit]
    doc_ids = read_ids(semantic_source / "doc_ids.txt")
    embeddings = np.load(semantic_source / "corpus_embeddings.npy", mmap_mode="r")
    query_embedding_name = (
        "query_embeddings_dev.npy"
        if dataset == "msmarco" and split == "dev"
        else "query_embeddings.npy"
    )
    queries = np.load(semantic_source / query_embedding_name, mmap_mode="r")
    queries = np.asarray(queries[: len(query_ids)], dtype=np.float32)
    if dataset == "msmarco" and split == "dev":
        # Full scanning 8.84M vectors for 6,980 queries is outside the
        # candidate-computation budget.  The audited global-DPE IVF output is
        # used only as the semantic retention reference; compartment rankings
        # are generated afresh below.
        semantic_reference_path = (
            ROOT / "results" / "msmarco_dev_full_semantic" / "rankings.npz"
        )
        exact_dense = np.asarray(
            np.load(semantic_reference_path)["semantic"][0, : len(query_ids)],
            dtype=np.int32,
        )
        semantic_reference_name = "pre-compartment global-DPE IVF"
    else:
        existing_rankings_path = (
            ROOT
            / "results"
            / "full_semantic_hybrid"
            / SEMANTIC_RESULT_NAMES[dataset]
            / "rankings.npz"
        )
        exact_dense = np.asarray(
            np.load(existing_rankings_path)["exact_dense"][: len(query_ids)],
            dtype=np.int32,
        )
        semantic_reference_name = "exact dense"
    semantic_setup = build_semantic_compartment_cache(
        semantic_source, semantic_cache_destination, args.projection_dimension
    )
    lexical_setup = build_lexical_compartment_cache(
        lexical_source,
        lexical_cache_destination,
        args.projection_dimension,
        args.lexical_cells,
    )
    semantic_outputs, semantic_timing = semantic_compartment_retrieval(
        semantic_source,
        semantic_cache_destination,
        queries,
        embeddings,
        exact_dense,
        projection_dimension=args.projection_dimension,
        probes=args.semantic_probes,
        top_per_cell_values=args.semantic_top_per_cell,
        output_depth=args.semantic_output_depth,
        repeats=args.repeats,
        beta=args.beta,
        scale=args.scale,
        seed=args.seed,
        device=args.device,
        latency_samples=args.latency_samples,
    )
    workload_label = (
        "msmarco_dev_full"
        if dataset == "msmarco" and split == "dev"
        else f"{dataset}_full"
    )
    raw_path = ROOT / "results" / "budgeted_lexical" / f"{workload_label}.raw.npz"
    raw_candidates = np.asarray(
        np.load(raw_path)[f"budget_{args.posting_budget}"][: len(query_ids), :1000],
        dtype=np.int32,
    )
    lexical_outputs, lexical_timing = lexical_compartment_retrieval(
        dataset,
        lexical_source,
        lexical_cache_destination,
        raw_candidates,
        query_texts,
        documents=documents,
        projection_dimension=args.projection_dimension,
        cells=args.lexical_cells,
        top_per_cell_values=args.lexical_top_per_cell,
        output_depth=args.lexical_output_depth,
        repeats=args.repeats,
        beta=args.beta,
        scale=args.scale,
        seed=args.seed + 9000,
    )

    dense_metrics = evaluate(exact_dense, doc_ids, query_ids, qrels)
    raw_lexical_metrics = evaluate(raw_candidates[:, :100], doc_ids, query_ids, qrels)
    raw_coverage = relevant_coverage(raw_candidates, doc_ids, query_ids, qrels)
    semantic_rows = []
    for top in args.semantic_top_per_cell:
        metrics = metric_mean(
            semantic_outputs[top], doc_ids, query_ids, qrels
        )
        semantic_rows.append(
            {
                "top_per_cell": top,
                "metrics": metrics,
                "ndcg_retention_vs_exact_dense": metrics["nDCG@10"]
                / max(dense_metrics["nDCG@10"], 1e-12),
                "recall100_retention_vs_exact_dense": metrics["Recall@100"]
                / max(dense_metrics["Recall@100"], 1e-12),
                **semantic_timing["by_top_per_cell"][top],
            }
        )
    lexical_rows = []
    for top in args.lexical_top_per_cell:
        metrics = metric_mean(lexical_outputs[top], doc_ids, query_ids, qrels)
        coverage = float(
            np.mean(
                [
                    relevant_coverage(row, doc_ids, query_ids, qrels)
                    for row in lexical_outputs[top]
                ]
            )
        )
        lexical_rows.append(
            {
                "top_per_cell": top,
                "metrics": metrics,
                "ndcg_retention_vs_budget_bm25": metrics["nDCG@10"]
                / max(raw_lexical_metrics["nDCG@10"], 1e-12),
                "recall100_retention_vs_budget_bm25": metrics["Recall@100"]
                / max(raw_lexical_metrics["Recall@100"], 1e-12),
                "relevant_candidate_coverage": coverage,
                "coverage_retention_vs_budget_candidates": coverage
                / max(raw_coverage, 1e-12),
                **lexical_timing["by_top_per_cell"][top],
            }
        )
    selected_semantic = choose_operating_point(
        semantic_rows, dense_metrics["nDCG@10"], dense_metrics["Recall@100"]
    )
    selected_lexical = choose_operating_point(
        lexical_rows,
        raw_lexical_metrics["nDCG@10"],
        raw_lexical_metrics["Recall@100"],
    )
    semantic_selected = semantic_outputs[selected_semantic]
    lexical_selected = lexical_outputs[selected_lexical]
    semantic_rankings_path = destination / "semantic_compartment.rankings.npz"
    lexical_rankings_path = destination / "lexical_compartment.rankings.npz"
    np.savez_compressed(semantic_rankings_path, semantic=semantic_selected)
    np.savez_compressed(lexical_rankings_path, lexical_dpe=lexical_selected)

    fusion_payload = None
    recall_payload = None
    if args.query_limit is None and not args.skip_duetrank:
        trace_path = (
            ROOT / "results" / "budgeted_lexical" / f"{workload_label}.trace.json"
        )
        fusion_payload = run_duetrank(
            dataset,
            lexical_source="budgeted-dpe",
            scale="full",
            lexical_rankings_path=lexical_rankings_path,
            split=split,
            semantic_rankings_path=semantic_rankings_path,
            trace_path=trace_path,
            budget_key=args.posting_budget,
            model_tag="dual-compartment",
        )
        atomic_json(destination / "duetrank.json", {"datasets": [fusion_payload]})
        recall_payload = reconstruct_duetrank_and_recall(
            dataset,
            semantic_selected,
            lexical_selected,
            fusion_payload,
            query_ids,
            query_texts,
            qrels,
            doc_ids,
            trace_path,
            str(args.posting_budget),
            destination / "duetrank.rankings.npz",
        )

    semantic_entries = int(semantic_setup["posting_entries"])
    semantic_storage = (
        int(semantic_setup["ciphertext_bytes"])
        + int(semantic_setup["norm_bytes"])
        + (semantic_source / "semantic_postings.npy").stat().st_size
        + (semantic_source / "semantic_offsets.npy").stat().st_size
        + 16 * semantic_entries
    )
    lexical_full_storage = documents * (
        args.projection_dimension * 2 + 4 + 2 + 16
    )
    fts_bytes = (
        ROOT / "results" / "keyed_fts5" / f"{dataset}_full.sqlite3"
    ).stat().st_size
    communication = communication_bytes(
        selected_semantic,
        selected_lexical,
        args.semantic_probes,
        args.lexical_cells,
        args.projection_dimension,
    )
    lookup_trace = json.loads(
        (
            ROOT / "results" / "budgeted_lexical" / f"{workload_label}.trace.json"
        ).read_text(encoding="utf-8")
    )[str(args.posting_budget)]
    lexical_lookup_mean = float(np.mean(lookup_trace["latency_ms"][: len(query_ids)]))
    selected_semantic_row = next(
        row for row in semantic_rows if row["top_per_cell"] == selected_semantic
    )
    selected_lexical_row = next(
        row for row in lexical_rows if row["top_per_cell"] == selected_lexical
    )
    report = {
        "experiment": "complete-corpus dual-path Compartment-DPE",
        "dataset": dataset,
        "documents": documents,
        "queries": len(query_ids),
        "configuration": {
            "split": split,
            "semantic_retention_reference": semantic_reference_name,
            "projection_dimension": args.projection_dimension,
            "semantic_cells": int(semantic_setup["cells"]),
            "semantic_probes": args.semantic_probes,
            "semantic_top_per_cell_sweep": args.semantic_top_per_cell,
            "lexical_cells": args.lexical_cells,
            "lexical_top_per_cell_sweep": args.lexical_top_per_cell,
            "posting_budget": args.posting_budget,
            "repeats": args.repeats,
            "selection_rule": "smallest local depth retaining >=98% nDCG@10 and >=95% Recall@100 of the path-specific plaintext reference",
        },
        "setup": {
            "semantic": semantic_setup,
            "lexical_candidate_scoped": lexical_setup,
            "estimated_full_server_storage_bytes": int(
                semantic_storage + lexical_full_storage + fts_bytes
            ),
            "estimated_full_server_storage_gib": (
                semantic_storage + lexical_full_storage + fts_bytes
            )
            / (1024**3),
            "storage_scope_note": f"semantic compartments are fully materialized; lexical full-deployment coordinates are estimated for one keyed compartment copy per document, while the executed cache materializes every row touched by the complete {split} workload",
        },
        "rq1_utility": {
            "exact_dense_metrics": dense_metrics,
            "budget_bm25_metrics": raw_lexical_metrics,
            "budget_candidate_relevant_coverage": raw_coverage,
            "semantic_sweep": semantic_rows,
            "lexical_sweep": lexical_rows,
            "selected_semantic_top_per_cell": selected_semantic,
            "selected_lexical_top_per_cell": selected_lexical,
            "selected_semantic": selected_semantic_row,
            "selected_lexical": selected_lexical_row,
            "duetrank": fusion_payload,
            "matched_budget_recall": recall_payload,
        },
        "rq2_efficiency": {
            "semantic": semantic_timing,
            "lexical": lexical_timing,
            "lexical_candidate_lookup_mean_ms": lexical_lookup_mean,
            "communication": communication,
            "server_online_mean_ms_batched": semantic_timing[
                "batch_server_ms_per_query_repeat"
            ]
            + lexical_timing["server_latency_ms"]["mean"]
            + lexical_lookup_mean,
            "server_online_mean_ms_interactive_sample": semantic_timing[
                "interactive_server_latency_ms"
            ]["mean"]
            + lexical_timing["server_latency_ms"]["mean"]
            + lexical_lookup_mean,
        },
        "caveat": "Both paths expose cell labels, cell sizes, probes, access timing, and within-cell distance order. Candidate-scoped lexical materialization preserves online results but does not measure full offline build time.",
    }
    atomic_json(destination / "results.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=FULL_DOCUMENTS, required=True)
    parser.add_argument("--split", choices=["test", "dev"], default="test")
    parser.add_argument("--projection-dimension", type=int, default=64)
    parser.add_argument("--semantic-probes", type=int, default=128)
    parser.add_argument(
        "--semantic-top-per-cell",
        nargs="+",
        type=int,
        default=[2, 4, 8, 16, 32, 64, 128],
    )
    parser.add_argument(
        "--lexical-top-per-cell", nargs="+", type=int, default=[2, 4, 8, 16]
    )
    parser.add_argument("--lexical-cells", type=int, default=64)
    parser.add_argument("--semantic-output-depth", type=int, default=100)
    parser.add_argument("--lexical-output-depth", type=int, default=300)
    parser.add_argument("--posting-budget", type=int, default=50_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--beta", type=float, default=0.10)
    parser.add_argument("--scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--latency-samples", type=int, default=64)
    parser.add_argument("--query-limit", type=int, default=None)
    parser.add_argument("--skip-duetrank", action="store_true")
    parser.add_argument(
        "--cache-root", type=Path, default=ROOT / "cache" / "dual_compartment_full"
    )
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "results" / "dual_compartment_full"
    )
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    result = run(parsed)
    print(
        json.dumps(
            {
                "dataset": result["dataset"],
                "selected_semantic_top_per_cell": result["rq1_utility"][
                    "selected_semantic_top_per_cell"
                ],
                "selected_lexical_top_per_cell": result["rq1_utility"][
                    "selected_lexical_top_per_cell"
                ],
                "server_online_mean_ms_batched": result["rq2_efficiency"][
                    "server_online_mean_ms_batched"
                ],
                "communication_mib": result["rq2_efficiency"]["communication"][
                    "total_mib"
                ],
            },
            indent=2,
        ),
        flush=True,
    )
