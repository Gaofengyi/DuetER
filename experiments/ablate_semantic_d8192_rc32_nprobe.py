"""Semantic CCADPE nprobe sweep at d=8192 and configurable r_c.

The full 8192-D database intermediate is never materialized.  For every IVF
cell, this script evaluates exactly the selected rows of the normalized
Walsh--Hadamard DPE transform, adds the exact selected-coordinate marginal of the
isotropic database noise, and stores only the cell-local FP16 coordinates.

All nprobe operating points share one routing run and one set of local scores.
Both posting-entry scan rate (the convention used by the paper) and the exact
unique-document scan rate are reported.  The latter accounts for the two IVF
replicas of every semantic document.
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


ROOT = Path(__file__).resolve().parent

# Import the host CUDA build before the legacy modules prepend their dependency
# directories.  The CPU runtime remains a valid fallback.
try:
    import torch
except ImportError:
    if (ROOT / ".deps").exists():
        sys.path.insert(0, str(ROOT / ".deps"))
    import torch

import numpy as np

from benchmark_million_semantic_hybrid import (
    IvfFiles,
    atomic_json,
    ball_noise,
    dpe_transform,
    evaluate,
    load_queries_qrels,
    read_ids,
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
WORK_DIMENSION = 8192


def command_line_integer(name: str, default: int) -> int:
    if name not in sys.argv:
        return default
    position = sys.argv.index(name)
    if position + 1 >= len(sys.argv):
        raise ValueError(f"missing value after {name}")
    return int(sys.argv[position + 1])


PROJECTION_DIMENSION = command_line_integer("--projection-dimension", 32)
CONFIG_TAG = f"semantic_d8192_rc{PROJECTION_DIMENSION}"
SEMANTIC_CELLS = 2048
PROBES = (128, 64, 32, 16, 8)
GLOBAL_DPE_SEED = 20260823
QUERY_NOISE_SEED = 20260917 + 2027
BETA = 0.10
SCALE = 3.0
DATABASE_NOISE_RADIUS = 3.0 * SCALE * BETA / 8.0
QUERY_NOISE_RADIUS = SCALE * BETA / 8.0

CONFIG = {
    "nq": {"split": "test", "top": 16, "result": "nq"},
    "hotpotqa": {"split": "test", "top": 16, "result": "hotpotqa"},
    "msmarco": {"split": "dev", "top": 32, "result": "msmarco_dev"},
}


@dataclass(frozen=True)
class CellTransform:
    coordinates: np.ndarray
    signs: np.ndarray
    multiplier: float
    translation: np.ndarray


def cell_transform(
    key: bytes, cell: int, work_dimension: int, projection_dimension: int
) -> CellTransform:
    digest = hashlib.sha256(
        key
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


def top_indices(scores: np.ndarray, depth: int) -> np.ndarray:
    if depth <= 0:
        return np.empty(0, dtype=np.int64)
    if depth >= len(scores):
        return np.argsort(-scores, kind="stable")
    selected = np.argpartition(-scores, depth - 1)[:depth]
    return selected[np.argsort(-scores[selected], kind="stable")]


def route_semantic_queries(
    ivf: IvfFiles, queries: np.ndarray, probes: int, device: str
) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    residual = np.asarray(queries, dtype=np.float32) - ivf.mean[None, :]
    residual /= np.maximum(np.linalg.norm(residual, axis=1, keepdims=True), 1e-12)
    with torch.inference_mode():
        query_device = torch.from_numpy(residual).to(device)
        centroid_device = torch.from_numpy(ivf.centroids).to(device)
        cells = torch.topk(query_device @ centroid_device.T, k=probes, dim=1).indices
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
    repeats, query_count, _, _ = emitted.shape
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
            maximum_candidates = np.unique(
                emitted[repeat, query_index][emitted[repeat, query_index] >= 0]
            )
            maximum_scores = (
                np.asarray(embeddings[maximum_candidates], dtype=np.float32)
                @ queries[query_index]
                if len(maximum_candidates)
                else np.empty(0, dtype=np.float32)
            )
            for top in top_per_cell_values:
                subset = emitted[repeat, query_index, :, :top].reshape(-1)
                candidates = np.unique(subset[subset >= 0])
                diagnostics[top]["candidate_counts"].append(float(len(candidates)))
                diagnostics[top]["dense_coverages"].append(
                    len(
                        set(map(int, candidates)).intersection(
                            map(int, exact_dense[query_index])
                        )
                    )
                    / max(len(exact_dense[query_index]), 1)
                )
                if not len(candidates):
                    continue
                positions = np.searchsorted(maximum_candidates, candidates)
                scores = maximum_scores[positions]
                order = top_indices(scores, min(output_depth, len(candidates)))
                outputs[top][repeat, query_index, : len(order)] = candidates[order]
    summarized = {
        top: {
            "candidate_mean": float(np.mean(values["candidate_counts"])),
            "candidate_p95": float(np.percentile(values["candidate_counts"], 95)),
            "dense_top100_coverage_mean": float(np.mean(values["dense_coverages"])),
        }
        for top, values in diagnostics.items()
    }
    return outputs, summarized, time.perf_counter() - started


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


def global_key() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(GLOBAL_DPE_SEED)
    sign1 = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), WORK_DIMENSION)
    sign2 = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), WORK_DIMENSION)
    permutation = rng.permutation(WORK_DIMENSION)
    return sign1, sign2, permutation


GLOBAL_SIGN1, GLOBAL_SIGN2, GLOBAL_PERMUTATION = global_key()
PARITY_TABLE = np.asarray(
    [int(value).bit_count() & 1 for value in range(WORK_DIMENSION)], dtype=np.int8
)


def selected_dpe_matrix(output_coordinates: np.ndarray, input_dimension: int) -> np.ndarray:
    """Map unpadded semantic inputs to selected 8192-D DPE coordinates."""
    source = GLOBAL_PERMUTATION[np.asarray(output_coordinates, dtype=np.int64)]
    inputs = np.arange(input_dimension, dtype=np.int64)
    parity = PARITY_TABLE[np.bitwise_and(source[:, None], inputs[None, :])]
    hadamard = (1.0 - 2.0 * parity.astype(np.float32)) / math.sqrt(WORK_DIMENSION)
    return (
        hadamard
        * GLOBAL_SIGN1[:input_dimension][None, :]
        * GLOBAL_SIGN2[source][:, None]
    ).astype(np.float32)


def selected_ball_noise(
    rng: np.random.Generator,
    rows: int,
    selected_dimension: int,
    radius: float,
) -> np.ndarray:
    """Exact marginal of uniform 8192-ball noise on selected coordinates."""
    selected = rng.normal(size=(rows, selected_dimension)).astype(np.float32)
    omitted = rng.chisquare(
        WORK_DIMENSION - selected_dimension, size=rows
    ).astype(np.float32)
    norm = np.sqrt(np.sum(selected * selected, axis=1) + omitted).astype(np.float32)
    radii = (
        radius * np.power(rng.random(rows), 1.0 / WORK_DIMENSION)
    ).astype(np.float32)
    return selected * (radii / np.maximum(norm, 1e-12))[:, None]


def build_local_cache(source: Path, destination: Path, device: str) -> dict[str, object]:
    destination.mkdir(parents=True, exist_ok=True)
    metadata_path = destination / "metadata.json"
    local_path = destination / "semantic_compartment_fp16.npy"
    norms_path = destination / "semantic_compartment_norms.npy"
    if all(path.exists() for path in (metadata_path, local_path, norms_path)):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {
            "work_dimension": WORK_DIMENSION,
            "projection_dimension": PROJECTION_DIMENSION,
            "cells": SEMANTIC_CELLS,
        }
        if all(int(metadata[key]) == value for key, value in expected.items()):
            return metadata
        raise RuntimeError(
            f"semantic d8192/r{PROJECTION_DIMENSION} cache has incompatible metadata"
        )

    embeddings = np.load(source / "corpus_embeddings.npy", mmap_mode="r")
    postings = np.load(source / "semantic_postings.npy", mmap_mode="r")
    offsets = np.load(source / "semantic_offsets.npy")
    local = np.lib.format.open_memmap(
        local_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(postings), PROJECTION_DIMENSION),
    )
    norms = np.lib.format.open_memmap(
        norms_path, mode="w+", dtype=np.float32, shape=(len(postings),)
    )
    started = time.perf_counter()
    use_cuda = device.startswith("cuda")
    with torch.inference_mode():
        for cell in range(len(offsets) - 1):
            first, last = int(offsets[cell]), int(offsets[cell + 1])
            if first == last:
                continue
            transform = cell_transform(
                KEY_SEMANTIC, cell, WORK_DIMENSION, PROJECTION_DIMENSION
            )
            matrix = selected_dpe_matrix(transform.coordinates, embeddings.shape[1])
            document_rows = np.asarray(postings[first:last], dtype=np.int64)
            values = np.asarray(embeddings[document_rows], dtype=np.float32)
            if use_cuda:
                values_device = torch.from_numpy(values).to(device)
                matrix_device = torch.from_numpy(matrix.T.copy()).to(device)
                selected = (SCALE * (values_device @ matrix_device)).cpu().numpy()
            else:
                selected = SCALE * (values @ matrix.T)
            # Each cell receives the exact local marginal.  Replicated copies
            # are independently sampled, which leaves every within-cell score
            # distribution unchanged and is recorded in the metadata.
            noise_rng = np.random.default_rng(GLOBAL_DPE_SEED + 1009 + cell * 100003)
            selected += selected_ball_noise(
                noise_rng, len(values), PROJECTION_DIMENSION, DATABASE_NOISE_RADIUS
            )
            projected = (
                selected * transform.signs * transform.multiplier
                + transform.translation
            ).astype(np.float32)
            stored = projected.astype(np.float16)
            local[first:last] = stored
            norms[first:last] = np.sum(stored.astype(np.float32) ** 2, axis=1)
            if cell == 0 or cell + 1 == len(offsets) - 1 or (cell + 1) % 128 == 0:
                local.flush()
                norms.flush()
                print(
                    f"[cache d8192/r{PROJECTION_DIMENSION}] "
                    f"cell {cell + 1:,}/{len(offsets) - 1:,}",
                    flush=True,
                )
    if use_cuda:
        torch.cuda.synchronize()
    local.flush()
    norms.flush()
    metadata = {
        "variant": "semantic selected-row CCADPE",
        "work_dimension": WORK_DIMENSION,
        "projection_dimension": PROJECTION_DIMENSION,
        "cells": int(len(offsets) - 1),
        "documents": int(len(embeddings)),
        "posting_entries": int(len(postings)),
        "input_dimension": int(embeddings.shape[1]),
        "beta": BETA,
        "scale": SCALE,
        "database_noise_radius": DATABASE_NOISE_RADIUS,
        "storage_dtype": "float16",
        "selected_transform_evaluation": True,
        "replica_noise_note": (
            "Each posting copy samples the exact selected-coordinate marginal independently; "
            "this preserves the within-compartment score distribution used for retrieval."
        ),
        "build_seconds": time.perf_counter() - started,
        "ciphertext_bytes": int(local_path.stat().st_size),
        "norm_bytes": int(norms_path.stat().st_size),
    }
    atomic_json(metadata_path, metadata)
    return metadata


def reference_ranking(dataset: str, query_count: int) -> tuple[np.ndarray, str]:
    if dataset == "msmarco":
        path = ROOT / "results" / "msmarco_dev_full_semantic" / "rankings.npz"
        return (
            np.asarray(np.load(path)["semantic"][0, :query_count], dtype=np.int32),
            "pre-compartment global-DPE IVF",
        )
    path = (
        ROOT
        / "results"
        / "full_semantic_hybrid"
        / SEMANTIC_RESULT_NAMES[dataset]
        / "rankings.npz"
    )
    return (
        np.asarray(np.load(path)["exact_dense"][:query_count], dtype=np.int32),
        "exact dense",
    )


def exact_scan_prefixes(
    assignments: np.ndarray,
    offsets: np.ndarray,
    query_cells: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Return posting and unique-document reads for every nprobe prefix."""
    cells = len(offsets) - 1
    first = np.minimum(assignments[:, 0], assignments[:, 1]).astype(np.int64)
    second = np.maximum(assignments[:, 0], assignments[:, 1]).astype(np.int64)
    upper = np.bincount(first * cells + second, minlength=cells * cells).reshape(cells, cells)
    pair_counts = upper + upper.T
    diagonal = np.diag_indices(cells)
    pair_counts[diagonal] = upper[diagonal]
    cell_sizes = np.diff(offsets).astype(np.int64)
    posting = {probe: np.empty(len(query_cells), dtype=np.int64) for probe in PROBES}
    unique = {probe: np.empty(len(query_cells), dtype=np.int64) for probe in PROBES}
    wanted = {probe: probe - 1 for probe in PROBES}
    for query_index, ordered in enumerate(query_cells):
        sub = pair_counts[np.ix_(ordered, ordered)]
        duplicates = np.cumsum(np.tril(sub, k=-1).sum(axis=1))
        reads = np.cumsum(cell_sizes[ordered])
        uniques = reads - duplicates
        for probe, position in wanted.items():
            posting[probe][query_index] = reads[position]
            unique[probe][query_index] = uniques[position]
    return posting, unique


def score_all_cells(
    source: Path,
    cache: Path,
    queries: np.ndarray,
    probes: int,
    local_depth: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    ivf = IvfFiles.load(source)
    query_cells, route_seconds = route_semantic_queries(ivf, queries, probes, device)
    global_queries = SCALE * dpe_transform(
        np.asarray(queries, dtype=np.float32),
        GLOBAL_SIGN1,
        GLOBAL_SIGN2,
        GLOBAL_PERMUTATION,
    )
    query_rng = np.random.default_rng(QUERY_NOISE_SEED)
    global_queries += ball_noise(
        query_rng, len(queries), WORK_DIMENSION, QUERY_NOISE_RADIUS
    )
    local = np.load(cache / "semantic_compartment_fp16.npy", mmap_mode="r")
    norms = np.load(cache / "semantic_compartment_norms.npy", mmap_mode="r")
    emitted = np.full(
        (len(queries), probes, local_depth), -1, dtype=np.int32
    )
    by_cell: list[list[tuple[int, int]]] = [[] for _ in range(len(ivf.offsets) - 1)]
    for query_index, cells in enumerate(query_cells):
        for ordinal, cell in enumerate(cells):
            by_cell[int(cell)].append((query_index, ordinal))

    started = time.perf_counter()
    with torch.inference_mode():
        for cell, locations in enumerate(by_cell):
            if not locations:
                continue
            first, last = int(ivf.offsets[cell]), int(ivf.offsets[cell + 1])
            if first == last:
                continue
            q_indices = np.asarray([item[0] for item in locations], dtype=np.int64)
            ordinals = np.asarray([item[1] for item in locations], dtype=np.int64)
            transform = cell_transform(
                KEY_SEMANTIC, cell, WORK_DIMENSION, PROJECTION_DIMENSION
            )
            q_local = (
                global_queries[q_indices][:, transform.coordinates]
                * transform.signs
                * transform.multiplier
                + transform.translation
            ).astype(np.float16)
            rows_device = torch.from_numpy(
                np.asarray(local[first:last], dtype=np.float16).copy()
            ).to(device)
            norms_device = torch.from_numpy(
                np.asarray(norms[first:last], dtype=np.float32).copy()
            ).to(device)
            q_device = torch.from_numpy(q_local).to(device)
            scores = 2.0 * (q_device @ rows_device.T).float() - norms_device[None, :]
            depth = min(local_depth, last - first)
            selected = torch.topk(scores, k=depth, dim=1).indices.cpu().numpy()
            documents = np.asarray(ivf.postings[first:last], dtype=np.int32)
            emitted[q_indices, ordinals, :depth] = documents[selected]
            if cell == 0 or cell + 1 == len(by_cell) or (cell + 1) % 128 == 0:
                print(
                    f"[query d8192/r{PROJECTION_DIMENSION}] "
                    f"cell {cell + 1:,}/{len(by_cell):,}",
                    flush=True,
                )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return emitted, query_cells, route_seconds + (time.perf_counter() - started)


def run_dataset(dataset: str, device: str) -> dict[str, object]:
    cfg = CONFIG[dataset]
    documents = FULL_DOCUMENTS[dataset]
    source = ROOT / "cache" / "full_semantic" / f"{dataset}_{documents}"
    cache = ROOT / "cache" / CONFIG_TAG / str(cfg["result"])
    destination = ROOT / "results" / f"{CONFIG_TAG}_nprobe" / str(cfg["result"])
    destination.mkdir(parents=True, exist_ok=True)
    setup = build_local_cache(source, cache, device)

    split = str(cfg["split"])
    local_depth = int(cfg["top"])
    query_ids, _, qrels, _ = load_queries_qrels(ROOT / "data" / dataset, split)
    doc_ids = read_ids(source / "doc_ids.txt")
    embeddings = np.load(source / "corpus_embeddings.npy", mmap_mode="r")
    query_name = "query_embeddings_dev.npy" if split == "dev" else "query_embeddings.npy"
    queries = np.asarray(
        np.load(source / query_name, mmap_mode="r")[: len(query_ids)], dtype=np.float32
    )
    reference, reference_name = reference_ranking(dataset, len(query_ids))
    reference_metrics = evaluate(reference, doc_ids, query_ids, qrels)

    emitted, query_cells, score_seconds = score_all_cells(
        source,
        cache,
        queries,
        max(PROBES),
        local_depth,
        device,
    )
    offsets = np.load(source / "semantic_offsets.npy")
    assignments = np.load(source / "semantic_assignments.npy", mmap_mode="r")
    posting_reads, unique_reads = exact_scan_prefixes(assignments, offsets, query_cells)

    rows: list[dict[str, object]] = []
    rankings: dict[str, np.ndarray] = {}
    for probes in PROBES:
        outputs, diagnostics, rerank_seconds = exact_semantic_rerank(
            emitted[None, :, :probes, :],
            embeddings,
            queries,
            reference,
            [local_depth],
            100,
        )
        ranking = outputs[local_depth]
        metrics = metric_mean(ranking, doc_ids, query_ids, qrels)
        diagnostic = diagnostics[local_depth]
        row = {
            "nprobe": probes,
            "local_depth": local_depth,
            "nDCG@10": float(metrics["nDCG@10"]),
            "nDCG@10_retention": float(metrics["nDCG@10"])
            / max(float(reference_metrics["nDCG@10"]), 1e-12),
            "Recall@100": float(metrics["Recall@100"]),
            "Recall@100_retention": float(metrics["Recall@100"])
            / max(float(reference_metrics["Recall@100"]), 1e-12),
            "posting_entries_read_mean": float(np.mean(posting_reads[probes])),
            "posting_scan_percent": 100.0
            * float(np.mean(posting_reads[probes]))
            / documents,
            "unique_documents_scanned_mean": float(np.mean(unique_reads[probes])),
            "unique_document_scan_percent": 100.0
            * float(np.mean(unique_reads[probes]))
            / documents,
            "returned_candidate_union_mean": float(diagnostic["candidate_mean"]),
            "reference_top100_coverage": float(
                diagnostic["dense_top100_coverage_mean"]
            ),
            "client_rerank_seconds": float(rerank_seconds),
        }
        rows.append(row)
        rankings[f"nprobe_{probes}"] = ranking[0]
        print(f"[{dataset}] {json.dumps(row, ensure_ascii=False)}", flush=True)

    np.savez_compressed(destination / "rankings.npz", **rankings)
    result = {
        "experiment": (
            f"semantic CCADPE nprobe sweep at d=8192, "
            f"r_c={PROJECTION_DIMENSION}"
        ),
        "dataset": dataset,
        "documents": documents,
        "queries": len(query_ids),
        "split": split,
        "parameters": {
            "work_dimension": WORK_DIMENSION,
            "projection_dimension": PROJECTION_DIMENSION,
            "semantic_cells": SEMANTIC_CELLS,
            "ivf_assignments_per_document": 2,
            "local_depth": local_depth,
            "output_depth": 100,
            "beta": BETA,
            "scale": SCALE,
            "global_dpe_seed": GLOBAL_DPE_SEED,
            "query_noise_seed": QUERY_NOISE_SEED,
        },
        "scan_definition": {
            "posting_scan_percent": (
                "mean posting entries read divided by total corpus documents"
            ),
            "unique_document_scan_percent": (
                "mean number of distinct corpus documents touched by probed cells "
                "divided by total corpus documents; exact for the two-assignment IVF"
            ),
        },
        "reference": reference_name,
        "reference_metrics": reference_metrics,
        "cache": setup,
        "shared_score_seconds_for_max_nprobe": float(score_seconds),
        "rows": rows,
    }
    atomic_json(destination / "results.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(CONFIG) + ("all",), default="all")
    parser.add_argument(
        "--projection-dimension",
        type=int,
        default=PROJECTION_DIMENSION,
        choices=(32, 64),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=("cuda", "cpu"),
    )
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="merge completed per-dataset result files without rerunning retrieval",
    )
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    selected = tuple(CONFIG) if args.dataset == "all" else (args.dataset,)
    summary_path = (
        ROOT / "results" / f"{CONFIG_TAG}_nprobe" / "summary.json"
    )
    if summary_path.exists():
        report = json.loads(summary_path.read_text(encoding="utf-8"))
        report["device"] = args.device
    else:
        report = {
            "experiment": (
                f"semantic CCADPE nprobe sweep at d=8192, "
                f"r_c={PROJECTION_DIMENSION}"
            ),
            "device": args.device,
            "datasets": {},
        }
    if args.merge_existing:
        for dataset, cfg in CONFIG.items():
            result_path = (
                ROOT
                / "results"
                / f"{CONFIG_TAG}_nprobe"
                / str(cfg["result"])
                / "results.json"
            )
            if result_path.exists():
                report["datasets"][dataset] = json.loads(
                    result_path.read_text(encoding="utf-8")
                )
        atomic_json(summary_path, report)
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
        return
    for dataset in selected:
        report["datasets"][dataset] = run_dataset(dataset, args.device)
        atomic_json(summary_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
