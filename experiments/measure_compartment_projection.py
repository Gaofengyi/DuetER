"""Measure coherence and coordinate-sampling distortion for final CCADPE runs.

The diagnostic samples actual query--candidate intermediate differences from
the complete-corpus semantic and lexical workloads.  It reports only the
rank-reducing projection error; bounded DPE noise is already present in the
intermediate coordinates and is handled separately by the paper's beta term.
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for dependency in (ROOT / ".deps", ROOT / ".gpu_deps"):
    if dependency.exists():
        sys.path.insert(0, str(dependency))

import numpy as np

from benchmark_dual_compartment_full import (
    FULL_DOCUMENTS,
    KEY_LEXICAL,
    KEY_SEMANTIC,
    SEMANTIC_RESULT_NAMES,
    cell_transform,
    keyed_lexical_cells,
)
from benchmark_million_lexical_dpe import encrypt_query_matrix, query_mips
from benchmark_million_semantic_hybrid import IvfFiles, encrypt_queries, load_queries_qrels


PROJECTION_DIMENSION = 256
SEMANTIC_PROBES = 128
MAX_QUERIES = 64
SEMANTIC_CELLS_PER_QUERY = 16
SEMANTIC_DOCS_PER_CELL = 8
LEXICAL_DOCS_PER_QUERY = 128
BASE_SEED = 20260917


def summarize(coherences: list[float], errors: list[float]) -> dict[str, object]:
    coherence = np.asarray(coherences, dtype=np.float64)
    error = np.asarray(errors, dtype=np.float64)
    return {
        "sampled_differences": int(len(error)),
        "coherence": {
            "mean": float(np.mean(coherence)),
            "p95": float(np.percentile(coherence, 95)),
            "p99": float(np.percentile(coherence, 99)),
            "max": float(np.max(coherence)),
        },
        "relative_norm_error": {
            "mean": float(np.mean(error)),
            "p95": float(np.percentile(error, 95)),
            "p99": float(np.percentile(error, 99)),
            "max": float(np.max(error)),
            "fraction_gt_0_10": float(np.mean(error > 0.10)),
            "fraction_gt_0_20": float(np.mean(error > 0.20)),
        },
    }


def record_difference(
    z: np.ndarray,
    coordinates: np.ndarray,
    multiplier: float,
    coherences: list[float],
    errors: list[float],
) -> None:
    z = np.asarray(z, dtype=np.float64)
    norm = float(np.linalg.norm(z))
    if norm <= 1e-12:
        return
    projected = float(np.linalg.norm(z[coordinates]) * multiplier)
    coherences.append(float(len(z) * np.max(z * z) / (norm * norm)))
    errors.append(abs(projected / norm - 1.0))


def sampled_query_indices(count: int) -> np.ndarray:
    take = min(MAX_QUERIES, count)
    return np.unique(np.linspace(0, count - 1, take, dtype=np.int64))


def semantic_diagnostic(dataset: str, documents: int) -> dict[str, object]:
    source = ROOT / "cache" / "full_semantic" / f"{dataset}_{documents}"
    _, query_texts, _, _ = load_queries_qrels(ROOT / "data" / dataset, "test")
    queries = np.load(source / "query_embeddings.npy", mmap_mode="r")
    query_indices = sampled_query_indices(min(len(query_texts), len(queries)))
    query_vectors = np.asarray(queries[query_indices], dtype=np.float32)
    global_queries = encrypt_queries(
        query_vectors,
        source / "semantic_dpe_key.npz",
        0.10,
        3.0,
        BASE_SEED + 2027,
    )
    base = np.load(source / "semantic_cipher_fp16.npy", mmap_mode="r")
    ivf = IvfFiles.load(source)
    residual = query_vectors - ivf.mean[None, :]
    residual /= np.maximum(np.linalg.norm(residual, axis=1, keepdims=True), 1e-12)
    scores = residual @ np.asarray(ivf.centroids, dtype=np.float32).T
    cells = np.argpartition(-scores, SEMANTIC_PROBES - 1, axis=1)[:, :SEMANTIC_PROBES]

    coherences: list[float] = []
    errors: list[float] = []
    cell_ordinals = np.linspace(
        0, SEMANTIC_PROBES - 1, SEMANTIC_CELLS_PER_QUERY, dtype=np.int64
    )
    for local_query, query_cells in zip(global_queries, cells, strict=True):
        selected_cells = query_cells[cell_ordinals]
        for raw_cell in selected_cells:
            cell = int(raw_cell)
            first, last = int(ivf.offsets[cell]), int(ivf.offsets[cell + 1])
            if first == last:
                continue
            count = min(SEMANTIC_DOCS_PER_CELL, last - first)
            positions = np.linspace(first, last - 1, count, dtype=np.int64)
            rows = np.asarray(ivf.postings[positions], dtype=np.int64)
            transform = cell_transform(
                KEY_SEMANTIC, cell, int(base.shape[1]), PROJECTION_DIMENSION
            )
            for row in rows:
                record_difference(
                    local_query - np.asarray(base[int(row)], dtype=np.float32),
                    transform.coordinates,
                    transform.multiplier,
                    coherences,
                    errors,
                )
    output = summarize(coherences, errors)
    output.update(
        {
            "base_dimension": int(base.shape[1]),
            "projection_dimension": PROJECTION_DIMENSION,
            "sampled_queries": int(len(query_indices)),
        }
    )
    return output


def lexical_diagnostic(dataset: str, documents: int) -> dict[str, object]:
    source = ROOT / "cache" / "full_candidate_lexical_dpe" / f"{dataset}_full_h1024"
    compartment = ROOT / "cache" / "dual_compartment_full_p256" / f"{dataset}_{documents}"
    _, query_texts, _, _ = load_queries_qrels(ROOT / "data" / dataset, "test")
    query_indices = sampled_query_indices(len(query_texts))
    chosen_texts = [query_texts[int(index)] for index in query_indices]
    connection = sqlite3.connect(ROOT / "results" / "keyed_fts5" / f"{dataset}_full.sqlite3")
    document_frequency = {
        str(term): int(df) for term, df in connection.execute("SELECT term, doc FROM vocab")
    }
    connection.close()
    plain_queries = np.stack(
        [query_mips(text, document_frequency, documents, 1024) for text in chosen_texts]
    )
    global_queries = encrypt_query_matrix(
        plain_queries,
        source / "lexical_dpe_key.npz",
        0.10,
        3.0,
        BASE_SEED + 9000 + 2027,
    )
    raw = np.load(ROOT / "results" / "budgeted_lexical" / f"{dataset}_full.raw.npz")
    candidates = np.asarray(raw["budget_50000"][query_indices, :1000], dtype=np.int32)
    candidate_rows = np.load(source / "candidate_rows.npy", mmap_mode="r")
    base = np.load(source / "lexical_cipher_fp16.npy", mmap_mode="r")
    assignment_path = compartment / "lexical_compartment_cells.npy"
    if assignment_path.exists():
        assignments = np.load(assignment_path, mmap_mode="r")
    else:
        assignments = keyed_lexical_cells(np.asarray(candidate_rows), 64)

    transforms = [
        cell_transform(KEY_LEXICAL, cell, int(base.shape[1]), PROJECTION_DIMENSION)
        for cell in range(64)
    ]
    coherences: list[float] = []
    errors: list[float] = []
    for query, raw_candidates in zip(global_queries, candidates, strict=True):
        valid = raw_candidates[raw_candidates >= 0]
        if len(valid) > LEXICAL_DOCS_PER_QUERY:
            ordinals = np.linspace(0, len(valid) - 1, LEXICAL_DOCS_PER_QUERY, dtype=np.int64)
            valid = valid[ordinals]
        positions = np.searchsorted(candidate_rows, valid)
        if np.any(np.asarray(candidate_rows[positions]) != valid):
            raise RuntimeError("candidate absent from lexical scoped cache")
        for position in positions:
            cell = int(assignments[int(position)])
            transform = transforms[cell]
            record_difference(
                query - np.asarray(base[int(position)], dtype=np.float32),
                transform.coordinates,
                transform.multiplier,
                coherences,
                errors,
            )
    output = summarize(coherences, errors)
    output.update(
        {
            "base_dimension": int(base.shape[1]),
            "projection_dimension": PROJECTION_DIMENSION,
            "sampled_queries": int(len(query_indices)),
        }
    )
    return output


def conservative_bound(summary: dict[str, object], delta: float = 0.05) -> dict[str, object]:
    """Evaluate the paper's union bound using the sampled maximum coherence."""
    count = int(summary["sampled_differences"])
    coherence = float(summary["coherence"]["max"])
    dimension = int(summary["projection_dimension"])

    def failure(epsilon: float) -> float:
        kappa = 2.0 * epsilon - epsilon * epsilon
        exponent = -dimension * kappa * kappa / (2.0 * coherence * (1.0 + kappa / 3.0))
        return min(1.0, 2.0 * count * math.exp(exponent))

    grid = np.linspace(0.001, 0.999, 999)
    sufficient = next((float(epsilon) for epsilon in grid if failure(float(epsilon)) <= delta), None)
    return {
        "workload_size": count,
        "empirical_max_coherence": coherence,
        "union_failure_at_epsilon_0_10": failure(0.10),
        "union_failure_at_epsilon_0_20": failure(0.20),
        "smallest_epsilon_with_delta_at_most_0_05": sufficient,
        "non_vacuous_below_epsilon_1": sufficient is not None,
    }


def main() -> None:
    report: dict[str, object] = {
        "experiment": "CCADPE projection coherence and distortion diagnostic",
        "sampling": {
            "max_queries_per_dataset": MAX_QUERIES,
            "semantic_cells_per_query": SEMANTIC_CELLS_PER_QUERY,
            "semantic_documents_per_cell": SEMANTIC_DOCS_PER_CELL,
            "lexical_documents_per_query": LEXICAL_DOCS_PER_QUERY,
            "deterministic": True,
        },
        "datasets": {},
    }
    for dataset in ("nq", "hotpotqa", "msmarco"):
        documents = FULL_DOCUMENTS[dataset]
        print(f"[{dataset}] semantic", flush=True)
        semantic = semantic_diagnostic(dataset, documents)
        semantic["analytic_union_bound_on_sample"] = conservative_bound(semantic)
        print(f"[{dataset}] lexical", flush=True)
        lexical = lexical_diagnostic(dataset, documents)
        lexical["analytic_union_bound_on_sample"] = conservative_bound(lexical)
        report["datasets"][dataset] = {"semantic": semantic, "lexical": lexical}
        print(json.dumps(report["datasets"][dataset], indent=2), flush=True)

    output = ROOT / "results" / "ccadpe_projection_diagnostic.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
