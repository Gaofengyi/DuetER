"""Same-host FiQA benchmark for DuetER and the official Pisces workload.

The script consumes the preprocessed FiQA JSONL files used by the official
Pisces artifact, builds the final dual-compartment DuetER indexes, and runs the
same three queries with k=10 for five repetitions.  Neural encoding is excluded
because both systems consume precomputed Granite embeddings/tokenization.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
try:
    import resource
except ImportError:  # Windows utility reruns; the controlled VM run still uses resource.
    resource = None
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans


ROOT = Path(os.environ.get("DUETER_FIQA_ROOT", Path.cwd()))
DATA = Path(os.environ.get("DUETER_FIQA_DATA", str(ROOT / "data")))
OUTPUT = Path(os.environ.get("DUETER_FIQA_OUTPUT", str(ROOT / "dueter_fiqa_same_host.json")))
QRELS = os.environ.get("DUETER_FIQA_QRELS")
N_DOCS = int(os.environ.get("DUETER_FIQA_N_DOCS", "57638"))
N_QUERIES = int(os.environ.get("DUETER_FIQA_N_QUERIES", "3"))
REPEATS = int(os.environ.get("DUETER_FIQA_REPEATS", "5"))
K = int(os.environ.get("DUETER_FIQA_K", "10"))
SEMANTIC_CELLS = 2_048
SEMANTIC_PROBES = 128
SEMANTIC_ASSIGNMENTS = 2
LEXICAL_CELLS = 64
LOCAL_DEPTH = 16
PROJECTION_DIM = 256
SEMANTIC_WORK_DIM = 512
LEXICAL_HASH_DIM = 1_024
LEXICAL_WORK_DIM = 2_048
POSTING_BUDGET = 50_000
BETA = 0.10
SCALE = 3.0
SEED = 20260826
KEY_SEMANTIC = b"DuetER-semantic-ccadpe-fiqa-v1"
KEY_LEXICAL = b"DuetER-lexical-ccadpe-fiqa-v1"
KEY_FEATURE = b"DuetDPE-Lexical-v1"


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return (values / np.maximum(norms, 1e-12)).astype(np.float32)


def top_indices(scores: np.ndarray, depth: int) -> np.ndarray:
    depth = min(max(int(depth), 0), len(scores))
    if depth == 0:
        return np.empty(0, dtype=np.int64)
    if depth == len(scores):
        return np.argsort(-scores, kind="stable").astype(np.int64)
    selected = np.argpartition(-scores, depth - 1)[:depth]
    return selected[np.argsort(-scores[selected], kind="stable")].astype(np.int64)


def fwht(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32).copy()
    width = result.shape[1]
    step = 1
    while step < width:
        shaped = result.reshape(len(result), -1, 2 * step)
        left = shaped[:, :, :step].copy()
        right = shaped[:, :, step : 2 * step].copy()
        shaped[:, :, :step] = left + right
        shaped[:, :, step : 2 * step] = left - right
        result = shaped.reshape(len(result), width)
        step *= 2
    return result / math.sqrt(width)


def cell_seed(key: bytes, cell: int, purpose: bytes) -> int:
    digest = hashlib.sha256(
        key + purpose + int(cell).to_bytes(8, "big") + SEED.to_bytes(8, "big")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def cell_parameters(key: bytes, cell: int, work_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cell_seed(key, cell, b"transform"))
    flatten_sign = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), work_dim)
    coordinates = np.sort(rng.choice(work_dim, PROJECTION_DIM, replace=False)).astype(np.int32)
    output_sign = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), PROJECTION_DIM)
    translation = rng.normal(0.0, SCALE, PROJECTION_DIM).astype(np.float32)
    return flatten_sign, coordinates, output_sign, translation


def ball_noise(rng: np.random.Generator, rows: int, radius: float) -> np.ndarray:
    if radius <= 0:
        return np.zeros((rows, PROJECTION_DIM), dtype=np.float32)
    direction = normalize_rows(rng.normal(size=(rows, PROJECTION_DIM)).astype(np.float32))
    radii = radius * np.power(rng.random(rows), 1.0 / PROJECTION_DIM)
    return (direction * radii[:, None]).astype(np.float32)


def ccadpe_transform(
    values: np.ndarray,
    *,
    key: bytes,
    cell: int,
    work_dim: int,
    query: bool,
    nonce: int,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    padded = np.zeros((len(values), work_dim), dtype=np.float32)
    padded[:, : values.shape[1]] = values
    flatten_sign, coordinates, output_sign, translation = cell_parameters(key, cell, work_dim)
    flattened = fwht(padded * flatten_sign[None, :])
    selected = (
        flattened[:, coordinates]
        * output_sign[None, :]
        * math.sqrt(work_dim / PROJECTION_DIM)
    )
    radius = SCALE * BETA / 8.0 if query else 3.0 * SCALE * BETA / 8.0
    rng = np.random.default_rng(cell_seed(key, cell, b"query" if query else b"database") ^ int(nonce))
    return (
        SCALE * selected
        + translation[None, :]
        + ball_noise(rng, len(values), radius)
    ).astype(np.float32)


def load_inputs() -> tuple[
    np.ndarray,
    np.ndarray,
    list[dict[str, int]],
    list[dict[str, int]],
    list[str],
    list[str],
]:
    documents = []
    document_ids = []
    with (DATA / "sim_corpus.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            documents.append(row["context_embeddings_norm"])
            document_ids.append(str(row["context_id"]))
    queries = []
    query_ids = []
    with (DATA / "sim_query.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            queries.append(row["query_embeddings_norm"])
            query_ids.append(str(row["id"]))
    doc_counts = []
    with (DATA / "bm25_corpus.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            doc_counts.append({str(k): int(v) for k, v in row["bert_token_counts"].items()})
    query_counts = []
    with (DATA / "bm25_query.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            query_counts.append({str(k): int(v) for k, v in row["query_bert_token_counts"].items()})
    if not (len(documents) == len(doc_counts) == N_DOCS):
        raise RuntimeError("FiQA corpus size does not match the Pisces workload")
    if not (len(queries) == len(query_counts) == N_QUERIES):
        raise RuntimeError("FiQA query count does not match the Pisces workload")
    return (
        normalize_rows(np.asarray(documents, dtype=np.float32)),
        normalize_rows(np.asarray(queries, dtype=np.float32)),
        doc_counts,
        query_counts,
        document_ids,
        query_ids,
    )


def load_qrels() -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = defaultdict(set)
    if not QRELS:
        return qrels
    with Path(QRELS).open(encoding="utf-8") as handle:
        next(handle, None)
        for line in handle:
            query_id, document_id, score = line.rstrip("\n").split("\t")
            if float(score) > 0:
                qrels[str(query_id)].add(str(document_id))
    return qrels


def build_semantic(documents: np.ndarray) -> dict[str, object]:
    started = time.perf_counter()
    mean = documents.mean(axis=0).astype(np.float32)
    residuals = normalize_rows(documents - mean[None, :])
    model = MiniBatchKMeans(
        n_clusters=SEMANTIC_CELLS,
        init="k-means++",
        n_init=3,
        max_iter=100,
        batch_size=6_144,
        random_state=SEED,
        reassignment_ratio=0.01,
    )
    model.fit(residuals)
    centroids = normalize_rows(model.cluster_centers_)
    assignments = np.empty((len(documents), SEMANTIC_ASSIGNMENTS), dtype=np.int32)
    for first in range(0, len(documents), 2_048):
        last = min(first + 2_048, len(documents))
        similarity = residuals[first:last] @ centroids.T
        assignments[first:last] = np.argpartition(
            -similarity, SEMANTIC_ASSIGNMENTS - 1, axis=1
        )[:, :SEMANTIC_ASSIGNMENTS]
    postings: dict[int, np.ndarray] = {}
    ciphertexts: dict[int, np.ndarray] = {}
    for cell in range(SEMANTIC_CELLS):
        doc_ids = np.flatnonzero(np.any(assignments == cell, axis=1)).astype(np.int32)
        if not len(doc_ids):
            continue
        postings[cell] = doc_ids
        ciphertexts[cell] = ccadpe_transform(
            documents[doc_ids], key=KEY_SEMANTIC, cell=cell,
            work_dim=SEMANTIC_WORK_DIM, query=False, nonce=0,
        ).astype(np.float16)
    return {
        "mean": mean,
        "centroids": centroids,
        "postings": postings,
        "ciphertexts": ciphertexts,
        "build_seconds": time.perf_counter() - started,
        "posting_entries": int(sum(len(x) for x in postings.values())),
    }


def feature(term: str) -> tuple[int, float]:
    digest = hashlib.blake2b(term.encode("utf-8"), key=KEY_FEATURE, digest_size=16).digest()
    return int.from_bytes(digest[:8], "little") % LEXICAL_HASH_DIM, (1.0 if digest[8] & 1 else -1.0)


def lexical_cell(doc_id: int) -> int:
    digest = hashlib.blake2b(
        int(doc_id).to_bytes(8, "big"), key=KEY_LEXICAL, digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") % LEXICAL_CELLS


def build_lexical(doc_counts: list[dict[str, int]]) -> dict[str, object]:
    started = time.perf_counter()
    doc_lengths = np.asarray([sum(row.values()) for row in doc_counts], dtype=np.float32)
    average_length = float(np.mean(doc_lengths))
    temporary: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for doc_id, counts in enumerate(doc_counts):
        for term, frequency in counts.items():
            temporary[term].append((doc_id, frequency))
    postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    idf: dict[str, float] = {}
    for term, pairs in temporary.items():
        ids = np.fromiter((pair[0] for pair in pairs), dtype=np.int32)
        tf = np.fromiter((pair[1] for pair in pairs), dtype=np.float32)
        postings[term] = (ids, tf)
        frequency = len(ids)
        idf[term] = math.log(1.0 + (N_DOCS - frequency + 0.5) / (frequency + 0.5))
    vectors = np.zeros((N_DOCS, LEXICAL_HASH_DIM), dtype=np.float32)
    k1, b = 1.2, 0.75
    for doc_id, counts in enumerate(doc_counts):
        normalizer = k1 * (1.0 - b + b * doc_lengths[doc_id] / average_length)
        for term, frequency in counts.items():
            coordinate, sign = feature(term)
            value = idf[term] * frequency * (k1 + 1.0) / (frequency + normalizer)
            vectors[doc_id, coordinate] += sign * value
    maximum_norm = float(np.max(np.linalg.norm(vectors, axis=1))) * (1.0 + 1e-6)
    vectors /= max(maximum_norm, 1e-12)
    tail = np.sqrt(np.maximum(0.0, 1.0 - np.sum(vectors * vectors, axis=1)))
    cells = np.fromiter((lexical_cell(i) for i in range(N_DOCS)), dtype=np.int32)
    ciphertext = np.empty((N_DOCS, PROJECTION_DIM), dtype=np.float16)
    for cell in range(LEXICAL_CELLS):
        ids = np.flatnonzero(cells == cell)
        base = np.zeros((len(ids), LEXICAL_HASH_DIM + 1), dtype=np.float32)
        base[:, :LEXICAL_HASH_DIM] = vectors[ids]
        base[:, LEXICAL_HASH_DIM] = tail[ids]
        ciphertext[ids] = ccadpe_transform(
            base, key=KEY_LEXICAL, cell=cell,
            work_dim=LEXICAL_WORK_DIM, query=False, nonce=0,
        ).astype(np.float16)
    return {
        "doc_lengths": doc_lengths,
        "average_length": average_length,
        "postings": postings,
        "idf": idf,
        "vectors": vectors,
        "tail": tail,
        "cells": cells,
        "ciphertext": ciphertext,
        "maximum_norm": maximum_norm,
        "build_seconds": time.perf_counter() - started,
    }


def semantic_query(
    query: np.ndarray,
    index: dict[str, object],
    documents: np.ndarray,
    *,
    nonce: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    server_started = time.perf_counter()
    residual = normalize_rows(query - index["mean"])[0]
    cells = top_indices(index["centroids"] @ residual, SEMANTIC_PROBES)
    emitted = []
    posting_entries = 0
    for ordinal, raw_cell in enumerate(cells):
        cell = int(raw_cell)
        ids = index["postings"].get(cell)
        if ids is None:
            continue
        posting_entries += len(ids)
        encrypted_query = ccadpe_transform(
            query, key=KEY_SEMANTIC, cell=cell,
            work_dim=SEMANTIC_WORK_DIM, query=True,
            nonce=nonce * 1_009 + ordinal,
        )[0]
        delta = index["ciphertexts"][cell].astype(np.float32) - encrypted_query[None, :]
        distance = np.einsum("ij,ij->i", delta, delta)
        emitted.append(ids[top_indices(-distance, min(LOCAL_DEPTH, len(ids)))])
    candidates = np.unique(np.concatenate(emitted)) if emitted else np.empty(0, dtype=np.int64)
    server_seconds = time.perf_counter() - server_started
    client_started = time.perf_counter()
    ranking = candidates[top_indices(documents[candidates] @ query, min(K, len(candidates)))]
    client_seconds = time.perf_counter() - client_started
    return ranking, {
        "server_seconds": server_seconds,
        "client_seconds": client_seconds,
        "posting_entries": posting_entries,
        "candidates": len(candidates),
    }


def lexical_plain_scores(
    query_counts: dict[str, int], index: dict[str, object]
) -> np.ndarray:
    scores = np.zeros(N_DOCS, dtype=np.float32)
    k1, b = 1.2, 0.75
    for term in query_counts:
        posting = index["postings"].get(term)
        if posting is None:
            continue
        ids, tf = posting
        normalizer = k1 * (
            1.0 - b + b * index["doc_lengths"][ids] / index["average_length"]
        )
        scores[ids] += (
            index["idf"][term]
            * tf
            * (k1 + 1.0)
            / (tf + normalizer)
        ).astype(np.float32)
    return scores


def lexical_query(
    query_counts: dict[str, int],
    index: dict[str, object],
    *,
    nonce: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    server_started = time.perf_counter()
    available = [term for term in query_counts if term in index["postings"]]
    available.sort(key=lambda term: len(index["postings"][term][0]))
    selected_terms = []
    consumed = 0
    for term in available:
        posting_size = len(index["postings"][term][0])
        if selected_terms and consumed + posting_size > POSTING_BUDGET:
            continue
        selected_terms.append(term)
        consumed += posting_size
    proxy = np.zeros(N_DOCS, dtype=np.float32)
    touched = np.zeros(N_DOCS, dtype=np.bool_)
    k1, b = 1.2, 0.75
    for term in selected_terms:
        ids, tf = index["postings"][term]
        normalizer = k1 * (
            1.0 - b + b * index["doc_lengths"][ids] / index["average_length"]
        )
        # This is the floating-point counterpart of the 16-bit quantized
        # keyed BM25 posting weight used by the deployed candidate planner.
        proxy[ids] += (
            float(query_counts[term])
            * index["idf"][term]
            * tf
            * (k1 + 1.0)
            / (tf + normalizer)
        ).astype(np.float32)
        touched[ids] = True
    candidates = np.flatnonzero(touched)
    if len(candidates) > 1_000:
        candidates = candidates[top_indices(proxy[candidates], 1_000)]
    query_vector = np.zeros(LEXICAL_HASH_DIM, dtype=np.float32)
    for term, frequency in query_counts.items():
        if term not in index["idf"]:
            continue
        coordinate, sign = feature(term)
        query_vector[coordinate] += sign * float(frequency)
    norm = float(np.linalg.norm(query_vector))
    if norm > 0:
        query_vector /= norm
    base_query = np.concatenate([query_vector, np.zeros(1, dtype=np.float32)])
    emitted = []
    active_cells = np.unique(index["cells"][candidates])
    for ordinal, raw_cell in enumerate(active_cells):
        cell = int(raw_cell)
        ids = candidates[index["cells"][candidates] == cell]
        encrypted_query = ccadpe_transform(
            base_query, key=KEY_LEXICAL, cell=cell,
            work_dim=LEXICAL_WORK_DIM, query=True,
            nonce=nonce * 1_009 + ordinal,
        )[0]
        delta = index["ciphertext"][ids].astype(np.float32) - encrypted_query[None, :]
        distance = np.einsum("ij,ij->i", delta, delta)
        emitted.append(ids[top_indices(-distance, min(LOCAL_DEPTH, len(ids)))])
    rerank_candidates = np.unique(np.concatenate(emitted)) if emitted else np.empty(0, dtype=np.int64)
    server_seconds = time.perf_counter() - server_started
    client_started = time.perf_counter()
    exact_scores = lexical_plain_scores(query_counts, index)
    ranking = rerank_candidates[
        top_indices(exact_scores[rerank_candidates], min(K, len(rerank_candidates)))
    ]
    client_seconds = time.perf_counter() - client_started
    return ranking, {
        "server_seconds": server_seconds,
        "client_seconds": client_seconds,
        "posting_entries": consumed,
        "candidates": len(candidates),
        "rerank_candidates": len(rerank_candidates),
        "exact_payload_bucket_mean": float(
            np.mean(
                np.ceil(
                    (4.0 + 4.0 * index["doc_lengths"][rerank_candidates]) / 256.0
                )
                * 256.0
            )
        ) if len(rerank_candidates) else 256.0,
        "active_cells": len(active_cells),
    }


def percentile(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
    }


def communication(mean_exact_payload_bucket: float) -> dict[str, float | int]:
    label_bytes = 16
    local_query_bytes = PROJECTION_DIM * 2
    semantic_records = SEMANTIC_PROBES * LOCAL_DEPTH
    lexical_records = LEXICAL_CELLS * LOCAL_DEPTH
    semantic_record_bytes = 16 + 12 + (384 * 2 + 16) + 12 + (20 + 16)
    lexical_record_bytes = 92.0 + mean_exact_payload_bucket
    upload = (
        SEMANTIC_PROBES * (label_bytes + local_query_bytes)
        + LEXICAL_CELLS * (label_bytes + local_query_bytes)
        + 32
    )
    download = 16 + semantic_records * semantic_record_bytes + lexical_records * lexical_record_bytes
    total = upload + download
    return {
        "upload_bytes": upload,
        "download_bytes": download,
        "total_bytes": total,
        "total_mib": total / (1024**2),
        "mean_exact_lexical_payload_bucket_bytes": mean_exact_payload_bucket,
        "semantic_padded_records": semantic_records,
        "lexical_padded_records": lexical_records,
    }


def main() -> None:
    np.random.seed(SEED)
    load_started = time.perf_counter()
    documents, queries, doc_counts, query_counts, document_ids, query_ids = load_inputs()
    qrels = load_qrels()
    load_seconds = time.perf_counter() - load_started
    semantic_index = build_semantic(documents)
    lexical_index = build_lexical(doc_counts)
    exact_semantic = [top_indices(documents @ query, K) for query in queries]
    exact_lexical = [top_indices(lexical_plain_scores(query, lexical_index), K) for query in query_counts]

    # Untimed warm-up.
    semantic_query(queries[0], semantic_index, documents, nonce=900_000)
    lexical_query(query_counts[0], lexical_index, nonce=900_000)

    observations = []
    semantic_overlaps = []
    lexical_overlaps = []
    for repeat in range(REPEATS):
        for query_id in range(N_QUERIES):
            semantic_rank, semantic_trace = semantic_query(
                queries[query_id], semantic_index, documents,
                nonce=repeat * N_QUERIES + query_id + 1,
            )
            lexical_rank, lexical_trace = lexical_query(
                query_counts[query_id], lexical_index,
                nonce=repeat * N_QUERIES + query_id + 1,
            )
            fusion_started = time.perf_counter()
            dual_union = np.unique(np.concatenate([semantic_rank, lexical_rank]))
            fusion_seconds = time.perf_counter() - fusion_started
            semantic_overlap = len(set(map(int, semantic_rank)) & set(map(int, exact_semantic[query_id]))) / K
            lexical_overlap = len(set(map(int, lexical_rank)) & set(map(int, exact_lexical[query_id]))) / K
            semantic_overlaps.append(semantic_overlap)
            lexical_overlaps.append(lexical_overlap)
            semantic_document_ids = [document_ids[int(index)] for index in semantic_rank]
            lexical_document_ids = [document_ids[int(index)] for index in lexical_rank]
            union_document_ids = sorted(set(semantic_document_ids) | set(lexical_document_ids))
            relevant = qrels.get(query_ids[query_id], set())
            observations.append(
                {
                    "repeat": repeat + 1,
                    "query": query_id + 1,
                    "semantic_server_seconds": semantic_trace["server_seconds"],
                    "semantic_client_seconds": semantic_trace["client_seconds"],
                    "lexical_server_seconds": lexical_trace["server_seconds"],
                    "lexical_client_seconds": lexical_trace["client_seconds"],
                    "fusion_seconds": fusion_seconds,
                    "online_seconds": semantic_trace["server_seconds"]
                    + semantic_trace["client_seconds"]
                    + lexical_trace["server_seconds"]
                    + lexical_trace["client_seconds"]
                    + fusion_seconds,
                    "semantic_candidates": semantic_trace["candidates"],
                    "semantic_posting_entries": semantic_trace["posting_entries"],
                    "lexical_candidates": lexical_trace["candidates"],
                    "lexical_rerank_candidates": lexical_trace["rerank_candidates"],
                    "lexical_posting_entries": lexical_trace["posting_entries"],
                    "lexical_active_cells": lexical_trace["active_cells"],
                    "lexical_payload_bucket_mean": lexical_trace["exact_payload_bucket_mean"],
                    "dual_output_documents": len(dual_union),
                    "semantic_top10_overlap": semantic_overlap,
                    "lexical_top10_overlap": lexical_overlap,
                    "query_id": query_ids[query_id],
                    "semantic_top10": semantic_document_ids,
                    "lexical_top10": lexical_document_ids,
                    "dual_union": union_document_ids,
                    "relevant_documents": sorted(relevant),
                    "semantic_recall": len(relevant & set(semantic_document_ids)) / max(len(relevant), 1),
                    "lexical_recall": len(relevant & set(lexical_document_ids)) / max(len(relevant), 1),
                    "union_recall": len(relevant & set(union_document_ids)) / max(len(relevant), 1),
                }
            )
            print(
                f"repeat={repeat + 1} query={query_id + 1} "
                f"online={observations[-1]['online_seconds']:.6f}s "
                f"S={semantic_overlap:.3f} L={lexical_overlap:.3f}",
                flush=True,
            )

    def field(name: str) -> list[float]:
        return [float(row[name]) for row in observations]

    report = {
        "experiment": "DuetER same-host controlled FiQA execution",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn": __import__("sklearn").__version__,
            "peak_rss_mib": (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
                if resource is not None else None
            ),
        },
        "workload": {
            "dataset": "BEIR FiQA",
            "documents": N_DOCS,
            "queries": N_QUERIES,
            "repeats": REPEATS,
            "observations": len(observations),
            "k_per_path": K,
            "same_preprocessed_inputs_as_official_pisces": True,
        },
        "configuration": {
            "semantic_cells": SEMANTIC_CELLS,
            "semantic_probes": SEMANTIC_PROBES,
            "semantic_assignments": SEMANTIC_ASSIGNMENTS,
            "lexical_cells": LEXICAL_CELLS,
            "local_depth": LOCAL_DEPTH,
            "projection_dimension": PROJECTION_DIM,
            "posting_budget": POSTING_BUDGET,
            "bm25_query_weight": "binary unique-term",
            "beta": BETA,
            "scale": SCALE,
        },
        "offline": {
            "load_seconds": load_seconds,
            "semantic_build_seconds": semantic_index["build_seconds"],
            "semantic_posting_entries": semantic_index["posting_entries"],
            "lexical_build_seconds": lexical_index["build_seconds"],
        },
        "online": {
            "semantic_server": percentile(field("semantic_server_seconds")),
            "semantic_client": percentile(field("semantic_client_seconds")),
            "lexical_server": percentile(field("lexical_server_seconds")),
            "lexical_client": percentile(field("lexical_client_seconds")),
            "fusion": percentile(field("fusion_seconds")),
            "total": percentile(field("online_seconds")),
        },
        "communication": communication(float(np.mean(field("lexical_payload_bucket_mean")))),
        "retrieval_fidelity": {
            "semantic_top10_overlap_mean": float(np.mean(semantic_overlaps)),
            "lexical_top10_overlap_mean": float(np.mean(lexical_overlaps)),
            "semantic_recall_mean": float(np.mean(field("semantic_recall"))),
            "lexical_recall_mean": float(np.mean(field("lexical_recall"))),
            "union_recall_mean": float(np.mean(field("union_recall"))),
        },
        "candidate_means": {
            "semantic": float(np.mean(field("semantic_candidates"))),
            "semantic_posting_entries": float(np.mean(field("semantic_posting_entries"))),
            "lexical": float(np.mean(field("lexical_candidates"))),
            "lexical_rerank": float(np.mean(field("lexical_rerank_candidates"))),
            "lexical_posting_entries": float(np.mean(field("lexical_posting_entries"))),
            "dual_output_documents": float(np.mean(field("dual_output_documents"))),
        },
        "observations": observations,
        "scope": "Online time excludes neural encoding and offline index construction; both systems consume the same preprocessed embeddings and BERT token counts.",
    }
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"online": report["online"], "communication": report["communication"], "retrieval_fidelity": report["retrieval_fidelity"]}, indent=2))


if __name__ == "__main__":
    main()
