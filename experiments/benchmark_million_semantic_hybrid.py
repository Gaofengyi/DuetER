"""Million-document Granite semantic and dual-path retrieval benchmark.

The benchmark uses the same relevance-preserving subset policy as
``benchmark_keyed_fts5_scale.py``: every positive test-qrel document plus the
earliest non-qrel documents required to reach ``--max-docs``.  It is designed
to be resumable because encoding one million documents is the dominant cost.

Semantic path
-------------
* stream Granite embeddings to a NumPy memmap;
* train a mean-centered spherical IVF codebook on a deterministic sample;
* assign documents to multiple cells in GPU-sized blocks;
* expose only HMAC cell labels and CSR posting lists to the candidate lookup;
* refine candidates using materialized conditional-DPE ciphertexts; and
* rerank the returned top-k exactly on the client.

Lexical path
------------
The script reuses the existing contentless HMAC-FTS5 million-document index.
Two rare keyed query terms generate at most 200 candidates, matching the
audited lexical scale experiment.  FTS5's exact BM25 order is retained as the
lexical path order; this is explicitly distinct from the in-memory 16-bit
lexical-DPE prototype and is not mislabeled as cryptographic DPE execution.

The two path rankings are fused locally with reciprocal-rank fusion.  NumPy
PCG64 supplies reproducible bounded DPE noise, as in the rest of this research
prototype; it is not a production PRF.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import hmac
import json
import math
import os
import platform
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from run_experiment import fwht_batch, normalize_rows, tokenize


ROOT = Path(__file__).resolve().parent
SEMANTIC_KEY = b"DuetDPE-semantic-residual-ivf-v1"
LEXICAL_KEY = b"DuetDPE-keyed-vocabulary-tail-v2"
FULL_CORPUS_DOCS = {
    "msmarco": 8_841_823,
    "nq": 2_681_468,
    "hotpotqa": 5_233_329,
}


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def load_queries_qrels(
    data_dir: Path,
    split: str = "test",
) -> tuple[list[str], list[str], dict[str, dict[str, int]], set[str]]:
    query_lookup: dict[str, str] = {}
    with (data_dir / "queries.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            query_lookup[str(row["_id"])] = str(row["text"])
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with (data_dir / "qrels" / f"{split}.tsv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            qrels[str(row["query-id"])][str(row["corpus-id"])] = int(row["score"])
    query_ids = [qid for qid in query_lookup if qid in qrels]
    query_texts = [query_lookup[qid] for qid in query_ids]
    relevant_ids = {
        doc_id
        for truth in qrels.values()
        for doc_id, gain in truth.items()
        if gain > 0
    }
    return query_ids, query_texts, dict(qrels), relevant_ids


def selected_records(
    corpus_path: Path, relevant_ids: set[str], max_docs: int
) -> Iterator[tuple[str, str]]:
    if len(relevant_ids) > max_docs:
        raise ValueError("max_docs is smaller than the number of relevant documents")
    filler_budget = max_docs - len(relevant_ids)
    selected_fillers = 0
    with corpus_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            doc_id = str(row["_id"])
            is_relevant = doc_id in relevant_ids
            if not is_relevant and selected_fillers >= filler_budget:
                continue
            if not is_relevant:
                selected_fillers += 1
            title = str(row.get("title", ""))
            text = f"{title}. {row.get('text', '')}".strip()
            yield doc_id, text


def read_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def encode_subset(
    *,
    data_dir: Path,
    cache_dir: Path,
    relevant_ids: set[str],
    query_texts: list[str],
    model_path: Path,
    max_docs: int,
    batch_size: int,
    encoder_chunk: int,
    device: str,
) -> dict[str, object]:
    """Stream and resume corpus encoding without retaining corpus text."""

    from sentence_transformers import SentenceTransformer

    cache_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = cache_dir / "corpus_embeddings.npy"
    queries_path = cache_dir / "query_embeddings.npy"
    ids_path = cache_dir / "doc_ids.txt"
    progress_path = cache_dir / "encoding_progress.json"
    manifest_path = cache_dir / "encoding_manifest.json"

    if all(
        path.exists()
        for path in (embeddings_path, queries_path, ids_path, progress_path, manifest_path)
    ):
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if progress.get("complete") and int(progress["completed_documents"]) == max_docs:
            return manifest

    completed = 0
    elapsed_before = 0.0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        completed = int(progress["completed_documents"])
        elapsed_before = float(progress.get("elapsed_seconds", 0.0))
    if ids_path.exists():
        id_lines = sum(1 for _ in ids_path.open("r", encoding="utf-8"))
        if id_lines != completed:
            raise RuntimeError(
                f"resume mismatch: progress={completed}, doc_ids lines={id_lines}"
            )
    elif completed:
        raise RuntimeError("encoding progress exists but doc_ids.txt is missing")

    model = SentenceTransformer(
        str(model_path), cache_folder=str(ROOT / "cache" / "models"), device=device
    )
    dimension = int(model.get_embedding_dimension())
    if embeddings_path.exists():
        embeddings = np.lib.format.open_memmap(embeddings_path, mode="r+")
        if embeddings.shape != (max_docs, dimension):
            raise RuntimeError(f"unexpected embedding shape {embeddings.shape}")
    else:
        embeddings = np.lib.format.open_memmap(
            embeddings_path,
            mode="w+",
            dtype=np.float32,
            shape=(max_docs, dimension),
        )

    started = time.perf_counter()
    buffer_ids: list[str] = []
    buffer_texts: list[str] = []
    selected_count = 0
    ids_mode = "a" if completed else "w"
    with ids_path.open(ids_mode, encoding="utf-8", newline="\n") as id_handle:
        for doc_id, text in selected_records(
            data_dir / "corpus.jsonl", relevant_ids, max_docs
        ):
            if selected_count < completed:
                selected_count += 1
                continue
            buffer_ids.append(doc_id)
            buffer_texts.append(text)
            selected_count += 1
            if len(buffer_texts) < encoder_chunk:
                continue
            encoded = model.encode(
                buffer_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)
            first = selected_count - len(buffer_texts)
            embeddings[first:selected_count] = encoded
            embeddings.flush()
            id_handle.writelines(f"{value}\n" for value in buffer_ids)
            id_handle.flush()
            elapsed = elapsed_before + time.perf_counter() - started
            atomic_json(
                progress_path,
                {
                    "completed_documents": selected_count,
                    "elapsed_seconds": elapsed,
                    "dimension": dimension,
                },
            )
            print(
                f"[encode] {selected_count:,}/{max_docs:,} "
                f"({selected_count / max(elapsed, 1e-9):.1f} docs/s)",
                flush=True,
            )
            buffer_ids.clear()
            buffer_texts.clear()
        if buffer_texts:
            encoded = model.encode(
                buffer_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)
            first = selected_count - len(buffer_texts)
            embeddings[first:selected_count] = encoded
            embeddings.flush()
            id_handle.writelines(f"{value}\n" for value in buffer_ids)
            id_handle.flush()

    if selected_count != max_docs:
        raise RuntimeError(f"selected {selected_count} documents, expected {max_docs}")
    corpus_seconds = elapsed_before + time.perf_counter() - started
    atomic_json(
        progress_path,
        {
            "completed_documents": selected_count,
            "elapsed_seconds": corpus_seconds,
            "dimension": dimension,
            "complete": True,
        },
    )
    if not queries_path.exists():
        q_started = time.perf_counter()
        query_embeddings = model.encode(
            query_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        np.save(queries_path, query_embeddings)
        query_seconds = time.perf_counter() - q_started
    else:
        query_embeddings = np.load(queries_path, mmap_mode="r")
        query_seconds = 0.0
    del model
    gc.collect()

    doc_ids_hash = hashlib.sha256(ids_path.read_bytes()).hexdigest()
    manifest = {
        "documents": max_docs,
        "queries": len(query_texts),
        "dimension": dimension,
        "model": str(model_path),
        "device": device,
        "batch_size": batch_size,
        "encoder_chunk": encoder_chunk,
        "corpus_encoding_seconds": corpus_seconds,
        "query_encoding_seconds_last_run": query_seconds,
        "subset_policy": (
            "complete corpus in released order"
            if FULL_CORPUS_DOCS.get(data_dir.name) == max_docs
            else "all positive test-qrel documents plus earliest non-qrel corpus records"
        ),
        "doc_ids_sha256": doc_ids_hash,
        "embedding_dtype": "float32",
        "embedding_bytes": embeddings_path.stat().st_size,
    }
    atomic_json(manifest_path, manifest)
    return manifest


def _normalize_torch(values):
    import torch

    return values / torch.linalg.vector_norm(values, dim=1, keepdim=True).clamp_min(1e-12)


def build_semantic_index(
    *,
    cache_dir: Path,
    clusters: int,
    assignments: int,
    train_sample: int,
    epochs: int,
    block_size: int,
    seed: int,
    device: str,
) -> dict[str, object]:
    import torch

    metadata_path = cache_dir / "semantic_index.json"
    required = [
        cache_dir / "semantic_mean.npy",
        cache_dir / "semantic_centroids.npy",
        cache_dir / "semantic_offsets.npy",
        cache_dir / "semantic_postings.npy",
        cache_dir / "semantic_labels.npy",
    ]
    if metadata_path.exists() and all(path.exists() for path in required):
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    embeddings = np.load(cache_dir / "corpus_embeddings.npy", mmap_mode="r")
    documents, dimension = embeddings.shape
    if clusters > documents:
        raise ValueError("clusters exceeds document count")
    started = time.perf_counter()
    total = np.zeros(dimension, dtype=np.float64)
    for first in range(0, documents, block_size):
        total += np.asarray(embeddings[first : first + block_size], dtype=np.float32).sum(
            axis=0, dtype=np.float64
        )
    mean = (total / documents).astype(np.float32)
    np.save(cache_dir / "semantic_mean.npy", mean)

    rng = np.random.default_rng(seed)
    sample_count = min(train_sample, documents)
    sample_indices = np.sort(rng.choice(documents, size=sample_count, replace=False))
    sample = np.asarray(embeddings[sample_indices], dtype=np.float32)
    sample = normalize_rows(sample - mean[None, :])
    sample_gpu = torch.from_numpy(sample).to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    initial = torch.randperm(sample_count, generator=generator, device=device)[:clusters]
    centroids = sample_gpu[initial].clone()
    centroids = _normalize_torch(centroids)
    torch.set_float32_matmul_precision("high")
    train_batch = min(block_size, 8192)
    for epoch in range(epochs):
        sums = torch.zeros((clusters, dimension), dtype=torch.float32, device=device)
        counts = torch.zeros(clusters, dtype=torch.float32, device=device)
        order = torch.randperm(sample_count, generator=generator, device=device)
        for first in range(0, sample_count, train_batch):
            rows = sample_gpu[order[first : first + train_batch]]
            labels = torch.argmax(rows @ centroids.T, dim=1)
            sums.index_add_(0, labels, rows)
            counts += torch.bincount(labels, minlength=clusters).to(torch.float32)
        nonempty = counts > 0
        updated = sums[nonempty] / counts[nonempty, None]
        centroids[nonempty] = _normalize_torch(updated)
        if bool((~nonempty).any()):
            replacements = torch.randint(
                sample_count,
                (int((~nonempty).sum().item()),),
                generator=generator,
                device=device,
            )
            centroids[~nonempty] = sample_gpu[replacements]
        print(
            f"[ivf-train] epoch {epoch + 1}/{epochs}, "
            f"nonempty={int(nonempty.sum().item())}/{clusters}",
            flush=True,
        )
    centroids_np = centroids.cpu().numpy().astype(np.float32)
    np.save(cache_dir / "semantic_centroids.npy", centroids_np)
    del sample_gpu, sample, centroids
    torch.cuda.empty_cache() if device.startswith("cuda") else None

    assignment_path = cache_dir / "semantic_assignments.npy"
    assignment_map = np.lib.format.open_memmap(
        assignment_path,
        mode="w+",
        dtype=np.uint16 if clusters <= np.iinfo(np.uint16).max else np.uint32,
        shape=(documents, assignments),
    )
    centroid_gpu = torch.from_numpy(centroids_np).to(device)
    mean_gpu = torch.from_numpy(mean).to(device)
    for first in range(0, documents, block_size):
        last = min(first + block_size, documents)
        rows = torch.from_numpy(
            np.asarray(embeddings[first:last], dtype=np.float32).copy()
        ).to(device)
        rows = _normalize_torch(rows - mean_gpu)
        cells = torch.topk(rows @ centroid_gpu.T, k=assignments, dim=1).indices
        assignment_map[first:last] = cells.cpu().numpy().astype(assignment_map.dtype)
        if first == 0 or last == documents or (last // block_size) % 20 == 0:
            print(f"[ivf-assign] {last:,}/{documents:,}", flush=True)
    assignment_map.flush()

    flat_cells = np.asarray(assignment_map).reshape(-1)
    flat_docs = np.repeat(np.arange(documents, dtype=np.int32), assignments)
    order = np.argsort(flat_cells, kind="stable")
    postings = flat_docs[order]
    counts = np.bincount(flat_cells.astype(np.int64), minlength=clusters)
    offsets = np.empty(clusters + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    labels = np.stack(
        [
            np.frombuffer(
                hmac.new(
                    SEMANTIC_KEY,
                    b"semantic-residual-ivf-v1" + cell.to_bytes(4, "big"),
                    hashlib.sha256,
                ).digest()[:16],
                dtype=np.uint8,
            )
            for cell in range(clusters)
        ]
    )
    np.save(cache_dir / "semantic_postings.npy", postings)
    np.save(cache_dir / "semantic_offsets.npy", offsets)
    np.save(cache_dir / "semantic_labels.npy", labels)
    setup = {
        "variant": "mean-centered multi-assignment spherical IVF with HMAC cell labels",
        "documents": documents,
        "dimension": dimension,
        "clusters": clusters,
        "assignments": assignments,
        "train_sample": sample_count,
        "training_epochs": epochs,
        "seed": seed,
        "posting_entries": int(len(postings)),
        "mean_posting_size": float(counts.mean()),
        "largest_posting_size": int(counts.max()),
        "nonempty_cells": int(np.count_nonzero(counts)),
        "server_estimated_bytes": int(postings.nbytes + offsets.nbytes + labels.nbytes),
        "client_centroid_mean_bytes": int(centroids_np.nbytes + mean.nbytes),
        "build_seconds": time.perf_counter() - started,
    }
    atomic_json(metadata_path, setup)
    return setup


def dpe_transform(
    vectors: np.ndarray,
    sign1: np.ndarray,
    sign2: np.ndarray,
    permutation: np.ndarray,
) -> np.ndarray:
    rows = len(vectors)
    padded = np.zeros((rows, len(sign1)), dtype=np.float32)
    padded[:, : vectors.shape[1]] = vectors
    padded *= sign1
    padded = fwht_batch(padded)
    padded *= sign2
    return padded[:, permutation]


def ball_noise(
    rng: np.random.Generator, rows: int, dimension: int, radius: float
) -> np.ndarray:
    directions = rng.normal(size=(rows, dimension)).astype(np.float32)
    directions = normalize_rows(directions)
    radii = radius * np.power(rng.random(rows), 1.0 / dimension)
    return directions * radii[:, None].astype(np.float32)


def build_dpe_ciphertexts(
    *, cache_dir: Path, beta: float, scale: float, block_size: int, seed: int
) -> dict[str, object]:
    metadata_path = cache_dir / "semantic_dpe.json"
    cipher_path = cache_dir / "semantic_cipher_fp16.npy"
    norms_path = cache_dir / "semantic_cipher_norms.npy"
    key_path = cache_dir / "semantic_dpe_key.npz"
    if all(path.exists() for path in (metadata_path, cipher_path, norms_path, key_path)):
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    embeddings = np.load(cache_dir / "corpus_embeddings.npy", mmap_mode="r")
    documents, input_dimension = embeddings.shape
    work_dimension = 1 << (input_dimension - 1).bit_length()
    rng = np.random.default_rng(seed)
    sign1 = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), work_dimension)
    sign2 = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), work_dimension)
    permutation = rng.permutation(work_dimension)
    db_rng = np.random.default_rng(seed + 1009)
    np.savez(key_path, sign1=sign1, sign2=sign2, permutation=permutation)
    cipher = np.lib.format.open_memmap(
        cipher_path,
        mode="w+",
        dtype=np.float16,
        shape=(documents, work_dimension),
    )
    norms = np.lib.format.open_memmap(
        norms_path, mode="w+", dtype=np.float32, shape=(documents,)
    )
    started = time.perf_counter()
    radius = 3.0 * scale * beta / 8.0
    for first in range(0, documents, block_size):
        last = min(first + block_size, documents)
        values = np.asarray(embeddings[first:last], dtype=np.float32)
        encrypted = scale * dpe_transform(values, sign1, sign2, permutation)
        encrypted += ball_noise(db_rng, len(values), work_dimension, radius)
        stored = encrypted.astype(np.float16)
        cipher[first:last] = stored
        norms[first:last] = np.sum(stored.astype(np.float32) ** 2, axis=1)
        if first == 0 or last == documents or (last // block_size) % 20 == 0:
            cipher.flush()
            norms.flush()
            print(f"[dpe-database] {last:,}/{documents:,}", flush=True)
    cipher.flush()
    norms.flush()
    metadata = {
        "input_dimension": input_dimension,
        "work_dimension": work_dimension,
        "beta": beta,
        "scale": scale,
        "seed": seed,
        "database_noise_radius": radius,
        "ciphertext_storage_dtype": "float16",
        "ciphertext_bytes": cipher_path.stat().st_size,
        "norm_bytes": norms_path.stat().st_size,
        "build_seconds": time.perf_counter() - started,
        "quantization_note": "server ciphertext coordinates are stored as float16; norms are computed after float16 quantization",
    }
    atomic_json(metadata_path, metadata)
    return metadata


@dataclass
class IvfFiles:
    mean: np.ndarray
    centroids: np.ndarray
    offsets: np.ndarray
    postings: np.ndarray
    labels: np.ndarray
    token_to_cell: dict[bytes, int]

    @classmethod
    def load(cls, cache_dir: Path) -> "IvfFiles":
        labels = np.load(cache_dir / "semantic_labels.npy")
        return cls(
            mean=np.load(cache_dir / "semantic_mean.npy"),
            centroids=np.load(cache_dir / "semantic_centroids.npy"),
            offsets=np.load(cache_dir / "semantic_offsets.npy", mmap_mode="r"),
            postings=np.load(cache_dir / "semantic_postings.npy", mmap_mode="r"),
            labels=labels,
            token_to_cell={bytes(row): idx for idx, row in enumerate(labels)},
        )

    def candidates(self, query: np.ndarray, probes: int) -> tuple[np.ndarray, int]:
        residual = query.astype(np.float32) - self.mean
        residual /= max(float(np.linalg.norm(residual)), 1e-12)
        scores = self.centroids @ residual
        cells = np.argpartition(-scores, probes - 1)[:probes]
        # The owner sends HMAC tokens; the server resolves only opaque labels.
        resolved = [self.token_to_cell[bytes(self.labels[cell])] for cell in cells]
        lists = [self.postings[self.offsets[cell] : self.offsets[cell + 1]] for cell in resolved]
        reads = sum(len(values) for values in lists)
        return np.unique(np.concatenate(lists)).astype(np.int64), int(reads)


def exact_dense_topk_gpu(
    embeddings: np.ndarray,
    queries: np.ndarray,
    depth: int,
    batch_size: int,
    document_block: int,
    device: str,
) -> tuple[np.ndarray, float]:
    """Exact dense top-k with one host-to-device transfer per document block.

    The former implementation placed the complete corpus on the GPU and could
    not exceed roughly five million 384-D vectors on an 8 GiB card.  This
    document-major implementation keeps all query/top-k state on the GPU and
    streams each corpus block once, so its device memory is independent of N.
    """
    import torch

    started = time.perf_counter()
    query_gpu = torch.from_numpy(np.asarray(queries, dtype=np.float32)).to(device)
    best_scores = torch.full((len(queries), depth), -torch.inf, device=device)
    best_indices = torch.full((len(queries), depth), -1, dtype=torch.int64, device=device)
    with torch.inference_mode():
        for doc_first in range(0, len(embeddings), document_block):
            doc_last = min(doc_first + document_block, len(embeddings))
            document_gpu = torch.from_numpy(
                np.asarray(embeddings[doc_first:doc_last], dtype=np.float32).copy()
            ).to(device)
            block_depth = min(depth, doc_last - doc_first)
            for query_first in range(0, len(queries), batch_size):
                query_last = min(query_first + batch_size, len(queries))
                scores = query_gpu[query_first:query_last] @ document_gpu.T
                block_scores, block_local = torch.topk(scores, k=block_depth, dim=1)
                block_indices = block_local.to(torch.int64) + doc_first
                merged_scores = torch.cat(
                    (best_scores[query_first:query_last], block_scores), dim=1
                )
                merged_indices = torch.cat(
                    (best_indices[query_first:query_last], block_indices), dim=1
                )
                values, positions = torch.topk(merged_scores, k=depth, dim=1)
                best_scores[query_first:query_last] = values
                best_indices[query_first:query_last] = torch.gather(
                    merged_indices, 1, positions
                )
            if (
                doc_first == 0
                or doc_last == len(embeddings)
                or (doc_last // document_block) % 10 == 0
            ):
                print(f"[dense-exact] documents {doc_last:,}/{len(embeddings):,}", flush=True)
    output = best_indices.cpu().numpy().astype(np.int32)
    return output, time.perf_counter() - started


def keyed_term(value: str) -> str:
    return hmac.new(LEXICAL_KEY, value.encode("utf-8"), hashlib.sha256).hexdigest()[:32]


def lexical_rankings(
    *,
    db_path: Path,
    query_texts: list[str],
    expected_doc_ids: list[str],
    cap: int,
    max_probe_terms: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA cache_size=-262144")
    connection.execute("CREATE VIRTUAL TABLE IF NOT EXISTS vocab USING fts5vocab(postings, 'row')")
    connection.commit()
    document_count = int(connection.execute("SELECT count(*) FROM docmap").fetchone()[0])
    if document_count != len(expected_doc_ids):
        raise RuntimeError(
            f"lexical/semantic document-count mismatch: {document_count} vs {len(expected_doc_ids)}"
        )
    alignment_rows = sorted(
        set(
            [1, document_count, document_count // 2]
            + [1 + int(value) for value in np.linspace(0, document_count - 1, 14)]
        )
    )
    observed = {
        int(rowid): str(doc_id)
        for rowid, doc_id in connection.execute(
            f"SELECT rowid, doc_id FROM docmap WHERE rowid IN ({','.join('?' for _ in alignment_rows)})",
            alignment_rows,
        ).fetchall()
    }
    for rowid in alignment_rows:
        if observed.get(rowid) != expected_doc_ids[rowid - 1]:
            raise RuntimeError(f"lexical/semantic subset order mismatch at rowid {rowid}")
    rankings = np.full((len(query_texts), cap), -1, dtype=np.int32)
    latencies: list[float] = []
    posting_reads: list[int] = []
    probe_counts: list[int] = []
    for qi, query in enumerate(query_texts):
        terms = list(dict.fromkeys(keyed_term(token) for token in tokenize(query)))
        if terms:
            placeholders = ",".join("?" for _ in terms)
            frequencies = {
                str(term): int(doc_frequency)
                for term, doc_frequency in connection.execute(
                    f"SELECT term, doc FROM vocab WHERE term IN ({placeholders})", terms
                ).fetchall()
            }
        else:
            frequencies = {}
        matched = sorted(
            (term for term in terms if term in frequencies), key=lambda term: frequencies[term]
        )
        probes = matched[:max_probe_terms] if max_probe_terms else matched
        expression = " OR ".join(probes)
        probe_counts.append(len(probes))
        posting_reads.append(sum(frequencies[term] for term in probes))
        if expression:
            started = time.perf_counter()
            rows = connection.execute(
                "SELECT p.rowid FROM postings p WHERE postings MATCH ? "
                "ORDER BY bm25(postings) LIMIT ?",
                (expression, cap),
            ).fetchall()
            latencies.append((time.perf_counter() - started) * 1000.0)
            values = np.asarray([int(row[0]) - 1 for row in rows], dtype=np.int32)
            rankings[qi, : len(values)] = values
        else:
            latencies.append(0.0)
        if qi == 0 or qi + 1 == len(query_texts) or (qi + 1) % 500 == 0:
            print(f"[lexical] {qi + 1:,}/{len(query_texts):,}", flush=True)
    connection.close()
    return rankings, {
        "documents": document_count,
        "alignment_checks": len(alignment_rows),
        "alignment_status": "passed",
        "queries": len(query_texts),
        "cap": cap,
        "max_rarest_probe_terms": max_probe_terms,
        "probe_terms_mean": float(np.mean(probe_counts)),
        "estimated_posting_entries_read_mean": float(np.mean(posting_reads)),
        "candidate_latency_mean_ms": float(np.mean(latencies)),
        "candidate_latency_p50_ms": float(np.percentile(latencies, 50)),
        "candidate_latency_p95_ms": float(np.percentile(latencies, 95)),
    }


def rrf_pair(first: np.ndarray, second: np.ndarray, depth: int, constant: int = 60) -> np.ndarray:
    scores: dict[int, float] = {}
    for ranking in (first, second):
        for position, value in enumerate(ranking):
            doc = int(value)
            if doc < 0:
                continue
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (constant + position + 1)
    ordered = sorted(scores, key=lambda doc: (-scores[doc], doc))[:depth]
    result = np.full(depth, -1, dtype=np.int32)
    result[: len(ordered)] = ordered
    return result


def fuse_all(semantic: np.ndarray, lexical: np.ndarray, depth: int) -> np.ndarray:
    output = np.empty((len(semantic), depth), dtype=np.int32)
    for row in range(len(semantic)):
        output[row] = rrf_pair(semantic[row], lexical[row], depth)
    return output


def evaluate(
    rankings: np.ndarray,
    doc_ids: list[str],
    query_ids: list[str],
    qrels: dict[str, dict[str, int]],
) -> dict[str, float]:
    ndcg: list[float] = []
    mrr: list[float] = []
    recall10: list[float] = []
    recall100: list[float] = []
    for row, qid in zip(rankings, query_ids):
        truth = qrels[qid]
        relevant = {doc_id for doc_id, gain in truth.items() if gain > 0}
        ranked_ids = [doc_ids[int(idx)] for idx in row if int(idx) >= 0]
        observed = [truth.get(doc_id, 0) for doc_id in ranked_ids[:10]]
        ideal = sorted(truth.values(), reverse=True)[:10]
        dcg = sum((2.0**gain - 1.0) / math.log2(rank + 2) for rank, gain in enumerate(observed))
        idcg = sum((2.0**gain - 1.0) / math.log2(rank + 2) for rank, gain in enumerate(ideal))
        ndcg.append(dcg / idcg if idcg else 0.0)
        reciprocal = next(
            (1.0 / (rank + 1) for rank, doc_id in enumerate(ranked_ids[:10]) if doc_id in relevant),
            0.0,
        )
        mrr.append(reciprocal)
        recall10.append(len(relevant.intersection(ranked_ids[:10])) / len(relevant))
        recall100.append(len(relevant.intersection(ranked_ids[:100])) / len(relevant))
    return {
        "nDCG@10": float(np.mean(ndcg)),
        "MRR@10": float(np.mean(mrr)),
        "Recall@10": float(np.mean(recall10)),
        "Recall@100": float(np.mean(recall100)),
    }


def encrypt_queries(
    queries: np.ndarray,
    key_path: Path,
    beta: float,
    scale: float,
    seed: int,
) -> np.ndarray:
    key = np.load(key_path)
    transformed = dpe_transform(
        np.asarray(queries, dtype=np.float32),
        key["sign1"],
        key["sign2"],
        key["permutation"],
    )
    rng = np.random.default_rng(seed)
    noise = ball_noise(rng, len(queries), transformed.shape[1], scale * beta / 8.0)
    return (scale * transformed + noise).astype(np.float32)


def probe_sweep(
    *,
    ivf: IvfFiles,
    queries: np.ndarray,
    exact_dense: np.ndarray,
    relevant_indices: list[set[int]],
    probe_values: list[int],
) -> tuple[list[dict[str, float | int]], int]:
    rows: list[dict[str, float | int]] = []
    document_count = int(np.max(ivf.postings)) + 1
    for probes in probe_values:
        counts: list[int] = []
        reads: list[int] = []
        dense_coverage: list[float] = []
        relevant_coverage: list[float] = []
        latencies: list[float] = []
        for qi, query in enumerate(queries):
            started = time.perf_counter()
            candidates, posting_reads = ivf.candidates(query, probes)
            latencies.append((time.perf_counter() - started) * 1000.0)
            candidate_set = set(map(int, candidates))
            counts.append(len(candidates))
            reads.append(posting_reads)
            dense_coverage.append(
                len(candidate_set.intersection(map(int, exact_dense[qi]))) / len(exact_dense[qi])
            )
            truth = relevant_indices[qi]
            relevant_coverage.append(
                len(candidate_set.intersection(truth)) / len(truth) if truth else 1.0
            )
        row = {
            "nprobe": probes,
            "candidate_mean": float(np.mean(counts)),
            "candidate_p95": float(np.percentile(counts, 95)),
            "candidate_fraction_mean": float(np.mean(counts) / document_count),
            "posting_entries_read_mean": float(np.mean(reads)),
            "dense_top100_coverage_mean": float(np.mean(dense_coverage)),
            "dense_top100_coverage_p05": float(np.percentile(dense_coverage, 5)),
            "relevant_document_coverage_mean": float(np.mean(relevant_coverage)),
            "lookup_mean_ms": float(np.mean(latencies)),
            "lookup_p95_ms": float(np.percentile(latencies, 95)),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    eligible = [
        row
        for row in rows
        if row["dense_top100_coverage_mean"] >= 0.95
        and row["relevant_document_coverage_mean"] >= 0.95
    ]
    selected = int(eligible[0]["nprobe"] if eligible else rows[-1]["nprobe"])
    return rows, selected


def semantic_dpe_retrieval_repeats(
    *,
    ivf: IvfFiles,
    queries: np.ndarray,
    query_ciphers: np.ndarray,
    probes: int,
    candidate_depth: int,
    output_depth: int,
    embeddings: np.ndarray,
    cipher: np.ndarray,
    norms: np.ndarray,
    device: str,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Execute independent DPE query noises while sharing immutable row I/O.

    Each repeat performs its own encrypted distance top-k and exact reranking.
    Candidate generation and memmap materialization do not depend on DPE query
    noise, so they are loaded once per query.  Reported online times include an
    equal amortized share of that immutable I/O and identify this convention.
    """
    import torch

    repeats = int(query_ciphers.shape[0])
    output = np.full((repeats, len(queries), output_depth), -1, dtype=np.int32)
    candidate_counts: list[int] = []
    posting_reads: list[int] = []
    shared_latencies: list[float] = []
    repeat_latencies: list[list[float]] = [[] for _ in range(repeats)]
    torch.cuda.synchronize() if device.startswith("cuda") else None
    with torch.inference_mode():
        for qi, query in enumerate(queries):
            shared_started = time.perf_counter()
            candidates, reads = ivf.candidates(query, probes)
            candidate_counts.append(len(candidates))
            posting_reads.append(reads)
            candidate_rows = np.asarray(cipher[candidates], dtype=np.float16)
            candidate_norms = np.asarray(norms[candidates], dtype=np.float32)
            rows_gpu = torch.from_numpy(candidate_rows).to(device)
            norms_gpu = torch.from_numpy(candidate_norms).to(device)
            torch.cuda.synchronize() if device.startswith("cuda") else None
            shared_elapsed = (time.perf_counter() - shared_started) * 1000.0
            shared_latencies.append(shared_elapsed)
            cloud_depth = min(candidate_depth, len(candidates))
            final_depth = min(output_depth, cloud_depth)
            for repeat in range(repeats):
                started = time.perf_counter()
                q_cipher_gpu = torch.from_numpy(query_ciphers[repeat, qi]).to(
                    device=device, dtype=torch.float16
                )
                scores = 2.0 * torch.mv(rows_gpu, q_cipher_gpu).float() - norms_gpu
                cloud_local = torch.topk(scores, k=cloud_depth).indices.cpu().numpy()
                cloud = candidates[cloud_local]
                exact_rows = torch.from_numpy(
                    np.asarray(embeddings[cloud], dtype=np.float32)
                ).to(device)
                q_gpu = torch.from_numpy(queries[qi]).to(device)
                exact_scores = torch.mv(exact_rows, q_gpu)
                final_local = torch.topk(exact_scores, k=final_depth).indices.cpu().numpy()
                result = cloud[final_local].astype(np.int32)
                output[repeat, qi, : len(result)] = result
                torch.cuda.synchronize() if device.startswith("cuda") else None
                repeat_latencies[repeat].append(
                    (time.perf_counter() - started) * 1000.0 + shared_elapsed / repeats
                )
            if qi == 0 or qi + 1 == len(queries) or (qi + 1) % 500 == 0:
                print(f"[semantic-dpe x{repeats}] {qi + 1:,}/{len(queries):,}", flush=True)
    rows: list[dict[str, float]] = []
    for repeat, latencies in enumerate(repeat_latencies):
        rows.append(
            {
                "repeat": repeat,
                "candidate_mean": float(np.mean(candidate_counts)),
                "candidate_p95": float(np.percentile(candidate_counts, 95)),
                "posting_entries_read_mean": float(np.mean(posting_reads)),
                "shared_candidate_materialization_mean_ms": float(np.mean(shared_latencies)),
                "online_mean_ms": float(np.mean(latencies)),
                "online_p50_ms": float(np.percentile(latencies, 50)),
                "online_p95_ms": float(np.percentile(latencies, 95)),
                "timing_convention": "independent DPE scoring plus 1/repeats of shared candidate generation and immutable memmap row materialization",
            }
        )
    return output, rows


def semantic_ivf_exact_retrieval(
    *,
    ivf: IvfFiles,
    queries: np.ndarray,
    probes: int,
    output_depth: int,
    embeddings: np.ndarray,
    device: str,
) -> tuple[np.ndarray, dict[str, float]]:
    """Score the entire IVF candidate union exactly to isolate index loss."""
    import torch

    output = np.full((len(queries), output_depth), -1, dtype=np.int32)
    latencies: list[float] = []
    with torch.inference_mode():
        for qi, query in enumerate(queries):
            torch.cuda.synchronize() if device.startswith("cuda") else None
            started = time.perf_counter()
            candidates, _ = ivf.candidates(query, probes)
            q_gpu = torch.from_numpy(query).to(device)
            candidate_rows = torch.from_numpy(
                np.asarray(embeddings[candidates], dtype=np.float32)
            ).to(device)
            scores = torch.mv(candidate_rows, q_gpu)
            depth = min(output_depth, len(candidates))
            local = torch.topk(scores, k=depth).indices
            result = candidates[local.cpu().numpy()].astype(np.int32)
            output[qi, : len(result)] = result
            torch.cuda.synchronize() if device.startswith("cuda") else None
            latencies.append((time.perf_counter() - started) * 1000.0)
            if qi == 0 or qi + 1 == len(queries) or (qi + 1) % 500 == 0:
                print(f"[semantic-ivf-exact] {qi + 1:,}/{len(queries):,}", flush=True)
    return output, {
        "online_mean_ms": float(np.mean(latencies)),
        "online_p50_ms": float(np.percentile(latencies, 50)),
        "online_p95_ms": float(np.percentile(latencies, 95)),
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    import torch

    data_dir = ROOT / "data" / args.dataset
    cache_dir = args.cache_root / f"{args.dataset}_{args.max_docs}"
    result_dir = args.results_root / f"{args.dataset}_{args.max_docs}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    query_ids, query_texts, qrels, relevant_ids = load_queries_qrels(data_dir)

    encoding = encode_subset(
        data_dir=data_dir,
        cache_dir=cache_dir,
        relevant_ids=relevant_ids,
        query_texts=query_texts,
        model_path=args.model,
        max_docs=args.max_docs,
        batch_size=args.batch_size,
        encoder_chunk=args.encoder_chunk,
        device=args.device,
    )
    index_setup = build_semantic_index(
        cache_dir=cache_dir,
        clusters=args.clusters,
        assignments=args.assignments,
        train_sample=args.train_sample,
        epochs=args.kmeans_epochs,
        block_size=args.index_block,
        seed=args.seed,
        device=args.device,
    )
    dpe_setup = build_dpe_ciphertexts(
        cache_dir=cache_dir,
        beta=args.beta,
        scale=args.scale,
        block_size=args.dpe_block,
        seed=args.seed + 1,
    )

    doc_ids = read_ids(cache_dir / "doc_ids.txt")
    if len(doc_ids) != args.max_docs or len(set(doc_ids)) != args.max_docs:
        raise RuntimeError("doc ID manifest is incomplete or contains duplicates")
    doc_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
    missing_relevant = sorted(relevant_ids.difference(doc_index))
    if missing_relevant:
        raise RuntimeError(f"subset omitted {len(missing_relevant)} relevant documents")
    relevant_indices = [
        {doc_index[doc_id] for doc_id, gain in qrels[qid].items() if gain > 0}
        for qid in query_ids
    ]

    embeddings = np.load(cache_dir / "corpus_embeddings.npy", mmap_mode="r")
    queries = np.load(cache_dir / "query_embeddings.npy")
    exact_dense, dense_seconds = exact_dense_topk_gpu(
        embeddings,
        queries,
        args.output_depth,
        args.dense_query_batch,
        args.dense_document_block,
        args.device,
    )

    lexical_suffix = (
        "full" if FULL_CORPUS_DOCS.get(args.dataset) == args.max_docs else str(args.max_docs)
    )
    lexical_db = (
        ROOT
        / "results"
        / "keyed_fts5"
        / f"{args.dataset}_{lexical_suffix}.sqlite3"
    )
    if not lexical_db.exists():
        raise FileNotFoundError(f"missing lexical index {lexical_db}")
    lexical, lexical_summary = lexical_rankings(
        db_path=lexical_db,
        query_texts=query_texts,
        expected_doc_ids=doc_ids,
        cap=args.candidate_depth,
        max_probe_terms=args.lexical_probe_terms,
    )
    if lexical_summary["documents"] != args.max_docs:
        raise RuntimeError("lexical and semantic subset sizes differ")

    ivf = IvfFiles.load(cache_dir)
    sweep, selected_probes = probe_sweep(
        ivf=ivf,
        queries=queries,
        exact_dense=exact_dense,
        relevant_indices=relevant_indices,
        probe_values=args.probe_sweep,
    )

    if args.skip_ivf_exact:
        ivf_exact = None
        ivf_exact_online = None
    else:
        ivf_exact, ivf_exact_online = semantic_ivf_exact_retrieval(
            ivf=ivf,
            queries=queries,
            probes=selected_probes,
            output_depth=args.output_depth,
            embeddings=embeddings,
            device=args.device,
        )

    cipher = np.load(cache_dir / "semantic_cipher_fp16.npy", mmap_mode="r")
    norms = np.load(cache_dir / "semantic_cipher_norms.npy", mmap_mode="r")
    exact_hybrid = fuse_all(exact_dense, lexical, args.output_depth)
    ivf_exact_hybrid = (
        None if ivf_exact is None else fuse_all(ivf_exact, lexical, args.output_depth)
    )
    dense_metrics = evaluate(exact_dense, doc_ids, query_ids, qrels)
    ivf_exact_metrics = (
        None if ivf_exact is None else evaluate(ivf_exact, doc_ids, query_ids, qrels)
    )
    lexical_metrics = evaluate(lexical[:, : args.output_depth], doc_ids, query_ids, qrels)
    exact_hybrid_metrics = evaluate(exact_hybrid, doc_ids, query_ids, qrels)
    ivf_exact_hybrid_metrics = (
        None
        if ivf_exact_hybrid is None
        else evaluate(ivf_exact_hybrid, doc_ids, query_ids, qrels)
    )

    repeat_rows: list[dict[str, object]] = []
    semantic_rankings: list[np.ndarray] = []
    hybrid_rankings: list[np.ndarray] = []
    query_ciphers = np.stack(
        [
            encrypt_queries(
                queries,
                cache_dir / "semantic_dpe_key.npz",
                args.beta,
                args.scale,
                args.seed + 2027 + repeat * 100003,
            )
            for repeat in range(args.repeats)
        ]
    )
    semantic_all, online_all = semantic_dpe_retrieval_repeats(
        ivf=ivf,
        queries=queries,
        query_ciphers=query_ciphers,
        probes=selected_probes,
        candidate_depth=args.candidate_depth,
        output_depth=args.output_depth,
        embeddings=embeddings,
        cipher=cipher,
        norms=norms,
        device=args.device,
    )
    for repeat in range(args.repeats):
        semantic = semantic_all[repeat]
        online = online_all[repeat]
        hybrid = fuse_all(semantic, lexical, args.output_depth)
        sem_metrics = evaluate(semantic, doc_ids, query_ids, qrels)
        hybrid_metrics = evaluate(hybrid, doc_ids, query_ids, qrels)
        repeat_rows.append(
            {
                "repeat": repeat,
                "semantic_metrics": sem_metrics,
                "hybrid_metrics": hybrid_metrics,
                "semantic_online": online,
            }
        )
        semantic_rankings.append(semantic)
        hybrid_rankings.append(hybrid)
        atomic_json(result_dir / "partial_results.json", repeat_rows)
        print(
            json.dumps(
                {"repeat": repeat, "semantic": sem_metrics, "hybrid": hybrid_metrics}
            ),
            flush=True,
        )

    metric_names = ("nDCG@10", "MRR@10", "Recall@10", "Recall@100")
    semantic_mean = {
        metric: float(np.mean([row["semantic_metrics"][metric] for row in repeat_rows]))
        for metric in metric_names
    }
    semantic_sd = {
        metric: float(np.std([row["semantic_metrics"][metric] for row in repeat_rows], ddof=1))
        if args.repeats > 1
        else 0.0
        for metric in metric_names
    }
    hybrid_mean = {
        metric: float(np.mean([row["hybrid_metrics"][metric] for row in repeat_rows]))
        for metric in metric_names
    }
    hybrid_sd = {
        metric: float(np.std([row["hybrid_metrics"][metric] for row in repeat_rows], ddof=1))
        if args.repeats > 1
        else 0.0
        for metric in metric_names
    }
    output = {
        "dataset": args.dataset,
        "documents": args.max_docs,
        "queries": len(query_ids),
        "qrels": sum(len(value) for value in qrels.values()),
        "subset_policy": encoding["subset_policy"],
        "model": str(args.model),
        "embedding_dimension": int(queries.shape[1]),
        "configuration": {
            "clusters": args.clusters,
            "assignments": args.assignments,
            "probe_sweep": args.probe_sweep,
            "selected_probes": selected_probes,
            "candidate_depth": args.candidate_depth,
            "output_depth": args.output_depth,
            "beta_semantic": args.beta,
            "dpe_scale": args.scale,
            "repeats": args.repeats,
            "lexical_rarest_probe_terms": args.lexical_probe_terms,
            "seed": args.seed,
            "skip_ivf_exact": args.skip_ivf_exact,
            "dense_document_block": args.dense_document_block,
        },
        "encoding": encoding,
        "semantic_index_setup": index_setup,
        "semantic_dpe_setup": dpe_setup,
        "probe_sweep": sweep,
        "exact_dense_seconds": dense_seconds,
        "lexical_index": lexical_summary,
        "reference_metrics": {
            "dense_exact": dense_metrics,
            "semantic_ivf_exact_refinement": ivf_exact_metrics,
            "keyed_lexical_p2": lexical_metrics,
            "hybrid_dense_exact_plus_keyed_lexical_p2": exact_hybrid_metrics,
            "hybrid_ivf_exact_plus_keyed_lexical_p2": ivf_exact_hybrid_metrics,
        },
        "semantic_ivf_exact_online": ivf_exact_online,
        "semantic_dpe_mean": semantic_mean,
        "semantic_dpe_sd": semantic_sd,
        "hybrid_dpe_plus_keyed_lexical_mean": hybrid_mean,
        "hybrid_dpe_plus_keyed_lexical_sd": hybrid_sd,
        "retention": {
            "semantic_ndcg_vs_dense": semantic_mean["nDCG@10"] / max(dense_metrics["nDCG@10"], 1e-12),
            "semantic_recall100_vs_dense": semantic_mean["Recall@100"] / max(dense_metrics["Recall@100"], 1e-12),
            "hybrid_ndcg_vs_exact_semantic_same_lexical": hybrid_mean["nDCG@10"] / max(exact_hybrid_metrics["nDCG@10"], 1e-12),
            "hybrid_recall100_vs_exact_semantic_same_lexical": hybrid_mean["Recall@100"] / max(exact_hybrid_metrics["Recall@100"], 1e-12),
        },
        "repeats": repeat_rows,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "interpretation": {
            "semantic": "actual residual-IVF candidate generation, stored conditional-DPE ciphertext refinement, and exact client reranking",
            "lexical": "contentless HMAC-FTS5 candidate retrieval with exact FTS5 BM25 order; not the in-memory lexical-DPE implementation",
            "hybrid": "client-side RRF of the preceding independent paths",
            "cryptographic_warning": "NumPy PCG64 models bounded DPE noise and is not a production PRF",
        },
    }
    atomic_json(result_dir / "results.json", output)
    ranking_payload = {
        "exact_dense": exact_dense,
        "lexical": lexical,
        "exact_hybrid": exact_hybrid,
        "semantic": np.stack(semantic_rankings),
        "hybrid": np.stack(hybrid_rankings),
    }
    if ivf_exact is not None and ivf_exact_hybrid is not None:
        ranking_payload["ivf_exact"] = ivf_exact
        ranking_payload["ivf_exact_hybrid"] = ivf_exact_hybrid
    np.savez_compressed(result_dir / "rankings.npz", **ranking_payload)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("msmarco", "nq", "hotpotqa"), required=True)
    parser.add_argument("--max-docs", type=int, default=1_000_000)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "cache" / "models" / "granite-embedding-small-english-r2",
    )
    parser.add_argument("--cache-root", type=Path, default=ROOT / "cache" / "million_semantic")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results" / "million_semantic_hybrid")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--encoder-chunk", type=int, default=2048)
    parser.add_argument("--clusters", type=int, default=2048)
    parser.add_argument("--assignments", type=int, default=2)
    parser.add_argument("--train-sample", type=int, default=65536)
    parser.add_argument("--kmeans-epochs", type=int, default=6)
    parser.add_argument("--index-block", type=int, default=4096)
    parser.add_argument("--probe-sweep", type=str, default="32,64,128")
    parser.add_argument("--beta", type=float, default=0.10)
    parser.add_argument("--scale", type=float, default=3.0)
    parser.add_argument("--dpe-block", type=int, default=4096)
    parser.add_argument("--candidate-depth", type=int, default=200)
    parser.add_argument("--output-depth", type=int, default=100)
    parser.add_argument("--lexical-probe-terms", type=int, default=2)
    parser.add_argument("--dense-query-batch", type=int, default=8)
    parser.add_argument("--dense-document-block", type=int, default=65536)
    parser.add_argument(
        "--skip-ivf-exact",
        action="store_true",
        help="skip the redundant exact-within-IVF intermediate on very large corpora",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()
    args.probe_sweep = [int(value) for value in args.probe_sweep.split(",")]
    if sorted(args.probe_sweep) != args.probe_sweep:
        raise ValueError("probe-sweep must be ascending")
    return args


def main() -> None:
    args = parse_args()
    result = run_benchmark(args)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
