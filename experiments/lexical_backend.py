"""Streaming lexical backend used by the complete-corpus DuetER experiment.

The module provides signed feature-hashed BM25 construction, MIPS-to-L2
conversion, stored transformed coordinates, query encryption, and server-side
candidate ranking. Large arrays are memory-mapped so construction is resumable.
NumPy PCG64 is used only for deterministic experiments and is not a production
cryptographic PRF.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import hmac
import json
import math
import platform
import sqlite3
import time
from pathlib import Path
from typing import Iterator

import numpy as np

from semantic_backend import (
    LEXICAL_KEY,
    atomic_json,
    ball_noise,
    dpe_transform,
    evaluate,
    load_queries_qrels,
    read_ids,
)
from common import tokenize


ROOT = Path(__file__).resolve().parent
FEATURE_KEY = b"DuetDPE-Lexical-feature-hash-v1"


def keyed_term(term: str) -> str:
    return hmac.new(LEXICAL_KEY, term.encode("utf-8"), hashlib.sha256).hexdigest()[:32]


def feature(term_token: str, dimension: int) -> tuple[int, float]:
    digest = hmac.new(FEATURE_KEY, term_token.encode("ascii"), hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big") % dimension, (1.0 if digest[8] & 1 else -1.0)


def fts5_idf(documents: int, document_frequency: int) -> float:
    value = math.log((documents - document_frequency + 0.5) / (document_frequency + 0.5))
    return max(value, 1e-6)


def selected_binary_records(
    corpus_path: Path, relevant_ids: set[str], expected_ids: list[str]
) -> Iterator[tuple[int, int, dict[str, object]]]:
    filler_budget = len(expected_ids) - len(relevant_ids)
    fillers = 0
    selected = 0
    with corpus_path.open("rb") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            row = json.loads(line)
            doc_id = str(row["_id"])
            is_relevant = doc_id in relevant_ids
            if not is_relevant and fillers >= filler_budget:
                continue
            if selected >= len(expected_ids) or expected_ids[selected] != doc_id:
                raise RuntimeError(f"semantic/lexical row mismatch at {selected}: {doc_id}")
            yield selected, offset, row
            selected += 1
            if not is_relevant:
                fillers += 1
    if selected != len(expected_ids):
        raise RuntimeError(f"selected {selected} records, expected {len(expected_ids)}")


def document_vector(
    text: str,
    document_frequency: dict[str, int],
    documents: int,
    average_length: float,
    dimension: int,
    k1: float,
    b: float,
) -> tuple[np.ndarray, int]:
    counts = collections.Counter(tokenize(text))
    length = sum(counts.values())
    vector = np.zeros(dimension, dtype=np.float32)
    normalizer = k1 * (1.0 - b + b * length / max(average_length, 1e-12))
    for term, frequency in counts.items():
        token = keyed_term(term)
        df = document_frequency.get(token)
        if df is None:
            continue
        coordinate, sign = feature(token, dimension)
        weight = fts5_idf(documents, df) * frequency / (frequency + normalizer)
        vector[coordinate] += sign * weight
    return vector, length


def build_plain_sketches(
    *,
    data_dir: Path,
    cache_dir: Path,
    relevant_ids: set[str],
    expected_ids: list[str],
    document_frequency: dict[str, int],
    average_length: float,
    dimension: int,
    chunk: int,
    k1: float,
    b: float,
) -> dict[str, object]:
    path = cache_dir / "lexical_sketch_fp16.npy"
    offsets_path = cache_dir / "corpus_offsets.npy"
    progress_path = cache_dir / "lexical_sketch_progress.json"
    metadata_path = cache_dir / "lexical_sketch.json"
    if metadata_path.exists() and path.exists() and offsets_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    documents = len(expected_ids)
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        completed = int(progress["completed"])
        maximum_norm = float(progress["maximum_norm"])
        sketches = np.load(path, mmap_mode="r+")
        offsets = np.load(offsets_path, mmap_mode="r+")
    else:
        completed = 0
        maximum_norm = 0.0
        sketches = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float16, shape=(documents, dimension)
        )
        offsets = np.lib.format.open_memmap(
            offsets_path, mode="w+", dtype=np.int64, shape=(documents,)
        )
    started = time.perf_counter()
    last_report = started
    lengths = 0
    for row_index, offset, row in selected_binary_records(
        data_dir / "corpus.jsonl", relevant_ids, expected_ids
    ):
        if row_index < completed:
            continue
        text = f"{row.get('title', '')} {row.get('text', '')}".strip()
        vector, length = document_vector(
            text,
            document_frequency,
            documents,
            average_length,
            dimension,
            k1,
            b,
        )
        stored = vector.astype(np.float16)
        sketches[row_index] = stored
        offsets[row_index] = offset
        maximum_norm = max(
            maximum_norm,
            float(np.linalg.vector_norm(stored.astype(np.float32))),
        )
        lengths += length
        completed = row_index + 1
        if completed % chunk == 0 or completed == documents:
            sketches.flush()
            offsets.flush()
            atomic_json(
                progress_path,
                {"completed": completed, "maximum_norm": maximum_norm},
            )
            now = time.perf_counter()
            rate = chunk / max(now - last_report, 1e-12)
            last_report = now
            print(f"[lexical-sketch] {completed:,}/{documents:,} ({rate:.1f} docs/s)", flush=True)
    metadata = {
        "documents": documents,
        "hash_dimension": dimension,
        "storage_dtype": "float16",
        "maximum_quantized_sketch_norm": maximum_norm,
        "bytes": path.stat().st_size,
        "offset_bytes": offsets_path.stat().st_size,
        "build_seconds_last_invocation": time.perf_counter() - started,
        "bm25": {"k1": k1, "b": b, "average_document_length": average_length},
        "feature_hash": "HMAC-SHA256 coordinate and sign; collisions are retained",
    }
    atomic_json(metadata_path, metadata)
    return metadata


def build_ciphertexts(
    *, cache_dir: Path, dimension: int, beta: float, scale: float, block: int, seed: int
) -> dict[str, object]:
    metadata_path = cache_dir / "lexical_dpe.json"
    cipher_path = cache_dir / "lexical_cipher_fp16.npy"
    norm_path = cache_dir / "lexical_cipher_norms.npy"
    key_path = cache_dir / "lexical_dpe_key.npz"
    progress_path = cache_dir / "lexical_dpe_progress.json"
    if metadata_path.exists() and cipher_path.exists() and norm_path.exists() and key_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    sketches = np.load(cache_dir / "lexical_sketch_fp16.npy", mmap_mode="r")
    sketch_metadata = json.loads((cache_dir / "lexical_sketch.json").read_text(encoding="utf-8"))
    maximum_norm = max(float(sketch_metadata["maximum_quantized_sketch_norm"]), 1e-12)
    documents = len(sketches)
    input_dimension = dimension + 1
    work_dimension = 1 << (input_dimension - 1).bit_length()
    rng = np.random.default_rng(seed)
    sign1 = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), work_dimension)
    sign2 = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), work_dimension)
    permutation = rng.permutation(work_dimension)
    if not key_path.exists():
        np.savez(key_path, sign1=sign1, sign2=sign2, permutation=permutation)
    else:
        key = np.load(key_path)
        sign1, sign2, permutation = key["sign1"], key["sign2"], key["permutation"]
    if progress_path.exists():
        completed = int(json.loads(progress_path.read_text(encoding="utf-8"))["completed"])
        cipher = np.load(cipher_path, mmap_mode="r+")
        norms = np.load(norm_path, mmap_mode="r+")
    else:
        completed = 0
        cipher = np.lib.format.open_memmap(
            cipher_path, mode="w+", dtype=np.float16, shape=(documents, work_dimension)
        )
        norms = np.lib.format.open_memmap(
            norm_path, mode="w+", dtype=np.float32, shape=(documents,)
        )
    noise_rng = np.random.default_rng(seed + 1009)
    # Advance deterministic noise when resuming without materializing prior rows.
    for first in range(0, completed, block):
        rows = min(block, completed - first)
        ball_noise(noise_rng, rows, work_dimension, 1.0)
    radius = 3.0 * scale * beta / 8.0
    started = time.perf_counter()
    for first in range(completed, documents, block):
        last = min(first + block, documents)
        base = np.asarray(sketches[first:last], dtype=np.float32) / maximum_norm
        base_norm = np.sum(base * base, axis=1)
        mips = np.zeros((len(base), input_dimension), dtype=np.float32)
        mips[:, :dimension] = base
        mips[:, dimension] = np.sqrt(np.maximum(0.0, 1.0 - base_norm))
        encrypted = scale * dpe_transform(mips, sign1, sign2, permutation)
        encrypted += ball_noise(noise_rng, len(base), work_dimension, radius)
        stored = encrypted.astype(np.float16)
        cipher[first:last] = stored
        norms[first:last] = np.sum(stored.astype(np.float32) ** 2, axis=1)
        if last % (block * 20) == 0 or last == documents:
            cipher.flush()
            norms.flush()
            atomic_json(progress_path, {"completed": last})
            print(f"[lexical-dpe-database] {last:,}/{documents:,}", flush=True)
    metadata = {
        "documents": documents,
        "hash_dimension": dimension,
        "mips_dimension": input_dimension,
        "work_dimension": work_dimension,
        "maximum_sketch_norm": maximum_norm,
        "beta": beta,
        "scale": scale,
        "database_noise_radius": radius,
        "ciphertext_dtype": "float16",
        "ciphertext_bytes": cipher_path.stat().st_size,
        "norm_bytes": norm_path.stat().st_size,
        "build_seconds_last_invocation": time.perf_counter() - started,
    }
    atomic_json(metadata_path, metadata)
    return metadata


def query_mips(
    text: str, document_frequency: dict[str, int], documents: int, dimension: int
) -> np.ndarray:
    vector = np.zeros(dimension + 1, dtype=np.float32)
    for term in dict.fromkeys(tokenize(text)):
        token = keyed_term(term)
        if token not in document_frequency:
            continue
        coordinate, sign = feature(token, dimension)
        vector[coordinate] += sign
    norm = float(np.linalg.vector_norm(vector[:dimension]))
    if norm > 0:
        vector[:dimension] /= norm
    return vector


def encrypt_query_matrix(
    queries: np.ndarray, key_path: Path, beta: float, scale: float, seed: int
) -> np.ndarray:
    key = np.load(key_path)
    transformed = dpe_transform(queries, key["sign1"], key["sign2"], key["permutation"])
    noise = ball_noise(
        np.random.default_rng(seed), len(queries), transformed.shape[1], scale * beta / 8.0
    )
    return (scale * transformed + noise).astype(np.float32)


def exact_bm25_for_rows(
    *,
    stream,
    offsets: np.ndarray,
    row_indices: np.ndarray,
    query_terms: set[str],
    document_frequency: dict[str, int],
    documents: int,
    average_length: float,
    k1: float,
    b: float,
) -> np.ndarray:
    scores = np.zeros(len(row_indices), dtype=np.float32)
    for position, row_index in enumerate(row_indices):
        stream.seek(int(offsets[int(row_index)]))
        row = json.loads(stream.readline())
        counts = collections.Counter(
            tokenize(f"{row.get('title', '')} {row.get('text', '')}".strip())
        )
        length = sum(counts.values())
        normalizer = k1 * (1.0 - b + b * length / max(average_length, 1e-12))
        score = 0.0
        for term in query_terms:
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            token = keyed_term(term)
            df = document_frequency.get(token)
            if df is not None:
                score += fts5_idf(documents, df) * frequency / (frequency + normalizer)
        scores[position] = score
    return scores


def run_queries(
    *,
    data_dir: Path,
    db_path: Path,
    cache_dir: Path,
    doc_ids: list[str],
    query_ids: list[str],
    query_texts: list[str],
    qrels: dict[str, dict[str, int]],
    document_frequency: dict[str, int],
    average_length: float,
    dimension: int,
    beta: float,
    scale: float,
    repeats: int,
    probe_terms: int,
    max_candidates: int,
    cloud_depth: int,
    output_depth: int,
    k1: float,
    b: float,
    seed: int,
    device: str,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    import torch

    documents = len(doc_ids)
    doc_index = {doc_id: index for index, doc_id in enumerate(doc_ids)}
    query_plain = np.stack(
        [query_mips(text, document_frequency, documents, dimension) for text in query_texts]
    )
    query_ciphers = np.stack(
        [
            encrypt_query_matrix(
                query_plain,
                cache_dir / "lexical_dpe_key.npz",
                beta,
                scale,
                seed + 2027 + repeat * 100003,
            )
            for repeat in range(repeats)
        ]
    )
    cipher = np.load(cache_dir / "lexical_cipher_fp16.npy", mmap_mode="r")
    norms = np.load(cache_dir / "lexical_cipher_norms.npy", mmap_mode="r")
    sketches = np.load(cache_dir / "lexical_sketch_fp16.npy", mmap_mode="r")
    offsets = np.load(cache_dir / "corpus_offsets.npy", mmap_mode="r")
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA cache_size=-262144")
    reference = np.full((len(query_ids), output_depth), -1, dtype=np.int32)
    fts5_reference = np.full((len(query_ids), output_depth), -1, dtype=np.int32)
    rankings = np.full((repeats, len(query_ids), output_depth), -1, dtype=np.int32)
    candidate_counts: list[int] = []
    relevant_coverages: list[float] = []
    truncated_queries = 0
    latencies: list[list[float]] = [[] for _ in range(repeats)]
    with (data_dir / "corpus.jsonl").open("rb") as corpus_stream, torch.inference_mode():
        for qi, (qid, text) in enumerate(zip(query_ids, query_texts)):
            terms = list(dict.fromkeys(keyed_term(term) for term in tokenize(text)))
            if terms:
                placeholders = ",".join("?" for _ in terms)
                frequencies = {
                    str(term): int(df)
                    for term, df in connection.execute(
                        f"SELECT term, doc FROM vocab WHERE term IN ({placeholders})", terms
                    ).fetchall()
                }
            else:
                frequencies = {}
            probes = sorted(frequencies, key=frequencies.get)[:probe_terms]
            expression = " OR ".join(probes)
            if expression:
                rows = connection.execute(
                    "SELECT rowid FROM postings WHERE postings MATCH ? LIMIT ?",
                    (expression, max_candidates + 1),
                ).fetchall()
                truncated_queries += int(len(rows) > max_candidates)
                candidates = np.asarray(
                    [int(row[0]) - 1 for row in rows[:max_candidates]], dtype=np.int64
                )
                ref_rows = connection.execute(
                    "SELECT rowid FROM postings WHERE postings MATCH ? "
                    "ORDER BY bm25(postings) LIMIT ?",
                    (expression, output_depth),
                ).fetchall()
                fts5_reference[qi, : len(ref_rows)] = np.asarray(
                    [int(row[0]) - 1 for row in ref_rows], dtype=np.int32
                )
            else:
                candidates = np.empty(0, dtype=np.int64)
            candidate_counts.append(len(candidates))
            relevant = {
                doc_index[doc_id]
                for doc_id, gain in qrels[qid].items()
                if gain > 0 and doc_id in doc_index
            }
            relevant_coverages.append(
                len(relevant.intersection(map(int, candidates))) / len(relevant)
                if relevant
                else 1.0
            )
            if len(candidates):
                shared_started = time.perf_counter()
                rows_gpu = torch.from_numpy(
                    np.asarray(cipher[candidates], dtype=np.float16)
                ).to(device)
                norms_gpu = torch.from_numpy(
                    np.asarray(norms[candidates], dtype=np.float32)
                ).to(device)
                torch.cuda.synchronize() if device.startswith("cuda") else None
                shared_ms = (time.perf_counter() - shared_started) * 1000.0
                plain_scores = (
                    np.asarray(sketches[candidates], dtype=np.float32)
                    @ query_plain[qi, :dimension]
                )
                plain_depth = min(cloud_depth, len(candidates))
                plain_local = np.argpartition(-plain_scores, plain_depth - 1)[:plain_depth]
                plain_cloud = candidates[plain_local]
                clouds: list[np.ndarray] = []
                for repeat in range(repeats):
                    started = time.perf_counter()
                    q_gpu = torch.from_numpy(query_ciphers[repeat, qi]).to(
                        device=device, dtype=torch.float16
                    )
                    scores = 2.0 * torch.mv(rows_gpu, q_gpu).float() - norms_gpu
                    depth = min(cloud_depth, len(candidates))
                    local = torch.topk(scores, k=depth).indices.cpu().numpy()
                    clouds.append(candidates[local])
                    torch.cuda.synchronize() if device.startswith("cuda") else None
                    latencies[repeat].append(
                        (time.perf_counter() - started) * 1000.0 + shared_ms / repeats
                    )
                unique_cloud = np.unique(np.concatenate([plain_cloud, *clouds]))
                exact_scores = exact_bm25_for_rows(
                    stream=corpus_stream,
                    offsets=offsets,
                    row_indices=unique_cloud,
                    query_terms=set(tokenize(text)),
                    document_frequency=document_frequency,
                    documents=documents,
                    average_length=average_length,
                    k1=k1,
                    b=b,
                )
                score_lookup = dict(zip(map(int, unique_cloud), map(float, exact_scores)))
                plain_exact = np.asarray(
                    [score_lookup[int(row)] for row in plain_cloud], dtype=np.float32
                )
                plain_order = np.argsort(-plain_exact, kind="stable")[:output_depth]
                reference[qi, : len(plain_order)] = plain_cloud[plain_order].astype(np.int32)
                for repeat, cloud in enumerate(clouds):
                    cloud_scores = np.asarray([score_lookup[int(row)] for row in cloud])
                    order = np.argsort(-cloud_scores, kind="stable")[:output_depth]
                    rankings[repeat, qi, : len(order)] = cloud[order].astype(np.int32)
            if qi == 0 or qi + 1 == len(query_ids) or (qi + 1) % 500 == 0:
                print(f"[lexical-dpe-query x{repeats}] {qi + 1:,}/{len(query_ids):,}", flush=True)
    connection.close()
    reference_metrics = evaluate(reference, doc_ids, query_ids, qrels)
    fts5_reference_metrics = evaluate(fts5_reference, doc_ids, query_ids, qrels)
    repeat_rows = []
    for repeat in range(repeats):
        metrics = evaluate(rankings[repeat], doc_ids, query_ids, qrels)
        repeat_rows.append(
            {
                "repeat": repeat,
                "metrics": metrics,
                "online_mean_ms": float(np.mean(latencies[repeat])) if latencies[repeat] else 0.0,
                "online_p95_ms": float(np.percentile(latencies[repeat], 95)) if latencies[repeat] else 0.0,
            }
        )
    metric_names = ("nDCG@10", "MRR@10", "Recall@10", "Recall@100")
    mean = {
        name: float(np.mean([row["metrics"][name] for row in repeat_rows]))
        for name in metric_names
    }
    sd = {
        name: float(np.std([row["metrics"][name] for row in repeat_rows], ddof=1))
        if repeats > 1
        else 0.0
        for name in metric_names
    }
    return (
        {
            "candidate_generation": {
                "variant": "complete union of rare HMAC-term postings without BM25 ordering",
                "probe_terms": probe_terms,
                "safety_cap": max_candidates,
                "candidate_mean": float(np.mean(candidate_counts)),
                "candidate_p95": float(np.percentile(candidate_counts, 95)),
                "truncated_queries": truncated_queries,
                "relevant_document_coverage_mean": float(np.mean(relevant_coverages)),
            },
            "reference_plaintext_sketch_same_candidates": reference_metrics,
            "external_fts5_two_probe_reference": fts5_reference_metrics,
            "lexical_dpe_mean": mean,
            "lexical_dpe_sd": sd,
            "retention": {
                "ndcg": mean["nDCG@10"] / max(reference_metrics["nDCG@10"], 1e-12),
                "recall100": mean["Recall@100"] / max(reference_metrics["Recall@100"], 1e-12),
            },
            "repeats": repeat_rows,
        },
        {
            "plaintext_sketch_reference": reference,
            "fts5_two_probe_reference": fts5_reference,
            "lexical_dpe": rankings,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("msmarco", "nq", "hotpotqa"), required=True)
    parser.add_argument("--max-docs", type=int, default=1_000_000)
    parser.add_argument("--hash-dimension", type=int, default=1024)
    parser.add_argument("--beta", type=float, default=0.10)
    parser.add_argument("--scale", type=float, default=3.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--probe-terms", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=100_000)
    parser.add_argument("--cloud-depth", type=int, default=200)
    parser.add_argument("--output-depth", type=int, default=100)
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--dpe-block", type=int, default=2048)
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--cache-root", type=Path, default=ROOT / "cache" / "million_lexical_dpe"
    )
    parser.add_argument(
        "--results-root", type=Path, default=ROOT / "results" / "million_lexical_dpe"
    )
    args = parser.parse_args()

    data_dir = ROOT / "data" / args.dataset
    semantic_cache = ROOT / "cache" / "million_semantic" / f"{args.dataset}_{args.max_docs}"
    doc_ids = read_ids(semantic_cache / "doc_ids.txt")
    if len(doc_ids) != args.max_docs:
        raise RuntimeError("the complete-corpus semantic doc-ID manifest is required")
    query_ids, query_texts, qrels, relevant_ids = load_queries_qrels(data_dir)
    db_path = ROOT / "results" / "keyed_fts5" / f"{args.dataset}_{args.max_docs}.sqlite3"
    connection = sqlite3.connect(db_path)
    documents = int(connection.execute("SELECT count(*) FROM docmap").fetchone()[0])
    document_frequency = {
        str(term): int(df) for term, df in connection.execute("SELECT term, doc FROM vocab")
    }
    connection.close()
    build_json = json.loads(
        (ROOT / "results" / "keyed_fts5" / f"{args.dataset}_{args.max_docs}_p2.json").read_text(
            encoding="utf-8"
        )
    )
    average_length = float(build_json["build"]["tokens"]) / documents
    cache_dir = args.cache_root / f"{args.dataset}_{args.max_docs}_h{args.hash_dimension}"
    result_dir = args.results_root / f"{args.dataset}_{args.max_docs}_h{args.hash_dimension}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    sketch = build_plain_sketches(
        data_dir=data_dir,
        cache_dir=cache_dir,
        relevant_ids=relevant_ids,
        expected_ids=doc_ids,
        document_frequency=document_frequency,
        average_length=average_length,
        dimension=args.hash_dimension,
        chunk=args.chunk,
        k1=args.k1,
        b=args.b,
    )
    dpe = build_ciphertexts(
        cache_dir=cache_dir,
        dimension=args.hash_dimension,
        beta=args.beta,
        scale=args.scale,
        block=args.dpe_block,
        seed=args.seed + 2,
    )
    query_result, rankings = run_queries(
        data_dir=data_dir,
        db_path=db_path,
        cache_dir=cache_dir,
        doc_ids=doc_ids,
        query_ids=query_ids,
        query_texts=query_texts,
        qrels=qrels,
        document_frequency=document_frequency,
        average_length=average_length,
        dimension=args.hash_dimension,
        beta=args.beta,
        scale=args.scale,
        repeats=args.repeats,
        probe_terms=args.probe_terms,
        max_candidates=args.max_candidates,
        cloud_depth=args.cloud_depth,
        output_depth=args.output_depth,
        k1=args.k1,
        b=args.b,
        seed=args.seed,
        device=args.device,
    )
    output = {
        "dataset": args.dataset,
        "documents": documents,
        "queries": len(query_ids),
        "subset_policy": "all positive test-qrel documents plus earliest non-qrel records",
        "configuration": vars(args) | {"cache_root": str(args.cache_root), "results_root": str(args.results_root)},
        "sketch_setup": sketch,
        "dpe_setup": dpe,
        **query_result,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "scope": {
            "candidate_order": "unordered HMAC posting union; FTS5 BM25 is not used for DPE candidates",
            "server_refinement": "actual stored conditional-DPE ciphertext L2 ranking",
            "client_refinement": "exact un-hashed BM25 over server top-200 using corpus byte offsets",
            "primary_reference": "plaintext signed-hash MIPS over the identical unordered candidate set and identical exact-BM25 client reranking",
            "external_reference": "exact FTS5 BM25 top-100 for the two rare probe terms only; it is not used to compute DPE retention",
            "warning": "bounded noise uses NumPy PCG64, not a production PRF",
        },
    }
    atomic_json(result_dir / "results.json", output)
    np.savez_compressed(result_dir / "rankings.npz", **rankings)
    print(json.dumps(output, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
