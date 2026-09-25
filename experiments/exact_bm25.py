"""Exact candidate-local BM25 reranking for the final DuetER CCADPE runs.

The compartment experiment previously preserved the order produced by the
HMAC-FTS5 candidate planner.  This script implements the protocol described in
the revised paper: feature-hashed CCADPE coordinates are used only for cloud
candidate refinement, while the client decrypts a compact token-ID payload and
recomputes unsketched BM25 over the returned candidate set.

The script reuses the final semantic and lexical candidate sets.  It streams
the corpus once to materialize, only for documents that occur in a returned
lexical set, the term frequencies needed by the evaluated queries.  The full
encrypted payload is modelled as the document's keyed uint32 token-ID sequence,
which is sufficient to reconstruct exact term frequencies and is padded to a
public 256-byte length bucket before AEAD protection.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import sqlite3
import time
from array import array
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse

from run_full_corpus import (
    communication_bytes,
    metric_mean,
    reconstruct_duetrank_and_recall,
)
from semantic_backend import (
    LEXICAL_KEY,
    atomic_json,
    load_queries_qrels,
    read_ids,
)
from duetrank import run_dataset as run_duetrank
from common import tokenize


ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parent
SOURCE_ROOT = REPOSITORY_ROOT / "results" / "generated" / "dual_compartment_full"
DESTINATION_ROOT = REPOSITORY_ROOT / "results" / "generated" / "exact_bm25_client_rerank"
CACHE_ROOT = ROOT / "cache" / "exact_bm25_client_rerank"

CONFIG = {
    "nq": {
        "source": "nq",
        "split": "test",
        "data": "nq",
        "fts": "nq_full.sqlite3",
        "posting_budget": 50_000,
    },
    "hotpotqa": {
        "source": "hotpotqa",
        "split": "test",
        "data": "hotpotqa",
        "fts": "hotpotqa_full.sqlite3",
        "posting_budget": 50_000,
    },
    "msmarco": {
        "source": "msmarco_dev",
        "split": "dev",
        "data": "msmarco",
        "fts": "msmarco_full.sqlite3",
        "posting_budget": 250_000,
    },
}


def keyed_term(value: str) -> str:
    return hmac.new(
        LEXICAL_KEY, value.encode("utf-8"), hashlib.sha256
    ).hexdigest()[:32]


def decode_first_fts5_varint(blob: bytes) -> int:
    value = 0
    for byte in blob:
        value = (value << 7) | (byte & 0x7F)
        if byte < 0x80:
            return value
    raise ValueError("unterminated FTS5 docsize varint")


def load_document_lengths(database: Path, cache: Path) -> np.ndarray:
    if cache.exists():
        return np.load(cache, mmap_mode="r")
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    count = int(connection.execute("SELECT count(*) FROM postings_docsize").fetchone()[0])
    lengths = np.empty(count, dtype=np.uint32)
    cursor = connection.execute("SELECT id, sz FROM postings_docsize ORDER BY id")
    seen = 0
    for rowid, blob in cursor:
        if int(rowid) != seen + 1:
            raise RuntimeError("FTS5 rowids are not dense")
        lengths[seen] = decode_first_fts5_varint(blob)
        seen += 1
        if seen % 1_000_000 == 0:
            print(f"[doc lengths] {seen:,}/{count:,}", flush=True)
    connection.close()
    if seen != count:
        raise RuntimeError("incomplete FTS5 docsize scan")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, lengths)
    return np.load(cache, mmap_mode="r")


def query_vocabulary(query_texts: list[str]) -> tuple[list[str], dict[str, int]]:
    terms = sorted({term for text in query_texts for term in tokenize(text)})
    return terms, {term: index for index, term in enumerate(terms)}


def load_document_frequencies(
    database: Path, terms: list[str]
) -> np.ndarray:
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    output = np.zeros(len(terms), dtype=np.uint32)
    keyed_to_index = {keyed_term(term): index for index, term in enumerate(terms)}
    keyed = list(keyed_to_index)
    for start in range(0, len(keyed), 800):
        batch = keyed[start : start + 800]
        placeholders = ",".join("?" for _ in batch)
        for label, frequency in connection.execute(
            f"SELECT term, doc FROM vocab WHERE term IN ({placeholders})", batch
        ):
            output[keyed_to_index[str(label)]] = int(frequency)
    connection.close()
    return output


def selected_document_rows(rankings: np.ndarray) -> np.ndarray:
    values = rankings[rankings >= 0]
    return np.unique(values.astype(np.int32, copy=False))


def build_candidate_term_matrix(
    corpus: Path,
    selected_rows: np.ndarray,
    term_to_index: dict[str, int],
    destination: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows_path = destination / "rows.npy"
    lengths_path = destination / "token_lengths.npy"
    indptr_path = destination / "indptr.npy"
    indices_path = destination / "indices.npy"
    counts_path = destination / "counts.npy"
    if all(
        path.exists()
        for path in (rows_path, lengths_path, indptr_path, indices_path, counts_path)
    ):
        return (
            np.load(lengths_path, mmap_mode="r"),
            np.load(indptr_path, mmap_mode="r"),
            np.load(indices_path, mmap_mode="r"),
            np.load(counts_path, mmap_mode="r"),
        )

    destination.mkdir(parents=True, exist_ok=True)
    np.save(rows_path, selected_rows)
    token_lengths = np.empty(len(selected_rows), dtype=np.uint32)
    indptr = np.empty(len(selected_rows) + 1, dtype=np.uint64)
    indptr[0] = 0
    indices = array("I")
    counts = array("I")
    target = 0
    started = time.perf_counter()
    with corpus.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if target >= len(selected_rows):
                break
            wanted = int(selected_rows[target])
            if row_index < wanted:
                continue
            if row_index != wanted:
                raise RuntimeError("corpus row order disagrees with candidate row")
            record = json.loads(line)
            text = f"{record.get('title', '')} {record.get('text', '')}".strip()
            tokens = tokenize(text)
            token_lengths[target] = len(tokens)
            local = Counter(
                term_to_index[token] for token in tokens if token in term_to_index
            )
            for term_index, frequency in sorted(local.items()):
                indices.append(int(term_index))
                counts.append(int(frequency))
            indptr[target + 1] = len(indices)
            target += 1
            if target % 100_000 == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"[exact payload] {target:,}/{len(selected_rows):,} documents "
                    f"({elapsed:.1f}s)",
                    flush=True,
                )
    if target != len(selected_rows):
        raise RuntimeError(f"materialized {target:,}/{len(selected_rows):,} rows")
    np.save(lengths_path, token_lengths)
    np.save(indptr_path, indptr)
    np.save(indices_path, np.frombuffer(indices, dtype=np.uint32).copy())
    np.save(counts_path, np.frombuffer(counts, dtype=np.uint32).copy())
    return (
        np.load(lengths_path, mmap_mode="r"),
        np.load(indptr_path, mmap_mode="r"),
        np.load(indices_path, mmap_mode="r"),
        np.load(counts_path, mmap_mode="r"),
    )


def exact_bm25_rerank(
    rankings: np.ndarray,
    selected_rows: np.ndarray,
    token_lengths: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    counts: np.ndarray,
    query_texts: list[str],
    term_to_index: dict[str, int],
    document_frequencies: np.ndarray,
    documents: int,
    average_length: float,
    *,
    k1: float = 1.2,
    b: float = 0.75,
) -> tuple[np.ndarray, dict[str, float]]:
    matrix = sparse.csr_matrix(
        (counts.astype(np.float32), indices.astype(np.int32), indptr.astype(np.int64)),
        shape=(len(selected_rows), len(term_to_index)),
    )
    output = np.full_like(rankings, -1)
    latencies = []
    for query_index, text in enumerate(query_texts):
        query_counts = Counter(tokenize(text))
        query_pairs = [
            (term_to_index[term], 1.0)
            for term in query_counts
            if term in term_to_index
            and document_frequencies[term_to_index[term]] > 0
        ]
        query_terms = [index for index, _ in query_pairs]
        union = np.unique(rankings[:, query_index])
        union = union[union >= 0]
        positions = np.searchsorted(selected_rows, union)
        if np.any(selected_rows[positions] != union):
            raise RuntimeError("returned document absent from exact payload cache")
        started = time.perf_counter()
        if len(query_terms):
            tf = matrix[positions][:, query_terms].toarray().astype(np.float64)
            dl = np.asarray(token_lengths[positions], dtype=np.float64)[:, None]
            dfs = document_frequencies[query_terms].astype(np.float64)
            idf = np.log1p((documents - dfs + 0.5) / (dfs + 0.5))[None, :]
            query_weights = np.asarray(
                [frequency for _, frequency in query_pairs],
                dtype=np.float64,
            )[None, :]
            norm = k1 * (1.0 - b + b * dl / max(average_length, 1e-12))
            scores = np.sum(
                idf * query_weights * tf * (k1 + 1.0) / np.maximum(tf + norm, 1e-12),
                axis=1,
            )
        else:
            scores = np.zeros(len(union), dtype=np.float64)
        score_order = np.lexsort((union, -scores))
        global_order = union[score_order]
        for repeat in range(len(rankings)):
            candidates = rankings[repeat, query_index]
            candidates = candidates[candidates >= 0]
            keep = np.isin(global_order, candidates, assume_unique=True)
            ordered = global_order[keep]
            depth = min(len(ordered), output.shape[-1])
            output[repeat, query_index, :depth] = ordered[:depth]
        latencies.append((time.perf_counter() - started) * 1000.0)
        if query_index == 0 or query_index + 1 == len(query_texts) or (query_index + 1) % 500 == 0:
            print(f"[client BM25] {query_index + 1:,}/{len(query_texts):,}", flush=True)
    return output, {
        "mean_ms": float(np.mean(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
    }


def exact_payload_communication(
    baseline: dict[str, float | int],
    rankings: np.ndarray,
    selected_rows: np.ndarray,
    token_lengths: np.ndarray,
    *,
    bucket_bytes: int = 256,
) -> dict[str, float | int | str]:
    positions = np.searchsorted(selected_rows, rankings[rankings >= 0])
    lengths = np.asarray(token_lengths[positions], dtype=np.int64)
    plaintext = 4 + 4 * lengths
    bucketed = ((plaintext + bucket_bytes - 1) // bucket_bytes) * bucket_bytes
    average_bucket = float(np.mean(bucketed))
    p95_bucket = float(np.percentile(bucketed, 95))
    lexical_records = int(baseline["lexical_padded_records"])
    semantic_records = int(baseline["semantic_padded_records"])
    semantic_record_bytes = 16 + 12 + (384 * 2 + 16) + 12 + (20 + 16)
    # Alias + AEAD nonce/tag + join-capsule nonce/ciphertext = 92 bytes.
    lexical_record_bytes = 92.0 + average_bucket
    download = 16.0 + semantic_records * semantic_record_bytes + lexical_records * lexical_record_bytes
    total = float(baseline["upload_bytes"]) + download
    return {
        "payload_encoding": "uint32 keyed token-ID sequence plus uint32 document length",
        "payload_padding_bucket_bytes": bucket_bytes,
        "mean_plaintext_payload_bytes": float(np.mean(plaintext)),
        "mean_padded_payload_bytes": average_bucket,
        "p95_padded_payload_bytes": p95_bucket,
        "semantic_padded_records": semantic_records,
        "lexical_padded_records": lexical_records,
        "upload_bytes": int(baseline["upload_bytes"]),
        "download_bytes": int(round(download)),
        "total_bytes": int(round(total)),
        "total_mib": total / (1024**2),
        "accounting_note": "fixed local record counts; exact lexical payloads use public 256-byte length buckets",
    }


def run_dataset(dataset: str) -> dict[str, object]:
    cfg = CONFIG[dataset]
    source = SOURCE_ROOT / cfg["source"]
    destination = DESTINATION_ROOT / cfg["source"]
    cache = CACHE_ROOT / cfg["source"]
    destination.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    query_ids, query_texts, qrels, _ = load_queries_qrels(
        ROOT / "data" / cfg["data"], cfg["split"]
    )
    lexical_path = source / "lexical_compartment.rankings.npz"
    semantic_path = source / "semantic_compartment.rankings.npz"
    lexical = np.asarray(np.load(lexical_path)["lexical_dpe"], dtype=np.int32)
    semantic = np.asarray(np.load(semantic_path)["semantic"], dtype=np.int32)
    rows = selected_document_rows(lexical)
    terms, term_to_index = query_vocabulary(query_texts)
    database = ROOT / "results" / "keyed_fts5" / cfg["fts"]
    full_lengths = load_document_lengths(database, cache / "full_doc_lengths_v2.npy")
    documents = len(full_lengths)
    average_length = float(np.mean(full_lengths))
    document_frequencies = load_document_frequencies(database, terms)
    token_lengths, indptr, indices, counts = build_candidate_term_matrix(
        ROOT / "data" / cfg["data"] / "corpus.jsonl",
        rows,
        term_to_index,
        cache,
    )
    exact, latency = exact_bm25_rerank(
        lexical,
        rows,
        token_lengths,
        indptr,
        indices,
        counts,
        query_texts,
        term_to_index,
        document_frequencies,
        documents,
        average_length,
    )
    exact_path = destination / "lexical_exact_bm25.rankings.npz"
    np.savez_compressed(exact_path, lexical_dpe=exact)

    doc_ids = read_ids(
        ROOT / "cache" / "full_semantic" / f"{cfg['data']}_{documents}" / "doc_ids.txt"
    )
    old_metrics = metric_mean(lexical, doc_ids, query_ids, qrels)
    exact_metrics = metric_mean(exact, doc_ids, query_ids, qrels)

    result_payload = json.loads((source / "results.json").read_text(encoding="utf-8"))
    old_communication = result_payload["rq2_efficiency"]["communication"]
    communication = exact_payload_communication(
        old_communication, exact, rows, token_lengths
    )

    workload = "msmarco_dev_full" if dataset == "msmarco" else f"{dataset}_full"
    trace_path = ROOT / "results" / "budgeted_lexical" / f"{workload}.trace.json"
    fusion = run_duetrank(
        cfg["data"],
        lexical_source="budgeted-dpe",
        scale="full",
        lexical_rankings_path=exact_path,
        split=cfg["split"],
        semantic_rankings_path=semantic_path,
        trace_path=trace_path,
        budget_key=cfg["posting_budget"],
        model_tag="exact-client-bm25",
    )
    atomic_json(destination / "duetrank.json", {"datasets": [fusion]})
    recall = reconstruct_duetrank_and_recall(
        cfg["data"],
        semantic,
        exact,
        fusion,
        query_ids,
        query_texts,
        qrels,
        doc_ids,
        trace_path,
        str(cfg["posting_budget"]),
        destination / "duetrank.rankings.npz",
    )

    report = {
        "experiment": "candidate-local exact BM25 reranking after lexical CCADPE",
        "dataset": dataset,
        "documents": documents,
        "queries": len(query_ids),
        "repeats": len(exact),
        "returned_depth": exact.shape[-1],
        "candidate_documents_materialized": len(rows),
        "average_document_tokens": average_length,
        "bm25": {
            "k1": 1.2,
            "b": 0.75,
            "idf": "log(1+(N-df+0.5)/(df+0.5))",
            "query_weight": "binary unique-term",
        },
        "client_exact_bm25_latency": latency,
        "lexical_metrics_before": old_metrics,
        "lexical_metrics_exact_bm25": exact_metrics,
        "metric_delta": {
            key: float(exact_metrics[key] - old_metrics[key]) for key in exact_metrics
        },
        "communication_before": old_communication,
        "communication_exact_payload": communication,
        "duetrank": fusion,
        "matched_budget_recall": recall,
        "scope": "exact BM25 ordering within the returned lexical candidate set; candidate omissions remain unrecoverable",
    }
    atomic_json(destination / "results.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+", choices=tuple(CONFIG), default=list(CONFIG)
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    all_results = [run_dataset(dataset) for dataset in args.datasets]
    atomic_json(DESTINATION_ROOT / "summary.json", {"datasets": all_results})
    print(json.dumps({"datasets": all_results}, indent=2))
