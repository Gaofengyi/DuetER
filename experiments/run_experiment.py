"""Pilot evaluation for DuetDPE on a dataset in standard BEIR format.

This script deliberately separates offline encoding/index construction from online
retrieval.  It implements the ranking behavior of the proposed construction,
including signed feature hashing, BM25-to-L2 reduction, conditional bounded-noise
distance encryption, keyed LSH candidate generation, client reranking, RRF, and a
cross-path co-occurrence linkage attack.

The NumPy generator is used to make the experiment reproducible.  It is NOT a
production cryptographic PRF; a deployed implementation must replace it with a
CSPRNG keyed through HKDF/PRF as specified in the paper.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import math
import os
import platform
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
DEPS = ROOT / ".deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))
GPU_DEPS = ROOT / ".gpu_deps"
if os.environ.get("DUETDPE_GPU_DEPS") == "1" and GPU_DEPS.exists():
    sys.path.insert(0, str(GPU_DEPS))

import numpy as np

from lexical_mips_index import KeyedBM25TailIndex
from pisces_reimplementation import PiscesResearchReimplementation
from semantic_candidate_index import KeyedResidualSphericalIVF


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norms, eps)).astype(np.float32)


def normalize_vector(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return (x / max(float(np.linalg.norm(x)), eps)).astype(np.float32)


def top_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, scores.shape[0])
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    if k == scores.shape[0]:
        return np.argsort(-scores, kind="stable")
    chosen = np.argpartition(-scores, k - 1)[:k]
    return chosen[np.argsort(-scores[chosen], kind="stable")]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def load_beir(data_dir: Path, split: str = "test") -> tuple[list[str], list[str], list[str], list[str], dict[str, dict[str, int]]]:
    corpus_rows = load_jsonl(data_dir / "corpus.jsonl")
    query_rows = load_jsonl(data_dir / "queries.jsonl")
    corpus_ids = [str(row["_id"]) for row in corpus_rows]
    corpus_texts = [f"{row.get('title', '')}. {row.get('text', '')}".strip() for row in corpus_rows]
    query_lookup = {str(row["_id"]): row["text"] for row in query_rows}

    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with (data_dir / "qrels" / f"{split}.tsv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            qrels[str(row["query-id"])][str(row["corpus-id"])] = int(row["score"])

    query_ids = [qid for qid in query_lookup if qid in qrels]
    query_texts = [query_lookup[qid] for qid in query_ids]
    return corpus_ids, corpus_texts, query_ids, query_texts, dict(qrels)


def load_scifact(data_dir: Path) -> tuple[list[str], list[str], list[str], list[str], dict[str, dict[str, int]]]:
    """Backward-compatible alias used by the existing unit tests."""
    return load_beir(data_dir)


class BM25Index:
    def __init__(self, documents: list[str], k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(text) for text in documents]
        self.doc_len = np.asarray([len(tokens) for tokens in self.doc_tokens], dtype=np.float32)
        self.avgdl = float(np.mean(self.doc_len))
        self.n_docs = len(documents)
        self.term_counts: list[Counter[str]] = [Counter(tokens) for tokens in self.doc_tokens]
        df: Counter[str] = Counter()
        for counts in self.term_counts:
            df.update(counts.keys())
        self.idf = {
            term: math.log(1.0 + (self.n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }
        postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        temp: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc_idx, counts in enumerate(self.term_counts):
            for term, tf in counts.items():
                temp[term].append((doc_idx, tf))
        for term, pairs in temp.items():
            postings[term] = (
                np.asarray([p[0] for p in pairs], dtype=np.int32),
                np.asarray([p[1] for p in pairs], dtype=np.float32),
            )
        self.postings = postings

    def score(self, query: str) -> np.ndarray:
        scores = np.zeros(self.n_docs, dtype=np.float32)
        qtf = Counter(tokenize(query))
        for term, query_count in qtf.items():
            posting = self.postings.get(term)
            if posting is None:
                continue
            doc_idx, tf = posting
            denom = tf + self.k1 * (
                1.0 - self.b + self.b * self.doc_len[doc_idx] / self.avgdl
            )
            scores[doc_idx] += (
                query_count * self.idf[term] * tf * (self.k1 + 1.0) / denom
            ).astype(np.float32)
        return scores

    @staticmethod
    def _feature(term: str, dim: int, seed: bytes) -> tuple[int, float]:
        digest = hashlib.blake2b(term.encode("utf-8"), key=seed, digest_size=16).digest()
        bucket = int.from_bytes(digest[:8], "little") % dim
        sign = 1.0 if digest[8] & 1 else -1.0
        return bucket, sign

    def hashed_document_vectors(self, dim: int, seed: bytes) -> np.ndarray:
        vectors = np.zeros((self.n_docs, dim), dtype=np.float32)
        for doc_idx, counts in enumerate(self.term_counts):
            norm_term = self.k1 * (
                1.0 - self.b + self.b * self.doc_len[doc_idx] / self.avgdl
            )
            for term, tf_int in counts.items():
                tf = float(tf_int)
                weight = self.idf[term] * tf * (self.k1 + 1.0) / (tf + norm_term)
                bucket, sign = self._feature(term, dim, seed)
                vectors[doc_idx, bucket] += sign * weight
        return vectors

    def hashed_query_vector(self, query: str, dim: int, seed: bytes) -> np.ndarray:
        vector = np.zeros(dim, dtype=np.float32)
        for term, count in Counter(tokenize(query)).items():
            if term not in self.idf:
                continue
            bucket, sign = self._feature(term, dim, seed)
            vector[bucket] += sign * float(count)
        return vector


def mips_document_transform(x: np.ndarray) -> tuple[np.ndarray, float]:
    norms = np.linalg.norm(x, axis=1)
    scale = float(np.max(norms)) * (1.0 + 1e-6)
    first = x / max(scale, 1e-12)
    tail = np.sqrt(np.maximum(0.0, 1.0 - np.sum(first * first, axis=1, keepdims=True)))
    return np.concatenate([first, tail], axis=1).astype(np.float32), scale


def mips_query_transform(y: np.ndarray) -> np.ndarray:
    first = normalize_vector(y)
    return np.concatenate([first, np.zeros(1, dtype=np.float32)]).astype(np.float32)


def fwht_batch(x: np.ndarray) -> np.ndarray:
    """Normalized Walsh-Hadamard transform over the last dimension."""
    y = np.asarray(x, dtype=np.float32).copy()
    n = y.shape[1]
    if n & (n - 1):
        raise ValueError("FWHT dimension must be a power of two")
    h = 1
    while h < n:
        y = y.reshape(y.shape[0], -1, 2 * h)
        a = y[:, :, :h].copy()
        b = y[:, :, h : 2 * h].copy()
        y[:, :, :h] = a + b
        y[:, :, h : 2 * h] = a - b
        y = y.reshape(y.shape[0], n)
        h *= 2
    return y / math.sqrt(n)


class ConditionalDPE:
    """Research prototype of the conditional scale-and-perturb construction."""

    def __init__(self, input_dim: int, beta: float, scale: float, seed: int):
        self.input_dim = input_dim
        self.work_dim = 1 << (input_dim - 1).bit_length()
        self.beta = float(beta)
        self.scale = float(scale)
        rng = np.random.default_rng(seed)
        self.sign1 = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), self.work_dim)
        self.sign2 = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), self.work_dim)
        self.permutation = rng.permutation(self.work_dim)
        self.db_rng = np.random.default_rng(seed + 1009)
        self.query_rng = np.random.default_rng(seed + 2027)

    def _pad(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x[None, :]
        if x.shape[1] == self.work_dim:
            return np.asarray(x, dtype=np.float32)
        out = np.zeros((x.shape[0], self.work_dim), dtype=np.float32)
        out[:, : x.shape[1]] = x
        return out

    def transform(self, x: np.ndarray) -> np.ndarray:
        y = self._pad(x) * self.sign1
        y = fwht_batch(y)
        y *= self.sign2
        return y[:, self.permutation]

    @staticmethod
    def _ball_noise(rng: np.random.Generator, rows: int, dim: int, radius: float) -> np.ndarray:
        directions = rng.normal(size=(rows, dim)).astype(np.float32)
        directions = normalize_rows(directions)
        radii = radius * np.power(rng.random(rows), 1.0 / dim)
        return directions * radii[:, None].astype(np.float32)

    def encrypt_database(self, vectors: np.ndarray) -> np.ndarray:
        transformed = self.transform(vectors)
        noise = self._ball_noise(
            self.db_rng,
            transformed.shape[0],
            transformed.shape[1],
            3.0 * self.scale * self.beta / 8.0,
        )
        return (self.scale * transformed + noise).astype(np.float32)

    def encrypt_queries(self, vectors: np.ndarray) -> np.ndarray:
        transformed = self.transform(vectors)
        noise = self._ball_noise(
            self.query_rng,
            transformed.shape[0],
            transformed.shape[1],
            self.scale * self.beta / 8.0,
        )
        return (self.scale * transformed + noise).astype(np.float32)


class KeyedPStableLSH:
    """Keyed multi-probe Euclidean LSH using Gaussian (2-stable) projections."""

    def __init__(
        self,
        dim: int,
        tables: int,
        projections: int,
        width: float,
        seed: int,
        key: bytes,
    ):
        if projections < 1:
            raise ValueError("projections must be positive")
        if width <= 0:
            raise ValueError("width must be positive")
        self.dim = dim
        self.tables = tables
        self.projections = projections
        self.width = float(width)
        self.key = key
        rng = np.random.default_rng(seed)
        self.planes = rng.normal(size=(tables, projections, dim)).astype(np.float32)
        self.offsets = rng.uniform(0.0, self.width, size=(tables, projections)).astype(
            np.float32
        )
        self.buckets: list[dict[bytes, list[int]]] = [defaultdict(list) for _ in range(tables)]

    def _token(self, table: int, code: np.ndarray) -> bytes:
        payload = table.to_bytes(2, "big") + np.asarray(code, dtype=">i8").tobytes()
        return hmac.new(self.key, payload, hashlib.sha256).digest()[:16]

    def _codes(self, vectors: np.ndarray, table: int) -> tuple[np.ndarray, np.ndarray]:
        positions = (vectors @ self.planes[table].T + self.offsets[table]) / self.width
        codes = np.floor(positions).astype(np.int64)
        fractions = positions - codes
        return codes, fractions

    def build(self, vectors: np.ndarray) -> None:
        for table in range(self.tables):
            codes, _ = self._codes(vectors, table)
            for doc_idx, code in enumerate(codes):
                self.buckets[table][self._token(table, code)].append(doc_idx)

    def candidates(self, query: np.ndarray, probes_per_table: int) -> np.ndarray:
        found: set[int] = set()
        query_2d = query[None, :]
        for table in range(self.tables):
            codes, fractions = self._codes(query_2d, table)
            code = codes[0]
            probe_codes = [code]
            neighbors: list[tuple[float, int, int]] = []
            for coordinate, fraction in enumerate(fractions[0]):
                neighbors.append((float(fraction), coordinate, -1))
                neighbors.append((float(1.0 - fraction), coordinate, 1))
            neighbors.sort(key=lambda item: item[0])
            for _, coordinate, direction in neighbors[: max(0, probes_per_table - 1)]:
                adjacent = code.copy()
                adjacent[coordinate] += direction
                probe_codes.append(adjacent)
            tokens = [self._token(table, probe_code) for probe_code in probe_codes]
            for token in tokens:
                found.update(self.buckets[table].get(token, ()))
        return np.fromiter(found, dtype=np.int64)


def rrf(rank_a: Iterable[int], rank_b: Iterable[int], depth: int, alpha: float, c: int = 60) -> list[int]:
    fused: dict[int, float] = defaultdict(float)
    for rank, doc_idx in enumerate(list(rank_a)[:depth], start=1):
        fused[int(doc_idx)] += alpha / (c + rank)
    for rank, doc_idx in enumerate(list(rank_b)[:depth], start=1):
        fused[int(doc_idx)] += (1.0 - alpha) / (c + rank)
    return [item[0] for item in sorted(fused.items(), key=lambda item: (-item[1], item[0]))]


def evaluate(
    rankings: dict[str, list[int]],
    query_ids: list[str],
    corpus_ids: list[str],
    qrels: dict[str, dict[str, int]],
) -> dict[str, float]:
    doc_id_by_idx = np.asarray(corpus_ids)
    ndcg10, mrr10, recall10, recall100 = [], [], [], []
    for qid in query_ids:
        relevant = qrels[qid]
        ranked_ids = [str(x) for x in doc_id_by_idx[np.asarray(rankings[qid], dtype=np.int64)]]
        gains = [relevant.get(doc_id, 0) for doc_id in ranked_ids[:10]]
        dcg = sum((2**gain - 1) / math.log2(rank + 2) for rank, gain in enumerate(gains))
        ideal = sorted(relevant.values(), reverse=True)[:10]
        idcg = sum((2**gain - 1) / math.log2(rank + 2) for rank, gain in enumerate(ideal))
        ndcg10.append(dcg / idcg if idcg else 0.0)
        reciprocal = 0.0
        for rank, doc_id in enumerate(ranked_ids[:10], start=1):
            if relevant.get(doc_id, 0) > 0:
                reciprocal = 1.0 / rank
                break
        mrr10.append(reciprocal)
        relevant_set = {doc_id for doc_id, score in relevant.items() if score > 0}
        recall10.append(len(relevant_set.intersection(ranked_ids[:10])) / len(relevant_set))
        recall100.append(len(relevant_set.intersection(ranked_ids[:100])) / len(relevant_set))
    return {
        "nDCG@10": float(np.mean(ndcg10)),
        "MRR@10": float(np.mean(mrr10)),
        "Recall@10": float(np.mean(recall10)),
        "Recall@100": float(np.mean(recall100)),
    }


def neighbor_overlap(plain: np.ndarray, encrypted: np.ndarray, sample_size: int, k: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    sample = rng.choice(plain.shape[0], min(sample_size, plain.shape[0]), replace=False)
    overlaps = []
    for idx in sample:
        plain_scores = plain @ plain[idx]
        encrypted_dist = np.linalg.norm(encrypted - encrypted[idx], axis=1)
        plain_rank = top_indices(plain_scores, k + 1)
        plain_rank = [int(x) for x in plain_rank if int(x) != int(idx)][:k]
        encrypted_rank = np.argsort(encrypted_dist)
        encrypted_rank = [int(x) for x in encrypted_rank if int(x) != int(idx)][:k]
        overlaps.append(len(set(plain_rank).intersection(encrypted_rank)) / k)
    return float(np.mean(overlaps))


def cooccurrence_link_attack(
    semantic_results: list[np.ndarray],
    lexical_results: list[np.ndarray],
    sem_alias: np.ndarray,
    lex_alias: np.ndarray,
    attack_depth: int,
) -> dict[str, float | int]:
    counts: dict[int, Counter[int]] = defaultdict(Counter)
    for sem_docs, lex_docs in zip(semantic_results, lexical_results):
        s_aliases = sem_alias[sem_docs[:attack_depth]]
        l_aliases = lex_alias[lex_docs[:attack_depth]]
        for sa in s_aliases:
            counts[int(sa)].update(int(x) for x in l_aliases)
    inverse_sem = {int(alias): idx for idx, alias in enumerate(sem_alias)}
    correct = 0
    for sa, counter in counts.items():
        if not counter:
            continue
        predicted = counter.most_common(1)[0][0]
        doc_idx = inverse_sem[sa]
        correct += int(predicted == int(lex_alias[doc_idx]))
    evaluated = sum(bool(counter) for counter in counts.values())
    return {
        "evaluated_semantic_aliases": evaluated,
        "top1_link_accuracy": correct / evaluated if evaluated else 0.0,
    }


@dataclass
class Timings:
    corpus_dense_encoding_seconds: float = 0.0
    query_dense_encoding_seconds: float = 0.0
    lexical_vector_build_seconds: float = 0.0
    semantic_dpe_setup_seconds: float = 0.0
    lexical_dpe_setup_seconds: float = 0.0
    semantic_query_encryption_seconds: float = 0.0
    lexical_query_encryption_seconds: float = 0.0
    perturbed_query_encryption_seconds: float = 0.0
    lsh_build_seconds: float = 0.0
    semantic_residual_ivf_build_seconds: float = 0.0
    lexical_tail_index_build_seconds: float = 0.0
    pisces_reimplementation_setup_seconds: float = 0.0


def encode_dense(
    corpus_texts: list[str],
    query_texts: list[str],
    cache_dir: Path,
    model_name: str,
    batch_size: int,
    backend: str,
    cache_namespace: str,
    device: str,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    if backend == "lsa":
        encoder_tag = "lsa256"
    else:
        encoder_tag = "st-" + hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:12]
    cache_tag = f"{cache_namespace}-{encoder_tag}"
    corpus_cache = cache_dir / f"{cache_tag}_corpus_embeddings.npy"
    query_cache = cache_dir / f"{cache_tag}_query_embeddings.npy"
    if corpus_cache.exists() and query_cache.exists():
        return np.load(corpus_cache), np.load(query_cache), 0.0, 0.0

    if backend == "lsa":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b[A-Za-z0-9]+\b",
            ngram_range=(1, 2),
            min_df=2,
            max_features=30000,
            sublinear_tf=True,
        )
        start = time.perf_counter()
        corpus_sparse = vectorizer.fit_transform(corpus_texts)
        svd = TruncatedSVD(n_components=256, n_iter=7, random_state=20260822)
        corpus = normalize_rows(svd.fit_transform(corpus_sparse).astype(np.float32))
        corpus_time = time.perf_counter() - start
        start = time.perf_counter()
        queries = normalize_rows(svd.transform(vectorizer.transform(query_texts)).astype(np.float32))
        query_time = time.perf_counter() - start
        np.save(corpus_cache, corpus)
        np.save(query_cache, queries)
        return corpus, queries, corpus_time, query_time

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, cache_folder=str(cache_dir / "models"), device=device)
    start = time.perf_counter()
    corpus = model.encode(
        corpus_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    corpus_time = time.perf_counter() - start
    start = time.perf_counter()
    queries = model.encode(
        query_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    query_time = time.perf_counter() - start
    np.save(corpus_cache, corpus)
    np.save(query_cache, queries)
    return corpus, queries, corpus_time, query_time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "scifact")
    parser.add_argument("--dataset-name", default=None, help="Label stored in reports; defaults to the data directory name.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument(
        "--dense-backend",
        choices=("lsa", "sentence-transformer"),
        default="lsa",
        help="Use offline LSA for the pilot or a downloaded Sentence Transformer.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto",
        help="Sentence Transformer device. 'auto' uses CUDA when available.",
    )
    parser.add_argument("--lexical-dim", type=int, default=1024)
    parser.add_argument("--beta-semantic", type=float, default=0.10)
    parser.add_argument("--beta-lexical", type=float, default=0.10)
    parser.add_argument("--dpe-scale", type=float, default=3.0)
    parser.add_argument("--dp-sigma", type=float, default=0.015)
    parser.add_argument("--lsh-tables", type=int, default=24)
    parser.add_argument(
        "--lsh-bits",
        type=int,
        default=4,
        help="Number of concatenated p-stable projections per table (legacy option name).",
    )
    parser.add_argument("--lsh-width-semantic", type=float, default=1.0)
    parser.add_argument("--lsh-width-lexical", type=float, default=1.0)
    parser.add_argument("--lsh-probes", type=int, default=9)
    parser.add_argument("--candidate-depth", type=int, default=200)
    parser.add_argument("--semantic-ivf-clusters", type=int, default=256)
    parser.add_argument("--semantic-ivf-assignments", type=int, default=4)
    parser.add_argument("--semantic-ivf-probes", type=int, default=32)
    parser.add_argument("--lexical-index-cap", type=int, default=1600)
    parser.add_argument("--lexical-dummy-features", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument(
        "--rankings-output",
        type=Path,
        default=None,
        help="Optionally export per-query rankings and candidate traces for held-out fusion experiments.",
    )
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    dataset_name = args.dataset_name or args.data_dir.name
    corpus_ids, corpus_texts, query_ids, query_texts, qrels = load_beir(args.data_dir, args.split)
    timings = Timings()
    if args.device == "auto":
        try:
            import torch
            encoder_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            encoder_device = "cpu"
    else:
        encoder_device = args.device
    dense_docs, dense_queries, timings.corpus_dense_encoding_seconds, timings.query_dense_encoding_seconds = encode_dense(
        corpus_texts,
        query_texts,
        args.cache_dir,
        args.model,
        args.batch_size,
        args.dense_backend,
        dataset_name,
        encoder_device,
    )
    dense_docs = normalize_rows(dense_docs)
    dense_queries = normalize_rows(dense_queries)

    bm25 = BM25Index(corpus_texts)
    pisces_bm25 = BM25Index(corpus_texts, k1=1.5, b=0.75)
    start = time.perf_counter()
    lexical_hashed = bm25.hashed_document_vectors(args.lexical_dim, b"DuetDPE-Lexical-v1")
    lexical_docs, lexical_scale = mips_document_transform(lexical_hashed)
    lexical_queries = np.stack(
        [
            mips_query_transform(
                bm25.hashed_query_vector(query, args.lexical_dim, b"DuetDPE-Lexical-v1")
            )
            for query in query_texts
        ]
    )
    timings.lexical_vector_build_seconds = time.perf_counter() - start

    semantic_dpe = ConditionalDPE(
        dense_docs.shape[1], args.beta_semantic, args.dpe_scale, args.seed + 1
    )
    lexical_dpe = ConditionalDPE(
        lexical_docs.shape[1], args.beta_lexical, args.dpe_scale, args.seed + 2
    )
    start = time.perf_counter()
    semantic_cipher = semantic_dpe.encrypt_database(dense_docs)
    timings.semantic_dpe_setup_seconds = time.perf_counter() - start
    start = time.perf_counter()
    lexical_cipher = lexical_dpe.encrypt_database(lexical_docs)
    timings.lexical_dpe_setup_seconds = time.perf_counter() - start

    sem_lsh = KeyedPStableLSH(
        dense_docs.shape[1], args.lsh_tables, args.lsh_bits, args.lsh_width_semantic,
        args.seed + 3, b"semantic-bucket-key"
    )
    lex_lsh = KeyedPStableLSH(
        lexical_docs.shape[1], args.lsh_tables, args.lsh_bits, args.lsh_width_lexical,
        args.seed + 4, b"lexical-bucket-key"
    )
    start = time.perf_counter()
    sem_lsh.build(dense_docs)
    lex_lsh.build(lexical_docs)
    timings.lsh_build_seconds = time.perf_counter() - start

    semantic_residual_ivf = KeyedResidualSphericalIVF(
        b"DuetDPE-semantic-residual-ivf-v1",
        n_clusters=args.semantic_ivf_clusters,
        doc_assignments=args.semantic_ivf_assignments,
        nprobe=args.semantic_ivf_probes,
        centered=True,
        seed=args.seed + 30,
    )
    start = time.perf_counter()
    semantic_residual_ivf_setup = semantic_residual_ivf.build(dense_docs)
    timings.semantic_residual_ivf_build_seconds = time.perf_counter() - start

    lexical_tail_index = KeyedBM25TailIndex(
        b"DuetDPE-keyed-vocabulary-tail-v2"
    )
    start = time.perf_counter()
    lexical_tail_index.build(
        bm25.postings, bm25.idf, bm25.doc_len, bm25.avgdl,
        bm25.n_docs, bm25.k1, bm25.b,
    )
    timings.lexical_tail_index_build_seconds = time.perf_counter() - start

    pisces = PiscesResearchReimplementation(
        dense_documents=dense_docs,
        term_counts=pisces_bm25.term_counts,
        document_lengths=pisces_bm25.doc_len,
        average_document_length=pisces_bm25.avgdl,
        tokenizer=tokenize,
        k1=pisces_bm25.k1,
        b=pisces_bm25.b,
        seed=args.seed + 20,
    )
    start = time.perf_counter()
    pisces_setup = pisces.build()
    timings.pisces_reimplementation_setup_seconds = time.perf_counter() - start

    rng = np.random.default_rng(args.seed + 5)
    sem_perm = rng.permutation(len(corpus_ids))
    lex_perm = rng.permutation(len(corpus_ids))

    methods = [
        "bm25_plain",
        "dense_plain",
        "hybrid_plain",
        "pisces_bm25_plain_adapter",
        "pisces_hybrid_plain_adapter",
        "lexical_sketch_plain",
        "semantic_dpe_full",
        "lexical_dpe_full",
        "dual_dpe_full",
        "semantic_dpe_lsh",
        "semantic_dpe_residual_ivf",
        "lexical_dpe_lsh",
        "dual_dpe_lsh",
        "lexical_dpe_tail_index",
        "dual_dpe_tail_index",
        "duetdpe_tail_index_dp",
        "pisces_semantic_reimpl",
        "pisces_lexical_reimpl",
        "pisces_dual_reimpl",
    ]
    rankings: dict[str, dict[str, list[int]]] = {method: {} for method in methods}
    query_latencies: dict[str, list[float]] = {method: [] for method in methods}
    sem_cloud_results: list[np.ndarray] = []
    lex_cloud_results: list[np.ndarray] = []
    candidate_counts_sem: list[int] = []
    candidate_counts_lex: list[int] = []
    candidate_recall_sem: list[float] = []
    candidate_counts_sem_ivf: list[int] = []
    candidate_recall_sem_ivf: list[float] = []
    candidate_relevant_recall_sem_ivf: list[float] = []
    semantic_ivf_posting_reads: list[int] = []
    candidate_recall_lex: list[float] = []
    candidate_counts_tail: list[int] = []
    candidate_recall_tail: list[float] = []
    candidate_positive_recall_tail: list[float] = []
    tail_query_tokens: list[int] = []
    tail_posting_reads: list[int] = []
    pisces_candidate_counts: list[int] = []
    pisces_operation_counts: Counter[str] = Counter()

    dp_rng = np.random.default_rng(args.seed + 6)
    start = time.perf_counter()
    query_sem_cipher = semantic_dpe.encrypt_queries(dense_queries)
    timings.semantic_query_encryption_seconds = time.perf_counter() - start
    start = time.perf_counter()
    query_lex_cipher = lexical_dpe.encrypt_queries(lexical_queries)
    timings.lexical_query_encryption_seconds = time.perf_counter() - start

    dp_dense_queries = normalize_rows(
        dense_queries + dp_rng.normal(0.0, args.dp_sigma, dense_queries.shape).astype(np.float32)
    )
    dp_lexical_queries = lexical_queries.copy()
    vocabulary = np.asarray(list(bm25.idf), dtype=object)
    dp_lexical_cover_terms: list[list[str]] = []
    for row in range(dp_lexical_queries.shape[0]):
        block = dp_lexical_queries[row, :-1]
        nonzero = np.flatnonzero(np.abs(block) > 1e-10)
        block[nonzero] += dp_rng.normal(0.0, args.dp_sigma, len(nonzero)).astype(np.float32)
        dummy_count = min(args.lexical_dummy_features, len(vocabulary))
        dummy_terms: list[str] = []
        if dummy_count:
            dummy_terms = [str(term) for term in dp_rng.choice(vocabulary, dummy_count, replace=False)]
            for term in dummy_terms:
                coordinate, sign = bm25._feature(term, len(block), b"DuetDPE-Lexical-v1")
                block[coordinate] += sign * args.dp_sigma
        dp_lexical_cover_terms.append(tokenize(query_texts[row]) + dummy_terms)
        dp_lexical_queries[row, :-1] = normalize_vector(block)
        dp_lexical_queries[row, -1] = 0.0
    start = time.perf_counter()
    dp_sem_cipher = semantic_dpe.encrypt_queries(dp_dense_queries)
    dp_lex_cipher = lexical_dpe.encrypt_queries(dp_lexical_queries)
    timings.perturbed_query_encryption_seconds = time.perf_counter() - start
    semantic_cipher_norm = np.sum(semantic_cipher * semantic_cipher, axis=1)
    lexical_cipher_norm = np.sum(lexical_cipher * lexical_cipher, axis=1)
    corpus_index_by_id = {doc_id: idx for idx, doc_id in enumerate(corpus_ids)}

    for qi, (qid, query) in enumerate(zip(query_ids, query_texts)):
        start = time.perf_counter()
        bm25_scores = bm25.score(query)
        bm25_rank = top_indices(bm25_scores, 100)
        bm25_elapsed = time.perf_counter() - start
        query_latencies["bm25_plain"].append(bm25_elapsed)
        rankings["bm25_plain"][qid] = bm25_rank.tolist()

        start = time.perf_counter()
        pisces_bm25_scores = pisces_bm25.score(query)
        pisces_bm25_rank = top_indices(pisces_bm25_scores, 100)
        pisces_bm25_elapsed = time.perf_counter() - start
        query_latencies["pisces_bm25_plain_adapter"].append(pisces_bm25_elapsed)
        rankings["pisces_bm25_plain_adapter"][qid] = pisces_bm25_rank.tolist()

        start = time.perf_counter()
        dense_scores = dense_docs @ dense_queries[qi]
        dense_rank = top_indices(dense_scores, 100)
        dense_elapsed = time.perf_counter() - start
        query_latencies["dense_plain"].append(dense_elapsed)
        rankings["dense_plain"][qid] = dense_rank.tolist()

        start = time.perf_counter()
        hybrid_rank = rrf(bm25_rank, dense_rank, depth=100, alpha=0.5)
        hybrid_fusion_elapsed = time.perf_counter() - start
        query_latencies["hybrid_plain"].append(
            bm25_elapsed + dense_elapsed + hybrid_fusion_elapsed
        )
        rankings["hybrid_plain"][qid] = hybrid_rank

        start = time.perf_counter()
        pisces_hybrid_rank = rrf(pisces_bm25_rank, dense_rank, depth=100, alpha=0.5)
        pisces_hybrid_fusion = time.perf_counter() - start
        query_latencies["pisces_hybrid_plain_adapter"].append(
            pisces_bm25_elapsed + dense_elapsed + pisces_hybrid_fusion
        )
        rankings["pisces_hybrid_plain_adapter"][qid] = pisces_hybrid_rank

        start = time.perf_counter()
        lexical_sketch_scores = lexical_docs @ lexical_queries[qi]
        lexical_sketch_rank = top_indices(lexical_sketch_scores, 100)
        query_latencies["lexical_sketch_plain"].append(time.perf_counter() - start)
        rankings["lexical_sketch_plain"][qid] = lexical_sketch_rank.tolist()

        # Full-scan DPE isolates encryption/ranking loss from ANN candidate loss.
        start = time.perf_counter()
        sem_full_scores = 2.0 * (semantic_cipher @ query_sem_cipher[qi]) - semantic_cipher_norm
        sem_full_cloud = top_indices(sem_full_scores, args.candidate_depth)
        sem_full_exact = dense_docs[sem_full_cloud] @ dense_queries[qi]
        sem_full_rank = sem_full_cloud[
            top_indices(sem_full_exact, min(100, len(sem_full_cloud)))
        ]
        sem_full_elapsed = time.perf_counter() - start
        query_latencies["semantic_dpe_full"].append(sem_full_elapsed)
        rankings["semantic_dpe_full"][qid] = sem_full_rank.tolist()

        start = time.perf_counter()
        lex_full_scores = 2.0 * (lexical_cipher @ query_lex_cipher[qi]) - lexical_cipher_norm
        lex_full_cloud = top_indices(lex_full_scores, args.candidate_depth)
        lex_full_exact = bm25_scores[lex_full_cloud]
        lex_full_rank = lex_full_cloud[
            top_indices(lex_full_exact, min(100, len(lex_full_cloud)))
        ]
        lex_full_elapsed = time.perf_counter() - start
        query_latencies["lexical_dpe_full"].append(lex_full_elapsed)
        rankings["lexical_dpe_full"][qid] = lex_full_rank.tolist()

        start = time.perf_counter()
        dual_full_rank = rrf(sem_full_rank, lex_full_rank, depth=100, alpha=0.5)
        dual_full_fusion = time.perf_counter() - start
        query_latencies["dual_dpe_full"].append(
            sem_full_elapsed + lex_full_elapsed + dual_full_fusion
        )
        rankings["dual_dpe_full"][qid] = dual_full_rank

        # LSH-DPE measures candidate generation plus encrypted ranking and reranking.
        sem_start = time.perf_counter()
        sem_candidates = sem_lsh.candidates(dense_queries[qi], args.lsh_probes)
        sem_dist = np.linalg.norm(semantic_cipher[sem_candidates] - query_sem_cipher[qi], axis=1)
        sem_cloud = sem_candidates[top_indices(-sem_dist, args.candidate_depth)]
        sem_exact = dense_docs[sem_cloud] @ dense_queries[qi]
        sem_rank = sem_cloud[top_indices(sem_exact, min(100, len(sem_cloud)))]
        sem_lsh_elapsed = time.perf_counter() - sem_start
        query_latencies["semantic_dpe_lsh"].append(sem_lsh_elapsed)
        rankings["semantic_dpe_lsh"][qid] = sem_rank.tolist()

        # Mean-centering removes Granite's common direction.  Multi-assignment
        # spherical IVF then yields balanced keyed postings instead of the
        # near-full-scan union produced by the legacy p-stable LSH.
        sem_ivf_start = time.perf_counter()
        sem_ivf_candidates, sem_ivf_trace = semantic_residual_ivf.candidates(
            dense_queries[qi]
        )
        sem_ivf_dist = np.linalg.norm(
            semantic_cipher[sem_ivf_candidates] - query_sem_cipher[qi], axis=1
        )
        sem_ivf_cloud = sem_ivf_candidates[
            top_indices(-sem_ivf_dist, args.candidate_depth)
        ]
        sem_ivf_exact = dense_docs[sem_ivf_cloud] @ dense_queries[qi]
        sem_ivf_rank = sem_ivf_cloud[
            top_indices(sem_ivf_exact, min(100, len(sem_ivf_cloud)))
        ]
        sem_ivf_elapsed = time.perf_counter() - sem_ivf_start
        query_latencies["semantic_dpe_residual_ivf"].append(sem_ivf_elapsed)
        rankings["semantic_dpe_residual_ivf"][qid] = sem_ivf_rank.tolist()
        candidate_counts_sem_ivf.append(len(sem_ivf_candidates))
        candidate_recall_sem_ivf.append(
            len(set(map(int, dense_rank)).intersection(map(int, sem_ivf_candidates)))
            / len(dense_rank)
        )
        relevant_indices = {
            corpus_index_by_id[doc_id]
            for doc_id, gain in qrels[qid].items()
            if gain > 0 and doc_id in corpus_index_by_id
        }
        candidate_relevant_recall_sem_ivf.append(
            len(relevant_indices.intersection(map(int, sem_ivf_candidates)))
            / len(relevant_indices) if relevant_indices else 1.0
        )
        semantic_ivf_posting_reads.append(sem_ivf_trace.posting_entries_read)

        lex_start = time.perf_counter()
        lex_candidates = lex_lsh.candidates(lexical_queries[qi], args.lsh_probes)
        lex_dist = np.linalg.norm(lexical_cipher[lex_candidates] - query_lex_cipher[qi], axis=1)
        lex_cloud = lex_candidates[top_indices(-lex_dist, args.candidate_depth)]
        lex_exact_scores = bm25_scores[lex_cloud]
        lex_rank = lex_cloud[top_indices(lex_exact_scores, min(100, len(lex_cloud)))]
        lex_lsh_elapsed = time.perf_counter() - lex_start
        query_latencies["lexical_dpe_lsh"].append(lex_lsh_elapsed)
        rankings["lexical_dpe_lsh"][qid] = lex_rank.tolist()

        candidate_counts_sem.append(len(sem_candidates))
        candidate_counts_lex.append(len(lex_candidates))
        candidate_recall_sem.append(
            len(set(map(int, dense_rank)).intersection(map(int, sem_candidates)))
            / len(dense_rank)
        )
        candidate_recall_lex.append(
            len(set(map(int, bm25_rank)).intersection(map(int, lex_candidates)))
            / len(bm25_rank)
        )
        start = time.perf_counter()
        dual_rank = rrf(sem_rank, lex_rank, depth=100, alpha=0.5)
        dual_fusion_elapsed = time.perf_counter() - start
        query_latencies["dual_dpe_lsh"].append(
            sem_lsh_elapsed + lex_lsh_elapsed + dual_fusion_elapsed
        )
        rankings["dual_dpe_lsh"][qid] = dual_rank

        # Tail-orthogonal candidate generation ignores the MIPS-to-L2 residual
        # coordinate, then retains the same encrypted-L2 refinement and exact BM25 rerank.
        tail_start = time.perf_counter()
        tail_candidates, tail_trace = lexical_tail_index.candidates(
            tokenize(query), args.lexical_index_cap
        )
        tail_dist = np.linalg.norm(
            lexical_cipher[tail_candidates] - query_lex_cipher[qi], axis=1
        )
        tail_cloud = tail_candidates[top_indices(-tail_dist, args.candidate_depth)]
        tail_exact = bm25_scores[tail_cloud]
        tail_rank = tail_cloud[top_indices(tail_exact, min(100, len(tail_cloud)))]
        tail_elapsed = time.perf_counter() - tail_start
        query_latencies["lexical_dpe_tail_index"].append(tail_elapsed)
        rankings["lexical_dpe_tail_index"][qid] = tail_rank.tolist()
        candidate_counts_tail.append(len(tail_candidates))
        candidate_recall_tail.append(
            len(set(map(int, bm25_rank)).intersection(map(int, tail_candidates)))
            / len(bm25_rank)
        )
        positive_bm25_rank = top_indices(
            bm25_scores, min(100, int(np.count_nonzero(bm25_scores > 0)))
        )
        candidate_positive_recall_tail.append(
            len(set(map(int, positive_bm25_rank)).intersection(map(int, tail_candidates)))
            / len(positive_bm25_rank) if len(positive_bm25_rank) else 1.0
        )
        tail_query_tokens.append(tail_trace["query_feature_tokens"])
        tail_posting_reads.append(tail_trace["posting_entries_read"])
        sem_cloud_results.append(sem_ivf_cloud)
        lex_cloud_results.append(tail_cloud)

        start = time.perf_counter()
        dual_tail_rank = rrf(sem_ivf_rank, tail_rank, depth=100, alpha=0.5)
        dual_tail_fusion = time.perf_counter() - start
        query_latencies["dual_dpe_tail_index"].append(
            sem_ivf_elapsed + tail_elapsed + dual_tail_fusion
        )
        rankings["dual_dpe_tail_index"][qid] = dual_tail_rank

        start = time.perf_counter()
        sem_dp_candidates, _ = semantic_residual_ivf.candidates(dp_dense_queries[qi])
        lex_dp_candidates, _ = lexical_tail_index.candidates(
            dp_lexical_cover_terms[qi], args.lexical_index_cap
        )
        sem_dp_dist = np.linalg.norm(
            semantic_cipher[sem_dp_candidates] - dp_sem_cipher[qi], axis=1
        )
        lex_dp_dist = np.linalg.norm(
            lexical_cipher[lex_dp_candidates] - dp_lex_cipher[qi], axis=1
        )
        sem_dp_cloud = sem_dp_candidates[top_indices(-sem_dp_dist, args.candidate_depth)]
        lex_dp_cloud = lex_dp_candidates[top_indices(-lex_dp_dist, args.candidate_depth)]
        sem_dp_exact = dense_docs[sem_dp_cloud] @ dense_queries[qi]
        sem_dp_rank = sem_dp_cloud[top_indices(sem_dp_exact, min(100, len(sem_dp_cloud)))]
        lex_dp_exact = bm25_scores[lex_dp_cloud]
        lex_dp_rank = lex_dp_cloud[top_indices(lex_dp_exact, min(100, len(lex_dp_cloud)))]
        duet_rank = rrf(sem_dp_rank, lex_dp_rank, depth=100, alpha=0.5)
        query_latencies["duetdpe_tail_index_dp"].append(time.perf_counter() - start)
        rankings["duetdpe_tail_index_dp"][qid] = duet_rank

        pisces_result = pisces.retrieve(query, dense_queries[qi], depth=100)
        rankings["pisces_semantic_reimpl"][qid] = pisces_result.semantic_rank.tolist()
        rankings["pisces_lexical_reimpl"][qid] = pisces_result.lexical_rank.tolist()
        rankings["pisces_dual_reimpl"][qid] = pisces_result.dual_rank
        query_latencies["pisces_semantic_reimpl"].append(
            (pisces_result.timings_ms["coarse_filter"] + pisces_result.timings_ms["fine_cosine"])
            / 1000.0
        )
        query_latencies["pisces_lexical_reimpl"].append(
            (pisces_result.timings_ms["lpsi_equivalent"] + pisces_result.timings_ms["fixed_point_bm25"])
            / 1000.0
        )
        query_latencies["pisces_dual_reimpl"].append(
            pisces_result.timings_ms["total_emulator"] / 1000.0
        )
        pisces_candidate_counts.append(len(pisces_result.semantic_candidates))
        pisces_operation_counts.update(pisces_result.operation_counts)

    if args.rankings_output is not None:
        args.rankings_output.parent.mkdir(parents=True, exist_ok=True)

        def padded_rankings(method: str, depth: int = 100) -> np.ndarray:
            packed = np.full((len(query_ids), depth), -1, dtype=np.int32)
            for row, qid in enumerate(query_ids):
                values = np.asarray(rankings[method][qid][:depth], dtype=np.int32)
                packed[row, : len(values)] = values
            return packed

        np.savez_compressed(
            args.rankings_output,
            query_ids=np.asarray(query_ids, dtype=str),
            query_texts=np.asarray(query_texts, dtype=str),
            doc_ids=np.asarray(corpus_ids, dtype=str),
            semantic_dpe=padded_rankings("semantic_dpe_residual_ivf"),
            lexical_dpe=padded_rankings("lexical_dpe_tail_index"),
            semantic_plain=padded_rankings("dense_plain"),
            lexical_plain=padded_rankings("bm25_plain"),
            tail_query_tokens=np.asarray(tail_query_tokens, dtype=np.int32),
            tail_posting_reads=np.asarray(tail_posting_reads, dtype=np.int64),
            semantic_ivf_posting_reads=np.asarray(
                semantic_ivf_posting_reads, dtype=np.int64
            ),
        )

    metrics = {method: evaluate(rankings[method], query_ids, corpus_ids, qrels) for method in methods}
    latency_metrics = {
        method: {
            "mean_ms": float(np.mean(values) * 1000.0),
            "p50_ms": float(np.percentile(values, 50) * 1000.0),
            "p95_ms": float(np.percentile(values, 95) * 1000.0),
        }
        for method, values in query_latencies.items()
    }

    link_attack = cooccurrence_link_attack(
        sem_cloud_results, lex_cloud_results, sem_perm, lex_perm, attack_depth=20
    )
    privacy = {
        "naive_shared_identifier_direct_link_accuracy": 1.0,
        "unlinkable_alias_cooccurrence_attack": link_attack,
        "semantic_database_top10_neighbor_overlap": neighbor_overlap(
            dense_docs, semantic_cipher, sample_size=200, k=10, seed=args.seed + 7
        ),
        "lexical_database_top10_neighbor_overlap": neighbor_overlap(
            lexical_docs, lexical_cipher, sample_size=200, k=10, seed=args.seed + 8
        ),
    }

    comparison_pairs = {
        "semantic_full_vs_dense_plain": ("semantic_dpe_full", "dense_plain"),
        "lexical_full_vs_bm25_plain": ("lexical_dpe_full", "bm25_plain"),
        "dual_full_vs_hybrid_plain": ("dual_dpe_full", "hybrid_plain"),
        "semantic_lsh_vs_semantic_full": ("semantic_dpe_lsh", "semantic_dpe_full"),
        "semantic_residual_ivf_vs_semantic_full": ("semantic_dpe_residual_ivf", "semantic_dpe_full"),
        "semantic_residual_ivf_vs_legacy_lsh": ("semantic_dpe_residual_ivf", "semantic_dpe_lsh"),
        "legacy_lexical_lsh_vs_lexical_full": ("lexical_dpe_lsh", "lexical_dpe_full"),
        "tail_index_vs_legacy_lexical_lsh": ("lexical_dpe_tail_index", "lexical_dpe_lsh"),
        "tail_index_vs_lexical_full": ("lexical_dpe_tail_index", "lexical_dpe_full"),
        "dual_tail_index_vs_dual_full": ("dual_dpe_tail_index", "dual_dpe_full"),
        "query_perturbation_vs_dual_tail_index": ("duetdpe_tail_index_dp", "dual_dpe_tail_index"),
        "pisces_semantic_vs_dense_plain": ("pisces_semantic_reimpl", "dense_plain"),
        "pisces_lexical_vs_matching_plain": ("pisces_lexical_reimpl", "pisces_bm25_plain_adapter"),
        "pisces_dual_vs_matching_hybrid": ("pisces_dual_reimpl", "pisces_hybrid_plain_adapter"),
    }
    comparisons = {}
    for label, (method, reference) in comparison_pairs.items():
        comparisons[label] = {
            "method": method,
            "reference": reference,
            "ndcg10_retention": metrics[method]["nDCG@10"]
            / max(metrics[reference]["nDCG@10"], 1e-12),
            "ndcg10_delta": metrics[method]["nDCG@10"] - metrics[reference]["nDCG@10"],
            "recall100_delta": metrics[method]["Recall@100"]
            - metrics[reference]["Recall@100"],
            "latency_ratio": latency_metrics[method]["mean_ms"]
            / max(latency_metrics[reference]["mean_ms"], 1e-12),
        }

    communication = {
        "semantic_query_cipher_bytes": int(semantic_cipher.shape[1] * 4),
        "lexical_query_cipher_bytes": int(lexical_cipher.shape[1] * 4),
        "two_path_query_cipher_bytes": int((semantic_cipher.shape[1] + lexical_cipher.shape[1]) * 4),
        "candidate_depth_per_path": args.candidate_depth,
        "estimated_two_path_candidate_vector_bytes": int(
            args.candidate_depth * (semantic_cipher.shape[1] + lexical_cipher.shape[1]) * 4
        ),
        "tail_index_query_tokens_mean": float(np.mean(tail_query_tokens)),
        "tail_index_posting_entries_read_mean": float(np.mean(tail_posting_reads)),
        "semantic_ivf_probe_tokens": args.semantic_ivf_probes,
        "semantic_ivf_posting_entries_read_mean": float(np.mean(semantic_ivf_posting_reads)),
    }

    status = {
        "Pisces_official": {
            "status": "not_run",
            "reason": "Official implementation requires Linux/macOS, Bazel 6+, and C++17; current host is Windows without WSL.",
            "official_repository": "https://github.com/ant-intl/Pisces",
        },
        "Pisces_manual_reimplementation": {
            "status": "functional_reimplementation_run",
            "reason": "Protocols 1-4 are independently reimplemented at the retrieval-functionality level. HMAC/direct dictionaries and local arithmetic model ideal OPRF/OKVS/MPC/Top-K/PIR outputs; they do not provide production cryptographic security or comparable cryptographic latency.",
            "setup": pisces_setup,
        },
        "PRAG": {
            "status": "not_run",
            "reason": "No verified official implementation was found during this pilot; no surrogate is reported as PRAG.",
        },
        "ppRAG_CAPRISE": {
            "status": "implemented_behaviorally",
            "reason": "The semantic_dpe_full baseline implements bounded-noise conditional distance ranking behavior; it is not claimed to be the authors' code.",
        },
    }

    output = {
        "dataset": {
            "name": f"BEIR/{dataset_name}",
            "documents": len(corpus_ids),
            "test_queries": len(query_ids),
            "qrels": sum(len(x) for x in qrels.values()),
        },
        "dense_backend": "TF-IDF + TruncatedSVD(256)" if args.dense_backend == "lsa" else args.model,
        "dense_encoder_device": encoder_device,
        "configuration": vars(args) | {"data_dir": str(args.data_dir), "cache_dir": str(args.cache_dir), "results_dir": str(args.results_dir)},
        "offline_timings": asdict(timings),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
        },
        "retrieval_metrics": metrics,
        "comparisons": comparisons,
        "online_latency": latency_metrics,
        "candidate_counts": {
            "semantic_mean": float(np.mean(candidate_counts_sem)),
            "semantic_p95": float(np.percentile(candidate_counts_sem, 95)),
            "lexical_mean": float(np.mean(candidate_counts_lex)),
            "lexical_p95": float(np.percentile(candidate_counts_lex, 95)),
            "semantic_plain_top100_recall_mean": float(np.mean(candidate_recall_sem)),
            "semantic_residual_ivf_mean": float(np.mean(candidate_counts_sem_ivf)),
            "semantic_residual_ivf_p95": float(np.percentile(candidate_counts_sem_ivf, 95)),
            "semantic_residual_ivf_plain_top100_recall_mean": float(np.mean(candidate_recall_sem_ivf)),
            "semantic_residual_ivf_relevant_document_recall_mean": float(np.mean(candidate_relevant_recall_sem_ivf)),
            "semantic_residual_ivf_posting_entries_read_mean": float(np.mean(semantic_ivf_posting_reads)),
            "lexical_plain_top100_recall_mean": float(np.mean(candidate_recall_lex)),
            "tail_index_mean": float(np.mean(candidate_counts_tail)),
            "tail_index_p95": float(np.percentile(candidate_counts_tail, 95)),
            "tail_index_plain_top100_recall_mean": float(np.mean(candidate_recall_tail)),
            "tail_index_positive_bm25_top100_recall_mean": float(np.mean(candidate_positive_recall_tail)),
            "tail_index_query_tokens_mean": float(np.mean(tail_query_tokens)),
            "tail_index_posting_entries_read_mean": float(np.mean(tail_posting_reads)),
            "pisces_semantic_mean": float(np.mean(pisces_candidate_counts)),
            "pisces_semantic_p95": float(np.percentile(pisces_candidate_counts, 95)),
        },
        "lexical_tail_index_storage": lexical_tail_index.storage_summary(),
        "semantic_residual_ivf_setup": semantic_residual_ivf_setup,
        "semantic_residual_ivf_storage": semantic_residual_ivf.storage_summary(),
        "semantic_residual_ivf_variant": "mean-centered multi-assignment spherical IVF with HMAC-keyed cell labels",
        "lexical_tail_index_variant": "collision-free HMAC-keyed BM25 vocabulary postings; MIPS-to-L2 tail omitted",
        "pisces_operation_counts_total": dict(pisces_operation_counts),
        "privacy_metrics": privacy,
        "communication": communication,
        "external_baseline_status": status,
        "prototype_warning": "NumPy PCG64 is used for reproducible DPE noise; the lexical and semantic indexes leak keyed label equality, posting sizes, access patterns, and candidate overlap; and the Pisces implementation is a functional protocol emulator without production OKVS/OPRF/HE/GC/PIR security.",
    }

    with (args.results_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False, default=str)

    with (args.results_dir / "retrieval_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "nDCG@10", "MRR@10", "Recall@10", "Recall@100", "mean_ms", "p50_ms", "p95_ms"])
        for method in methods:
            writer.writerow(
                [
                    method,
                    metrics[method]["nDCG@10"],
                    metrics[method]["MRR@10"],
                    metrics[method]["Recall@10"],
                    metrics[method]["Recall@100"],
                    latency_metrics[method]["mean_ms"],
                    latency_metrics[method]["p50_ms"],
                    latency_metrics[method]["p95_ms"],
                ]
            )

    with (args.results_dir / "comparisons.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["comparison", "method", "reference", "nDCG@10 retention", "nDCG@10 delta", "Recall@100 delta", "latency ratio"]
        )
        for label, values in comparisons.items():
            writer.writerow(
                [
                    label,
                    values["method"],
                    values["reference"],
                    values["ndcg10_retention"],
                    values["ndcg10_delta"],
                    values["recall100_delta"],
                    values["latency_ratio"],
                ]
            )

    report_lines = [
        f"# DuetDPE {dataset_name} pilot results",
        "",
        f"Dataset: {len(corpus_ids):,} documents, {len(query_ids):,} test queries.",
        "",
        "| Method | nDCG@10 | MRR@10 | Recall@10 | Recall@100 | Mean ms | p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        m = metrics[method]
        t = latency_metrics[method]
        report_lines.append(
            f"| {method} | {m['nDCG@10']:.4f} | {m['MRR@10']:.4f} | {m['Recall@10']:.4f} | {m['Recall@100']:.4f} | {t['mean_ms']:.3f} | {t['p95_ms']:.3f} |"
        )
    report_lines.extend(
        [
            "",
            "## Relative comparisons",
            "",
            "| Comparison | nDCG@10 retention | nDCG@10 delta | Recall@100 delta | Latency ratio |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label, values in comparisons.items():
        report_lines.append(
            f"| {label} | {values['ndcg10_retention']:.4f} | {values['ndcg10_delta']:+.4f} | {values['recall100_delta']:+.4f} | {values['latency_ratio']:.2f}x |"
        )
    report_lines.extend(
        [
            "",
            "## Privacy observations",
            "",
            f"- Naive shared-ID direct linkage accuracy: {privacy['naive_shared_identifier_direct_link_accuracy']:.4f}",
            f"- Unlinkable-alias co-occurrence attack accuracy: {link_attack['top1_link_accuracy']:.4f} over {link_attack['evaluated_semantic_aliases']} observed aliases.",
            f"- Semantic encrypted database top-10 neighbor overlap: {privacy['semantic_database_top10_neighbor_overlap']:.4f}",
            f"- Lexical encrypted database top-10 neighbor overlap: {privacy['lexical_database_top10_neighbor_overlap']:.4f}",
            "",
            "## Candidate-index diagnostics",
            "",
            f"- Legacy semantic LSH candidate count: mean {np.mean(candidate_counts_sem):.1f}, p95 {np.percentile(candidate_counts_sem, 95):.1f}; plaintext top-100 coverage {np.mean(candidate_recall_sem):.4f}.",
            f"- Residual spherical IVF candidate count: mean {np.mean(candidate_counts_sem_ivf):.1f}, p95 {np.percentile(candidate_counts_sem_ivf, 95):.1f}; plaintext top-100 coverage {np.mean(candidate_recall_sem_ivf):.4f}, relevant-document coverage {np.mean(candidate_relevant_recall_sem_ivf):.4f}.",
            f"- Residual spherical IVF reads {np.mean(semantic_ivf_posting_reads):.1f} posting entries from {args.semantic_ivf_probes} keyed cells on average.",
            f"- Lexical candidate count: mean {np.mean(candidate_counts_lex):.1f}, p95 {np.percentile(candidate_counts_lex, 95):.1f}; plaintext top-100 coverage {np.mean(candidate_recall_lex):.4f}.",
            f"- Collision-free keyed BM25 tail-index candidate count: mean {np.mean(candidate_counts_tail):.1f}, p95 {np.percentile(candidate_counts_tail, 95):.1f}; padded plaintext top-100 coverage {np.mean(candidate_recall_tail):.4f}, positive-score top-100 coverage {np.mean(candidate_positive_recall_tail):.4f}.",
            f"- Tail index reads {np.mean(tail_posting_reads):.1f} posting entries from {np.mean(tail_query_tokens):.1f} keyed query-feature tokens on average.",
            f"- Pisces SimHash candidate count: mean {np.mean(pisces_candidate_counts):.1f}, p95 {np.percentile(pisces_candidate_counts, 95):.1f}.",
            "",
            "## Pisces implementation status",
            "",
            "- The manual implementation follows the retrieval functionality of Protocols 1-4 and includes a step-by-step labeled-PSI path checked against its accelerated equivalent path.",
            "- It does not instantiate real blind OPRF, OKVS, additive HE, garbled-circuit sorting, PIR-to-share, or network communication. Its latency is emulator latency, not a Pisces cryptographic benchmark.",
            "",
            "## Interpretation limits",
            "",
            "- Online ranking latency excludes query encoding, batched query encryption, and network transport; encoding and encryption are reported separately.",
            "- The DPE implementation reproduces ranking behavior but uses a seeded NumPy generator, not a production cryptographic PRF.",
            "- The official Pisces code was not run because it requires Linux/macOS and Bazel; the independently written functional reimplementation is reported under explicit reimpl method names.",
            "- This Python prototype is one component of a multi-dataset evaluation and is not, by itself, sufficient for a WWW submission.",
        ]
    )
    (args.results_dir / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
