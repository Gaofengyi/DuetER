"""Shared utilities used by the DuetER implementation and evaluations.

This module intentionally contains no implementation of an external baseline.
"""

from __future__ import annotations

import csv
import hashlib
import hmac
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


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
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def load_beir(
    data_dir: Path, split: str = "test"
) -> tuple[list[str], list[str], list[str], list[str], dict[str, dict[str, int]]]:
    corpus_rows = load_jsonl(data_dir / "corpus.jsonl")
    query_rows = load_jsonl(data_dir / "queries.jsonl")
    corpus_ids = [str(row["_id"]) for row in corpus_rows]
    corpus_texts = [
        f"{row.get('title', '')}. {row.get('text', '')}".strip()
        for row in corpus_rows
    ]
    query_lookup = {str(row["_id"]): row["text"] for row in query_rows}
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with (data_dir / "qrels" / f"{split}.tsv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            qrels[str(row["query-id"])][str(row["corpus-id"])] = int(row["score"])
    query_ids = [qid for qid in query_lookup if qid in qrels]
    query_texts = [query_lookup[qid] for qid in query_ids]
    return corpus_ids, corpus_texts, query_ids, query_texts, dict(qrels)


def load_scifact(
    data_dir: Path,
) -> tuple[list[str], list[str], list[str], list[str], dict[str, dict[str, int]]]:
    return load_beir(data_dir)


class BM25Index:
    def __init__(self, documents: list[str], k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(text) for text in documents]
        self.doc_len = np.asarray(
            [len(tokens) for tokens in self.doc_tokens], dtype=np.float32
        )
        self.avgdl = float(np.mean(self.doc_len))
        self.n_docs = len(documents)
        self.term_counts: list[Counter[str]] = [
            Counter(tokens) for tokens in self.doc_tokens
        ]
        df: Counter[str] = Counter()
        for counts in self.term_counts:
            df.update(counts.keys())
        self.idf = {
            term: math.log(1.0 + (self.n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }
        temporary: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc_idx, counts in enumerate(self.term_counts):
            for term, tf in counts.items():
                temporary[term].append((doc_idx, tf))
        self.postings = {
            term: (
                np.asarray([pair[0] for pair in pairs], dtype=np.int32),
                np.asarray([pair[1] for pair in pairs], dtype=np.float32),
            )
            for term, pairs in temporary.items()
        }

    def score(self, query: str) -> np.ndarray:
        scores = np.zeros(self.n_docs, dtype=np.float32)
        for term, query_count in Counter(tokenize(query)).items():
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
        digest = hashlib.blake2b(
            term.encode("utf-8"), key=seed, digest_size=16
        ).digest()
        return int.from_bytes(digest[:8], "little") % dim, (1.0 if digest[8] & 1 else -1.0)

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
            if term in self.idf:
                bucket, sign = self._feature(term, dim, seed)
                vector[bucket] += sign * float(count)
        return vector


def mips_document_transform(x: np.ndarray) -> tuple[np.ndarray, float]:
    scale = float(np.max(np.linalg.norm(x, axis=1))) * (1.0 + 1e-6)
    first = x / max(scale, 1e-12)
    tail = np.sqrt(
        np.maximum(0.0, 1.0 - np.sum(first * first, axis=1, keepdims=True))
    )
    return np.concatenate([first, tail], axis=1).astype(np.float32), scale


def mips_query_transform(y: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [normalize_vector(y), np.zeros(1, dtype=np.float32)]
    ).astype(np.float32)


def fwht_batch(x: np.ndarray) -> np.ndarray:
    """Normalized Walsh-Hadamard transform over the final dimension."""
    y = np.asarray(x, dtype=np.float32).copy()
    n = y.shape[1]
    if n & (n - 1):
        raise ValueError("FWHT dimension must be a power of two")
    h = 1
    while h < n:
        y = y.reshape(y.shape[0], -1, 2 * h)
        left = y[:, :, :h].copy()
        right = y[:, :, h : 2 * h].copy()
        y[:, :, :h] = left + right
        y[:, :, h : 2 * h] = left - right
        y = y.reshape(y.shape[0], n)
        h *= 2
    return y / math.sqrt(n)


class ConditionalDPE:
    """Reproducible experimental scale-and-perturb transform."""

    def __init__(self, input_dim: int, beta: float, scale: float, seed: int):
        self.input_dim = input_dim
        self.work_dim = 1 << (input_dim - 1).bit_length()
        self.beta = float(beta)
        self.scale = float(scale)
        rng = np.random.default_rng(seed)
        signs = np.asarray([-1.0, 1.0], dtype=np.float32)
        self.sign1 = rng.choice(signs, self.work_dim)
        self.sign2 = rng.choice(signs, self.work_dim)
        self.permutation = rng.permutation(self.work_dim)
        self.db_rng = np.random.default_rng(seed + 1009)
        self.query_rng = np.random.default_rng(seed + 2027)

    def _pad(self, x: np.ndarray) -> np.ndarray:
        values = x[None, :] if x.ndim == 1 else x
        if values.shape[1] == self.work_dim:
            return np.asarray(values, dtype=np.float32)
        output = np.zeros((values.shape[0], self.work_dim), dtype=np.float32)
        output[:, : values.shape[1]] = values
        return output

    def transform(self, x: np.ndarray) -> np.ndarray:
        values = fwht_batch(self._pad(x) * self.sign1)
        return (values * self.sign2)[:, self.permutation]

    @staticmethod
    def _ball_noise(
        rng: np.random.Generator, rows: int, dim: int, radius: float
    ) -> np.ndarray:
        directions = normalize_rows(rng.normal(size=(rows, dim)).astype(np.float32))
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
    def __init__(
        self,
        dim: int,
        tables: int,
        projections: int,
        width: float,
        seed: int,
        key: bytes,
    ) -> None:
        if projections < 1 or width <= 0:
            raise ValueError("projections and width must be positive")
        self.tables = tables
        self.projections = projections
        self.width = float(width)
        self.key = key
        rng = np.random.default_rng(seed)
        self.planes = rng.normal(size=(tables, projections, dim)).astype(np.float32)
        self.offsets = rng.uniform(0.0, width, size=(tables, projections)).astype(np.float32)
        self.buckets: list[dict[bytes, list[int]]] = [
            defaultdict(list) for _ in range(tables)
        ]

    def _token(self, table: int, code: np.ndarray) -> bytes:
        payload = table.to_bytes(2, "big") + np.asarray(code, dtype=">i8").tobytes()
        return hmac.new(self.key, payload, hashlib.sha256).digest()[:16]

    def _codes(self, vectors: np.ndarray, table: int) -> tuple[np.ndarray, np.ndarray]:
        positions = (vectors @ self.planes[table].T + self.offsets[table]) / self.width
        codes = np.floor(positions).astype(np.int64)
        return codes, positions - codes

    def build(self, vectors: np.ndarray) -> None:
        for table in range(self.tables):
            codes, _ = self._codes(vectors, table)
            for doc_idx, code in enumerate(codes):
                self.buckets[table][self._token(table, code)].append(doc_idx)

    def candidates(self, query: np.ndarray, probes_per_table: int) -> np.ndarray:
        found: set[int] = set()
        for table in range(self.tables):
            codes, fractions = self._codes(query[None, :], table)
            code = codes[0]
            probes = [code]
            neighbors: list[tuple[float, int, int]] = []
            for coordinate, fraction in enumerate(fractions[0]):
                neighbors.extend(
                    [(float(fraction), coordinate, -1), (float(1.0 - fraction), coordinate, 1)]
                )
            neighbors.sort(key=lambda item: item[0])
            for _, coordinate, direction in neighbors[: max(0, probes_per_table - 1)]:
                adjacent = code.copy()
                adjacent[coordinate] += direction
                probes.append(adjacent)
            for probe in probes:
                found.update(self.buckets[table].get(self._token(table, probe), ()))
        return np.fromiter(found, dtype=np.int64)


def rrf(
    rank_a: Iterable[int], rank_b: Iterable[int], depth: int, alpha: float, c: int = 60
) -> list[int]:
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
    for query_id in query_ids:
        relevant = qrels[query_id]
        ranked_ids = [
            str(value)
            for value in doc_id_by_idx[
                np.asarray(rankings[query_id], dtype=np.int64)
            ]
        ]
        gains = [relevant.get(doc_id, 0) for doc_id in ranked_ids[:10]]
        dcg = sum(
            (2**gain - 1) / math.log2(rank + 2)
            for rank, gain in enumerate(gains)
        )
        ideal = sorted(relevant.values(), reverse=True)[:10]
        idcg = sum(
            (2**gain - 1) / math.log2(rank + 2)
            for rank, gain in enumerate(ideal)
        )
        ndcg10.append(dcg / idcg if idcg else 0.0)
        reciprocal = next(
            (
                1.0 / rank
                for rank, doc_id in enumerate(ranked_ids[:10], start=1)
                if relevant.get(doc_id, 0) > 0
            ),
            0.0,
        )
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


def neighbor_overlap(
    plain: np.ndarray, encrypted: np.ndarray, sample_size: int, k: int, seed: int
) -> float:
    rng = np.random.default_rng(seed)
    sample = rng.choice(plain.shape[0], min(sample_size, plain.shape[0]), replace=False)
    overlaps = []
    for idx in sample:
        plain_rank = [
            int(value)
            for value in top_indices(plain @ plain[idx], k + 1)
            if int(value) != int(idx)
        ][:k]
        encrypted_rank = [
            int(value)
            for value in np.argsort(np.linalg.norm(encrypted - encrypted[idx], axis=1))
            if int(value) != int(idx)
        ][:k]
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
    for semantic_docs, lexical_docs in zip(semantic_results, lexical_results):
        for semantic_alias in sem_alias[semantic_docs[:attack_depth]]:
            counts[int(semantic_alias)].update(
                int(value) for value in lex_alias[lexical_docs[:attack_depth]]
            )
    inverse_semantic = {int(alias): idx for idx, alias in enumerate(sem_alias)}
    correct = 0
    for semantic_alias, counter in counts.items():
        if counter:
            predicted = counter.most_common(1)[0][0]
            correct += int(
                predicted == int(lex_alias[inverse_semantic[semantic_alias]])
            )
    evaluated = sum(bool(counter) for counter in counts.values())
    return {
        "evaluated_semantic_aliases": evaluated,
        "top1_link_accuracy": correct / evaluated if evaluated else 0.0,
    }

