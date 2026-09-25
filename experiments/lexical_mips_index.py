"""Tail-orthogonal sparse index for BM25 MIPS-to-L2 candidates.

For transformed documents P(x)=[x/R, sqrt(1-||x/R||^2)] and query
Q(q)=[q/||q||, 0], the tail coordinate contributes zero to the MIPS score.
Generic Euclidean LSH nevertheless hashes that high-variance tail coordinate.
This index instead uses keyed signed-feature postings over x/R, accumulates a
quantized positive-overlap score, and leaves final ordering to encrypted L2
distance plus client BM25 reranking.

The construction is deliberately leakage-aware: the server observes equality
and access patterns of keyed feature buckets.  It is not claimed to provide the
same leakage profile as an oblivious ANN index.
"""

from __future__ import annotations

import hashlib
import hmac
from collections import defaultdict

import numpy as np


class KeyedBM25TailIndex:
    """Collision-free keyed sparse index for the BM25 MIPS block.

    Each vocabulary coordinate is replaced by HMAC_k(term).  The document
    posting weight is the BM25 document-side coordinate and the query weight is
    term frequency.  The MIPS-to-L2 tail is intentionally not indexed because
    every transformed query has a zero in that coordinate.  Scores are only a
    quantized candidate-generation proxy; final ranking remains encrypted L2
    plus client-side BM25 in the full protocol.
    """

    def __init__(self, key: bytes, quantization_bits: int = 16):
        if quantization_bits < 8 or quantization_bits > 16:
            raise ValueError("quantization_bits must be in [8, 16]")
        self.key = key
        self.levels = (1 << quantization_bits) - 1
        self.postings: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}
        self.n_docs = 0

    def _token(self, term: str) -> bytes:
        return hmac.new(self.key, term.encode("utf-8"), hashlib.sha256).digest()[:16]

    def build(
        self,
        postings: dict[str, tuple[np.ndarray, np.ndarray]],
        idf: dict[str, float],
        doc_len: np.ndarray,
        avgdl: float,
        n_docs: int,
        k1: float,
        b: float,
    ) -> None:
        self.n_docs = int(n_docs)
        # tf*(k1+1)/(tf+normalizer) is bounded by k1+1, so this is a
        # dataset-wide public quantization bound and requires no dense matrix.
        max_weight = max(max(idf.values(), default=1.0) * (k1 + 1.0), 1e-12)
        encoded: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}
        for term, (doc_ids, tf) in postings.items():
            normalizer = k1 * (1.0 - b + b * doc_len[doc_ids] / avgdl)
            weights = idf[term] * tf * (k1 + 1.0) / (tf + normalizer)
            quantized = np.clip(
                np.rint(weights / max_weight * self.levels), 1, self.levels
            ).astype(np.uint16)
            encoded[self._token(term)] = (doc_ids.astype(np.int32, copy=False), quantized)
        self.postings = encoded

    def candidates(self, query_terms: list[str], max_candidates: int) -> tuple[np.ndarray, dict[str, int]]:
        counts: dict[bytes, int] = defaultdict(int)
        for term in query_terms:
            counts[self._token(term)] += 1
        scores = np.zeros(self.n_docs, dtype=np.float32)
        touched = np.zeros(self.n_docs, dtype=np.bool_)
        posting_entries = 0
        matched_tokens = 0
        for token, query_tf in counts.items():
            posting = self.postings.get(token)
            if posting is None:
                continue
            doc_ids, weights = posting
            scores[doc_ids] += query_tf * weights.astype(np.float32)
            touched[doc_ids] = True
            posting_entries += len(doc_ids)
            matched_tokens += 1
        touched_ids = np.flatnonzero(touched)
        if len(touched_ids) > max_candidates:
            local = np.argpartition(-scores[touched_ids], max_candidates - 1)[:max_candidates]
            candidates = touched_ids[local]
            candidates = candidates[np.argsort(-scores[candidates], kind="stable")]
        else:
            candidates = touched_ids[np.argsort(-scores[touched_ids], kind="stable")]
        return candidates.astype(np.int64), {
            "query_feature_tokens": len(counts),
            "matched_feature_tokens": matched_tokens,
            "posting_entries_read": posting_entries,
            "uncapped_candidates": len(touched_ids),
        }

    def storage_summary(self) -> dict[str, int]:
        entries = sum(len(ids) for ids, _ in self.postings.values())
        return {
            "keyed_buckets": len(self.postings),
            "posting_entries": entries,
            "quantization_levels": self.levels,
            "estimated_payload_bytes": entries * 6 + len(self.postings) * 16,
        }


class TailOrthogonalSparseMIPSIndex:
    def __init__(self, key: bytes, quantization_bits: int = 8, zero_epsilon: float = 1e-10):
        if quantization_bits < 2 or quantization_bits > 16:
            raise ValueError("quantization_bits must be in [2, 16]")
        self.key = key
        self.levels = (1 << quantization_bits) - 1
        self.zero_epsilon = float(zero_epsilon)
        self.postings: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}
        self.dim = 0
        self.n_docs = 0
        self.max_abs = 1.0

    def _token(self, coordinate: int, sign: int) -> bytes:
        payload = coordinate.to_bytes(4, "big") + bytes([1 if sign > 0 else 0])
        return hmac.new(self.key, payload, hashlib.sha256).digest()[:16]

    def build(self, document_mips_block: np.ndarray) -> None:
        docs = np.asarray(document_mips_block, dtype=np.float32)
        if docs.ndim != 2:
            raise ValueError("document_mips_block must be a matrix")
        self.n_docs, self.dim = docs.shape
        self.max_abs = max(float(np.max(np.abs(docs))), self.zero_epsilon)
        temporary: dict[bytes, list[tuple[int, int]]] = defaultdict(list)
        for doc_idx in range(self.n_docs):
            nonzero = np.flatnonzero(np.abs(docs[doc_idx]) > self.zero_epsilon)
            for coordinate in nonzero:
                value = float(docs[doc_idx, coordinate])
                quantized = max(1, int(round(abs(value) / self.max_abs * self.levels)))
                temporary[self._token(int(coordinate), 1 if value > 0 else -1)].append(
                    (doc_idx, min(self.levels, quantized))
                )
        self.postings = {
            token: (
                np.fromiter((pair[0] for pair in pairs), dtype=np.int32),
                np.fromiter((pair[1] for pair in pairs), dtype=np.uint16),
            )
            for token, pairs in temporary.items()
        }

    def candidates(
        self, query_mips_block: np.ndarray, max_candidates: int
    ) -> tuple[np.ndarray, dict[str, int]]:
        query = np.asarray(query_mips_block, dtype=np.float32)
        if query.ndim != 1 or query.shape[0] != self.dim:
            raise ValueError("query dimension does not match index")
        scores = np.zeros(self.n_docs, dtype=np.float32)
        touched = np.zeros(self.n_docs, dtype=np.bool_)
        posting_entries = 0
        query_features = 0
        for coordinate in np.flatnonzero(np.abs(query) > self.zero_epsilon):
            value = float(query[coordinate])
            posting = self.postings.get(
                self._token(int(coordinate), 1 if value > 0 else -1)
            )
            query_features += 1
            if posting is None:
                continue
            doc_ids, quantized_weights = posting
            scores[doc_ids] += abs(value) * quantized_weights.astype(np.float32)
            touched[doc_ids] = True
            posting_entries += len(doc_ids)
        touched_ids = np.flatnonzero(touched)
        if len(touched_ids) > max_candidates:
            local = np.argpartition(-scores[touched_ids], max_candidates - 1)[:max_candidates]
            candidates = touched_ids[local]
            candidates = candidates[np.argsort(-scores[candidates], kind="stable")]
        else:
            candidates = touched_ids[np.argsort(-scores[touched_ids], kind="stable")]
        return candidates.astype(np.int64), {
            "query_feature_tokens": query_features,
            "posting_entries_read": posting_entries,
            "uncapped_candidates": len(touched_ids),
        }

    def storage_summary(self) -> dict[str, int]:
        posting_entries = sum(len(ids) for ids, _ in self.postings.values())
        return {
            "keyed_buckets": len(self.postings),
            "posting_entries": posting_entries,
            "quantization_levels": self.levels,
        }
