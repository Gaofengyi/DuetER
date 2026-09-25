"""Independent research reimplementation of the Pisces retrieval functionality.

This module follows Pisces Protocols 1--4 at the *functional* level:

* SimHash plus 160 masked projections and a two-match oblivious-filter rule;
* coarse-to-fine cosine retrieval;
* reusable keyed storage and one ideal-OPRF call per unique query token;
* multi-instance labeled PSI term-frequency recovery;
* fixed-point/additive-share score reconstruction and private-top-k semantics.

It does NOT instantiate production OKVS, blind OPRF, additive HE, garbled
circuits, PIR-to-share, or a networked two-party runtime.  The direct dictionary
models OKVS functionality and HMAC models ideal PRF outputs.  Therefore its
retrieval accuracy is comparable, while its Python latency is only an emulator
measurement and must never be reported as Pisces cryptographic runtime.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np


def _hmac(key: bytes, payload: bytes, length: int = 32) -> bytes:
    return hmac.new(key, payload, hashlib.sha256).digest()[:length]


def _xor(left: bytes, right: bytes) -> bytes:
    return bytes(a ^ b for a, b in zip(left, right))


def additive_share_uint64(values: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    encoded = np.asarray(values, dtype=np.uint64)
    first = rng.integers(0, np.iinfo(np.uint64).max, encoded.shape, dtype=np.uint64)
    second = encoded - first
    return first, second


def reconstruct_uint64(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.asarray(first, dtype=np.uint64) + np.asarray(second, dtype=np.uint64)


class PiscesSimHashFilter:
    """Functional output of Pisces Protocol 3 without its HE/OKVS privacy."""

    def __init__(
        self,
        dimension: int,
        simhash_bits: int = 32,
        projection_num: int = 160,
        projection_weight: int = 16,
        min_matches: int = 2,
        min_candidate_fraction: float = 0.15,
        initial_hamming_radius: int = 5,
        maximum_hamming_radius: int = 10,
        seed: int = 20260823,
    ):
        if projection_weight > simhash_bits:
            raise ValueError("projection_weight exceeds SimHash length")
        self.simhash_bits = simhash_bits
        self.projection_num = projection_num
        self.projection_weight = projection_weight
        self.min_matches = min_matches
        self.min_candidate_fraction = float(min_candidate_fraction)
        self.initial_hamming_radius = int(initial_hamming_radius)
        self.maximum_hamming_radius = int(maximum_hamming_radius)
        rng = np.random.default_rng(seed)
        self.hyperplanes = rng.normal(size=(simhash_bits, dimension)).astype(np.float32)
        self.masks = np.stack(
            [
                np.sort(rng.choice(simhash_bits, projection_weight, replace=False))
                for _ in range(projection_num)
            ]
        )
        # The released C++ computes ``digest | mask``.  Therefore mask bits are
        # discarded (forced to one) and equality is tested on the complement.
        all_coordinates = np.arange(simhash_bits)
        self.projection_coordinates = np.stack(
            [np.setdiff1d(all_coordinates, mask, assume_unique=True) for mask in self.masks]
        )
        self.buckets: list[dict[int, np.ndarray]] = []
        self.document_bits: np.ndarray | None = None

    @staticmethod
    def _codes(bits: np.ndarray) -> np.ndarray:
        width = bits.shape[1]
        weights = (np.uint64(1) << np.arange(width, dtype=np.uint64))[None, :]
        return np.sum(bits.astype(np.uint64) * weights, axis=1, dtype=np.uint64)

    def build(self, documents: np.ndarray) -> None:
        self.document_bits = np.asarray(documents @ self.hyperplanes.T >= 0, dtype=np.bool_)
        self.buckets = []
        for coordinates in self.projection_coordinates:
            codes = self._codes(self.document_bits[:, coordinates])
            temporary: dict[int, list[int]] = defaultdict(list)
            for doc_idx, code in enumerate(codes):
                temporary[int(code)].append(doc_idx)
            self.buckets.append(
                {
                    code: np.asarray(doc_ids, dtype=np.int32)
                    for code, doc_ids in temporary.items()
                }
            )

    def candidates(self, query: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
        if self.document_bits is None:
            raise RuntimeError("index is not built")
        query_bits = np.asarray(query @ self.hyperplanes.T >= 0, dtype=np.bool_)
        counts = np.zeros(self.document_bits.shape[0], dtype=np.uint16)
        posting_entries = 0
        for coordinates, buckets in zip(self.projection_coordinates, self.buckets):
            code = int(self._codes(query_bits[None, coordinates])[0])
            docs = buckets.get(code)
            if docs is not None:
                counts[docs] += 1
                posting_entries += len(docs)
        seed_candidates = np.flatnonzero(counts >= self.min_matches)
        if len(seed_candidates) == 0:
            candidates = seed_candidates
            used_radius = self.initial_hamming_radius
        else:
            seed_bits = self.document_bits[seed_candidates]
            minimum_distances = np.full(self.document_bits.shape[0], self.simhash_bits, dtype=np.uint8)
            # Official Pisces expands neighborhoods around every fuzzy-PSI output.
            # Chunking avoids allocating N x |seed_candidates| x L at once.
            for offset in range(0, len(seed_bits), 64):
                block = seed_bits[offset : offset + 64]
                distances = np.count_nonzero(
                    self.document_bits[:, None, :] != block[None, :, :], axis=2
                )
                minimum_distances = np.minimum(
                    minimum_distances, np.min(distances, axis=1).astype(np.uint8)
                )
            target = math.ceil(self.document_bits.shape[0] * self.min_candidate_fraction)
            used_radius = self.initial_hamming_radius
            candidates = np.flatnonzero(minimum_distances <= used_radius)
            while len(candidates) < target and used_radius < self.maximum_hamming_radius:
                used_radius += 1
                candidates = np.flatnonzero(minimum_distances <= used_radius)
        return candidates.astype(np.int64), {
            "projection_tokens": self.projection_num,
            "posting_entries_read": posting_entries,
            "candidates": len(candidates),
            "fuzzy_seed_digests": len(seed_candidates),
            "expanded_hamming_radius": used_radius,
        }


class MultiInstanceLPSIReimplementation:
    """Protocol-4 functionality with explicit slow and accelerated paths."""

    def __init__(self, term_counts: list[Counter[str]], seed: int = 20260823):
        self.term_counts = term_counts
        self.n_docs = len(term_counts)
        seed_bytes = seed.to_bytes(16, "big")
        self.prf_key = _hmac(seed_bytes, b"pisces-oprf")
        self.kdf0_key = _hmac(seed_bytes, b"pisces-kdf0")
        self.kdf1_key = _hmac(seed_bytes, b"pisces-kdf1")
        self.miss_key = _hmac(seed_bytes, b"pisces-okvs-miss")
        self.okvs_functionality: dict[bytes, bytes] = {}
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _oprf(self, term: str) -> bytes:
        return _hmac(self.prf_key, term.encode("utf-8"))

    def _storage_key(self, doc_idx: int, oprf_value: bytes) -> bytes:
        return _hmac(self.kdf0_key, doc_idx.to_bytes(4, "big") + oprf_value, 16)

    def _label_mask(self, doc_idx: int, oprf_value: bytes) -> bytes:
        return _hmac(self.kdf1_key, doc_idx.to_bytes(4, "big") + oprf_value, 4)

    def build(self) -> dict[str, int]:
        postings_temp: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc_idx, counts in enumerate(self.term_counts):
            for term, tf in counts.items():
                oprf_value = self._oprf(term)
                key = self._storage_key(doc_idx, oprf_value)
                plaintext = b"\x00\x00" + min(int(tf), 65535).to_bytes(2, "big")
                self.okvs_functionality[key] = _xor(
                    plaintext, self._label_mask(doc_idx, oprf_value)
                )
                postings_temp[term].append((doc_idx, int(tf)))
        self.postings = {
            term: (
                np.fromiter((item[0] for item in items), dtype=np.int32),
                np.fromiter((item[1] for item in items), dtype=np.uint16),
            )
            for term, items in postings_temp.items()
        }
        return {
            "okvs_key_value_pairs": len(self.okvs_functionality),
            "vocabulary": len(self.postings),
        }

    def query_slow_protocol(self, unique_terms: list[str]) -> tuple[np.ndarray, dict[str, int]]:
        """Execute every OPRF/KDF/dictionary-decode step from Protocol 4."""
        result = np.zeros((self.n_docs, len(unique_terms)), dtype=np.uint16)
        for term_idx, term in enumerate(unique_terms):
            oprf_value = self._oprf(term)
            for doc_idx in range(self.n_docs):
                key = self._storage_key(doc_idx, oprf_value)
                cipher = self.okvs_functionality.get(key, _hmac(self.miss_key, key, 4))
                plaintext = _xor(cipher, self._label_mask(doc_idx, oprf_value))
                if plaintext[:2] == b"\x00\x00":
                    result[doc_idx, term_idx] = int.from_bytes(plaintext[2:], "big")
        return result, {
            "oprf_calls": len(unique_terms),
            "okvs_decodes": self.n_docs * len(unique_terms),
        }

    def query_equivalent(self, unique_terms: list[str]) -> tuple[np.ndarray, dict[str, int]]:
        """Vectorized output-equivalent path used for the full evaluation."""
        result = np.zeros((self.n_docs, len(unique_terms)), dtype=np.uint16)
        for term_idx, term in enumerate(unique_terms):
            posting = self.postings.get(term)
            if posting is not None:
                doc_ids, frequencies = posting
                result[doc_ids, term_idx] = frequencies
        return result, {
            "oprf_calls": len(unique_terms),
            "okvs_decodes": self.n_docs * len(unique_terms),
        }


@dataclass
class PiscesQueryResult:
    semantic_rank: np.ndarray
    lexical_rank: np.ndarray
    dual_rank: list[int]
    semantic_candidates: np.ndarray
    timings_ms: dict[str, float]
    operation_counts: dict[str, int]


class PiscesResearchReimplementation:
    def __init__(
        self,
        dense_documents: np.ndarray,
        term_counts: list[Counter[str]],
        document_lengths: np.ndarray,
        average_document_length: float,
        tokenizer,
        k1: float = 1.5,
        b: float = 0.75,
        seed: int = 20260823,
    ):
        self.dense_documents = np.asarray(dense_documents, dtype=np.float32)
        self.term_counts = term_counts
        self.document_lengths = np.asarray(document_lengths, dtype=np.float32)
        self.average_document_length = float(average_document_length)
        self.tokenizer = tokenizer
        self.k1 = float(k1)
        self.b = float(b)
        self.filter = PiscesSimHashFilter(self.dense_documents.shape[1], seed=seed)
        self.lpsi = MultiInstanceLPSIReimplementation(term_counts, seed=seed + 1)
        self.rng = np.random.default_rng(seed + 2)

    def build(self) -> dict[str, object]:
        start = time.perf_counter()
        self.filter.build(self.dense_documents)
        semantic_seconds = time.perf_counter() - start
        start = time.perf_counter()
        lexical = self.lpsi.build()
        lexical_seconds = time.perf_counter() - start
        return {
            "simhash_filter_seconds": semantic_seconds,
            "lpsi_setup_seconds": lexical_seconds,
            "lpsi": lexical,
            "security_status": "functional emulator; not a cryptographic Pisces deployment",
        }

    @staticmethod
    def _top(scores: np.ndarray, k: int) -> np.ndarray:
        k = min(k, len(scores))
        if k == 0:
            return np.empty(0, dtype=np.int64)
        selected = np.argpartition(-scores, k - 1)[:k]
        return selected[np.argsort(-scores[selected], kind="stable")]

    @staticmethod
    def _rrf(first: np.ndarray, second: np.ndarray, depth: int = 100, c: int = 60) -> list[int]:
        scores: dict[int, float] = defaultdict(float)
        for rank, doc in enumerate(first[:depth], 1):
            scores[int(doc)] += 0.5 / (c + rank)
        for rank, doc in enumerate(second[:depth], 1):
            scores[int(doc)] += 0.5 / (c + rank)
        return [doc for doc, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]

    def retrieve(self, query_text: str, dense_query: np.ndarray, depth: int = 100) -> PiscesQueryResult:
        start = time.perf_counter()
        semantic_candidates, semantic_trace = self.filter.candidates(dense_query)
        coarse_ms = (time.perf_counter() - start) * 1000.0
        start = time.perf_counter()
        semantic_scores = self.dense_documents[semantic_candidates] @ dense_query
        semantic_local = self._top(semantic_scores, depth)
        semantic_rank = semantic_candidates[semantic_local]
        semantic_fine_ms = (time.perf_counter() - start) * 1000.0

        # Released LpsiReceiverProcess stores query tokens in unordered_map,
        # so duplicate tokens collapse before BM25 computation.
        unique_terms = list(dict.fromkeys(self.tokenizer(query_text)))
        start = time.perf_counter()
        frequencies, lexical_trace = self.lpsi.query_equivalent(unique_terms)
        lpsi_ms = (time.perf_counter() - start) * 1000.0
        start = time.perf_counter()
        lexical_scores = np.zeros(len(self.term_counts), dtype=np.float64)
        for term_idx, term in enumerate(unique_terms):
            tf = frequencies[:, term_idx].astype(np.float64)
            df = int(np.count_nonzero(tf))
            if df == 0:
                continue
            idf = math.log(1.0 + (len(tf) - df + 0.5) / (df + 0.5))
            denominator = tf + self.k1 * (
                1.0 - self.b
                + self.b * self.document_lengths / self.average_document_length
            )
            lexical_scores += (
                idf
                * tf
                * (self.k1 + 1.0)
                / np.maximum(denominator, 1e-12)
            )
        # Official SPU uses 18 fractional bits and shifts by truncate_bit=8
        # before GC Top-K, giving an effective 10-bit fractional score.
        fixed = np.rint(lexical_scores * (1 << 10)).astype(np.uint64)
        share_server, share_client = additive_share_uint64(fixed, self.rng)
        reconstructed = reconstruct_uint64(share_server, share_client).astype(np.float64)
        lexical_rank = self._top(reconstructed, depth)
        lexical_score_ms = (time.perf_counter() - start) * 1000.0
        start = time.perf_counter()
        dual_rank = self._rrf(semantic_rank, lexical_rank, depth=depth)
        fusion_ms = (time.perf_counter() - start) * 1000.0
        return PiscesQueryResult(
            semantic_rank=semantic_rank,
            lexical_rank=lexical_rank,
            dual_rank=dual_rank,
            semantic_candidates=semantic_candidates,
            timings_ms={
                "coarse_filter": coarse_ms,
                "fine_cosine": semantic_fine_ms,
                "lpsi_equivalent": lpsi_ms,
                "fixed_point_bm25": lexical_score_ms,
                "fusion": fusion_ms,
                "total_emulator": coarse_ms + semantic_fine_ms + lpsi_ms + lexical_score_ms + fusion_ms,
            },
            operation_counts=semantic_trace | lexical_trace,
        )
