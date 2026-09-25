"""Keyed residual spherical IVF for private semantic candidate generation.

The owner removes the corpus-wide common direction, trains spherical cells, and
assigns every document to a small number of cells.  The cloud stores only HMAC
cell labels and document aliases.  A query probes its nearest cells and the
returned union is refined by encrypted distance, then reranked exactly client-side.

This is a research prototype.  HMAC labels hide centroid identities, but posting
sizes, repeated label equality, access patterns, and candidate-set overlap leak.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np


def _normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norms, eps)).astype(np.float32)


def _top_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(max(int(k), 0), len(scores))
    if k == 0:
        return np.empty(0, dtype=np.int64)
    if k == len(scores):
        return np.argsort(-scores, kind="stable").astype(np.int64)
    selected = np.argpartition(-scores, k - 1)[:k]
    return selected[np.argsort(-scores[selected], kind="stable")].astype(np.int64)


@dataclass(frozen=True)
class SemanticCandidateTrace:
    probed_cells: int
    posting_entries_read: int
    unique_candidates: int
    largest_probed_posting: int


class KeyedResidualSphericalIVF:
    """Mean-centered, multi-assignment spherical IVF with keyed cell labels.

    Centroids and the corpus mean remain at the query owner.  The untrusted
    server receives a map from truncated HMAC labels to document aliases.  The
    prototype returns local integer indices so it can be evaluated end to end.
    """

    def __init__(
        self,
        key: bytes,
        n_clusters: int = 256,
        doc_assignments: int = 2,
        nprobe: int = 16,
        centered: bool = True,
        seed: int = 20260822,
        max_iter: int = 100,
        batch_size: int = 1024,
    ) -> None:
        if not key:
            raise ValueError("key must be non-empty")
        if n_clusters < 2:
            raise ValueError("n_clusters must be at least 2")
        if not 1 <= doc_assignments <= n_clusters:
            raise ValueError("doc_assignments must be in [1, n_clusters]")
        if not 1 <= nprobe <= n_clusters:
            raise ValueError("nprobe must be in [1, n_clusters]")
        self.key = bytes(key)
        self.n_clusters = int(n_clusters)
        self.doc_assignments = int(doc_assignments)
        self.nprobe = int(nprobe)
        self.centered = bool(centered)
        self.seed = int(seed)
        self.max_iter = int(max_iter)
        self.batch_size = int(batch_size)
        self.mean: np.ndarray | None = None
        self.centroids: np.ndarray | None = None
        self.postings: dict[bytes, np.ndarray] = {}
        self.n_documents = 0
        self.dimension = 0
        self.build_seconds = 0.0

    def _token(self, cluster_id: int) -> bytes:
        payload = b"semantic-residual-ivf-v1" + int(cluster_id).to_bytes(4, "big")
        return hmac.new(self.key, payload, hashlib.sha256).digest()[:16]

    def _residualize(self, vectors: np.ndarray) -> np.ndarray:
        values = np.asarray(vectors, dtype=np.float32)
        if values.ndim == 1:
            values = values[None, :]
        if self.mean is None:
            raise RuntimeError("index has not been built")
        if values.shape[1] != self.mean.shape[0]:
            raise ValueError("vector dimension does not match the index")
        if self.centered:
            values = values - self.mean[None, :]
        return _normalize_rows(values)

    def build(self, vectors: np.ndarray) -> dict[str, float | int | bool]:
        from sklearn.cluster import MiniBatchKMeans

        started = time.perf_counter()
        documents = _normalize_rows(np.asarray(vectors, dtype=np.float32))
        if documents.ndim != 2:
            raise ValueError("vectors must be a two-dimensional matrix")
        if len(documents) < self.n_clusters:
            raise ValueError("n_clusters cannot exceed the number of documents")
        self.n_documents, self.dimension = documents.shape
        self.mean = documents.mean(axis=0).astype(np.float32) if self.centered else np.zeros(
            self.dimension, dtype=np.float32
        )
        residuals = self._residualize(documents)

        # Euclidean k-means on unit residuals is equivalent to spherical
        # assignment; centroids are normalized again before probing.
        model = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            init="k-means++",
            n_init=3,
            max_iter=self.max_iter,
            batch_size=min(max(self.batch_size, self.n_clusters * 3), len(documents)),
            random_state=self.seed,
            reassignment_ratio=0.01,
        )
        model.fit(residuals)
        self.centroids = _normalize_rows(model.cluster_centers_)

        similarities = residuals @ self.centroids.T
        if self.doc_assignments == 1:
            assignments = np.argmax(similarities, axis=1)[:, None]
        else:
            assignments = np.argpartition(
                -similarities, self.doc_assignments - 1, axis=1
            )[:, : self.doc_assignments]
        temporary: dict[bytes, list[int]] = defaultdict(list)
        for doc_id, cells in enumerate(assignments):
            for cell in cells:
                temporary[self._token(int(cell))].append(doc_id)
        self.postings = {
            token: np.asarray(doc_ids, dtype=np.int32)
            for token, doc_ids in temporary.items()
        }
        self.build_seconds = time.perf_counter() - started
        sizes = np.asarray([len(posting) for posting in self.postings.values()], dtype=np.int64)
        return {
            "centered": self.centered,
            "n_clusters": self.n_clusters,
            "doc_assignments": self.doc_assignments,
            "posting_entries": int(sizes.sum()),
            "nonempty_cells": int(len(sizes)),
            "mean_posting_size": float(sizes.mean()),
            "largest_posting_size": int(sizes.max()),
            "build_seconds": self.build_seconds,
        }

    def candidates(
        self, query: np.ndarray, nprobe: int | None = None
    ) -> tuple[np.ndarray, SemanticCandidateTrace]:
        if self.centroids is None:
            raise RuntimeError("index has not been built")
        probes = self.nprobe if nprobe is None else int(nprobe)
        if not 1 <= probes <= self.n_clusters:
            raise ValueError("nprobe must be in [1, n_clusters]")
        residual = self._residualize(np.asarray(query, dtype=np.float32))[0]
        cells = _top_indices(self.centroids @ residual, probes)
        posting_lists = [self.postings.get(self._token(int(cell)), np.empty(0, dtype=np.int32)) for cell in cells]
        entries = sum(len(posting) for posting in posting_lists)
        if entries:
            found = np.unique(np.concatenate(posting_lists)).astype(np.int64)
        else:
            found = np.empty(0, dtype=np.int64)
        trace = SemanticCandidateTrace(
            probed_cells=probes,
            posting_entries_read=int(entries),
            unique_candidates=int(len(found)),
            largest_probed_posting=max((len(x) for x in posting_lists), default=0),
        )
        return found, trace

    def storage_summary(self) -> dict[str, int | float | bool]:
        if self.centroids is None or self.mean is None:
            raise RuntimeError("index has not been built")
        posting_entries = sum(len(posting) for posting in self.postings.values())
        return {
            "centered": self.centered,
            "n_clusters": self.n_clusters,
            "doc_assignments": self.doc_assignments,
            "nprobe": self.nprobe,
            "server_posting_entries": int(posting_entries),
            "server_estimated_bytes": int(posting_entries * 4 + len(self.postings) * 16),
            "client_centroid_and_mean_bytes": int(self.centroids.nbytes + self.mean.nbytes),
            "largest_posting_size": max((len(x) for x in self.postings.values()), default=0),
        }
