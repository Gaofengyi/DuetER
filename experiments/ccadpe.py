"""Compartmentalized distance-preserving retrieval prototype.

Each residual-IVF cell has an independent, rank-reducing projection and affine
mask.  Ciphertext distances are compared only inside a cell.  The server emits
a fixed number of candidates per probed cell and the client reranks their union
with the original embeddings.  Cell labels and access patterns still leak.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np

from semantic_index import KeyedResidualSphericalIVF, _top_indices


def _normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(x, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return (values / np.maximum(norms, eps)).astype(np.float32)


def _ball_noise(
    rng: np.random.Generator, rows: int, dim: int, radius: float
) -> np.ndarray:
    if radius <= 0.0:
        return np.zeros((rows, dim), dtype=np.float32)
    directions = _normalize_rows(rng.normal(size=(rows, dim)).astype(np.float32))
    radii = radius * np.power(rng.random(rows), 1.0 / dim)
    return (directions * radii[:, None]).astype(np.float32)


@dataclass(frozen=True)
class CompartmentTrace:
    probed_cells: int
    nonempty_cells: int
    posting_entries_read: int
    ciphertext_distances: int
    emitted_entries: int
    unique_candidates: int


class CompartmentDPEIndex:
    """Independent low-rank DPE coordinates for every keyed IVF cell."""

    def __init__(
        self,
        *,
        key: bytes,
        projection_dim: int = 64,
        beta: float = 0.10,
        scale: float = 3.0,
        seed: int = 20260826,
    ) -> None:
        if not key:
            raise ValueError("key must be non-empty")
        if projection_dim < 2:
            raise ValueError("projection_dim must be at least 2")
        self.key = bytes(key)
        self.projection_dim = int(projection_dim)
        self.beta = float(beta)
        self.scale = float(scale)
        self.seed = int(seed)
        self.routing: KeyedResidualSphericalIVF | None = None
        self.documents: np.ndarray | None = None
        self.assignments: np.ndarray | None = None
        self.postings: dict[int, np.ndarray] = {}
        self.projections: dict[int, np.ndarray] = {}
        self.means: dict[int, np.ndarray] = {}
        self.translations: dict[int, np.ndarray] = {}
        self.ciphertexts: dict[int, np.ndarray] = {}
        self.build_seconds = 0.0

    def _cell_seed(self, cell: int, purpose: bytes) -> int:
        digest = hashlib.sha256(
            self.key
            + purpose
            + int(self.seed).to_bytes(8, "big", signed=False)
            + int(cell).to_bytes(4, "big", signed=False)
        ).digest()
        return int.from_bytes(digest[:8], "big", signed=False)

    def _projection(self, cell: int, input_dim: int) -> np.ndarray:
        rng = np.random.default_rng(self._cell_seed(cell, b"projection"))
        gaussian = rng.normal(size=(input_dim, self.projection_dim)).astype(np.float32)
        q, _ = np.linalg.qr(gaussian, mode="reduced")
        # Scaling makes squared projected distances unbiased under a random
        # rank-r subspace; it does not restore the discarded null space.
        return (np.sqrt(input_dim / self.projection_dim) * q.T).astype(np.float32)

    def _route(self, vectors: np.ndarray, count: int) -> np.ndarray:
        if self.routing is None or self.routing.centroids is None:
            raise RuntimeError("index has not been built")
        residuals = self.routing._residualize(vectors)
        similarities = residuals @ self.routing.centroids.T
        if count == 1:
            return np.argmax(similarities, axis=1)[:, None].astype(np.int32)
        return np.argpartition(-similarities, count - 1, axis=1)[:, :count].astype(
            np.int32
        )

    def build(
        self,
        documents: np.ndarray,
        routing: KeyedResidualSphericalIVF,
    ) -> dict[str, float | int]:
        started = time.perf_counter()
        if routing.centroids is None or routing.mean is None:
            raise ValueError("routing index must already be built")
        docs = _normalize_rows(documents)
        if self.projection_dim > docs.shape[1]:
            raise ValueError("projection_dim cannot exceed input dimension")
        if routing.n_documents != len(docs):
            raise ValueError("routing index and document matrix disagree")
        self.routing = routing
        self.documents = docs
        self.assignments = self._route(docs, routing.doc_assignments)

        for cell in range(routing.n_clusters):
            doc_ids = np.flatnonzero(np.any(self.assignments == cell, axis=1)).astype(
                np.int32
            )
            if not len(doc_ids):
                continue
            projection = self._projection(cell, docs.shape[1])
            mean = docs[doc_ids].mean(axis=0).astype(np.float32)
            translation_rng = np.random.default_rng(
                self._cell_seed(cell, b"translation")
            )
            translation = translation_rng.normal(
                0.0, self.scale, self.projection_dim
            ).astype(np.float32)
            noise_rng = np.random.default_rng(self._cell_seed(cell, b"database-noise"))
            projected = (docs[doc_ids] - mean[None, :]) @ projection.T
            noise = _ball_noise(
                noise_rng,
                len(doc_ids),
                self.projection_dim,
                3.0 * self.scale * self.beta / 8.0,
            )
            ciphertext = self.scale * projected + translation[None, :] + noise
            self.postings[cell] = doc_ids
            self.projections[cell] = projection
            self.means[cell] = mean
            self.translations[cell] = translation
            self.ciphertexts[cell] = ciphertext.astype(np.float32)

        self.build_seconds = time.perf_counter() - started
        entries = sum(len(ids) for ids in self.postings.values())
        cipher_bytes = sum(values.nbytes for values in self.ciphertexts.values())
        return {
            "n_documents": len(docs),
            "input_dimension": docs.shape[1],
            "projection_dimension": self.projection_dim,
            "n_clusters": routing.n_clusters,
            "doc_assignments": routing.doc_assignments,
            "posting_entries": entries,
            "server_ciphertext_bytes": cipher_bytes,
            "server_posting_bytes": entries * 4,
            "client_projection_bytes": sum(x.nbytes for x in self.projections.values()),
            "client_mean_translation_bytes": sum(
                self.means[cell].nbytes + self.translations[cell].nbytes
                for cell in self.means
            ),
            "build_seconds": self.build_seconds,
        }

    def query_cells(self, query: np.ndarray, nprobe: int) -> np.ndarray:
        return self._route(np.asarray(query, dtype=np.float32), nprobe)[0]

    def encrypt_query_for_cell(
        self, query: np.ndarray, cell: int, nonce: int = 0
    ) -> np.ndarray:
        if cell not in self.projections:
            raise KeyError(f"empty or unknown cell {cell}")
        vector = _normalize_rows(np.asarray(query, dtype=np.float32))[0]
        projected = (vector - self.means[cell]) @ self.projections[cell].T
        rng = np.random.default_rng(
            self._cell_seed(cell, b"query-noise") ^ int(nonce)
        )
        noise = _ball_noise(
            rng,
            1,
            self.projection_dim,
            self.scale * self.beta / 8.0,
        )[0]
        return (
            self.scale * projected + self.translations[cell] + noise
        ).astype(np.float32)

    def candidates(
        self,
        query: np.ndarray,
        *,
        nprobe: int,
        top_per_cell: int,
        nonce: int = 0,
    ) -> tuple[np.ndarray, CompartmentTrace]:
        if top_per_cell < 1:
            raise ValueError("top_per_cell must be positive")
        cells = self.query_cells(query, nprobe)
        emitted: list[np.ndarray] = []
        reads = 0
        nonempty = 0
        for ordinal, raw_cell in enumerate(cells):
            cell = int(raw_cell)
            doc_ids = self.postings.get(cell)
            if doc_ids is None:
                continue
            nonempty += 1
            reads += len(doc_ids)
            query_cipher = self.encrypt_query_for_cell(
                query, cell, nonce=nonce * 1009 + ordinal
            )
            delta = self.ciphertexts[cell] - query_cipher[None, :]
            distances = np.einsum("ij,ij->i", delta, delta)
            local = _top_indices(-distances, min(top_per_cell, len(doc_ids)))
            emitted.append(doc_ids[local])
        raw = np.concatenate(emitted) if emitted else np.empty(0, dtype=np.int32)
        unique = np.unique(raw).astype(np.int64)
        return unique, CompartmentTrace(
            probed_cells=len(cells),
            nonempty_cells=nonempty,
            posting_entries_read=reads,
            ciphertext_distances=reads,
            emitted_entries=len(raw),
            unique_candidates=len(unique),
        )

    def primary_view(
        self, vectors: np.ndarray, *, query: bool, nonce_offset: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return primary cell IDs and their local coordinate views."""
        values = _normalize_rows(vectors)
        cells = self._route(values, 1)[:, 0]
        if query and any(int(cell) not in self.projections for cell in cells):
            # MiniBatchKMeans can leave a centroid without a primary document.
            # Route an external query to its nearest nonempty compartment.
            if self.routing is None or self.routing.centroids is None:
                raise RuntimeError("index has not been built")
            similarities = self.routing._residualize(values) @ self.routing.centroids.T
            nonempty = np.asarray(sorted(self.projections), dtype=np.int32)
            replacement = np.argmax(similarities[:, nonempty], axis=1)
            cells = nonempty[replacement]
        views = np.empty((len(values), self.projection_dim), dtype=np.float32)
        if query:
            for row, cell in enumerate(cells):
                views[row] = self.encrypt_query_for_cell(
                    values[row], int(cell), nonce=nonce_offset + row
                )
        else:
            if self.assignments is None:
                raise RuntimeError("index has not been built")
            positions = {
                cell: {int(doc): pos for pos, doc in enumerate(doc_ids)}
                for cell, doc_ids in self.postings.items()
            }
            for row, cell in enumerate(cells):
                views[row] = self.ciphertexts[int(cell)][positions[int(cell)][row]]
        return cells.astype(np.int32), views
