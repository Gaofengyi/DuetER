from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from ccadpe import CompartmentDPEIndex  # noqa: E402
from semantic_index import KeyedResidualSphericalIVF  # noqa: E402


class SmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        self.documents = rng.normal(size=(48, 12)).astype(np.float32)
        self.documents /= np.linalg.norm(self.documents, axis=1, keepdims=True)

    def test_keyed_routing_is_deterministic(self) -> None:
        first = KeyedResidualSphericalIVF(
            key=b"routing-key", n_clusters=4, doc_assignments=2, nprobe=2, seed=11
        )
        second = KeyedResidualSphericalIVF(
            key=b"routing-key", n_clusters=4, doc_assignments=2, nprobe=2, seed=11
        )
        first.build(self.documents)
        second.build(self.documents)
        left, _ = first.candidates(self.documents[0])
        right, _ = second.candidates(self.documents[0])
        np.testing.assert_array_equal(left, right)
        self.assertTrue(all(len(label) == 16 for label in first.postings))

    def test_compartment_query_returns_bounded_unique_union(self) -> None:
        routing = KeyedResidualSphericalIVF(
            key=b"routing-key", n_clusters=4, doc_assignments=2, nprobe=2, seed=13
        )
        routing.build(self.documents)
        index = CompartmentDPEIndex(
            key=b"ccadpe-key", projection_dim=6, beta=0.0, scale=3.0, seed=17
        )
        metadata = index.build(self.documents, routing)
        candidates, trace = index.candidates(
            self.documents[0], nprobe=2, top_per_cell=3, nonce=0
        )
        self.assertEqual(metadata["projection_dimension"], 6)
        self.assertLessEqual(len(candidates), 6)
        self.assertEqual(len(candidates), len(np.unique(candidates)))
        self.assertEqual(trace.ciphertext_distances, trace.posting_entries_read)

    def test_independent_keys_change_local_coordinates(self) -> None:
        routing = KeyedResidualSphericalIVF(
            key=b"routing-key", n_clusters=4, doc_assignments=1, nprobe=1, seed=19
        )
        routing.build(self.documents)
        first = CompartmentDPEIndex(
            key=b"path-one", projection_dim=6, beta=0.0, scale=3.0, seed=23
        )
        second = CompartmentDPEIndex(
            key=b"path-two", projection_dim=6, beta=0.0, scale=3.0, seed=23
        )
        first.build(self.documents, routing)
        second.build(self.documents, routing)
        cells_one, view_one = first.primary_view(self.documents, query=False)
        cells_two, view_two = second.primary_view(self.documents, query=False)
        np.testing.assert_array_equal(cells_one, cells_two)
        self.assertFalse(np.allclose(view_one, view_two))


if __name__ == "__main__":
    unittest.main()

