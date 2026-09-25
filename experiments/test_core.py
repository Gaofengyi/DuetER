import sys
import tempfile
import unittest
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if (ROOT / ".deps").exists():
    sys.path.insert(0, str(ROOT / ".deps"))

import numpy as np

from benchmark_keyed_fts5_scale import KeyedTerms
from benchmark_exact_bm25_client_rerank import exact_bm25_rerank
from benchmark_budgeted_lexical_candidates import select_terms_under_budget
from benchmark_budgeted_lexical_dpe import build_candidate_scoped_cache
from benchmark_million_lexical_dpe import (
    build_ciphertexts as build_lexical_ciphertexts,
    build_plain_sketches,
    encrypt_query_matrix,
    keyed_term as lexical_keyed_term,
    query_mips,
)
from benchmark_million_semantic_hybrid import (
    dpe_transform,
    evaluate as evaluate_million,
    load_queries_qrels,
)
from benchmark_million_semantic_hybrid import rrf_pair, selected_records
from lexical_mips_index import KeyedBM25TailIndex, TailOrthogonalSparseMIPSIndex
from pisces_reimplementation import (
    MultiInstanceLPSIReimplementation,
    PiscesResearchReimplementation,
    PiscesSimHashFilter,
    additive_share_uint64,
    reconstruct_uint64,
)
from run_experiment import (
    ConditionalDPE,
    KeyedPStableLSH,
    load_scifact,
    mips_document_transform,
    mips_query_transform,
)
from semantic_candidate_index import KeyedResidualSphericalIVF
from compartment_dpe import CompartmentDPEIndex
from experiment_candidate_rank_fusion import document_feature_matrix, paired_bootstrap
from benchmark_msmarco_dev_batched import compact_unique_clouds


class CoreTests(unittest.TestCase):
    def test_candidate_local_exact_bm25_prefers_higher_term_frequency(self):
        rankings = np.asarray([[[0, 1]]], dtype=np.int32)
        reranked, _ = exact_bm25_rerank(
            rankings=rankings,
            selected_rows=np.asarray([0, 1], dtype=np.int32),
            token_lengths=np.asarray([100, 100], dtype=np.uint32),
            indptr=np.asarray([0, 1, 2], dtype=np.uint64),
            indices=np.asarray([0, 0], dtype=np.uint32),
            counts=np.asarray([1, 3], dtype=np.uint32),
            query_texts=["alpha"],
            term_to_index={"alpha": 0},
            document_frequencies=np.asarray([2], dtype=np.uint32),
            documents=10,
            average_length=100.0,
        )
        np.testing.assert_array_equal(reranked, np.asarray([[[1, 0]]], dtype=np.int32))

    def test_msmarco_dev_split_is_loaded_explicitly(self):
        query_ids, _, qrels, _ = load_queries_qrels(ROOT / "data" / "msmarco", "dev")
        self.assertEqual(len(query_ids), 6980)
        self.assertEqual(sum(len(row) for row in qrels.values()), 7437)

    def test_cell_major_cloud_compaction_removes_assignment_duplicates(self):
        scores = np.asarray([[[9.0, 8.0, 7.0, 6.0, 5.0]]], dtype=np.float32)
        documents = np.asarray([[[3, 3, 4, 5, 4]]], dtype=np.int32)
        compacted = compact_unique_clouds(scores, documents, 3)
        np.testing.assert_array_equal(compacted[0, 0], np.asarray([3, 4, 5]))

    def test_candidate_scoped_dpe_materializes_only_touched_rows(self):
        rows = [
            {"_id": "a", "title": "", "text": "alpha alpha"},
            {"_id": "b", "title": "", "text": "beta"},
            {"_id": "c", "title": "", "text": "alpha beta"},
        ]
        frequencies = {
            lexical_keyed_term("alpha"): 2,
            lexical_keyed_term("beta"): 2,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            cache = root / "cache"
            data.mkdir()
            (data / "corpus.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            metadata = build_candidate_scoped_cache(
                data_dir=data,
                cache_dir=cache,
                candidates=np.asarray([[2, 0, -1], [2, -1, -1]], dtype=np.int32),
                documents=3,
                document_frequency=frequencies,
                average_length=5.0 / 3.0,
                dimension=8,
                k1=1.2,
                b=0.75,
                beta=0.0,
                scale=3.0,
                seed=23,
                block=2,
            )
            np.testing.assert_array_equal(
                np.load(cache / "candidate_rows.npy"), np.asarray([0, 2])
            )
            self.assertEqual(metadata["materialized_candidate_rows"], 2)
            self.assertEqual(np.load(cache / "lexical_cipher_fp16.npy").shape, (2, 16))

    def test_duetrank_features_handle_missing_path_without_nan(self):
        features = document_feature_matrix(
            [1, 2, 3],
            {1: 0, 2: 4},
            {2: 1, 3: 7},
            np.asarray([0.25, 2.0]),
        )
        self.assertEqual(features.shape, (3, 13))
        self.assertTrue(np.all(np.isfinite(features)))

    def test_paired_bootstrap_detects_uniform_gain(self):
        baseline = np.zeros((2, 4), dtype=np.float64)
        system = np.full((2, 4), 0.1, dtype=np.float64)
        result = paired_bootstrap(
            baseline, system, np.asarray([True, True, True, True]), samples=100
        )
        self.assertAlmostEqual(result["mean_absolute_gain"], 0.1)
        self.assertGreater(result["ci95_low"], 0.0)

    def test_budgeted_term_selection_respects_cost_after_mandatory_term(self):
        frequencies = {"rare": 3, "medium": 10, "common": 100}
        selected = select_terms_under_budget(frequencies, budget=15, alpha=0.5)
        self.assertIn("rare", selected)
        self.assertIn("medium", selected)
        self.assertNotIn("common", selected)
        self.assertLessEqual(sum(frequencies[term] for term in selected), 15)

    def test_million_subset_keeps_late_relevant_documents(self):
        rows = [
            {"_id": "early-a", "title": "", "text": "a"},
            {"_id": "early-b", "title": "", "text": "b"},
            {"_id": "filler", "title": "", "text": "c"},
            {"_id": "late-relevant", "title": "", "text": "d"},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "corpus.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            selected = list(selected_records(path, {"late-relevant"}, 3))
        self.assertEqual([row[0] for row in selected], ["early-a", "early-b", "late-relevant"])

    def test_million_rrf_and_metrics(self):
        fused = rrf_pair(
            np.asarray([0, 1, 2], dtype=np.int32),
            np.asarray([2, 1, 3], dtype=np.int32),
            depth=4,
        )
        self.assertEqual(int(fused[0]), 2)
        metrics = evaluate_million(
            fused[None, :],
            ["a", "b", "c", "d"],
            ["q"],
            {"q": {"c": 1}},
        )
        self.assertEqual(metrics["Recall@10"], 1.0)
        self.assertEqual(metrics["MRR@10"], 1.0)

    def test_disk_keyed_terms_are_deterministic_and_opaque(self):
        keyed = KeyedTerms()
        first = keyed.term("alpha")
        self.assertEqual(first, keyed.term("alpha"))
        self.assertNotEqual(first, keyed.term("beta"))
        self.assertNotIn("alpha", first)
        self.assertEqual(len(first), 32)

    def test_scifact_shape(self):
        corpus_ids, _, query_ids, _, qrels = load_scifact(ROOT / "data" / "scifact")
        self.assertEqual(len(corpus_ids), 5183)
        self.assertEqual(len(query_ids), 300)
        self.assertEqual(len(qrels), 300)

    def test_mips_to_l2_preserves_order(self):
        docs = np.asarray([[1.0, 0.0], [0.2, 0.8], [-1.0, 0.0]], dtype=np.float32)
        query = np.asarray([1.0, 0.0], dtype=np.float32)
        transformed_docs, _ = mips_document_transform(docs)
        transformed_query = mips_query_transform(query)
        inner_order = np.argsort(-(docs @ query))
        distance_order = np.argsort(np.linalg.norm(transformed_docs - transformed_query, axis=1))
        np.testing.assert_array_equal(inner_order, distance_order)

    def test_padded_lexical_dpe_rotation_preserves_mips_order_without_noise(self):
        rng = np.random.default_rng(31)
        documents = rng.normal(size=(20, 32)).astype(np.float32)
        query = rng.normal(size=32).astype(np.float32)
        scale = float(np.linalg.vector_norm(documents, axis=1).max())
        transformed_documents = np.zeros((20, 33), dtype=np.float32)
        transformed_documents[:, :32] = documents / scale
        transformed_documents[:, 32] = np.sqrt(
            np.maximum(0.0, 1.0 - np.sum(transformed_documents[:, :32] ** 2, axis=1))
        )
        transformed_query = np.zeros((1, 33), dtype=np.float32)
        transformed_query[0, :32] = query / np.linalg.vector_norm(query)
        work_dimension = 64
        sign1 = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), work_dimension)
        sign2 = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), work_dimension)
        permutation = rng.permutation(work_dimension)
        encrypted_documents = dpe_transform(
            transformed_documents, sign1, sign2, permutation
        )
        encrypted_query = dpe_transform(transformed_query, sign1, sign2, permutation)[0]
        np.testing.assert_array_equal(
            np.argsort(-(documents @ query)),
            np.argsort(np.linalg.vector_norm(encrypted_documents - encrypted_query, axis=1)),
        )

    def test_disk_lexical_dpe_build_and_query_smoke(self):
        rows = [
            {"_id": "a", "title": "", "text": "alpha alpha"},
            {"_id": "b", "title": "", "text": "beta"},
            {"_id": "c", "title": "", "text": "alpha beta"},
        ]
        frequencies = {
            lexical_keyed_term("alpha"): 2,
            lexical_keyed_term("beta"): 2,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            cache = root / "cache"
            data.mkdir()
            cache.mkdir()
            (data / "corpus.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            build_plain_sketches(
                data_dir=data,
                cache_dir=cache,
                relevant_ids=set(),
                expected_ids=["a", "b", "c"],
                document_frequency=frequencies,
                average_length=5.0 / 3.0,
                dimension=8,
                chunk=2,
                k1=1.2,
                b=0.75,
            )
            build_lexical_ciphertexts(
                cache_dir=cache,
                dimension=8,
                beta=0.0,
                scale=3.0,
                block=2,
                seed=17,
            )
            query = query_mips("alpha", frequencies, 3, 8)[None, :]
            encrypted_query = encrypt_query_matrix(
                query, cache / "lexical_dpe_key.npz", 0.0, 3.0, 19
            )[0]
            cipher = np.load(cache / "lexical_cipher_fp16.npy")
            encrypted_order = np.argsort(np.linalg.vector_norm(cipher - encrypted_query, axis=1))
            sketches = np.load(cache / "lexical_sketch_fp16.npy").astype(np.float32)
            plain_order = np.argsort(-(sketches @ query[0, :8]))
            np.testing.assert_array_equal(encrypted_order, plain_order)

    def test_conditional_distance_order(self):
        query = np.asarray([[0.0, 0.0]], dtype=np.float32)
        docs = np.asarray([[0.1, 0.0], [0.9, 0.0]], dtype=np.float32)
        dpe = ConditionalDPE(input_dim=2, beta=0.2, scale=3.0, seed=7)
        encrypted_docs = dpe.encrypt_database(docs)
        encrypted_query = dpe.encrypt_queries(query)[0]
        distances = np.linalg.norm(encrypted_docs - encrypted_query, axis=1)
        self.assertLess(distances[0], distances[1])

    def test_pstable_lsh_returns_identical_vector(self):
        vectors = np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32
        )
        index = KeyedPStableLSH(
            2, tables=8, projections=2, width=1.0, seed=9, key=b"test"
        )
        index.build(vectors)
        candidates = index.candidates(vectors[0], probes_per_table=3)
        self.assertIn(0, set(map(int, candidates)))

    def test_residual_spherical_ivf_returns_identical_vector(self):
        rng = np.random.default_rng(21)
        common = np.ones(12, dtype=np.float32) * 4.0
        vectors = common + rng.normal(size=(80, 12)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        index = KeyedResidualSphericalIVF(
            b"unit-test-semantic-key",
            n_clusters=8,
            doc_assignments=2,
            nprobe=8,
            centered=True,
            seed=7,
        )
        setup = index.build(vectors)
        candidates, trace = index.candidates(vectors[0])
        self.assertIn(0, set(map(int, candidates)))
        self.assertEqual(trace.probed_cells, 8)
        self.assertEqual(setup["posting_entries"], 160)

    def test_residual_spherical_ivf_uses_keyed_labels(self):
        vectors = np.eye(16, dtype=np.float32)
        first = KeyedResidualSphericalIVF(b"first", n_clusters=4, nprobe=4, seed=5)
        second = KeyedResidualSphericalIVF(b"second", n_clusters=4, nprobe=4, seed=5)
        first.build(vectors)
        second.build(vectors)
        self.assertNotEqual(set(first.postings), set(second.postings))
        self.assertEqual(first.storage_summary()["server_posting_entries"], 32)

    def test_compartment_dpe_uses_incompatible_cell_coordinates(self):
        rng = np.random.default_rng(91)
        docs = rng.normal(size=(96, 16)).astype(np.float32)
        docs /= np.linalg.norm(docs, axis=1, keepdims=True)
        routing = KeyedResidualSphericalIVF(
            b"test-routing", n_clusters=8, doc_assignments=2, nprobe=3, seed=92
        )
        routing.build(docs)
        index = CompartmentDPEIndex(
            key=b"test-compartments", projection_dim=8, beta=0.05, seed=93
        )
        setup = index.build(docs, routing)
        candidates, trace = index.candidates(
            docs[0], nprobe=3, top_per_cell=5, nonce=7
        )
        self.assertGreater(len(candidates), 0)
        self.assertLessEqual(len(candidates), 15)
        self.assertLessEqual(trace.emitted_entries, 15)
        self.assertEqual(setup["posting_entries"], 2 * len(docs))
        first, second = sorted(index.projections)[:2]
        self.assertFalse(
            np.allclose(index.projections[first], index.projections[second])
        )

    def test_tail_orthogonal_index_prefers_signed_overlap(self):
        documents = np.asarray(
            [[0.8, 0.0, 0.1], [0.1, 0.7, 0.0], [-0.8, 0.0, 0.1]],
            dtype=np.float32,
        )
        index = TailOrthogonalSparseMIPSIndex(b"unit-test")
        index.build(documents)
        candidates, trace = index.candidates(
            np.asarray([1.0, 0.0, 0.0], dtype=np.float32), max_candidates=2
        )
        self.assertEqual(int(candidates[0]), 0)
        self.assertNotIn(2, set(map(int, candidates)))
        self.assertEqual(trace["query_feature_tokens"], 1)

    def test_keyed_bm25_index_matches_weighted_term_order(self):
        postings = {
            "alpha": (
                np.asarray([0, 1], dtype=np.int32),
                np.asarray([3.0, 1.0], dtype=np.float32),
            ),
            "beta": (
                np.asarray([1, 2], dtype=np.int32),
                np.asarray([1.0, 2.0], dtype=np.float32),
            ),
        }
        index = KeyedBM25TailIndex(b"unit-test-key")
        index.build(
            postings, {"alpha": 1.0, "beta": 1.0},
            np.ones(3, dtype=np.float32), 1.0, 3, 1.2, 0.75,
        )
        candidates, trace = index.candidates(["alpha", "alpha"], 3)
        self.assertEqual(int(candidates[0]), 0)
        self.assertEqual(trace["matched_feature_tokens"], 1)

    def test_pisces_lpsi_slow_matches_equivalent_path(self):
        from collections import Counter

        counts = [Counter("alpha beta beta".split()), Counter("beta gamma".split())]
        protocol = MultiInstanceLPSIReimplementation(counts, seed=11)
        protocol.build()
        slow, _ = protocol.query_slow_protocol(["beta", "missing"])
        fast, _ = protocol.query_equivalent(["beta", "missing"])
        np.testing.assert_array_equal(slow, fast)

    def test_pisces_simhash_filter_keeps_identical_vector(self):
        docs = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
        index = PiscesSimHashFilter(
            2, simhash_bits=32, projection_num=40, projection_weight=8, seed=3
        )
        index.build(docs)
        candidates, _ = index.candidates(docs[0])
        self.assertIn(0, set(map(int, candidates)))

    def test_pisces_defaults_match_official_end_to_end_parameters(self):
        index = PiscesSimHashFilter(2, seed=3)
        self.assertEqual(index.simhash_bits, 32)
        self.assertEqual(index.projection_num, 160)
        self.assertEqual(index.projection_weight, 16)
        self.assertEqual(index.min_matches, 2)
        self.assertEqual(index.initial_hamming_radius, 5)
        self.assertEqual(index.maximum_hamming_radius, 10)

    def test_additive_shares_reconstruct(self):
        values = np.asarray([0, 1, 17, 2**40], dtype=np.uint64)
        first, second = additive_share_uint64(values, np.random.default_rng(5))
        np.testing.assert_array_equal(reconstruct_uint64(first, second), values)

    def test_pisces_end_to_end_lexical_ranking(self):
        from collections import Counter

        term_counts = [
            Counter("alpha beta beta".split()),
            Counter("beta gamma".split()),
            Counter("gamma".split()),
        ]
        dense = np.asarray([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], dtype=np.float32)
        protocol = PiscesResearchReimplementation(
            dense,
            term_counts,
            np.asarray([3, 2, 1], dtype=np.float32),
            average_document_length=2.0,
            tokenizer=lambda text: text.split(),
            seed=19,
        )
        protocol.build()
        result = protocol.retrieve("beta beta", dense[0], depth=3)
        self.assertEqual(int(result.lexical_rank[0]), 0)


if __name__ == "__main__":
    unittest.main()
