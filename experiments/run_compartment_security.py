"""Security stress tests for Compartment DPE.

The experiments grant the attacker labeled auxiliary ciphertexts where needed.
They measure residual access-pattern leakage separately from local-coordinate
leakage and include both global anchor budgets and per-cell compromise tests.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEPS = ROOT / ".deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from compartment_dpe import CompartmentDPEIndex
from run_experiment import cooccurrence_link_attack, load_beir, normalize_rows, top_indices
from run_security_attacks import balanced_subset, wilson_interval
from semantic_candidate_index import KeyedResidualSphericalIVF

def block_sparse_view(cells: np.ndarray, coordinates: np.ndarray, n_cells: int) -> csr_matrix:
    """One independent linear feature block per observed cell label."""
    rows, cols, data = [], [], []
    width = coordinates.shape[1] + 1
    for row, cell in enumerate(cells):
        offset = int(cell) * width
        rows.extend([row] * width)
        cols.extend(range(offset, offset + width))
        data.extend([1.0, *map(float, coordinates[row])])
    return csr_matrix((data, (rows, cols)), shape=(len(cells), n_cells * width))


def one_hot_cells(cells: np.ndarray, n_cells: int) -> csr_matrix:
    rows = np.arange(len(cells), dtype=np.int32)
    return csr_matrix((np.ones(len(cells)), (rows, cells)), shape=(len(cells), n_cells))


def classify(train_x, train_y: np.ndarray, test_x, test_y: np.ndarray) -> dict[str, object]:
    classifier = LogisticRegression(max_iter=1500, solver="lbfgs", n_jobs=1)
    classifier.fit(train_x, train_y)
    predicted = classifier.predict(test_x)
    correct = int(np.count_nonzero(predicted == test_y))
    low, high = wilson_interval(correct, len(test_y))
    return {"accuracy": float(accuracy_score(test_y, predicted)), "wilson_ci95": [low, high]}


def topic_attack(args: argparse.Namespace) -> dict[str, object]:
    train = fetch_20newsgroups(
        data_home=ROOT / "data" / "20newsgroups",
        subset="train",
        remove=("headers", "footers", "quotes"),
        download_if_missing=False,
    )
    test = fetch_20newsgroups(
        data_home=ROOT / "data" / "20newsgroups",
        subset="test",
        remove=("headers", "footers", "quotes"),
        download_if_missing=False,
    )
    _, train_y = balanced_subset(
        train.data, np.asarray(train.target), args.topic_train_per_class, args.seed
    )
    _, test_y = balanced_subset(
        test.data, np.asarray(test.target), args.topic_test_per_class, args.seed + 1
    )
    train_x = normalize_rows(np.load(ROOT / "cache" / "security_20ng" / "granite_train_3000.npy"))
    test_x = normalize_rows(np.load(ROOT / "cache" / "security_20ng" / "granite_test_1500.npy"))
    if len(train_x) != len(train_y) or len(test_x) != len(test_y):
        raise ValueError("20 Newsgroups cache does not match requested balanced subset")

    routing = KeyedResidualSphericalIVF(
        b"DuetDPE-topic-compartment-routing-v1",
        n_clusters=args.topic_cells,
        doc_assignments=1,
        nprobe=1,
        centered=True,
        seed=args.seed + 100,
    )
    routing.build(train_x)
    index = CompartmentDPEIndex(
        key=b"DuetDPE-topic-compartment-coordinate-v1",
        projection_dim=args.projection_dim,
        beta=args.beta,
        scale=args.scale,
        seed=args.seed + 101,
    )
    index.build(train_x, routing)
    train_cells, train_view = index.primary_view(train_x, query=True, nonce_offset=10000)
    test_cells, test_view = index.primary_view(test_x, query=True, nonce_offset=20000)

    access_train = one_hot_cells(train_cells, args.topic_cells)
    access_test = one_hot_cells(test_cells, args.topic_cells)
    compartment_train = block_sparse_view(train_cells, train_view, args.topic_cells)
    compartment_test = block_sparse_view(test_cells, test_view, args.topic_cells)
    results = {
        "plaintext_embedding": classify(train_x, train_y, test_x, test_y),
        "cell_label_only": classify(access_train, train_y, access_test, test_y),
        "cell_label_plus_local_coordinate": classify(
            compartment_train, train_y, compartment_test, test_y
        ),
    }
    for name, row in results.items():
        print(f"topic {name:36s} accuracy={row['accuracy']:.4f}", flush=True)
    return {
        "dataset": "20 Newsgroups",
        "classes": 20,
        "train_queries": len(train_y),
        "test_queries": len(test_y),
        "cells": args.topic_cells,
        "random_accuracy": 0.05,
        "results": results,
    }


def primary_document_view(index: CompartmentDPEIndex, docs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return index.primary_view(docs, query=False)


def global_anchor_alignment(
    plain: np.ndarray,
    cells: np.ndarray,
    encrypted: np.ndarray,
    *,
    anchor_counts: list[int],
    targets: int,
    repeats: int,
    scale: float,
    seed: int,
) -> list[dict[str, object]]:
    rows = []
    max_anchors = max(anchor_counts)
    for count in anchor_counts:
        accuracies, covered = [], []
        for repeat in range(repeats):
            rng = np.random.default_rng(seed + repeat)
            order = rng.permutation(len(plain))
            anchors = order[:max_anchors][:count]
            target_ids = order[max_anchors : max_anchors + targets]
            cipher_order = rng.permutation(len(target_ids))
            cipher_ids = target_ids[cipher_order]
            cost = np.zeros((len(target_ids), len(cipher_ids)), dtype=np.float64)
            columns_with_anchor = 0
            for column, cipher_id in enumerate(cipher_ids):
                local = anchors[cells[anchors] == cells[cipher_id]]
                if not len(local):
                    cost[:, column] = rng.random(len(target_ids)) * 1e-8
                    continue
                columns_with_anchor += 1
                public_fp = cdist(plain[target_ids], plain[local])
                cipher_fp = cdist(encrypted[[cipher_id]], encrypted[local])[0] / scale
                cost[:, column] = np.mean(np.abs(public_fp - cipher_fp[None, :]), axis=1)
            public_rows, cipher_cols = linear_sum_assignment(cost)
            correct = target_ids[public_rows] == cipher_ids[cipher_cols]
            accuracies.append(float(np.mean(correct)))
            covered.append(columns_with_anchor / len(cipher_ids))
        rows.append(
            {
                "global_known_anchors": count,
                "unknown_targets": targets,
                "mean_alignment_accuracy": float(np.mean(accuracies)),
                "std_alignment_accuracy": float(np.std(accuracies, ddof=1)),
                "mean_target_fraction_with_local_anchor": float(np.mean(covered)),
                "trials": accuracies,
            }
        )
        print(
            f"global anchors={count:3d} accuracy={np.mean(accuracies):.4f} "
            f"covered={np.mean(covered):.4f}",
            flush=True,
        )
    return rows


def local_anchor_alignment(
    plain: np.ndarray,
    cells: np.ndarray,
    encrypted: np.ndarray,
    *,
    anchors_per_cell: list[int],
    scale: float,
    seed: int,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    rows = []
    for anchor_count in anchors_per_cell:
        correct_total = 0
        target_total = 0
        cells_tested = 0
        for cell in np.unique(cells):
            members = np.flatnonzero(cells == cell)
            if len(members) < anchor_count + 3:
                continue
            members = rng.permutation(members)
            anchors = members[:anchor_count]
            targets = members[anchor_count : anchor_count + min(20, len(members) - anchor_count)]
            shuffled = rng.permutation(targets)
            public_fp = cdist(plain[targets], plain[anchors])
            encrypted_fp = cdist(encrypted[shuffled], encrypted[anchors]) / scale
            public_rows, cipher_cols = linear_sum_assignment(cdist(public_fp, encrypted_fp))
            correct_total += int(np.count_nonzero(targets[public_rows] == shuffled[cipher_cols]))
            target_total += len(targets)
            cells_tested += 1
        rows.append(
            {
                "known_anchors_per_cell": anchor_count,
                "cells_tested": cells_tested,
                "unknown_targets": target_total,
                "local_alignment_accuracy": correct_total / target_total if target_total else None,
            }
        )
    return rows


def neighborhood_attack(
    plain: np.ndarray,
    cells: np.ndarray,
    encrypted: np.ndarray,
    *,
    sample_size: int,
    neighbors: int,
    seed: int,
    repeats: int,
) -> dict[str, float | int | list[float]]:
    trial_means, oracle_means, all_overlaps = [], [], []
    for repeat in range(repeats):
        rng = np.random.default_rng(seed + repeat)
        sampled = rng.choice(len(plain), min(sample_size, len(plain)), replace=False)
        true_dist = cdist(plain[sampled], plain)
        true_dist[np.arange(len(sampled)), sampled] = np.inf
        true_neighbors = np.argpartition(true_dist, neighbors - 1, axis=1)[:, :neighbors]
        overlaps, cell_oracle = [], []
        for row, doc_id in enumerate(sampled):
            members = np.flatnonzero(cells == cells[doc_id])
            members = members[members != doc_id]
            oracle_set = set(map(int, members))
            cell_oracle.append(
                len(set(map(int, true_neighbors[row])).intersection(oracle_set)) / neighbors
            )
            if not len(members):
                overlaps.append(0.0)
                continue
            distances = np.linalg.norm(encrypted[members] - encrypted[doc_id], axis=1)
            predicted = members[np.argsort(distances)[:neighbors]]
            overlaps.append(
                len(set(map(int, true_neighbors[row])).intersection(map(int, predicted)))
                / neighbors
            )
        trial_means.append(float(np.mean(overlaps)))
        oracle_means.append(float(np.mean(cell_oracle)))
        all_overlaps.extend(overlaps)
    return {
        "sample_size_per_repeat": min(sample_size, len(plain)),
        "neighbors": neighbors,
        "repeats": repeats,
        "encrypted_knn_global_recall": float(np.mean(trial_means)),
        "encrypted_knn_global_recall_std": float(np.std(trial_means, ddof=1)),
        "same_cell_oracle_global_recall": float(np.mean(oracle_means)),
        "encrypted_knn_p95": float(np.percentile(all_overlaps, 95)),
        "trials": trial_means,
    }


def global_known_query_recovery(
    index: CompartmentDPEIndex,
    documents: np.ndarray,
    document_cells: np.ndarray,
    document_cipher: np.ndarray,
    queries: np.ndarray,
    *,
    known_queries: int,
    document_sample: int,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    """Direct analogue of the global-DPE 384-known-query recovery attack.

    A separate affine inverse is estimated for every observed compartment.
    Targets in compartments with no known query receive the global known-query
    mean.  We report both all-target and covered-target accuracy so that a low
    number cannot be manufactured merely by declaring uncovered cells failed.
    """
    query_cells, query_cipher = index.primary_view(
        queries, query=True, nonce_offset=300_000
    )
    all_cosines, covered_cosines, coverages, baseline_cosines = [], [], [], []
    for repeat in range(repeats):
        rng = np.random.default_rng(seed + repeat)
        known_ids = rng.choice(
            len(queries), min(known_queries, len(queries)), replace=False
        )
        target_ids = rng.choice(
            len(documents), min(document_sample, len(documents)), replace=False
        )
        known_plain = queries[known_ids]
        fallback = normalize_rows(known_plain.mean(axis=0, keepdims=True))[0]
        estimate = np.repeat(fallback[None, :], len(target_ids), axis=0)
        baseline = estimate.copy()
        covered = np.zeros(len(target_ids), dtype=bool)
        for cell in np.unique(query_cells[known_ids]):
            local_known = known_ids[query_cells[known_ids] == cell]
            local_targets = np.flatnonzero(document_cells[target_ids] == cell)
            if not len(local_targets):
                continue
            design = np.column_stack(
                [query_cipher[local_known], np.ones(len(local_known))]
            )
            mapping = np.linalg.lstsq(design, queries[local_known], rcond=None)[0]
            prediction = np.column_stack(
                [document_cipher[target_ids[local_targets]], np.ones(len(local_targets))]
            ) @ mapping
            estimate[local_targets] = prediction
            local_mean = normalize_rows(
                queries[local_known].mean(axis=0, keepdims=True)
            )[0]
            baseline[local_targets] = local_mean
            covered[local_targets] = True
        estimate = normalize_rows(estimate)
        baseline = normalize_rows(baseline)
        actual = documents[target_ids]
        cosine = np.sum(actual * estimate, axis=1)
        baseline_cosine = np.sum(actual * baseline, axis=1)
        all_cosines.append(float(np.mean(cosine)))
        covered_cosines.append(float(np.mean(cosine[covered])) if np.any(covered) else 0.0)
        baseline_cosines.append(float(np.mean(baseline_cosine)))
        coverages.append(float(np.mean(covered)))
    return {
        "known_queries": min(known_queries, len(queries)),
        "available_queries": len(queries),
        "document_sample_per_repeat": min(document_sample, len(documents)),
        "repeats": repeats,
        "recovered_cosine_all_targets": float(np.mean(all_cosines)),
        "recovered_cosine_all_targets_std": float(np.std(all_cosines, ddof=1)),
        "recovered_cosine_covered_targets": float(np.mean(covered_cosines)),
        "known_query_mean_baseline_cosine": float(np.mean(baseline_cosines)),
        "target_compartment_coverage": float(np.mean(coverages)),
        "trials": all_cosines,
    }


def compartment_cloud_order(
    index: CompartmentDPEIndex,
    query: np.ndarray,
    *,
    nprobe: int,
    top_per_cell: int,
    nonce: int,
) -> np.ndarray:
    """Interleave local ranks and emit a distinct handle for every cell copy."""
    cells = index.query_cells(query, nprobe)
    local_ranks: list[tuple[int, np.ndarray]] = []
    for ordinal, raw_cell in enumerate(cells):
        cell = int(raw_cell)
        doc_ids = index.postings.get(cell)
        if doc_ids is None:
            continue
        query_cipher = index.encrypt_query_for_cell(
            query, cell, nonce=nonce * 1009 + ordinal
        )
        delta = index.ciphertexts[cell] - query_cipher[None, :]
        distances = np.einsum("ij,ij->i", delta, delta)
        local = top_indices(-distances, min(top_per_cell, len(doc_ids)))
        local_ranks.append((cell, doc_ids[local]))
    ordered: list[int] = []
    seen: set[int] = set()
    stride = len(index.documents)
    for rank in range(top_per_cell):
        for cell, values in local_ranks:
            if rank >= len(values):
                continue
            # The evaluator can recover the hidden document with modulo stride,
            # but the server receives an independently randomized alias for this
            # (cell, document) copy.  Copies of one document in two cells are
            # deliberately different handles.
            handle = cell * stride + int(values[rank])
            if handle not in seen:
                seen.add(handle)
                ordered.append(handle)
    return np.asarray(ordered, dtype=np.int64)


def alias_linkage_attack(
    index: CompartmentDPEIndex,
    queries: np.ndarray,
    *,
    nprobe: int,
    top_per_cell: int,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    ranking_path = ROOT / "results/scifact_granite_duetrank_rerun/rankings.npz"
    lexical = np.asarray(np.load(ranking_path)["lexical_dpe"], dtype=np.int32)
    semantic = [
        compartment_cloud_order(
            index,
            query,
            nprobe=nprobe,
            top_per_cell=top_per_cell,
            nonce=row + 1,
        )
        for row, query in enumerate(queries)
    ]
    lexical_rows = [row[row >= 0] for row in lexical]
    observed_handles = np.unique(np.concatenate(semantic)).astype(np.int64)
    handle_position = {int(handle): pos for pos, handle in enumerate(observed_handles)}
    stride = len(index.documents)
    accuracies, evaluated = [], []
    for repeat in range(repeats):
        rng = np.random.default_rng(seed + repeat)
        sem_alias_values = rng.permutation(len(observed_handles))
        lex_alias = rng.permutation(len(index.documents))
        counts: dict[int, dict[int, int]] = {}
        alias_to_document: dict[int, int] = {}
        for semantic_handles, lexical_docs in zip(semantic, lexical_rows):
            lexical_aliases = lex_alias[lexical_docs[:20]]
            for handle in semantic_handles[:20]:
                position = handle_position[int(handle)]
                alias = int(sem_alias_values[position])
                alias_to_document[alias] = int(handle) % stride
                counter = counts.setdefault(alias, {})
                for lexical_alias in lexical_aliases:
                    value = int(lexical_alias)
                    counter[value] = counter.get(value, 0) + 1
        correct = 0
        for alias, counter in counts.items():
            predicted = min(counter, key=lambda value: (-counter[value], value))
            correct += int(predicted == int(lex_alias[alias_to_document[alias]]))
        trial_evaluated = len(counts)
        accuracies.append(correct / trial_evaluated if trial_evaluated else 0.0)
        evaluated.append(trial_evaluated)
    return {
        "counterfactual_shared_identifier_direct_linkage": 1.0,
        "alias_scope": "independent alias per (semantic cell, document) copy",
        "independent_cell_local_alias_cooccurrence_top1": float(np.mean(accuracies)),
        "independent_cell_local_alias_cooccurrence_top1_std": float(
            np.std(accuracies, ddof=1)
        ),
        "observed_cell_local_handles": int(len(observed_handles)),
        "evaluated_semantic_cell_aliases_mean": float(np.mean(evaluated)),
        "attack_depth_per_path": 20,
        "repeats": repeats,
        "trials": accuracies,
    }


def local_known_pair_recovery(
    plain: np.ndarray,
    cells: np.ndarray,
    encrypted: np.ndarray,
    *,
    anchors_per_cell: list[int],
    seed: int,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    rows = []
    for count in anchors_per_cell:
        predicted_rows, actual_rows, mean_rows = [], [], []
        cells_tested = 0
        for cell in np.unique(cells):
            members = np.flatnonzero(cells == cell)
            if len(members) < count + 3:
                continue
            members = rng.permutation(members)
            known = members[:count]
            held = members[count : count + min(10, len(members) - count)]
            design = np.column_stack([encrypted[known], np.ones(len(known))])
            mapping = np.linalg.lstsq(design, plain[known], rcond=None)[0]
            estimate = np.column_stack([encrypted[held], np.ones(len(held))]) @ mapping
            predicted_rows.append(estimate)
            actual_rows.append(plain[held])
            mean_rows.append(np.repeat(plain[known].mean(axis=0)[None, :], len(held), axis=0))
            cells_tested += 1
        if not actual_rows:
            rows.append({"known_pairs_per_cell": count, "cells_tested": 0, "test_records": 0})
            continue
        actual = np.vstack(actual_rows)
        estimate = normalize_rows(np.vstack(predicted_rows))
        mean_estimate = normalize_rows(np.vstack(mean_rows))
        rows.append(
            {
                "known_pairs_per_cell": count,
                "cells_tested": cells_tested,
                "test_records": len(actual),
                "recovery_mean_cosine": float(np.mean(np.sum(actual * estimate, axis=1))),
                "known_record_mean_baseline_cosine": float(
                    np.mean(np.sum(actual * mean_estimate, axis=1))
                ),
                "recovery_mean_l2": float(np.mean(np.linalg.norm(actual - estimate, axis=1))),
            }
        )
    return rows


def document_attacks(args: argparse.Namespace) -> dict[str, object]:
    plain = normalize_rows(
        np.load(
            ROOT / "cache" / "granite_gpu" / "scifact-st-a7972eb5022e_corpus_embeddings.npy"
        ).astype(np.float32)
    )
    routing = KeyedResidualSphericalIVF(
        b"DuetDPE-document-compartment-routing-v1",
        n_clusters=args.document_cells,
        doc_assignments=args.document_assignments,
        nprobe=args.document_probes,
        centered=True,
        seed=(
            args.document_routing_seed
            if args.document_routing_seed is not None
            else args.seed + 200
        ),
    )
    routing.build(plain)
    index = CompartmentDPEIndex(
        key=b"DuetDPE-document-compartment-coordinate-v1",
        projection_dim=args.projection_dim,
        beta=args.beta,
        scale=args.scale,
        seed=args.seed + 201,
    )
    setup = index.build(plain, routing)
    cells, encrypted = primary_document_view(index, plain)
    global_alignment = global_anchor_alignment(
        plain,
        cells,
        encrypted,
        anchor_counts=args.global_anchor_counts,
        targets=args.known_targets,
        repeats=args.repeats,
        scale=args.scale,
        seed=args.seed + 202,
    )
    local_alignment = local_anchor_alignment(
        plain,
        cells,
        encrypted,
        anchors_per_cell=args.local_anchor_counts,
        scale=args.scale,
        seed=args.seed + 203,
    )
    neighborhood = neighborhood_attack(
        plain,
        cells,
        encrypted,
        sample_size=args.neighbor_sample,
        neighbors=args.neighbor_k,
        seed=args.seed + 204,
        repeats=args.repeats,
    )
    recovery = local_known_pair_recovery(
        plain,
        cells,
        encrypted,
        anchors_per_cell=args.local_anchor_counts,
        seed=args.seed + 205,
    )
    query_blocks = [
        np.load(ROOT / path).astype(np.float32) for path in args.known_query_files
    ]
    known_queries = normalize_rows(np.concatenate(query_blocks, axis=0))
    known_query_recovery = global_known_query_recovery(
        index,
        plain,
        cells,
        encrypted,
        known_queries,
        known_queries=args.known_query_count,
        document_sample=args.known_query_document_sample,
        repeats=args.repeats,
        seed=args.seed + 206,
    )
    _, _, _, query_texts, _ = load_beir(ROOT / "data/scifact")
    scifact_queries = normalize_rows(
        np.load(
            ROOT / "cache/granite_gpu/scifact-st-a7972eb5022e_query_embeddings.npy"
        ).astype(np.float32)
    )
    if len(scifact_queries) != len(query_texts):
        raise ValueError("SciFact query cache mismatch")
    alias_linkage = alias_linkage_attack(
        index,
        scifact_queries,
        nprobe=args.document_probes,
        top_per_cell=args.alias_top_per_cell,
        repeats=args.repeats,
        seed=args.seed + 207,
    )
    print(f"neighborhood recall={neighborhood['encrypted_knn_global_recall']:.4f}", flush=True)
    print(
        f"known-query m={args.known_query_count} "
        f"cos={known_query_recovery['recovered_cosine_all_targets']:.4f}",
        flush=True,
    )
    print(
        f"independent-alias linkage="
        f"{alias_linkage['independent_cell_local_alias_cooccurrence_top1']:.4f}",
        flush=True,
    )
    return {
        "dataset": "BEIR/SciFact Granite documents",
        "documents": len(plain),
        "setup": setup,
        "global_known_record_alignment": global_alignment,
        "local_cell_compromise_alignment": local_alignment,
        "neighborhood_reconstruction": neighborhood,
        "local_known_pair_recovery": recovery,
        "global_known_query_recovery": known_query_recovery,
        "cross_path_alias_linkage": alias_linkage,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--beta", type=float, default=0.10)
    parser.add_argument("--scale", type=float, default=3.0)
    parser.add_argument("--topic-cells", type=int, default=256)
    parser.add_argument("--topic-train-per-class", type=int, default=150)
    parser.add_argument("--topic-test-per-class", type=int, default=75)
    parser.add_argument("--document-cells", type=int, default=512)
    parser.add_argument("--document-assignments", type=int, default=2)
    parser.add_argument("--document-probes", type=int, default=16)
    parser.add_argument("--document-routing-seed", type=int, default=None)
    parser.add_argument("--global-anchor-counts", nargs="+", type=int, default=[2, 4, 8, 16, 32, 64])
    parser.add_argument("--local-anchor-counts", nargs="+", type=int, default=[2, 4, 8, 16])
    parser.add_argument("--known-targets", type=int, default=300)
    parser.add_argument("--neighbor-sample", type=int, default=512)
    parser.add_argument("--neighbor-k", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--known-query-count", type=int, default=384)
    parser.add_argument("--known-query-document-sample", type=int, default=1000)
    parser.add_argument(
        "--known-query-files",
        nargs="+",
        default=[
            "cache/granite_gpu/scifact-st-a7972eb5022e_query_embeddings.npy",
            "cache/granite_gpu/nfcorpus-st-a7972eb5022e_query_embeddings.npy",
        ],
    )
    parser.add_argument("--alias-top-per-cell", type=int, default=32)
    # The cached balanced 20NG embeddings were selected with this seed.  A
    # different value changes the sampled labels and invalidates the cache.
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "compartment_security.json")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    topic = topic_attack(parsed)
    documents = document_attacks(parsed)
    global_four = next(
        row for row in documents["global_known_record_alignment"] if row["global_known_anchors"] == 4
    )
    gates = {
        "topic_accuracy_at_most_0.35": topic["results"]["cell_label_plus_local_coordinate"]["accuracy"] <= 0.35,
        "four_global_anchor_alignment_at_most_0.20": global_four["mean_alignment_accuracy"] <= 0.20,
        "global_knn_recall_at_most_0.50": documents["neighborhood_reconstruction"]["encrypted_knn_global_recall"] <= 0.50,
    }
    report = {
        "configuration": vars(parsed) | {"output": str(parsed.output)},
        "predeclared_security_gates": gates,
        "security_gate_passed": all(gates.values()),
        "topic_inference": topic,
        "document_attacks": documents,
        "caveat": (
            "Compartment mode limits global geometry but does not hide cell labels, "
            "posting sizes, access patterns, or geometry inside a compromised cell."
        ),
    }
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    parsed.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"gates": gates, "passed": report["security_gate_passed"]}, indent=2))
