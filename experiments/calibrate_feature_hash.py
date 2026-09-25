"""Held-out calibration for signed feature-hashed BM25 score error."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from run_experiment import BM25Index, load_beir, tokenize, top_indices


ROOT = Path(__file__).resolve().parent


def document_norms_and_term_weights(index: BM25Index) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]:
    norm_sq = np.zeros(index.n_docs, dtype=np.float64)
    weighted_postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for term, (doc_ids, term_frequencies) in index.postings.items():
        tf = term_frequencies.astype(np.float64)
        denominator = tf + index.k1 * (
            1.0 - index.b + index.b * index.doc_len[doc_ids] / index.avgdl
        )
        weights = index.idf[term] * tf * (index.k1 + 1.0) / denominator
        weights = weights.astype(np.float64)
        norm_sq[doc_ids] += weights * weights
        weighted_postings[term] = (doc_ids, weights)
    return norm_sq, weighted_postings


def calibrate(data_dir: Path, dataset: str, dimension: int, failure_probability: float) -> dict:
    corpus_ids, corpus_texts, query_ids, query_texts, _ = load_beir(data_dir, "train")
    del corpus_ids
    index = BM25Index(corpus_texts)
    seed = b"DuetDPE-Lexical-v1"
    hashed_documents = index.hashed_document_vectors(dimension, seed).astype(np.float64)
    document_norm_sq, weighted_postings = document_norms_and_term_weights(index)

    absolute_errors: list[np.ndarray] = []
    active_errors: list[np.ndarray] = []
    query_union_max_errors: list[float] = []
    top100_coverages: list[float] = []
    chebyshev_covered = 0
    chebyshev_total = 0
    variance_sum = 0.0

    for query in query_texts:
        exact = index.score(query).astype(np.float64)
        hashed_query = index.hashed_query_vector(query, dimension, seed).astype(np.float64)
        approximate = hashed_documents @ hashed_query
        error = np.abs(approximate - exact)
        absolute_errors.append(error)
        active = exact > 0
        if np.any(active):
            active_errors.append(error[active])

        positive_depth = min(100, int(np.count_nonzero(exact > 0)))
        if positive_depth == 0:
            continue
        exact_top = top_indices(exact, positive_depth)
        approximate_top = top_indices(approximate, positive_depth)
        union = np.union1d(exact_top, approximate_top)
        query_union_max_errors.append(float(np.max(error[union])))
        top100_coverages.append(
            len(set(map(int, exact_top)).intersection(map(int, approximate_top)))
            / len(exact_top)
        )

        counts = Counter(tokenize(query))
        query_norm_sq = float(
            sum(count * count for term, count in counts.items() if term in index.idf)
        )
        diagonal = np.zeros(index.n_docs, dtype=np.float64)
        for term, count in counts.items():
            posting = weighted_postings.get(term)
            if posting is None:
                continue
            doc_ids, weights = posting
            diagonal[doc_ids] += (float(count) * weights) ** 2
        variance = (
            document_norm_sq * query_norm_sq + exact * exact - 2.0 * diagonal
        ) / dimension
        variance = np.maximum(variance, 0.0)
        bound = np.sqrt(variance / failure_probability)
        chebyshev_covered += int(np.count_nonzero(error <= bound + 1e-12))
        chebyshev_total += len(error)
        variance_sum += float(np.sum(variance))

    all_errors = np.concatenate(absolute_errors)
    all_active_errors = np.concatenate(active_errors) if active_errors else np.empty(0)
    per_query = np.asarray(query_union_max_errors)
    return {
        "dataset": dataset,
        "split": "train",
        "documents": index.n_docs,
        "queries": len(query_ids),
        "dimension": dimension,
        "hash_family_model": "keyed BLAKE2 treated as a PRF; bucket/sign bytes are pseudorandom",
        "absolute_score_error": {
            "mean": float(np.mean(all_errors)),
            "rmse": float(np.sqrt(np.mean(all_errors * all_errors))),
            "p95": float(np.percentile(all_errors, 95)),
            "p99": float(np.percentile(all_errors, 99)),
            "p999": float(np.percentile(all_errors, 99.9)),
            "max": float(np.max(all_errors)),
        },
        "active_document_absolute_error": {
            "mean": float(np.mean(all_active_errors)),
            "p95": float(np.percentile(all_active_errors, 95)),
            "p99": float(np.percentile(all_active_errors, 99)),
            "max": float(np.max(all_active_errors)),
        },
        "per_query_union_top100_max_error": {
            "median": float(np.percentile(per_query, 50)),
            "p95": float(np.percentile(per_query, 95)),
            "p99": float(np.percentile(per_query, 99)),
            "max": float(np.max(per_query)),
        },
        "positive_score_exact_top100_candidate_coverage_mean": float(np.mean(top100_coverages)),
        "chebyshev_diagnostic": {
            "per_pair_failure_probability": failure_probability,
            "empirical_fraction_within_pairwise_bound_for_one_fixed_seed": float(
                chebyshev_covered / chebyshev_total
            ),
            "mean_analytic_variance": float(variance_sum / chebyshev_total),
            "warning": "The probability is over setup hash randomness for each fixed pair; coverage across correlated pairs under one seed is diagnostic, not a simultaneous guarantee.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["scifact", "nfcorpus"])
    parser.add_argument("--dimensions", nargs="+", type=int, default=[512, 1024, 2048])
    parser.add_argument("--failure-probability", type=float, default=0.01)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "feature_hash_calibration.json"
    )
    args = parser.parse_args()
    if not 0.0 < args.failure_probability < 1.0:
        raise ValueError("failure probability must lie in (0,1)")
    rows = []
    for dataset in args.datasets:
        for dimension in args.dimensions:
            print(f"calibrating {dataset} m={dimension}", flush=True)
            rows.append(
                calibrate(
                    ROOT / "data" / dataset,
                    dataset,
                    dimension,
                    args.failure_probability,
                )
            )
    output = {
        "purpose": "held-out signed feature-hash calibration; no test labels used",
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
