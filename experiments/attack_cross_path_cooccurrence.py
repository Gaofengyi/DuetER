"""Cross-path linkage from synchronized candidate and response traces.

The attack models the honest-but-curious cloud after aliases have been
independently randomized.  It aligns semantic and lexical query episodes by
their simultaneous arrival, then represents each path-local document by its
candidate membership, response membership, and reciprocal response rank over
multiple queries.  Matching uses only cosine similarity between these trace
fingerprints.  True document rows are used solely to sample evaluation cohorts
and score the hidden mapping.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEPS = ROOT / ".deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))

import numpy as np
from scipy import sparse


CONFIGS = {
    "nq": {
        "documents": 2_681_468,
        "semantic_cache": ROOT / "cache/full_semantic/nq_2681468",
        "semantic_rankings": ROOT / "results/full_semantic_hybrid/nq_2681468/rankings.npz",
        "lexical_candidates": ROOT / "results/budgeted_lexical/nq_full.raw.npz",
        "lexical_candidate_key": "budget_50000",
        "lexical_rankings": ROOT / "results/budgeted_lexical_dpe/nq_full_d500_o300.rankings.npz",
        "lexical_rows": ROOT / "cache/full_candidate_lexical_dpe/nq_full_h1024/candidate_rows.npy",
        "query_embeddings": "query_embeddings.npy",
    },
    "hotpotqa": {
        "documents": 5_233_329,
        "semantic_cache": ROOT / "cache/full_semantic/hotpotqa_5233329",
        "semantic_rankings": ROOT / "results/full_semantic_hybrid/hotpotqa_5233329/rankings.npz",
        "lexical_candidates": ROOT / "results/budgeted_lexical/hotpotqa_full.raw.npz",
        "lexical_candidate_key": "budget_50000",
        "lexical_rankings": ROOT / "results/budgeted_lexical_dpe/hotpotqa_full_d200.rankings.npz",
        "lexical_rows": ROOT / "cache/full_candidate_lexical_dpe/hotpotqa_full_h1024/candidate_rows.npy",
        "query_embeddings": "query_embeddings.npy",
    },
    "msmarco": {
        "documents": 8_841_823,
        "semantic_cache": ROOT / "cache/full_semantic/msmarco_8841823",
        "semantic_rankings": ROOT / "results/msmarco_dev_full_semantic/rankings.npz",
        "lexical_candidates": ROOT / "results/budgeted_lexical/msmarco_dev_full.raw.npz",
        "lexical_candidate_key": "budget_250000",
        "lexical_rankings": ROOT / "results/budgeted_lexical_dpe/msmarco_dev_full_b250k_d500_o300.rankings.npz",
        "lexical_rows": ROOT / "cache/full_candidate_lexical_dpe/msmarco_dev_b250k_full_h1024/candidate_rows.npy",
        "query_embeddings": "query_embeddings_dev.npy",
    },
}


def route_semantic_queries(cache_dir: Path, query_filename: str, probes: int) -> np.ndarray:
    queries = np.load(cache_dir / query_filename).astype(np.float32)
    mean = np.load(cache_dir / "semantic_mean.npy").astype(np.float32)
    centroids = np.load(cache_dir / "semantic_centroids.npy").astype(np.float32)
    residual = queries - mean
    residual /= np.maximum(np.linalg.norm(residual, axis=1, keepdims=True), 1e-12)
    scores = residual @ centroids.T
    routed = np.argpartition(-scores, probes - 1, axis=1)[:, :probes]
    return routed.astype(np.int32)


def unique_valid(values: np.ndarray) -> np.ndarray:
    flattened = np.asarray(values).reshape(-1)
    return np.unique(flattened[flattened >= 0]).astype(np.int32)


def response_array(path: Path, key: str) -> np.ndarray:
    values = np.load(path)[key]
    return values[0] if values.ndim == 3 else values


def semantic_candidate_eligibility(
    documents: np.ndarray,
    assignments: np.ndarray,
    routed: np.ndarray,
    cells: int,
) -> np.ndarray:
    visited = np.zeros(cells, dtype=bool)
    visited[np.unique(routed)] = True
    document_cells = np.asarray(assignments[documents], dtype=np.int64)
    return np.any(visited[document_cells], axis=1)


def semantic_candidate_trace(
    documents: np.ndarray,
    assignments: np.ndarray,
    routed: np.ndarray,
    cells: int,
) -> sparse.csr_matrix:
    route_mask = np.zeros((len(routed), cells), dtype=bool)
    route_mask[np.arange(len(routed))[:, None], routed] = True
    document_cells = np.asarray(assignments[documents], dtype=np.int64)
    visible = np.zeros((len(documents), len(routed)), dtype=bool)
    for column in range(document_cells.shape[1]):
        visible |= route_mask[:, document_cells[:, column]].T
    return sparse.csr_matrix(visible, dtype=np.float32)


def trace_from_rows(
    documents: np.ndarray,
    per_query_rows: np.ndarray,
    rank_weighted: bool,
) -> sparse.csr_matrix:
    """Create a document-by-query sparse trace without exposing true rows to attack."""
    order = np.argsort(documents)
    sorted_documents = documents[order]
    out_rows: list[np.ndarray] = []
    out_columns: list[np.ndarray] = []
    out_values: list[np.ndarray] = []
    for query_index, row_values in enumerate(per_query_rows):
        valid = np.asarray(row_values[row_values >= 0], dtype=np.int64)
        if not len(valid):
            continue
        positions = np.searchsorted(sorted_documents, valid)
        in_range = positions < len(sorted_documents)
        matched = np.zeros(len(valid), dtype=bool)
        matched[in_range] = sorted_documents[positions[in_range]] == valid[in_range]
        if not np.any(matched):
            continue
        positions = positions[matched]
        original_rows = order[positions]
        candidate_ranks = np.flatnonzero(matched)
        values = (
            1.0 / (1.0 + candidate_ranks.astype(np.float32))
            if rank_weighted
            else np.ones(len(original_rows), dtype=np.float32)
        )
        out_rows.append(original_rows.astype(np.int32))
        out_columns.append(np.full(len(original_rows), query_index, dtype=np.int32))
        out_values.append(values)
    if not out_rows:
        return sparse.csr_matrix((len(documents), len(per_query_rows)), dtype=np.float32)
    matrix = sparse.coo_matrix(
        (
            np.concatenate(out_values),
            (np.concatenate(out_rows), np.concatenate(out_columns)),
        ),
        shape=(len(documents), len(per_query_rows)),
        dtype=np.float32,
    )
    matrix.sum_duplicates()
    if not rank_weighted:
        matrix.data[:] = 1.0
    return matrix.tocsr()


def sparse_cosine(first: sparse.csr_matrix, second: sparse.csr_matrix) -> np.ndarray:
    first = first.astype(np.float32, copy=False)
    second = second.astype(np.float32, copy=False)
    first_norm = np.sqrt(np.asarray(first.multiply(first).sum(axis=1)).ravel())
    second_norm = np.sqrt(np.asarray(second.multiply(second).sum(axis=1)).ravel())
    first_scaled = sparse.diags(1.0 / np.maximum(first_norm, 1e-12)) @ first
    second_scaled = sparse.diags(1.0 / np.maximum(second_norm, 1e-12)) @ second
    product = first_scaled @ second_scaled.T
    return product.toarray().astype(np.float32, copy=False)


def expected_matching_metrics(similarities: np.ndarray, permutation: np.ndarray) -> dict[str, float]:
    """Metrics with exact ties averaged over uniformly random tie breaking."""
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(len(permutation))
    true_score = similarities[np.arange(len(similarities)), inverse]
    greater = np.sum(similarities > true_score[:, None], axis=1)
    equal = np.sum(similarities == true_score[:, None], axis=1)
    equal = np.maximum(equal, 1)

    def expected_topk(k: int) -> float:
        remaining = np.clip(k - greater, 0, equal)
        return float(np.mean(remaining / equal))

    maximum_rank = len(permutation)
    harmonic = np.concatenate(
        [np.zeros(1, dtype=np.float64), np.cumsum(1.0 / np.arange(1, maximum_rank + 1))]
    )
    reciprocal = (harmonic[greater + equal] - harmonic[greater]) / equal
    expected_rank = greater + (equal + 1.0) / 2.0
    return {
        "top1": expected_topk(1),
        "top5": expected_topk(5),
        "top10": expected_topk(10),
        "mrr": float(np.mean(reciprocal)),
        "mean_rank": float(np.mean(expected_rank)),
        "median_expected_rank": float(np.median(expected_rank)),
    }


def random_baseline(pool_size: int) -> dict[str, float]:
    harmonic = float(np.sum(1.0 / np.arange(1, pool_size + 1, dtype=np.float64)))
    return {
        "top1": 1.0 / pool_size,
        "top5": min(5, pool_size) / pool_size,
        "top10": min(10, pool_size) / pool_size,
        "mrr": harmonic / pool_size,
    }


def summarize(trials: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    output = {}
    for key in trials[0]:
        values = np.asarray([trial[key] for trial in trials], dtype=np.float64)
        half = 0.0 if len(values) == 1 else 1.96 * np.std(values, ddof=1) / math.sqrt(len(values))
        output[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "normal_ci95_low": float(np.mean(values) - half),
            "normal_ci95_high": float(np.mean(values) + half),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return output


def evaluate_trace_set(
    traces: dict[str, sparse.csr_matrix],
    query_order: np.ndarray,
    query_counts: list[int],
    alias_permutation: np.ndarray,
    rng: np.random.Generator,
) -> dict[tuple[int, str], dict[str, float]]:
    output: dict[tuple[int, str], dict[str, float]] = {}
    for requested in query_counts:
        count = min(requested, len(query_order))
        columns = query_order[:count]
        semantic_candidate = traces["semantic_candidate"][:, columns]
        lexical_candidate = traces["lexical_candidate"][:, columns]
        semantic_response = traces["semantic_response"][:, columns]
        lexical_response = traces["lexical_response"][:, columns]
        semantic_rank = traces["semantic_rank"][:, columns]
        lexical_rank = traces["lexical_rank"][:, columns]

        candidate_similarity = sparse_cosine(semantic_candidate, lexical_candidate)
        response_similarity = sparse_cosine(semantic_response, lexical_response)
        rank_similarity = sparse_cosine(semantic_rank, lexical_rank)
        combined_similarity = 0.5 * candidate_similarity + 0.5 * rank_similarity
        for method, similarity in (
            ("candidate_membership", candidate_similarity),
            ("response_membership", response_similarity),
            ("response_reciprocal_rank", rank_similarity),
            ("candidate_plus_rank", combined_similarity),
        ):
            metrics = expected_matching_metrics(similarity[:, alias_permutation], alias_permutation)
            metrics["semantic_coverage"] = float(np.mean(np.asarray(
                (semantic_candidate if method.startswith("candidate") else semantic_response).getnnz(axis=1)
            ) > 0))
            metrics["lexical_coverage"] = float(np.mean(np.asarray(
                (lexical_candidate if method.startswith("candidate") else lexical_response).getnnz(axis=1)
            ) > 0))
            output[(count, method)] = metrics

        # Break simultaneous-query alignment while preserving both marginal traces.
        shuffled_columns = columns[rng.permutation(len(columns))]
        shuffled_lexical_candidate = traces["lexical_candidate"][:, shuffled_columns]
        shuffled_lexical_rank = traces["lexical_rank"][:, shuffled_columns]
        shuffled_similarity = 0.5 * sparse_cosine(
            semantic_candidate, shuffled_lexical_candidate
        ) + 0.5 * sparse_cosine(semantic_rank, shuffled_lexical_rank)
        metrics = expected_matching_metrics(
            shuffled_similarity[:, alias_permutation], alias_permutation
        )
        metrics["semantic_coverage"] = float(np.mean(np.asarray(semantic_candidate.getnnz(axis=1)) > 0))
        metrics["lexical_coverage"] = float(np.mean(np.asarray(shuffled_lexical_candidate.getnnz(axis=1)) > 0))
        output[(count, "candidate_plus_rank_time_shuffled_control")] = metrics
    return output


def run_dataset(name: str, args: argparse.Namespace) -> dict[str, object]:
    config = CONFIGS[name]
    cache_dir = Path(config["semantic_cache"])
    started = time.perf_counter()
    routed = route_semantic_queries(cache_dir, str(config["query_embeddings"]), args.probes)
    assignments = np.load(cache_dir / "semantic_assignments.npy", mmap_mode="r")
    cells = int(np.load(cache_dir / "semantic_centroids.npy", mmap_mode="r").shape[0])
    lexical_candidates = np.load(config["lexical_candidates"])[str(config["lexical_candidate_key"])]
    semantic_responses = response_array(Path(config["semantic_rankings"]), "semantic")
    lexical_responses = response_array(Path(config["lexical_rankings"]), "lexical_dpe")
    if not (
        len(routed)
        == len(lexical_candidates)
        == len(semantic_responses)
        == len(lexical_responses)
    ):
        raise ValueError(f"{name}: query counts do not agree across traces")

    lexical_candidate_documents = np.load(config["lexical_rows"], mmap_mode="r")
    eligible_candidate = np.asarray(lexical_candidate_documents, dtype=np.int32)
    eligible_candidate = eligible_candidate[
        semantic_candidate_eligibility(eligible_candidate, assignments, routed, cells)
    ]
    eligible_return = np.intersect1d(
        unique_valid(semantic_responses), unique_valid(lexical_responses), assume_unique=True
    ).astype(np.int32)

    cohorts = {
        "dual_candidate_observed": eligible_candidate,
        "dual_response_observed": eligible_return,
    }
    collected: dict[tuple[str, int, str], list[dict[str, float]]] = {}
    cohort_metadata = {}
    query_counts = sorted(set(min(value, len(routed)) for value in args.query_counts + [len(routed)]))

    for cohort_name, eligible in cohorts.items():
        if len(eligible) < args.pool_size:
            raise ValueError(f"{name}/{cohort_name}: only {len(eligible)} eligible documents")
        cohort_metadata[cohort_name] = {
            "eligible_documents": int(len(eligible)),
            "eligible_fraction_of_corpus": float(len(eligible) / int(config["documents"])),
        }
        for repeat in range(args.repeats):
            rng = np.random.default_rng(args.seed + 10007 * repeat + 101 * len(name) + len(cohort_name))
            documents = rng.choice(eligible, args.pool_size, replace=False).astype(np.int32)
            traces = {
                "semantic_candidate": semantic_candidate_trace(documents, assignments, routed, cells),
                "lexical_candidate": trace_from_rows(documents, lexical_candidates, False),
                "semantic_response": trace_from_rows(documents, semantic_responses, False),
                "lexical_response": trace_from_rows(documents, lexical_responses, False),
                "semantic_rank": trace_from_rows(documents, semantic_responses, True),
                "lexical_rank": trace_from_rows(documents, lexical_responses, True),
            }
            query_order = rng.permutation(len(routed))
            alias_permutation = rng.permutation(args.pool_size)
            trial = evaluate_trace_set(traces, query_order, query_counts, alias_permutation, rng)
            for (count, method), metrics in trial.items():
                collected.setdefault((cohort_name, count, method), []).append(metrics)
            final = trial[(len(routed), "candidate_plus_rank")]
            print(
                f"{name:9s} {cohort_name:24s} repeat={repeat + 1}/{args.repeats} "
                f"queries={len(routed)} top1={final['top1']:.4f} "
                f"top10={final['top10']:.4f} mrr={final['mrr']:.4f}",
                flush=True,
            )

    rows = []
    for (cohort_name, count, method), trials in sorted(collected.items()):
        rows.append(
            {
                "cohort": cohort_name,
                "queries": count,
                "method": method,
                "summary": summarize(trials),
                "trials": trials,
            }
        )
    return {
        "dataset": name,
        "documents": int(config["documents"]),
        "queries": int(len(routed)),
        "semantic_probes": args.probes,
        "pool_size": args.pool_size,
        "random_baseline": random_baseline(args.pool_size),
        "cohorts": cohort_metadata,
        "results": rows,
        "seconds": float(time.perf_counter() - started),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=sorted(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--query-counts", nargs="+", type=int, default=[10, 50, 100, 500, 1000])
    parser.add_argument("--pool-size", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--probes", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/cross_path_cooccurrence.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = {
        "attack": "cross-path candidate and response co-occurrence linkage",
        "attacker_view": (
            "synchronized query episodes, path-local candidate membership, returned handles, "
            "and returned rank; lexical aliases are independently permuted"
        ),
        "configuration": {
            "datasets": args.datasets,
            "query_counts": args.query_counts,
            "pool_size": args.pool_size,
            "repeats": args.repeats,
            "semantic_probes": args.probes,
            "seed": args.seed,
        },
        "datasets": [],
    }
    for dataset in args.datasets:
        report["datasets"].append(run_dataset(dataset, args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
