"""Sweep and evaluate the redesigned semantic candidate index on cached embeddings."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from dueter_common import evaluate, load_beir, normalize_rows, top_indices
from semantic_candidate_index import KeyedResidualSphericalIVF


ROOT = Path(__file__).resolve().parent


def run_configuration(
    documents: np.ndarray,
    queries: np.ndarray,
    query_ids: list[str],
    corpus_ids: list[str],
    qrels: dict[str, dict[str, int]],
    dense_top100: list[np.ndarray],
    *,
    centered: bool,
    n_clusters: int,
    assignments: int,
    probes: int,
    seed: int,
    index: KeyedResidualSphericalIVF | None = None,
    setup: dict | None = None,
) -> dict:
    if index is None:
        index = KeyedResidualSphericalIVF(
            b"DuetDPE-semantic-residual-ivf-v1",
            n_clusters=n_clusters,
            doc_assignments=assignments,
            nprobe=probes,
            centered=centered,
            seed=seed,
        )
        setup = index.build(documents)
    elif setup is None:
        raise ValueError("setup metadata is required with a prebuilt index")
    counts: list[int] = []
    reads: list[int] = []
    coverage: list[float] = []
    relevant_coverage: list[float] = []
    latencies: list[float] = []
    rankings: dict[str, list[int]] = {}
    corpus_id_to_index = {doc_id: idx for idx, doc_id in enumerate(corpus_ids)}
    for qi, qid in enumerate(query_ids):
        started = time.perf_counter()
        candidates, trace = index.candidates(queries[qi], nprobe=probes)
        latencies.append(time.perf_counter() - started)
        scores = documents[candidates] @ queries[qi]
        rank = candidates[top_indices(scores, min(100, len(candidates)))]
        rankings[qid] = rank.tolist()
        counts.append(len(candidates))
        reads.append(trace.posting_entries_read)
        reference = set(map(int, dense_top100[qi]))
        coverage.append(len(reference.intersection(map(int, candidates))) / len(reference))
        relevant = {
            corpus_id_to_index[doc_id]
            for doc_id, gain in qrels[qid].items()
            if gain > 0 and doc_id in corpus_id_to_index
        }
        relevant_coverage.append(
            len(relevant.intersection(map(int, candidates))) / len(relevant) if relevant else 1.0
        )
    storage = index.storage_summary()
    storage["nprobe"] = probes
    return {
        "configuration": {
            "centered": centered,
            "n_clusters": n_clusters,
            "doc_assignments": assignments,
            "nprobe": probes,
        },
        "setup": setup,
        "storage": storage,
        "candidate_mean": float(np.mean(counts)),
        "candidate_p50": float(np.percentile(counts, 50)),
        "candidate_p95": float(np.percentile(counts, 95)),
        "candidate_fraction_mean": float(np.mean(counts) / len(documents)),
        "posting_entries_read_mean": float(np.mean(reads)),
        "dense_top100_coverage_mean": float(np.mean(coverage)),
        "dense_top100_coverage_p05": float(np.percentile(coverage, 5)),
        "relevant_document_coverage_mean": float(np.mean(relevant_coverage)),
        "exact_rerank_metrics": evaluate(rankings, query_ids, corpus_ids, qrels),
        "candidate_lookup_mean_ms": float(np.mean(latencies) * 1000),
        "candidate_lookup_p95_ms": float(np.percentile(latencies, 95) * 1000),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("scifact", "nfcorpus"), required=True)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache" / "granite_gpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()

    data_dir = ROOT / "data" / args.dataset
    corpus_ids, _, query_ids, _, qrels = load_beir(data_dir)
    corpus_path = next(args.cache_dir.glob(f"{args.dataset}-st-*_corpus_embeddings.npy"))
    query_path = next(args.cache_dir.glob(f"{args.dataset}-st-*_query_embeddings.npy"))
    documents = normalize_rows(np.load(corpus_path))
    queries = normalize_rows(np.load(query_path))
    dense_top100 = [top_indices(documents @ query, 100) for query in queries]
    dense_rankings = {qid: dense_top100[i].tolist() for i, qid in enumerate(query_ids)}

    # Each server-side assignment shape is trained once and evaluated at four
    # query-side probe budgets.
    index_shapes = [
        (True, 128, 2),
        (True, 256, 1),
        (True, 256, 2),
        (True, 256, 4),
        (True, 512, 2),
        (False, 256, 2),
    ]
    results = []
    for centered, clusters, assignments in index_shapes:
        index = KeyedResidualSphericalIVF(
            b"DuetDPE-semantic-residual-ivf-v1",
            n_clusters=clusters,
            doc_assignments=assignments,
            nprobe=16,
            centered=centered,
            seed=args.seed + clusters + assignments,
        )
        setup = index.build(documents)
        for probes in (4, 8, 16, 32):
            if probes > clusters:
                continue
            print(
                f"[{args.dataset}] centered={centered} K={clusters} "
                f"r={assignments} nprobe={probes}",
                flush=True,
            )
            results.append(
                run_configuration(
                    documents,
                    queries,
                    query_ids,
                    corpus_ids,
                    qrels,
                    dense_top100,
                    centered=centered,
                    n_clusters=clusters,
                    assignments=assignments,
                    probes=probes,
                    seed=args.seed + clusters + assignments,
                    index=index,
                    setup=setup,
                )
            )

    dense_metrics = evaluate(dense_rankings, query_ids, corpus_ids, qrels)
    eligible = [
        row for row in results
        if row["exact_rerank_metrics"]["nDCG@10"]
        >= 0.98 * dense_metrics["nDCG@10"]
        and row["exact_rerank_metrics"]["Recall@100"]
        >= 0.95 * dense_metrics["Recall@100"]
        and row["candidate_fraction_mean"] <= 0.40
    ]
    if not eligible:
        eligible = sorted(
            results,
            key=lambda row: (
                -row["exact_rerank_metrics"]["nDCG@10"],
                -row["exact_rerank_metrics"]["Recall@100"],
                row["candidate_mean"],
            ),
        )[:1]
    recommended = min(
        eligible,
        key=lambda row: (
            row["candidate_mean"],
            -row["dense_top100_coverage_mean"],
            row["storage"]["server_estimated_bytes"],
        ),
    )
    output = {
        "dataset": args.dataset,
        "documents": len(corpus_ids),
        "queries": len(query_ids),
        "embedding_dimension": int(documents.shape[1]),
        "dense_plain_metrics": dense_metrics,
        "selection_rule": "minimum mean candidates subject to nDCG@10 retention >= 0.98, Recall@100 retention >= 0.95, and mean candidate fraction <= 0.40",
        "recommended": recommended,
        "sweep": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"recommended": recommended}, indent=2), flush=True)


if __name__ == "__main__":
    main()
