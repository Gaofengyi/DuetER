"""Re-evaluate lexical CCADPE local-depth ablations with exact client BM25."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("DUETER_FORCE_CPU_DEPS", "1")

from benchmark_budgeted_lexical_candidates import ROOT
from benchmark_dual_compartment_full import (
    FULL_DOCUMENTS,
    lexical_compartment_retrieval,
    metric_mean,
    read_ids,
)
from benchmark_exact_bm25_client_rerank import (
    build_candidate_term_matrix,
    exact_bm25_rerank,
    load_document_frequencies,
    load_document_lengths,
    query_vocabulary,
    selected_document_rows,
)
from benchmark_million_semantic_hybrid import atomic_json, load_queries_qrels


CONFIGS = {
    "nq": {"split": "test", "depths": [8, 16, 32], "budget": 50_000},
    "hotpotqa": {"split": "test", "depths": [4, 8, 16], "budget": 50_000},
    "msmarco": {"split": "dev", "depths": [8, 16, 32], "budget": 250_000},
}


def run_dataset(dataset: str) -> dict[str, object]:
    cfg = CONFIGS[dataset]
    split = str(cfg["split"])
    documents = FULL_DOCUMENTS[dataset]
    label = "msmarco_dev_b250k_full_h1024" if dataset == "msmarco" else f"{dataset}_full_h1024"
    workload = "msmarco_dev_full" if dataset == "msmarco" else f"{dataset}_full"
    cache_label = f"{dataset}_{documents}_{split}" if split != "test" else f"{dataset}_{documents}"
    lexical_source = ROOT / "cache" / "full_candidate_lexical_dpe" / label
    compartment = ROOT / "cache" / "dual_compartment_full_p256" / cache_label
    query_ids, query_texts, qrels, _ = load_queries_qrels(ROOT / "data" / dataset, split)
    raw = np.asarray(
        np.load(ROOT / "results" / "budgeted_lexical" / f"{workload}.raw.npz")[f"budget_{cfg['budget']}"][:, :1000],
        dtype=np.int32,
    )
    approximate, timing = lexical_compartment_retrieval(
        dataset,
        lexical_source,
        compartment,
        raw,
        query_texts,
        documents=documents,
        projection_dimension=256,
        cells=64,
        top_per_cell_values=list(cfg["depths"]),
        output_depth=300,
        repeats=3,
        beta=0.10,
        scale=3.0,
        seed=20260917 + 9000,
    )

    all_rankings = np.concatenate([approximate[depth] for depth in cfg["depths"]], axis=0)
    rows = selected_document_rows(all_rankings)
    destination = ROOT / "results" / "exact_bm25_local_depth" / dataset
    cache = ROOT / "cache" / "exact_bm25_local_depth" / dataset
    destination.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    terms, term_to_index = query_vocabulary(query_texts)
    database = ROOT / "results" / "keyed_fts5" / f"{dataset}_full.sqlite3"
    lengths = load_document_lengths(database, cache / "full_doc_lengths_v2.npy")
    average_length = float(np.mean(lengths))
    dfs = load_document_frequencies(database, terms)
    token_lengths, indptr, indices, counts = build_candidate_term_matrix(
        ROOT / "data" / dataset / "corpus.jsonl", rows, term_to_index, cache
    )
    doc_ids = read_ids(ROOT / "cache" / "full_semantic" / f"{dataset}_{documents}" / "doc_ids.txt")

    report_rows = []
    exact_outputs = {}
    for depth in cfg["depths"]:
        exact, latency = exact_bm25_rerank(
            approximate[depth], rows, token_lengths, indptr, indices, counts,
            query_texts, term_to_index, dfs, documents, average_length,
        )
        exact_outputs[f"lexical_t{depth}"] = exact
        metrics = metric_mean(exact, doc_ids, query_ids, qrels)
        report_rows.append(
            {
                "top_per_cell": depth,
                "metrics": metrics,
                "mean_union": float(timing["by_top_per_cell"][depth]["candidate_mean"]),
                "candidate_p95": float(timing["by_top_per_cell"][depth]["candidate_p95"]),
                "client_exact_bm25_latency_ms": latency,
                "cloud_latency_ms": timing["server_latency_ms"],
            }
        )
    np.savez_compressed(destination / "rankings.npz", **exact_outputs)
    report = {
        "experiment": "CCADPE lexical local-depth sensitivity with candidate-local exact BM25",
        "dataset": dataset,
        "documents": documents,
        "queries": len(query_ids),
        "repeats": 3,
        "bm25": {
            "k1": 1.2,
            "b": 0.75,
            "idf": "log(1+(N-df+0.5)/(df+0.5))",
            "query_weight": "binary unique-term",
        },
        "rows": report_rows,
        "scope": "exact BM25 only within each returned union; candidate omissions remain unrecoverable",
    }
    atomic_json(destination / "results.json", report)
    return report


def main() -> None:
    reports = [run_dataset(dataset) for dataset in CONFIGS]
    atomic_json(ROOT / "results" / "exact_bm25_local_depth" / "summary.json", {"datasets": reports})
    print(json.dumps({row["dataset"]: row["rows"] for row in reports}, indent=2))


if __name__ == "__main__":
    main()
