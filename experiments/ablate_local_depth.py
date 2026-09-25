"""Reproduce the lexical rows of the final compartment-depth ablation.

Semantic depth rows are emitted by the semantic sweep in ``run_full_corpus``;
this entry point evaluates the final 8192/32/16 lexical configuration and then
applies candidate-local exact BM25, as reported in the paper.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from exact_bm25 import (
    build_candidate_term_matrix,
    exact_bm25_rerank,
    load_document_frequencies,
    load_document_lengths,
    query_vocabulary,
    selected_document_rows,
)
from run_full_corpus import (
    FULL_DOCUMENTS,
    build_lexical_compartment_cache,
    lexical_compartment_retrieval,
    metric_mean,
)
from semantic_backend import atomic_json, load_queries_qrels, read_ids


ROOT = Path(__file__).resolve().parent
DEPTHS = (32, 64, 128)
CONFIG = {
    "nq": ("test", "nq_full", "nq_full_h1024", 50_000, 3),
    "hotpotqa": ("test", "hotpotqa_full", "hotpotqa_full_h1024", 50_000, 3),
    "msmarco": ("dev", "msmarco_dev_full", "msmarco_dev_b250k_full_h1024", 250_000, 1),
}


def run_dataset(dataset: str) -> dict[str, object]:
    split, workload, lexical_label, posting_budget, repeats = CONFIG[dataset]
    documents = FULL_DOCUMENTS[dataset]
    query_ids, query_texts, qrels, _ = load_queries_qrels(
        ROOT / "data" / dataset, split
    )
    lexical_source = ROOT / "cache" / "full_candidate_lexical_dpe" / lexical_label
    cache_label = f"{dataset}_{documents}_dev" if split == "dev" else f"{dataset}_{documents}"
    compartment = ROOT / "cache" / "dueter_final" / cache_label
    build_lexical_compartment_cache(
        lexical_source,
        compartment,
        work_dimension=8192,
        projection_dimension=32,
        cells=16,
        beta=0.10,
        scale=3.0,
        seed=20260917,
    )
    raw = np.asarray(
        np.load(ROOT / "results" / "budgeted_lexical" / f"{workload}.raw.npz")[
            f"budget_{posting_budget}"
        ][:, :1000],
        dtype=np.int32,
    )
    clouds, timing = lexical_compartment_retrieval(
        dataset,
        lexical_source,
        compartment,
        raw,
        query_texts,
        documents=documents,
        work_dimension=8192,
        projection_dimension=32,
        cells=16,
        top_per_cell_values=list(DEPTHS),
        output_depth=300,
        repeats=repeats,
        beta=0.10,
        scale=3.0,
        seed=20260917,
    )

    stacked = np.concatenate([clouds[depth] for depth in DEPTHS], axis=0)
    selected_rows = selected_document_rows(stacked)
    terms, term_to_index = query_vocabulary(query_texts)
    database = ROOT / "results" / "keyed_fts5" / f"{dataset}_full.sqlite3"
    payload_cache = ROOT / "cache" / "exact_bm25_depth" / f"{dataset}_{split}"
    full_lengths = load_document_lengths(database, payload_cache / "full_doc_lengths.npy")
    document_frequencies = load_document_frequencies(database, terms)
    token_lengths, indptr, indices, counts = build_candidate_term_matrix(
        ROOT / "data" / dataset / "corpus.jsonl",
        selected_rows,
        term_to_index,
        payload_cache,
    )
    exact, _ = exact_bm25_rerank(
        stacked,
        selected_rows,
        token_lengths,
        indptr,
        indices,
        counts,
        query_texts,
        term_to_index,
        document_frequencies,
        len(full_lengths),
        float(np.mean(full_lengths)),
    )
    doc_ids = read_ids(
        ROOT / "cache" / "full_semantic" / f"{dataset}_{documents}" / "doc_ids.txt"
    )
    rows = []
    for index, depth in enumerate(DEPTHS):
        first, last = index * repeats, (index + 1) * repeats
        metrics = metric_mean(exact[first:last], doc_ids, query_ids, qrels)
        rows.append(
            {
                "local_depth": depth,
                "nDCG@10": float(metrics["nDCG@10"]),
                "Recall@100": float(metrics["Recall@100"]),
                "mean_union": float(timing["by_top_per_cell"][depth]["candidate_mean"]),
            }
        )
    return {"dataset": dataset, "documents": documents, "queries": len(query_ids), "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=tuple(CONFIG), default=list(CONFIG))
    args = parser.parse_args()
    report = {
        "experiment": "final lexical compartment local-depth sensitivity",
        "fixed_parameters": {
            "work_dimension": 8192,
            "projection_dimension": 32,
            "compartments": 16,
        },
        "datasets": [run_dataset(dataset) for dataset in args.datasets],
    }
    destination = ROOT.parent / "results" / "generated" / "lexical_local_depth.json"
    atomic_json(destination, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
