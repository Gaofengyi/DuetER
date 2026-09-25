"""Local-depth ablation for the revised d=8192, r_c=32, B=16 setting."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if (ROOT / ".deps").exists():
    sys.path.insert(0, str(ROOT / ".deps"))
os.environ.setdefault("DUETER_FORCE_CPU_DEPS", "1")

import numpy as np

from benchmark_dual_compartment_full import KEY_LEXICAL, cell_transform, metric_mean, relevant_coverage
from benchmark_exact_bm25_client_rerank import (
    build_candidate_term_matrix, exact_bm25_rerank, load_document_frequencies,
    load_document_lengths, query_vocabulary, selected_document_rows,
)
from benchmark_million_lexical_dpe import query_mips
from benchmark_million_semantic_hybrid import atomic_json, load_queries_qrels, read_ids
from evaluate_lexical_d8192_rc32 import (
    GLOBAL_PERMUTATION, GLOBAL_SIGN1, GLOBAL_SIGN2, HASH_DIMENSION,
    PROJECTION_DIMENSION, WORK_DIMENSION, selected_dpe_matrix,
)
from evaluate_lexical_d8192_rc32_b_sweep import retrieve
from rerun_revised_main import CONFIG, CELLS, build_local_queries, lexical_document_frequencies, previous_metrics

DEPTHS = {
    "nq": (32, 64, 128),
    "hotpotqa": (16, 32, 64, 128),
    "msmarco": (32, 64, 128),
}


def run(dataset: str) -> dict[str, object]:
    cfg = CONFIG[dataset]
    started = time.perf_counter()
    source = ROOT / "cache" / "full_candidate_lexical_dpe" / cfg["source"]
    cache = ROOT / "cache" / "revised_d8192_rc32_b16" / cfg["result"]
    if dataset == "nq":
        cache = ROOT / "cache" / "lexical_d8192_rc32_b_sweep" / "nq_2681468_b16"
    database = ROOT / "results" / "keyed_fts5" / cfg["fts"]
    destination = ROOT / "results" / "revised_d8192_rc32_b16_depth_ablation" / cfg["result"]
    destination.mkdir(parents=True, exist_ok=True)
    query_ids, query_texts, qrels, _ = load_queries_qrels(ROOT / "data" / cfg["data"], cfg["split"])
    transforms = [cell_transform(KEY_LEXICAL, cell, WORK_DIMENSION, PROJECTION_DIMENSION) for cell in range(CELLS)]
    # Constructing these matrices also checks the same selector/key instance as the cache.
    _ = [selected_dpe_matrix(item.coordinates, GLOBAL_SIGN1, GLOBAL_SIGN2, GLOBAL_PERMUTATION, HASH_DIMENSION + 1) for item in transforms]
    df = lexical_document_frequencies(query_texts, database)
    plain = np.stack([query_mips(text, df, int(cfg["documents"]), HASH_DIMENSION) for text in query_texts])
    local_queries, query_timing = build_local_queries(plain, transforms)
    raw = np.asarray(np.load(ROOT / "results" / "budgeted_lexical" / cfg["raw"])[f"budget_{cfg['budget']}"][:, :1000], dtype=np.int32)
    clouds, partial = [], []
    for depth in DEPTHS[dataset]:
        cloud, timing = retrieve(source, cache, raw, local_queries, depth)
        clouds.append(cloud)
        partial.append({"local_depth": depth, "maximum_return_budget": CELLS * depth, "server_timing": timing})
    stacked = np.stack(clouds)
    rows = selected_document_rows(stacked)
    terms, term_to_index = query_vocabulary(query_texts)
    full_lengths = load_document_lengths(database, ROOT / "cache" / "exact_bm25_client_rerank" / cfg["result"] / "full_doc_lengths_v2.npy")
    exact_df = load_document_frequencies(database, terms)
    payload_cache = ROOT / "cache" / "exact_bm25_client_rerank" / f"{cfg['result']}_d8192_rc32_b16_depth_ablation"
    token_lengths, indptr, indices, counts = build_candidate_term_matrix(
        ROOT / "data" / cfg["data"] / "corpus.jsonl", rows, term_to_index, payload_cache
    )
    exact, exact_timing = exact_bm25_rerank(
        stacked, rows, token_lengths, indptr, indices, counts, query_texts,
        term_to_index, exact_df, len(full_lengths), float(np.mean(full_lengths)),
    )
    np.savez_compressed(destination / "rankings.npz", depths=np.asarray(DEPTHS[dataset]), cloud=stacked, lexical_dpe=exact)
    doc_ids = read_ids(ROOT / "cache" / "full_semantic" / f"{cfg['data']}_{cfg['documents']}" / "doc_ids.txt")
    baseline = previous_metrics(str(cfg["result"]))
    rows_report = []
    for index, base in enumerate(partial):
        metrics = metric_mean(exact[index:index + 1], doc_ids, query_ids, qrels)
        rows_report.append({
            **base,
            "cloud_candidate_metrics": metric_mean(stacked[index:index + 1], doc_ids, query_ids, qrels),
            "cloud_relevant_candidate_coverage": relevant_coverage(stacked[index], doc_ids, query_ids, qrels),
            "exact_bm25_metrics": metrics,
            "retention_vs_paper_configuration": None if baseline is None else {key: float(metrics[key] / baseline[key]) for key in metrics},
        })
    report = {
        "experiment": "revised lexical local-depth ablation", "dataset": dataset,
        "fixed_parameters": {"work_dimension": WORK_DIMENSION, "projection_dimension": PROJECTION_DIMENSION, "partitions": CELLS, "posting_budget": cfg["budget"]},
        "query_preparation_timing": query_timing, "paper_baseline": baseline,
        "configuration_results": rows_report, "exact_bm25_latency": exact_timing,
        "candidate_documents_materialized": int(len(rows)), "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(destination / "results.json", report)
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(CONFIG), required=True)
    args = parser.parse_args()
    run(args.dataset)


if __name__ == "__main__":
    main()
