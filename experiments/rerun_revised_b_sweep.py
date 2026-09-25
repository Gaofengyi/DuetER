"""Quality/security trade-off sweep over B at fixed lexical d=8192, r_c=32."""

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
from evaluate_lexical_d8192_rc32_b_sweep import build_local_cache, retrieve
from rerun_revised_main import CONFIG, build_local_queries, lexical_document_frequencies, previous_metrics

SWEEP = {
    "nq": ((16, 64), (32, 32), (64, 16)),
    "hotpotqa": ((16, 64), (32, 32), (64, 16)),
    "msmarco": ((16, 64), (32, 32), (64, 16)),
}


def run(dataset: str) -> dict[str, object]:
    cfg = CONFIG[dataset]
    started = time.perf_counter()
    source = ROOT / "cache" / "full_candidate_lexical_dpe" / cfg["source"]
    destination = ROOT / "results" / "revised_d8192_rc32_b_sweep" / cfg["result"]
    destination.mkdir(parents=True, exist_ok=True)
    database = ROOT / "results" / "keyed_fts5" / cfg["fts"]
    query_ids, query_texts, qrels, _ = load_queries_qrels(ROOT / "data" / cfg["data"], cfg["split"])
    df = lexical_document_frequencies(query_texts, database)
    plain = np.stack([query_mips(text, df, int(cfg["documents"]), HASH_DIMENSION) for text in query_texts])
    raw = np.asarray(np.load(ROOT / "results" / "budgeted_lexical" / cfg["raw"])[f"budget_{cfg['budget']}"][:, :1000], dtype=np.int32)
    clouds, partial = [], []
    for cells, depth in SWEEP[dataset]:
        if cells == 16:
            cache = ROOT / "cache" / "revised_d8192_rc32_b16" / cfg["result"]
            if dataset == "nq":
                cache = ROOT / "cache" / "lexical_d8192_rc32_b_sweep" / "nq_2681468_b16"
        else:
            cache = ROOT / "cache" / "revised_d8192_rc32_b_sweep" / f"{cfg['result']}_b{cells}"
        transforms = [cell_transform(KEY_LEXICAL, cell, WORK_DIMENSION, PROJECTION_DIMENSION) for cell in range(cells)]
        matrices = [selected_dpe_matrix(item.coordinates, GLOBAL_SIGN1, GLOBAL_SIGN2, GLOBAL_PERMUTATION, HASH_DIMENSION + 1) for item in transforms]
        setup = build_local_cache(source, cache, cells, transforms, matrices)
        local_queries, query_timing = build_local_queries(plain, transforms)
        cloud, timing = retrieve(source, cache, raw, local_queries, depth)
        clouds.append(cloud)
        partial.append({"partitions": cells, "local_depth": depth, "maximum_return_budget": cells * depth, "cache_setup": setup, "query_timing": query_timing, "server_timing": timing})
    stacked = np.stack(clouds)
    rows = selected_document_rows(stacked)
    terms, term_to_index = query_vocabulary(query_texts)
    full_lengths = load_document_lengths(database, ROOT / "cache" / "exact_bm25_client_rerank" / cfg["result"] / "full_doc_lengths_v2.npy")
    exact_df = load_document_frequencies(database, terms)
    # Dedicated rerun cache: selected-row unions differ from the preliminary
    # B-sweep and the exact-BM25 cache is intentionally immutable.
    payload_cache = ROOT / "cache" / "exact_bm25_client_rerank" / f"{cfg['result']}_d8192_rc32_b_sweep_rerun_v3"
    token_lengths, indptr, indices, counts = build_candidate_term_matrix(ROOT / "data" / cfg["data"] / "corpus.jsonl", rows, term_to_index, payload_cache)
    exact, exact_timing = exact_bm25_rerank(stacked, rows, token_lengths, indptr, indices, counts, query_texts, term_to_index, exact_df, len(full_lengths), float(np.mean(full_lengths)))
    np.savez_compressed(destination / "rankings.npz", configurations=np.asarray(SWEEP[dataset]), cloud=stacked, lexical_dpe=exact)
    doc_ids = read_ids(ROOT / "cache" / "full_semantic" / f"{cfg['data']}_{cfg['documents']}" / "doc_ids.txt")
    baseline = previous_metrics(str(cfg["result"]))
    result_rows = []
    for index, base in enumerate(partial):
        metrics = metric_mean(exact[index:index + 1], doc_ids, query_ids, qrels)
        result_rows.append({**base, "cloud_candidate_metrics": metric_mean(stacked[index:index + 1], doc_ids, query_ids, qrels), "cloud_relevant_candidate_coverage": relevant_coverage(stacked[index], doc_ids, query_ids, qrels), "exact_bm25_metrics": metrics, "retention_vs_paper_configuration": {key: float(metrics[key] / baseline[key]) for key in metrics}})
    report = {"experiment": "B sweep at lexical d=8192 r_c=32", "dataset": dataset, "paper_baseline": baseline, "configuration_results": result_rows, "exact_bm25_latency": exact_timing, "candidate_documents_materialized": int(len(rows)), "elapsed_seconds": time.perf_counter() - started}
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
