"""Compute the Table-1 plaintext/CCADPE nDCG gap for the revised lexical setup.

Both arms start from the same cumulative-posting candidate pool.  The reference
uses the unprojected signed-hash MIPS score inside each of the 16 keyed
partitions, whereas the revised arm uses the saved d=8192, r_c=32 CCADPE
result.  Both retain 64 records per partition, apply the same global 300-record
cap, and are finally reranked with exact BM25.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if (ROOT / ".deps").exists():
    sys.path.insert(0, str(ROOT / ".deps"))
os.environ.setdefault("DUETER_FORCE_CPU_DEPS", "1")

import numpy as np

from benchmark_dual_compartment_full import metric_mean, top_indices
from benchmark_exact_bm25_client_rerank import (
    build_candidate_term_matrix,
    exact_bm25_rerank,
    load_document_frequencies,
    load_document_lengths,
    query_vocabulary,
    selected_document_rows,
)
from benchmark_million_lexical_dpe import query_mips
from benchmark_million_semantic_hybrid import atomic_json, load_queries_qrels, read_ids
from rerun_revised_main import CONFIG, lexical_document_frequencies


CELLS = 16
LOCAL_DEPTH = 64
RETURNED_DEPTH = 300


def plaintext_candidates(dataset: str, query_texts: list[str]) -> np.ndarray:
    cfg = CONFIG[dataset]
    source = ROOT / "cache" / "full_candidate_lexical_dpe" / str(cfg["source"])
    database = ROOT / "results" / "keyed_fts5" / str(cfg["fts"])
    candidate_rows = np.load(source / "candidate_rows.npy", mmap_mode="r")
    sketches = np.load(source / "lexical_sketch_fp16.npy", mmap_mode="r")
    scope = json.loads((source / "candidate_scope.json").read_text(encoding="utf-8"))
    maximum_norm = float(scope["maximum_sketch_norm"])

    # The B=16 assignment used by the revised run is already persisted here.
    if dataset == "nq":
        local_cache = ROOT / "cache" / "lexical_d8192_rc32_b_sweep" / "nq_2681468_b16"
    else:
        local_cache = ROOT / "cache" / "revised_d8192_rc32_b16" / str(cfg["result"])
    assignments = np.load(local_cache / "lexical_compartment_cells.npy", mmap_mode="r")

    dfs = lexical_document_frequencies(query_texts, database)
    queries = np.stack(
        [query_mips(text, dfs, int(cfg["documents"]), 1024) for text in query_texts]
    )
    raw = np.asarray(
        np.load(ROOT / "results" / "budgeted_lexical" / str(cfg["raw"]))[
            f"budget_{cfg['budget']}"
        ][:, :1000],
        dtype=np.int32,
    )

    output = np.full((len(raw), RETURNED_DEPTH), -1, dtype=np.int32)
    for query_index, raw_row in enumerate(raw):
        candidates = np.asarray(raw_row[raw_row >= 0], dtype=np.int32)
        positions = np.searchsorted(candidate_rows, candidates)
        if np.any(np.asarray(candidate_rows[positions]) != candidates):
            raise RuntimeError("candidate absent from scoped lexical-sketch cache")
        candidate_cells = np.asarray(assignments[positions], dtype=np.int32)
        selected: list[int] = []
        query = np.asarray(queries[query_index, :1024], dtype=np.float32)
        for cell in np.unique(candidate_cells):
            ordinals = np.flatnonzero(candidate_cells == cell)
            local_positions = positions[ordinals]
            local_sketches = np.asarray(sketches[local_positions], dtype=np.float32)
            scores = (local_sketches @ query) / maximum_norm
            order = top_indices(scores, min(LOCAL_DEPTH, len(scores)))
            selected.extend(map(int, ordinals[order]))
        ordered = np.unique(selected)
        ordered.sort()
        documents = candidates[ordered]
        depth = min(RETURNED_DEPTH, len(documents))
        output[query_index, :depth] = documents[:depth]
        if query_index == 0 or query_index + 1 == len(raw) or (query_index + 1) % 250 == 0:
            print(f"[{dataset} plaintext reference] {query_index + 1:,}/{len(raw):,}", flush=True)
    return output


def revised_dpe_exact(dataset: str) -> np.ndarray:
    cfg = CONFIG[dataset]
    archive = np.load(
        ROOT / "results" / "revised_d8192_rc32_b_sweep" / str(cfg["result"]) / "rankings.npz"
    )
    configurations = np.asarray(archive["configurations"])
    matches = np.flatnonzero(
        (configurations[:, 0] == CELLS) & (configurations[:, 1] == LOCAL_DEPTH)
    )
    if len(matches) != 1:
        raise RuntimeError("missing unique B=16, t_l=64 revised result")
    return np.asarray(archive["lexical_dpe"][int(matches[0])], dtype=np.int32)


def run(dataset: str) -> dict[str, object]:
    cfg = CONFIG[dataset]
    destination = ROOT / "results" / "revised_d8192_rc32_b16_delta" / str(cfg["result"])
    destination.mkdir(parents=True, exist_ok=True)
    query_ids, query_texts, qrels, _ = load_queries_qrels(
        ROOT / "data" / str(cfg["data"]), str(cfg["split"])
    )
    plain_cloud_path = destination / "plaintext_cloud_candidates.npz"
    if plain_cloud_path.exists():
        plain_cloud = np.asarray(np.load(plain_cloud_path)["lexical"], dtype=np.int32)
    else:
        plain_cloud = plaintext_candidates(dataset, query_texts)
        np.savez_compressed(plain_cloud_path, lexical=plain_cloud)

    selected_rows = selected_document_rows(plain_cloud[None, :, :])
    terms, term_to_index = query_vocabulary(query_texts)
    database = ROOT / "results" / "keyed_fts5" / str(cfg["fts"])
    full_lengths = load_document_lengths(
        database,
        ROOT / "cache" / "exact_bm25_client_rerank" / str(cfg["result"]) / "full_doc_lengths_v2.npy",
    )
    exact_df = load_document_frequencies(database, terms)
    payload_cache = ROOT / "cache" / "exact_bm25_client_rerank" / f"{cfg['result']}_revised_plain_delta"
    token_lengths, indptr, indices, counts = build_candidate_term_matrix(
        ROOT / "data" / str(cfg["data"]) / "corpus.jsonl",
        selected_rows,
        term_to_index,
        payload_cache,
    )
    plain_exact, timing = exact_bm25_rerank(
        plain_cloud[None, :, :],
        selected_rows,
        token_lengths,
        indptr,
        indices,
        counts,
        query_texts,
        term_to_index,
        exact_df,
        len(full_lengths),
        float(np.mean(full_lengths)),
    )
    dpe_exact = revised_dpe_exact(dataset)
    doc_ids = read_ids(
        ROOT / "cache" / "full_semantic" / f"{cfg['data']}_{cfg['documents']}" / "doc_ids.txt"
    )
    plain_metrics = metric_mean(plain_exact, doc_ids, query_ids, qrels)
    dpe_metrics = metric_mean(dpe_exact[None, :, :], doc_ids, query_ids, qrels)
    delta = abs(float(plain_metrics["nDCG@10"]) - float(dpe_metrics["nDCG@10"]))
    report = {
        "experiment": "revised same-posting-pool plaintext-sketch versus CCADPE",
        "dataset": dataset,
        "parameters": {
            "hash_dimension": 1024,
            "dpe_work_dimension": 8192,
            "projection_dimension": 32,
            "partitions": CELLS,
            "local_depth": LOCAL_DEPTH,
            "returned_depth": RETURNED_DEPTH,
            "posting_budget": int(cfg["budget"]),
        },
        "plaintext_exact_bm25_metrics": plain_metrics,
        "revised_ccadpe_exact_bm25_metrics": dpe_metrics,
        "absolute_ndcg_at_10_delta": delta,
        "exact_bm25_timing": timing,
        "plaintext_candidate_documents_materialized": int(len(selected_rows)),
    }
    np.savez_compressed(destination / "plaintext_exact_bm25_rankings.npz", lexical=plain_exact)
    atomic_json(destination / "results.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(CONFIG), required=True)
    args = parser.parse_args()
    run(args.dataset)


if __name__ == "__main__":
    main()
