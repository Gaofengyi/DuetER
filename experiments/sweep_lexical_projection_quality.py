"""Full-NQ lexical retrieval quality sweep over CCADPE projection dimension."""

from __future__ import annotations

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

from benchmark_dual_compartment_full import (
    FULL_DOCUMENTS,
    build_lexical_compartment_cache,
    lexical_compartment_retrieval,
    metric_mean,
    relevant_coverage,
)
from benchmark_exact_bm25_client_rerank import (
    build_candidate_term_matrix,
    exact_bm25_rerank,
    load_document_frequencies,
    load_document_lengths,
    query_vocabulary,
    selected_document_rows,
)
from benchmark_million_semantic_hybrid import (
    atomic_json,
    load_queries_qrels,
    read_ids,
)


DATASET = "nq"
DOCUMENTS = FULL_DOCUMENTS[DATASET]
PROJECTIONS = (256, 128, 64, 32, 16)
LOCAL_DEPTH = 16
BASE_SEED = 20260917


def main() -> None:
    started = time.perf_counter()
    source = ROOT / "cache" / "full_candidate_lexical_dpe" / "nq_full_h1024"
    query_ids, query_texts, qrels, _ = load_queries_qrels(
        ROOT / "data" / DATASET, "test"
    )
    raw_path = ROOT / "results" / "budgeted_lexical" / "nq_full.raw.npz"
    raw_candidates = np.asarray(
        np.load(raw_path)["budget_50000"][:, :1000], dtype=np.int32
    )
    if len(raw_candidates) != len(query_ids):
        raise RuntimeError("raw candidate/query count mismatch")

    output_root = ROOT / "results" / "security" / "lexical_projection_quality"
    output_root.mkdir(parents=True, exist_ok=True)
    rankings_by_projection: list[np.ndarray] = []
    setup_rows: list[dict[str, object]] = []

    for projection in PROJECTIONS:
        cache = (
            ROOT
            / "cache"
            / f"dual_compartment_full_p{projection}"
            / f"nq_{DOCUMENTS}"
        )
        print(f"[projection {projection}] preparing cache", flush=True)
        setup = build_lexical_compartment_cache(source, cache, projection, 64)
        print(f"[projection {projection}] running all {len(query_ids)} queries", flush=True)
        outputs, timing = lexical_compartment_retrieval(
            DATASET,
            source,
            cache,
            raw_candidates,
            query_texts,
            documents=DOCUMENTS,
            projection_dimension=projection,
            cells=64,
            top_per_cell_values=[LOCAL_DEPTH],
            output_depth=300,
            repeats=1,
            beta=0.10,
            scale=3.0,
            seed=BASE_SEED,
        )
        ranking = np.asarray(outputs[LOCAL_DEPTH][0], dtype=np.int32)
        rankings_by_projection.append(ranking)
        np.savez_compressed(
            output_root / f"p{projection}_cloud_candidates.npz", lexical=ranking
        )
        setup_rows.append(
            {
                "projection_dimension": projection,
                "cache_build_seconds": float(setup.get("build_seconds", 0.0)),
                "cloud_candidate_metrics": None,
                "cloud_relevant_candidate_coverage": None,
                "server_latency_ms": timing["server_latency_ms"],
                "candidate_mean": timing["by_top_per_cell"][LOCAL_DEPTH][
                    "candidate_mean"
                ],
            }
        )

    # Treat projection settings as the leading dimension so exact BM25 can
    # rerank the union once while retaining a separate result for every r_c.
    stacked = np.stack(rankings_by_projection, axis=0)
    selected_rows = selected_document_rows(stacked)
    exact_cache = (
        ROOT / "cache" / "exact_bm25_client_rerank" / "nq_projection_sweep"
    )
    terms, term_to_index = query_vocabulary(query_texts)
    database = ROOT / "results" / "keyed_fts5" / "nq_full.sqlite3"
    full_lengths = load_document_lengths(
        database,
        ROOT
        / "cache"
        / "exact_bm25_client_rerank"
        / "nq"
        / "full_doc_lengths_v2.npy",
    )
    document_frequencies = load_document_frequencies(database, terms)
    token_lengths, indptr, indices, counts = build_candidate_term_matrix(
        ROOT / "data" / DATASET / "corpus.jsonl",
        selected_rows,
        term_to_index,
        exact_cache,
    )
    exact, bm25_latency = exact_bm25_rerank(
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
    np.savez_compressed(
        output_root / "exact_bm25_rankings.npz",
        projection_dimensions=np.asarray(PROJECTIONS, dtype=np.int32),
        lexical=exact,
    )

    doc_ids = read_ids(
        ROOT / "cache" / "full_semantic" / f"nq_{DOCUMENTS}" / "doc_ids.txt"
    )
    raw_metrics = metric_mean(raw_candidates[None, :, :300], doc_ids, query_ids, qrels)
    raw_coverage = relevant_coverage(
        raw_candidates[:, :300], doc_ids, query_ids, qrels
    )
    result_rows: list[dict[str, object]] = []
    for index, projection in enumerate(PROJECTIONS):
        cloud_metrics = metric_mean(
            stacked[index : index + 1], doc_ids, query_ids, qrels
        )
        exact_metrics = metric_mean(
            exact[index : index + 1], doc_ids, query_ids, qrels
        )
        coverage = relevant_coverage(stacked[index], doc_ids, query_ids, qrels)
        setup_rows[index]["cloud_candidate_metrics"] = cloud_metrics
        setup_rows[index]["cloud_relevant_candidate_coverage"] = coverage
        result_rows.append(
            {
                **setup_rows[index],
                "exact_bm25_metrics": exact_metrics,
                "ndcg_retention_vs_p256": None,
                "recall100_retention_vs_p256": None,
            }
        )
    baseline = result_rows[0]["exact_bm25_metrics"]
    for row in result_rows:
        metrics = row["exact_bm25_metrics"]
        row["ndcg_retention_vs_p256"] = float(
            metrics["nDCG@10"] / baseline["nDCG@10"]
        )
        row["recall100_retention_vs_p256"] = float(
            metrics["Recall@100"] / baseline["Recall@100"]
        )

    report = {
        "experiment": "lexical CCADPE projection-dimension retrieval-quality sweep",
        "dataset": DATASET,
        "documents": DOCUMENTS,
        "queries": len(query_ids),
        "fixed_parameters": {
            "work_dimension": 2048,
            "partitions": 64,
            "local_depth_per_partition": LOCAL_DEPTH,
            "posting_budget": 50000,
            "raw_candidates_per_query": 1000,
            "returned_depth": 300,
            "beta": 0.10,
            "scale": 3.0,
            "seed": BASE_SEED,
        },
        "plaintext_candidate_order_metrics_at_300": raw_metrics,
        "plaintext_candidate_relevant_coverage_at_300": raw_coverage,
        "exact_bm25_latency": bm25_latency,
        "candidate_documents_materialized": int(len(selected_rows)),
        "projection_results": result_rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(output_root / "results.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
