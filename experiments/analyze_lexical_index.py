"""Measure the MIPS-to-L2 tail distribution and new candidate-index coverage."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from lexical_mips_index import TailOrthogonalSparseMIPSIndex
from run_experiment import BM25Index, load_scifact, mips_document_transform, mips_query_transform, top_indices


ROOT = Path(__file__).resolve().parent


def main() -> None:
    corpus_ids, corpus_texts, query_ids, query_texts, _ = load_scifact(ROOT / "data" / "scifact")
    bm25 = BM25Index(corpus_texts)
    hashed = bm25.hashed_document_vectors(1024, b"DuetDPE-Lexical-v1")
    docs, scale = mips_document_transform(hashed)
    queries = np.stack(
        [
            mips_query_transform(
                bm25.hashed_query_vector(query, 1024, b"DuetDPE-Lexical-v1")
            )
            for query in query_texts
        ]
    )
    index = TailOrthogonalSparseMIPSIndex(b"DuetDPE-tail-orthogonal-index-v1")
    start = time.perf_counter()
    index.build(docs[:, :-1])
    build_seconds = time.perf_counter() - start

    rows = []
    for cap in (100, 200, 400, 800, 1600):
        coverages, latencies, posting_reads, uncapped = [], [], [], []
        for qi, query in enumerate(query_texts):
            plain_top100 = top_indices(bm25.score(query), 100)
            start = time.perf_counter()
            candidates, trace = index.candidates(queries[qi, :-1], cap)
            latencies.append((time.perf_counter() - start) * 1000.0)
            coverages.append(len(set(map(int, plain_top100)).intersection(map(int, candidates))) / 100.0)
            posting_reads.append(trace["posting_entries_read"])
            uncapped.append(trace["uncapped_candidates"])
        rows.append(
            {
                "cap": cap,
                "top100_coverage_mean": float(np.mean(coverages)),
                "latency_mean_ms": float(np.mean(latencies)),
                "latency_p95_ms": float(np.percentile(latencies, 95)),
                "posting_entries_mean": float(np.mean(posting_reads)),
                "uncapped_candidates_mean": float(np.mean(uncapped)),
            }
        )

    result = {
        "dataset": {"documents": len(corpus_ids), "queries": len(query_ids)},
        "mips_scale": scale,
        "first_block_norm_percentiles": {
            str(p): float(np.percentile(np.linalg.norm(docs[:, :-1], axis=1), p))
            for p in (0, 25, 50, 75, 95, 99, 100)
        },
        "tail_percentiles": {
            str(p): float(np.percentile(docs[:, -1], p))
            for p in (0, 25, 50, 75, 95, 99, 100)
        },
        "document_nonzeros_mean": float(np.mean(np.count_nonzero(docs[:, :-1], axis=1))),
        "query_nonzeros_mean": float(np.mean(np.count_nonzero(queries[:, :-1], axis=1))),
        "index_build_seconds": build_seconds,
        "storage": index.storage_summary(),
        "cap_sweep": rows,
    }
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "lexical_index_analysis.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    with (results_dir / "lexical_index_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
