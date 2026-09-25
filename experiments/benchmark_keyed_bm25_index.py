"""Cross-dataset benchmark of the collision-free keyed BM25 tail index."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from lexical_mips_index import KeyedBM25TailIndex
from run_experiment import BM25Index, load_beir, tokenize, top_indices


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--caps", nargs="+", type=int, default=[100, 200, 400, 800, 1600])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results" / "keyed_bm25")
    args = parser.parse_args()
    data_dir = ROOT / "data" / args.dataset
    corpus_ids, corpus_texts, query_ids, query_texts, _ = load_beir(data_dir)

    start = time.perf_counter()
    bm25 = BM25Index(corpus_texts)
    bm25_build = time.perf_counter() - start
    index = KeyedBM25TailIndex(b"DuetDPE-keyed-vocabulary-tail-v2")
    start = time.perf_counter()
    index.build(
        bm25.postings, bm25.idf, bm25.doc_len, bm25.avgdl,
        bm25.n_docs, bm25.k1, bm25.b,
    )
    index_build = time.perf_counter() - start

    cap_rows = []
    for cap in args.caps:
        padded_coverage, positive_coverage, reads, uncapped = [], [], [], []
        latency_by_repeat: list[list[float]] = [[] for _ in range(args.repeats)]
        queries_with_positive = 0
        for query in query_texts:
            exact_scores = bm25.score(query)
            exact = top_indices(exact_scores, 100)
            positive_count = int(np.count_nonzero(exact_scores > 0))
            exact_positive = top_indices(exact_scores, min(100, positive_count))
            candidates = np.empty(0, dtype=np.int64)
            trace: dict[str, int] = {}
            for repeat in range(args.repeats):
                start = time.perf_counter()
                candidates, trace = index.candidates(tokenize(query), cap)
                latency_by_repeat[repeat].append((time.perf_counter() - start) * 1000.0)
            candidate_set = set(map(int, candidates))
            padded_coverage.append(len(set(map(int, exact)).intersection(candidate_set)) / 100.0)
            if positive_count:
                queries_with_positive += 1
                positive_coverage.append(
                    len(set(map(int, exact_positive)).intersection(candidate_set)) / len(exact_positive)
                )
            reads.append(trace["posting_entries_read"])
            uncapped.append(trace["uncapped_candidates"])
        cap_rows.append({
            "cap": cap,
            "padded_top100_coverage_mean": float(np.mean(padded_coverage)),
            "positive_bm25_top100_coverage_mean": float(np.mean(positive_coverage)) if positive_coverage else 1.0,
            "queries_with_positive_bm25_scores": queries_with_positive,
            "timing_repeats": args.repeats,
            "candidate_latency_mean_ms": float(np.mean(latency_by_repeat)),
            "candidate_latency_repeat_mean_ms": [float(np.mean(row)) for row in latency_by_repeat],
            "candidate_latency_repeat_std_ms": float(np.std([np.mean(row) for row in latency_by_repeat], ddof=1)) if args.repeats > 1 else 0.0,
            "candidate_latency_p95_ms": float(np.percentile(latency_by_repeat, 95)),
            "posting_entries_read_mean": float(np.mean(reads)),
            "uncapped_candidates_mean": float(np.mean(uncapped)),
        })
    result = {
        "dataset": args.dataset,
        "documents": len(corpus_ids),
        "queries": len(query_ids),
        "bm25_build_seconds": bm25_build,
        "keyed_index_build_seconds": index_build,
        "storage": index.storage_summary(),
        "cap_sweep": cap_rows,
        "leakage": "keyed term equality, posting lengths, query access and repetition patterns",
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    output = args.results_dir / f"{args.dataset}.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
