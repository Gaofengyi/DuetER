#!/usr/bin/env python3
"""Summarize multi-scale runs of the unmodified official Pisces binaries."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_official_pisces import (  # noqa: E402
    bm25_plaintext,
    read_jsonl,
    resource_summary,
    similarity_plaintext,
    summarize_runs,
)


SCALES = (4096, 8192, 16384, 32768, 57638)
PATHS = {
    "bm25": ["lpsi", "bm25", "topk", "suda"],
    "similarity": ["fpsi", "sim", "topk", "suda"],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bm25_corpus = read_jsonl(args.data / "bm25_corpus.jsonl")
    bm25_queries = read_jsonl(args.data / "bm25_query.jsonl")
    sim_corpus = read_jsonl(args.data / "sim_corpus.jsonl")
    sim_queries = read_jsonl(args.data / "sim_query.jsonl")

    result: dict[str, Any] = {
        "protocol": "unmodified official Pisces benchmark binaries",
        "dataset": "BEIR FiQA, deterministic corpus prefixes",
        "parameters": {
            "scales": list(SCALES),
            "queries": len(bm25_queries),
            "k": 10,
            "repetitions": args.repetitions,
        },
        "results": {},
    }
    rows: list[dict[str, Any]] = []
    for size in SCALES:
        exact = {
            "bm25": bm25_plaintext(bm25_corpus[:size], bm25_queries, 10),
            "similarity": similarity_plaintext(sim_corpus[:size], sim_queries, 10)[0],
        }
        result["results"][str(size)] = {}
        for path_name, stages in PATHS.items():
            stem = f"{path_name}_fiqa_n{size}_q3_k10"
            run_files = [args.evidence / f"{stem}_rep{rep}.jsonl" for rep in range(1, args.repetitions + 1)]
            time_files = [args.evidence / f"{stem}_rep{rep}.time" for rep in range(1, args.repetitions + 1)]
            missing = [str(path) for path in (*run_files, *time_files) if not path.is_file()]
            if missing:
                raise FileNotFoundError("missing evidence: " + ", ".join(missing))
            item = summarize_runs(run_files, exact[path_name], stages)
            item["process_resources"] = resource_summary(time_files)
            result["results"][str(size)][path_name] = item
            row = {
                "documents": size,
                "path": path_name,
                "runs": item["run_count"],
                "query_observations": item["query_observations"],
                "online_mean_s": item["query_time_seconds"]["mean"],
                "online_std_s": item["query_time_seconds"]["std"],
                "online_p95_s": item["query_time_seconds"]["p95"],
                "upload_mean_mib": item["upload_mib"]["mean"],
                "download_mean_mib": item["download_mib"]["mean"],
                "recall_at_10_mean": item["quality"]["recall_at_10"]["mean"],
                "exact_top1_rate": item["quality"]["exact_top1"]["mean"],
                "process_wall_mean_s": item["process_resources"]["wall_seconds"]["mean"],
                "process_wall_p95_s": item["process_resources"]["wall_seconds"]["p95"],
                "max_rss_mean_mib": item["process_resources"]["max_rss_kib"]["mean"] / 1024,
                "max_rss_max_mib": item["process_resources"]["max_rss_kib"]["max"] / 1024,
            }
            if path_name == "similarity":
                row["candidate_count_mean"] = item["stages"]["sim"]["candidate_count"]["mean"]
            else:
                row["candidate_count_mean"] = ""
            rows.append(row)

    json_path = args.output_dir / "official_pisces_fiqa_scale_summary.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    csv_path = args.output_dir / "official_pisces_fiqa_scale_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Official Pisces FiQA scale experiment",
        "",
        "All rows are five executions of the unmodified official binaries; each execution processes three queries.",
        "",
        "| Documents | Path | Online mean ± sd (s/query) | p95 | Up / down (MiB/query) | Recall@10 | Process wall (s) | Peak RSS mean / max (MiB) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['documents']} | {row['path']} | {row['online_mean_s']:.3f} ± {row['online_std_s']:.3f} | "
            f"{row['online_p95_s']:.3f} | {row['upload_mean_mib']:.2f} / {row['download_mean_mib']:.2f} | "
            f"{row['recall_at_10_mean']:.3f} | {row['process_wall_mean_s']:.2f} | "
            f"{row['max_rss_mean_mib']:.0f} / {row['max_rss_max_mib']:.0f} |"
        )
    markdown = "\n".join(lines) + "\n"
    (args.output_dir / "official_pisces_fiqa_scale_summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
