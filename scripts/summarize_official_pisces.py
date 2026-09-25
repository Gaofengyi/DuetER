#!/usr/bin/env python3
"""Summarize official Pisces runs and compare them with plaintext rankings."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stats(values: Iterable[float]) -> dict[str, float | int]:
    data = [float(value) for value in values]
    ordered = sorted(data)
    if not data:
        return {"n": 0}
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "n": len(data),
        "mean": statistics.fmean(data),
        "std": statistics.stdev(data) if len(data) > 1 else 0.0,
        "median": statistics.median(data),
        "p95": ordered[p95_index],
        "min": ordered[0],
        "max": ordered[-1],
    }


def parse_elapsed(value: str) -> float:
    parts = [float(part) for part in value.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(value)


def parse_resource_file(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    # GNU time includes colons inside its explanatory label
    # ("h:mm:ss or m:ss"), so capture the final numeric field on the line.
    elapsed_match = re.search(
        r"(?m)^\s*Elapsed \(wall clock\) time \(.*\):\s*([0-9:.]+)\s*$",
        text,
    )
    rss_match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    result: dict[str, float] = {}
    if elapsed_match:
        result["wall_seconds"] = parse_elapsed(elapsed_match.group(1))
    if rss_match:
        result["max_rss_kib"] = float(rss_match.group(1))
    return result


def bm25_plaintext(corpus: list[dict[str, Any]], queries: list[dict[str, Any]], k: int) -> list[list[str]]:
    n_docs = len(corpus)
    counts = [row["bert_token_counts"] for row in corpus]
    lengths = [sum(int(value) for value in row.values()) for row in counts]
    average_length = statistics.fmean(lengths)
    doc_frequency: Counter[str] = Counter()
    for row in counts:
        doc_frequency.update(row.keys())

    rankings: list[list[str]] = []
    for query in queries:
        query_terms = list(dict.fromkeys(query["query_bert_tokens"]))
        scores: list[float] = []
        for token_counts, length in zip(counts, lengths, strict=True):
            factor = 1.5 * (1.0 - 0.75 + 0.75 * length / average_length)
            score = 0.0
            for token in query_terms:
                term_frequency = int(token_counts.get(token, 0))
                if term_frequency:
                    df = doc_frequency[token]
                    weight = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                    score += weight * term_frequency / (factor + term_frequency)
            scores.append(score)
        order = sorted(range(n_docs), key=lambda index: (-scores[index], index))[:k]
        rankings.append([str(corpus[index]["context_id"]) for index in order])
    return rankings


def similarity_plaintext(
    corpus: list[dict[str, Any]], queries: list[dict[str, Any]], k: int
) -> tuple[list[list[str]], list[np.ndarray]]:
    matrix = np.asarray([row["context_embeddings_norm"] for row in corpus], dtype=np.float64)
    rankings: list[list[str]] = []
    score_vectors: list[np.ndarray] = []
    for query in queries:
        vector = np.asarray(query["query_embeddings_norm"], dtype=np.float64)
        scores = matrix @ vector
        order = np.argsort(-scores, kind="stable")[:k]
        rankings.append([str(corpus[index]["context_id"]) for index in order])
        score_vectors.append(scores)
    return rankings, score_vectors


def quality_metrics(returned: list[str], exact: list[str]) -> dict[str, float]:
    returned = [str(item) for item in returned]
    exact = [str(item) for item in exact]
    overlap = len(set(returned) & set(exact))
    union = len(set(returned) | set(exact))
    reciprocal_rank = 0.0
    if exact[0] in returned:
        reciprocal_rank = 1.0 / (returned.index(exact[0]) + 1)
    return {
        "recall_at_10": overlap / len(exact),
        "jaccard_at_10": overlap / union,
        "exact_top1": float(bool(returned) and returned[0] == exact[0]),
        "rr_of_exact_top1": reciprocal_rank,
    }


def summarize_runs(
    run_files: list[Path], exact_rankings: list[list[str]], stage_names: list[str]
) -> dict[str, Any]:
    runs = [read_jsonl(path) for path in run_files]
    if any(len(run) != len(exact_rankings) for run in runs):
        raise ValueError("run/query count mismatch")
    records = [record for run in runs for record in run]

    summary: dict[str, Any] = {
        "run_count": len(runs),
        "query_observations": len(records),
        "query_time_seconds": stats(record["query_time"] for record in records),
        "upload_mib": stats(record["upload"] for record in records),
        "download_mib": stats(record["download"] for record in records),
        "stages": {},
    }
    for stage in stage_names:
        summary["stages"][stage] = {
            "time_seconds": stats(record[stage]["time"] for record in records),
            "upload_mib": stats(record[stage]["upload"] for record in records),
            "download_mib": stats(record[stage]["download"] for record in records),
        }
        if any("size" in record[stage] for record in records):
            summary["stages"][stage]["candidate_count"] = stats(
                record[stage]["size"] for record in records
            )

    qualities: list[dict[str, float]] = []
    rankings_by_query: list[set[tuple[str, ...]]] = [set() for _ in exact_rankings]
    for run in runs:
        for query_index, (record, exact) in enumerate(zip(run, exact_rankings, strict=True)):
            qualities.append(quality_metrics(record["results"], exact))
            rankings_by_query[query_index].add(tuple(str(item) for item in record["results"]))
    summary["quality"] = {
        name: stats(item[name] for item in qualities)
        for name in ("recall_at_10", "jaccard_at_10", "exact_top1", "rr_of_exact_top1")
    }
    summary["unique_rankings_per_query"] = [len(rankings) for rankings in rankings_by_query]
    summary["exact_top10"] = exact_rankings
    summary["first_run_top10"] = [record["results"] for record in runs[0]]
    return summary


def resource_summary(files: list[Path]) -> dict[str, Any]:
    parsed = [parse_resource_file(path) for path in files]
    return {
        "wall_seconds": stats(row["wall_seconds"] for row in parsed if "wall_seconds" in row),
        "max_rss_kib": stats(row["max_rss_kib"] for row in parsed if "max_rss_kib" in row),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--data-256", type=Path, required=True)
    parser.add_argument("--data-1990", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    datasets: dict[int, dict[str, Any]] = {}
    for size, path in ((256, args.data_256), (1990, args.data_1990)):
        datasets[size] = {
            "bm25_corpus": read_jsonl(path / "bm25_corpus.jsonl"),
            "bm25_query": read_jsonl(path / "bm25_query.jsonl"),
            "sim_corpus": read_jsonl(path / "sim_corpus.jsonl"),
            "sim_query": read_jsonl(path / "sim_query.jsonl"),
        }

    exact: dict[int, dict[str, list[list[str]]]] = {}
    for size, data in datasets.items():
        exact[size] = {
            "bm25": bm25_plaintext(data["bm25_corpus"], data["bm25_query"], 10),
            "similarity": similarity_plaintext(data["sim_corpus"], data["sim_query"], 10)[0],
        }

    result: dict[str, Any] = {"parameters": {"k": 10, "repetitions_full": 5}, "results": {}}
    for size in (256, 1990):
        result["results"][str(size)] = {}
        for path_name, stages in (
            ("bm25", ["lpsi", "bm25", "topk", "suda"]),
            ("similarity", ["fpsi", "sim", "topk", "suda"]),
        ):
            stem = f"{path_name}_official_clapnq_n{size}_q3_k10"
            run_files = [args.evidence / f"{stem}.jsonl"]
            if size == 1990:
                run_files.extend(args.evidence / f"{stem}_rep{rep}.jsonl" for rep in range(2, 6))
            result["results"][str(size)][path_name] = summarize_runs(
                run_files, exact[size][path_name], stages
            )
            resource_files = [args.evidence / f"{stem}.log"]
            if size == 1990:
                resource_files.extend(args.evidence / f"{stem}_rep{rep}.time" for rep in range(2, 6))
            result["results"][str(size)][path_name]["process_resources"] = resource_summary(resource_files)

    json_path = args.output_dir / "official_pisces_summary.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    rows: list[dict[str, Any]] = []
    for size, size_result in result["results"].items():
        for path_name, item in size_result.items():
            rows.append(
                {
                    "documents": size,
                    "path": path_name,
                    "runs": item["run_count"],
                    "query_observations": item["query_observations"],
                    "latency_mean_s": item["query_time_seconds"]["mean"],
                    "latency_std_s": item["query_time_seconds"]["std"],
                    "latency_p95_s": item["query_time_seconds"]["p95"],
                    "upload_mean_mib": item["upload_mib"]["mean"],
                    "download_mean_mib": item["download_mib"]["mean"],
                    "recall_at_10_mean": item["quality"]["recall_at_10"]["mean"],
                    "exact_top1_rate": item["quality"]["exact_top1"]["mean"],
                    "wall_mean_s": item["process_resources"]["wall_seconds"]["mean"],
                    "max_rss_mean_kib": item["process_resources"]["max_rss_kib"]["mean"],
                }
            )
    csv_path = args.output_dir / "official_pisces_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Official Pisces benchmark summary",
        "",
        "| Documents | Path | Runs | Query obs. | Latency mean ± sd (s) | p95 (s) | Upload (MiB) | Download (MiB) | Recall@10 | Top-1 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['documents']} | {row['path']} | {row['runs']} | {row['query_observations']} | "
            f"{row['latency_mean_s']:.3f} ± {row['latency_std_s']:.3f} | {row['latency_p95_s']:.3f} | "
            f"{row['upload_mean_mib']:.3f} | {row['download_mean_mib']:.3f} | "
            f"{row['recall_at_10_mean']:.3f} | {row['exact_top1_rate']:.3f} |"
        )
    (args.output_dir / "official_pisces_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
