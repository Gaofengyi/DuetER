#!/usr/bin/env python3
"""Summarize the paired 100-query FiQA Pisces--DuetER experiment."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_qrels(path: Path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    with path.open(encoding="utf-8") as handle:
        next(handle, None)
        for line in handle:
            query_id, document_id, score = line.rstrip("\n").split("\t")
            if float(score) > 0:
                result[str(query_id)].add(str(document_id))
    return result


def recall(results: set[str], relevant: set[str]) -> float:
    return len(results & relevant) / len(relevant)


def distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, repeats: int = 20_000) -> list[float]:
    indices = rng.integers(0, len(values), size=(repeats, len(values)))
    means = values[indices].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def bootstrap_difference(
    first: np.ndarray, second: np.ndarray, rng: np.random.Generator, repeats: int = 20_000
) -> list[float]:
    indices = rng.integers(0, len(first), size=(repeats, len(first)))
    values = (first[indices] - second[indices]).mean(axis=1)
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def bootstrap_ratio(
    numerator: np.ndarray, denominator: np.ndarray, rng: np.random.Generator,
    repeats: int = 20_000,
) -> list[float]:
    indices = rng.integers(0, len(numerator), size=(repeats, len(numerator)))
    ratios = numerator[indices].mean(axis=1) / denominator[indices].mean(axis=1)
    return [float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))]


def dueter_communication_mib(payload_bucket: float) -> float:
    semantic_probes = 128
    lexical_cells = 64
    local_depth = 16
    projection_dim = 256
    label_bytes = 16
    local_query_bytes = projection_dim * 2
    semantic_records = semantic_probes * local_depth
    lexical_records = lexical_cells * local_depth
    semantic_record_bytes = 16 + 12 + (384 * 2 + 16) + 12 + (20 + 16)
    lexical_record_bytes = 92.0 + payload_bucket
    upload = (
        semantic_probes * (label_bytes + local_query_bytes)
        + lexical_cells * (label_bytes + local_query_bytes)
        + 32
    )
    download = 16 + semantic_records * semantic_record_bytes + lexical_records * lexical_record_bytes
    return float((upload + download) / (1024**2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    qrels: dict[str, set[str]] = defaultdict(set)
    bm25: dict[str, dict] = {}
    semantic: dict[str, dict] = {}
    dueter: dict[str, dict] = {}
    fidelity_reports = []
    sample_seeds = []
    for directory in args.directory:
        local_qrels = load_qrels(directory / "qrels.tsv")
        if set(qrels) & set(local_qrels):
            raise RuntimeError("query IDs overlap across supplied experiment directories")
        qrels.update(local_qrels)
        bm25_path, = directory.glob("bm25_fiqa_n57638_q*_k10.jsonl")
        semantic_path, = directory.glob("similarity_fiqa_n57638_q*_k10.jsonl")
        dueter_path, = directory.glob("dueter_fiqa_q*.json")
        bm25.update({str(row["id"]): row for row in read_jsonl(bm25_path)})
        semantic.update({str(row["id"]): row for row in read_jsonl(semantic_path)})
        dueter_report = json.loads(dueter_path.read_text(encoding="utf-8"))
        dueter.update({str(row["query_id"]): row for row in dueter_report["observations"]})
        fidelity_reports.append(dueter_report["retrieval_fidelity"])
        sample_seeds.append(json.loads((directory / "manifest.json").read_text(encoding="utf-8"))["seed"])
    query_ids = sorted(qrels, key=int)
    if not (len(query_ids) == len(bm25) == len(semantic) == len(dueter)):
        raise RuntimeError("the paired experiment files have different query counts")
    if set(query_ids) != set(bm25) or set(query_ids) != set(semantic) or set(query_ids) != set(dueter):
        raise RuntimeError("query IDs differ across paired outputs")

    rows = []
    for query_id in query_ids:
        relevant = qrels[query_id]
        p_sem = set(map(str, semantic[query_id]["results"]))
        p_lex = set(map(str, bm25[query_id]["results"]))
        d_sem = set(map(str, dueter[query_id]["semantic_top10"]))
        d_lex = set(map(str, dueter[query_id]["lexical_top10"]))
        rows.append({
            "query_id": query_id,
            "relevant": len(relevant),
            "pisces_semantic_recall": recall(p_sem, relevant),
            "pisces_lexical_recall": recall(p_lex, relevant),
            "pisces_union_recall": recall(p_sem | p_lex, relevant),
            "pisces_union_size": len(p_sem | p_lex),
            "pisces_online_seconds": float(semantic[query_id]["query_time"] + bm25[query_id]["query_time"]),
            "pisces_communication_mib": float(
                semantic[query_id]["upload"] + semantic[query_id]["download"]
                + bm25[query_id]["upload"] + bm25[query_id]["download"]
            ),
            "dueter_semantic_recall": recall(d_sem, relevant),
            "dueter_lexical_recall": recall(d_lex, relevant),
            "dueter_union_recall": recall(d_sem | d_lex, relevant),
            "dueter_union_size": len(d_sem | d_lex),
            "dueter_online_seconds": float(dueter[query_id]["online_seconds"]),
            "dueter_communication_mib": dueter_communication_mib(
                float(dueter[query_id]["lexical_payload_bucket_mean"])
            ),
        })

    def vector(name: str) -> np.ndarray:
        return np.asarray([row[name] for row in rows], dtype=np.float64)

    rng = np.random.default_rng(20260921)
    systems = {}
    for system in ("pisces", "dueter"):
        systems[system] = {
            "semantic_recall": distribution(vector(f"{system}_semantic_recall")),
            "lexical_recall": distribution(vector(f"{system}_lexical_recall")),
            "union_recall": distribution(vector(f"{system}_union_recall")),
            "union_recall_mean_95ci": bootstrap_mean(vector(f"{system}_union_recall"), rng),
            "union_size": distribution(vector(f"{system}_union_size")),
            "online_seconds": distribution(vector(f"{system}_online_seconds")),
            "online_mean_95ci": bootstrap_mean(vector(f"{system}_online_seconds"), rng),
            "communication_mib": distribution(vector(f"{system}_communication_mib")),
            "communication_mean_95ci": bootstrap_mean(vector(f"{system}_communication_mib"), rng),
        }
    report = {
        "experiment": "paired official-Pisces versus DuetER FiQA test-query sample",
        "documents": 57_638,
        "queries": len(query_ids),
        "sample_seeds": sample_seeds,
        "output_policy": "unordered semantic@10 union lexical@10",
        "systems": systems,
        "paired_comparison": {
            "dueter_minus_pisces_union_recall": float(
                np.mean(vector("dueter_union_recall") - vector("pisces_union_recall"))
            ),
            "dueter_minus_pisces_union_recall_95ci": bootstrap_difference(
                vector("dueter_union_recall"), vector("pisces_union_recall"), rng
            ),
            "pisces_over_dueter_online_speedup": float(
                np.mean(vector("pisces_online_seconds")) / np.mean(vector("dueter_online_seconds"))
            ),
            "online_speedup_95ci": bootstrap_ratio(
                vector("pisces_online_seconds"), vector("dueter_online_seconds"), rng
            ),
            "pisces_over_dueter_communication_ratio": float(
                np.mean(vector("pisces_communication_mib"))
                / np.mean(vector("dueter_communication_mib"))
            ),
            "communication_ratio_95ci": bootstrap_ratio(
                vector("pisces_communication_mib"), vector("dueter_communication_mib"), rng
            ),
        },
        "dueter_fidelity_by_batch": fidelity_reports,
        "per_query": rows,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"systems": systems, "paired_comparison": report["paired_comparison"]}, indent=2))


if __name__ == "__main__":
    main()
