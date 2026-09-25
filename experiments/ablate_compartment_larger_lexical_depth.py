"""Run the missing larger lexical local-depth points for final-system ablation."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if (ROOT / ".deps").exists():
    sys.path.insert(0, str(ROOT / ".deps"))
if (ROOT / ".gpu_deps").exists():
    sys.path.insert(0, str(ROOT / ".gpu_deps"))

import numpy as np

from benchmark_dual_compartment_full import (
    FULL_DOCUMENTS,
    lexical_compartment_retrieval,
    metric_mean,
)
from benchmark_million_semantic_hybrid import (
    atomic_json,
    load_queries_qrels,
    read_ids,
)


CONFIG = {
    "nq": {
        "split": "test",
        "workload": "nq_full",
        "lexical_label": "nq_full_h1024",
        "posting_budget": 50_000,
        "repeats": 3,
        "cache_suffix": "",
    },
    "msmarco": {
        "split": "dev",
        "workload": "msmarco_dev_full",
        "lexical_label": "msmarco_dev_b250k_full_h1024",
        "posting_budget": 250_000,
        "repeats": 1,
        "cache_suffix": "_dev",
    },
}


def run_dataset(dataset: str, cfg: dict[str, object]) -> dict[str, object]:
    documents = FULL_DOCUMENTS[dataset]
    split = str(cfg["split"])
    query_ids, query_texts, qrels, _ = load_queries_qrels(
        ROOT / "data" / dataset, split
    )
    semantic_source = ROOT / "cache" / "full_semantic" / f"{dataset}_{documents}"
    lexical_source = (
        ROOT / "cache" / "full_candidate_lexical_dpe" / str(cfg["lexical_label"])
    )
    compartment = (
        ROOT
        / "cache"
        / "dual_compartment_full_p256"
        / f"{dataset}_{documents}{cfg['cache_suffix']}"
    )
    raw_path = (
        ROOT / "results" / "budgeted_lexical" / f"{cfg['workload']}.raw.npz"
    )
    raw_candidates = np.asarray(
        np.load(raw_path)[f"budget_{cfg['posting_budget']}"][: len(query_ids), :1000],
        dtype=np.int32,
    )
    outputs, timing = lexical_compartment_retrieval(
        dataset,
        lexical_source,
        compartment,
        raw_candidates,
        query_texts,
        documents=documents,
        projection_dimension=256,
        cells=64,
        top_per_cell_values=[32],
        output_depth=300,
        repeats=int(cfg["repeats"]),
        beta=0.10,
        scale=3.0,
        seed=20260917 + 9000,
    )
    doc_ids = read_ids(semantic_source / "doc_ids.txt")
    metrics = metric_mean(outputs[32], doc_ids, query_ids, qrels)
    return {
        "dataset": dataset,
        "documents": documents,
        "queries": len(query_ids),
        "split": split,
        "path": "lexical",
        "top_per_cell": 32,
        "repeats": int(cfg["repeats"]),
        "nDCG@10": float(metrics["nDCG@10"]),
        "Recall@100": float(metrics["Recall@100"]),
        "mean_union": float(timing["by_top_per_cell"][32]["candidate_mean"]),
        "candidate_p95": float(timing["by_top_per_cell"][32]["candidate_p95"]),
        "server_latency_ms": timing["server_latency_ms"],
    }


def main() -> None:
    destination = (
        ROOT
        / "results"
        / "dual_compartment_required_ablation"
        / "larger_lexical_depth.json"
    )
    report = {
        "experiment": "missing larger lexical local-depth points",
        "projection_dimension": 256,
        "lexical_cells": 64,
        "fixed_seed": 20260917,
        "datasets": {},
    }
    for dataset, cfg in CONFIG.items():
        report["datasets"][dataset] = run_dataset(dataset, cfg)
        atomic_json(destination, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
