"""Final-system semantic nprobe ablation for dual Compartment-DPE.

The script reuses the fully materialized 256-dimensional compartment cache and
varies only the number of probed residual-IVF cells.  A single fixed DPE seed is
used at every operating point so that changes are attributable to nprobe.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if (ROOT / ".deps").exists():
    sys.path.insert(0, str(ROOT / ".deps"))

import numpy as np

from benchmark_dual_compartment_full import (
    FULL_DOCUMENTS,
    SEMANTIC_RESULT_NAMES,
    metric_mean,
    semantic_compartment_retrieval,
)
from benchmark_million_semantic_hybrid import (
    atomic_json,
    evaluate,
    load_queries_qrels,
    read_ids,
)


CONFIG = {
    "nq": {"split": "test", "top": 16, "result": "nq"},
    "hotpotqa": {"split": "test", "top": 16, "result": "hotpotqa"},
    "msmarco": {"split": "dev", "top": 32, "result": "msmarco_dev"},
}
PROBES = (64, 128, 256)
DESTINATION = ROOT / "results" / "dual_compartment_required_ablation" / "nprobe.json"


def reference_ranking(dataset: str, query_count: int) -> tuple[np.ndarray, str]:
    if dataset == "msmarco":
        path = ROOT / "results" / "msmarco_dev_full_semantic" / "rankings.npz"
        return (
            np.asarray(np.load(path)["semantic"][0, :query_count], dtype=np.int32),
            "pre-compartment global-DPE IVF",
        )
    path = (
        ROOT
        / "results"
        / "full_semantic_hybrid"
        / SEMANTIC_RESULT_NAMES[dataset]
        / "rankings.npz"
    )
    return (
        np.asarray(np.load(path)["exact_dense"][:query_count], dtype=np.int32),
        "exact dense",
    )


def run_dataset(dataset: str) -> dict[str, object]:
    cfg = CONFIG[dataset]
    split = str(cfg["split"])
    top = int(cfg["top"])
    documents = FULL_DOCUMENTS[dataset]
    source = ROOT / "cache" / "full_semantic" / f"{dataset}_{documents}"
    compartment = (
        ROOT / "cache" / "dual_compartment_full_p256" / f"{dataset}_{documents}"
    )
    query_ids, _, qrels, _ = load_queries_qrels(ROOT / "data" / dataset, split)
    doc_ids = read_ids(source / "doc_ids.txt")
    embeddings = np.load(source / "corpus_embeddings.npy", mmap_mode="r")
    query_name = "query_embeddings_dev.npy" if split == "dev" else "query_embeddings.npy"
    queries = np.asarray(
        np.load(source / query_name, mmap_mode="r")[: len(query_ids)], dtype=np.float32
    )
    reference, reference_name = reference_ranking(dataset, len(query_ids))
    reference_metrics = evaluate(reference, doc_ids, query_ids, qrels)

    rows: list[dict[str, object]] = []
    for probes in PROBES:
        print(f"[{dataset}] nprobe={probes}", flush=True)
        outputs, timing = semantic_compartment_retrieval(
            source,
            compartment,
            queries,
            embeddings,
            reference,
            projection_dimension=256,
            probes=probes,
            top_per_cell_values=[top],
            output_depth=100,
            repeats=1,
            beta=0.10,
            scale=3.0,
            seed=20260917,
            device="cuda",
            latency_samples=5,
        )
        metrics = metric_mean(outputs[top], doc_ids, query_ids, qrels)
        diagnostic = timing["by_top_per_cell"][top]
        row = {
            "nprobe": probes,
            "local_depth": top,
            "scan_percent": 100.0
            * float(timing["posting_entries_read_mean"])
            / documents,
            "mean_union": float(diagnostic["candidate_mean"]),
            "reference_top100_coverage": float(
                diagnostic["dense_top100_coverage_mean"]
            ),
            "nDCG@10": float(metrics["nDCG@10"]),
            "nDCG@10_retention": float(metrics["nDCG@10"])
            / max(float(reference_metrics["nDCG@10"]), 1e-12),
            "Recall@100": float(metrics["Recall@100"]),
            "Recall@100_retention": float(metrics["Recall@100"])
            / max(float(reference_metrics["Recall@100"]), 1e-12),
            "batch_server_ms_per_query": float(
                timing["batch_server_ms_per_query_repeat"]
            ),
            "interactive_server_ms": float(
                timing["interactive_server_latency_ms"]["mean"]
            ),
            "client_rerank_ms_per_query": float(
                timing["client_rerank_ms_per_query_repeat"]
            ),
        }
        rows.append(row)
        partial = {
            "experiment": "final dual-Compartment-DPE nprobe ablation",
            "projection_dimension": 256,
            "semantic_cells": 2048,
            "fixed_seed": 20260917,
            "datasets": {dataset: {"rows": rows}},
        }
        atomic_json(DESTINATION.with_name(f"nprobe_{dataset}.json"), partial)
    return {
        "documents": documents,
        "queries": len(query_ids),
        "split": split,
        "reference": reference_name,
        "reference_metrics": reference_metrics,
        "rows": rows,
    }


def main() -> None:
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "final dual-Compartment-DPE nprobe ablation",
        "projection_dimension": 256,
        "semantic_cells": 2048,
        "fixed_seed": 20260917,
        "datasets": {},
    }
    for dataset in CONFIG:
        report["datasets"][dataset] = run_dataset(dataset)
        atomic_json(DESTINATION, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
