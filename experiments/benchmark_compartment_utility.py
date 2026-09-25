"""Utility, latency, and storage sweep for Compartment DPE."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from compartment_dpe import CompartmentDPEIndex
from dueter_common import ConditionalDPE, evaluate, load_beir, normalize_rows, top_indices
from semantic_candidate_index import KeyedResidualSphericalIVF


ROOT = Path(__file__).resolve().parent


def percentile_summary(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(data.mean()),
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
    }


def evaluate_configuration(
    index: CompartmentDPEIndex,
    documents: np.ndarray,
    queries: np.ndarray,
    corpus_ids: list[str],
    query_ids: list[str],
    qrels: dict[str, dict[str, int]],
    dense_top100: list[np.ndarray],
    *,
    nprobe: int,
    top_per_cell: int,
) -> dict[str, object]:
    rankings: dict[str, list[int]] = {}
    candidate_counts: list[int] = []
    reads: list[int] = []
    emitted: list[int] = []
    coverages: list[float] = []
    latencies: list[float] = []
    for qi, qid in enumerate(query_ids):
        started = time.perf_counter()
        candidates, trace = index.candidates(
            queries[qi], nprobe=nprobe, top_per_cell=top_per_cell, nonce=qi + 1
        )
        cloud_elapsed = time.perf_counter() - started
        started = time.perf_counter()
        exact_scores = documents[candidates] @ queries[qi]
        rank = candidates[top_indices(exact_scores, min(100, len(candidates)))]
        total_elapsed = cloud_elapsed + time.perf_counter() - started
        rankings[qid] = rank.tolist()
        candidate_counts.append(len(candidates))
        reads.append(trace.posting_entries_read)
        emitted.append(trace.emitted_entries)
        coverages.append(
            len(set(map(int, dense_top100[qi])).intersection(map(int, candidates)))
            / len(dense_top100[qi])
        )
        latencies.append(total_elapsed * 1000.0)
    return {
        "projection_dimension": index.projection_dim,
        "nprobe": nprobe,
        "top_per_cell": top_per_cell,
        "metrics": evaluate(rankings, query_ids, corpus_ids, qrels),
        "dense_top100_coverage_mean": float(np.mean(coverages)),
        "candidate_count": percentile_summary([float(x) for x in candidate_counts]),
        "candidate_fraction_mean": float(np.mean(candidate_counts) / len(documents)),
        "posting_reads": percentile_summary([float(x) for x in reads]),
        "emitted_entries": percentile_summary([float(x) for x in emitted]),
        "end_to_end_latency_ms": percentile_summary(latencies),
    }


def global_dpe_baseline(
    routing: KeyedResidualSphericalIVF,
    documents: np.ndarray,
    queries: np.ndarray,
    corpus_ids: list[str],
    query_ids: list[str],
    qrels: dict[str, dict[str, int]],
    *,
    nprobe: int,
    candidate_depth: int,
    beta: float,
    scale: float,
    seed: int,
) -> dict[str, object]:
    dpe = ConditionalDPE(documents.shape[1], beta, scale, seed)
    doc_cipher = dpe.encrypt_database(documents)
    query_cipher = dpe.encrypt_queries(queries)
    rankings: dict[str, list[int]] = {}
    latencies: list[float] = []
    counts: list[int] = []
    for qi, qid in enumerate(query_ids):
        started = time.perf_counter()
        candidates, _ = routing.candidates(queries[qi], nprobe=nprobe)
        distances = np.linalg.norm(doc_cipher[candidates] - query_cipher[qi], axis=1)
        cloud = candidates[top_indices(-distances, min(candidate_depth, len(candidates)))]
        exact = documents[cloud] @ queries[qi]
        rank = cloud[top_indices(exact, min(100, len(cloud)))]
        latencies.append((time.perf_counter() - started) * 1000.0)
        counts.append(len(candidates))
        rankings[qid] = rank.tolist()
    return {
        "metrics": evaluate(rankings, query_ids, corpus_ids, qrels),
        "candidate_count": percentile_summary([float(x) for x in counts]),
        "end_to_end_latency_ms": percentile_summary(latencies),
        "server_ciphertext_bytes": int(doc_cipher.nbytes),
        "ciphertext_dimension": int(doc_cipher.shape[1]),
    }


def run_dataset(args: argparse.Namespace) -> dict[str, object]:
    corpus_ids, _, query_ids, _, qrels = load_beir(ROOT / "data" / args.dataset)
    corpus_path = ROOT / "cache" / "granite_gpu" / (
        f"{args.dataset}-st-a7972eb5022e_corpus_embeddings.npy"
    )
    query_path = ROOT / "cache" / "granite_gpu" / (
        f"{args.dataset}-st-a7972eb5022e_query_embeddings.npy"
    )
    documents = normalize_rows(np.load(corpus_path).astype(np.float32))
    queries = normalize_rows(np.load(query_path).astype(np.float32))
    dense_top100 = [top_indices(documents @ query, 100) for query in queries]
    dense_rankings = {
        qid: dense_top100[row].tolist() for row, qid in enumerate(query_ids)
    }
    dense_metrics = evaluate(dense_rankings, query_ids, corpus_ids, qrels)

    routing = KeyedResidualSphericalIVF(
        b"DuetDPE-compartment-routing-v1",
        n_clusters=args.clusters,
        doc_assignments=args.assignments,
        nprobe=args.nprobe,
        centered=True,
        seed=args.routing_seed if args.routing_seed is not None else args.seed + 10,
    )
    routing_setup = routing.build(documents)
    baseline = global_dpe_baseline(
        routing,
        documents,
        queries,
        corpus_ids,
        query_ids,
        qrels,
        nprobe=args.nprobe,
        candidate_depth=args.candidate_depth,
        beta=args.beta,
        scale=args.scale,
        seed=args.seed + 20,
    )

    sweep: list[dict[str, object]] = []
    setups: dict[str, object] = {}
    for dimension in args.projection_dimensions:
        print(f"[{args.dataset}] building r={dimension}", flush=True)
        index = CompartmentDPEIndex(
            key=b"DuetDPE-compartment-coordinate-v1",
            projection_dim=dimension,
            beta=args.beta,
            scale=args.scale,
            seed=args.seed + dimension,
        )
        setups[str(dimension)] = index.build(documents, routing)
        for top_per_cell in args.top_per_cell:
            print(
                f"[{args.dataset}] r={dimension} nprobe={args.nprobe} "
                f"top/cell={top_per_cell}",
                flush=True,
            )
            row = evaluate_configuration(
                index,
                documents,
                queries,
                corpus_ids,
                query_ids,
                qrels,
                dense_top100,
                nprobe=args.nprobe,
                top_per_cell=top_per_cell,
            )
            row["server_ciphertext_bytes"] = setups[str(dimension)][
                "server_ciphertext_bytes"
            ]
            row["storage_ratio_vs_global_dpe"] = (
                row["server_ciphertext_bytes"] / baseline["server_ciphertext_bytes"]
            )
            sweep.append(row)

    utility_eligible = [
        row
        for row in sweep
        if row["metrics"]["nDCG@10"] >= 0.98 * dense_metrics["nDCG@10"]
        and row["metrics"]["Recall@100"] >= 0.95 * dense_metrics["Recall@100"]
        and row["candidate_fraction_mean"] <= 0.40
    ]
    recommended = min(
        utility_eligible,
        key=lambda row: (
            row["server_ciphertext_bytes"],
            row["candidate_count"]["mean"],
            -row["metrics"]["nDCG@10"],
        ),
    ) if utility_eligible else max(
        sweep,
        key=lambda row: (
            row["metrics"]["nDCG@10"] / dense_metrics["nDCG@10"],
            row["metrics"]["Recall@100"] / dense_metrics["Recall@100"],
        ),
    )
    return {
        "dataset": args.dataset,
        "documents": len(documents),
        "queries": len(queries),
        "embedding_dimension": documents.shape[1],
        "routing_configuration": {
            "clusters": args.clusters,
            "assignments": args.assignments,
            "nprobe": args.nprobe,
        },
        "routing_setup": routing_setup,
        "dense_plain_metrics": dense_metrics,
        "global_dpe_residual_ivf": baseline,
        "selection_rule": (
            "nDCG@10 retention >= 0.98, Recall@100 retention >= 0.95, "
            "candidate fraction <= 0.40; then minimize ciphertext storage and candidates"
        ),
        "utility_gate_passed": bool(utility_eligible),
        "recommended": recommended,
        "setups": setups,
        "sweep": sweep,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("scifact", "nfcorpus"), required=True)
    parser.add_argument("--clusters", type=int, required=True)
    parser.add_argument("--assignments", type=int, required=True)
    parser.add_argument("--nprobe", type=int, required=True)
    parser.add_argument("--projection-dimensions", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--top-per-cell", nargs="+", type=int, default=[4, 8, 16, 32])
    parser.add_argument("--candidate-depth", type=int, default=200)
    parser.add_argument("--beta", type=float, default=0.10)
    parser.add_argument("--scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--routing-seed", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    report = run_dataset(parsed)
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    parsed.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"gate": report["utility_gate_passed"], "recommended": report["recommended"]}, indent=2))
