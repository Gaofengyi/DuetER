"""Paired plaintext/DPE path-utility audit for the complete corpora.

The missing plaintext semantic reference is computed over exactly the same
mean-residual spherical-IVF candidate union used by semantic DPE.  The lexical
plaintext ranking, DPE repeats, outer split, and frozen DuetRank model are then
reused without parameter selection on the reporting holdout.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if (ROOT / ".deps").exists():
    sys.path.insert(0, str(ROOT / ".deps"))
if (ROOT / ".gpu_deps").exists():
    sys.path.insert(0, str(ROOT / ".gpu_deps"))

import joblib
import numpy as np

from benchmark_million_semantic_hybrid import (
    IvfFiles,
    atomic_json,
    load_queries_qrels,
    read_ids,
)
from benchmark_msmarco_dev_batched import compact_unique_clouds
from evaluate_duetrank_recall import FUSION_RESULTS, TRACES, metric_rows
from experiment_adaptive_fusion import query_features
from experiment_candidate_rank_fusion import (
    learned_rankings,
    mix_rankings,
    paired_bootstrap,
    split_mask,
)
from experiment_fixed_rrf_weight_sweep import CONFIGS

# Some imported legacy evaluators prepend the CPU dependency directory.  Put
# the CUDA wheel first again before the function-local ``import torch``.
GPU_DEPS = str(ROOT / ".gpu_deps")
if Path(GPU_DEPS).exists():
    while GPU_DEPS in sys.path:
        sys.path.remove(GPU_DEPS)
    sys.path.insert(0, GPU_DEPS)


CACHE_NAMES = {
    "nq": "nq_2681468",
    "hotpotqa": "hotpotqa_5233329",
    "msmarco": "msmarco_8841823",
}
QUERY_EMBEDDINGS = {
    "nq": "query_embeddings.npy",
    "hotpotqa": "query_embeddings.npy",
    "msmarco": "query_embeddings_dev.npy",
}


def cell_major_plaintext(
    *,
    ivf: IvfFiles,
    queries: np.ndarray,
    embeddings: np.ndarray,
    probes: int,
    output_depth: int,
    device: str,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Exact inner-product ranking over each query's full IVF candidate union.

    Cells are loaded once for all routed queries.  Documents have two IVF
    assignments, so retaining 2k scored occurrences is sufficient to recover
    k unique documents after deterministic duplicate removal.
    """
    import torch

    if probes > len(ivf.centroids):
        raise ValueError("probes exceeds the number of IVF cells")
    query_count = len(queries)
    keep = output_depth * 2
    residual = np.asarray(queries, dtype=np.float32) - ivf.mean
    residual /= np.maximum(np.linalg.norm(residual, axis=1, keepdims=True), 1e-12)
    routing_scores = residual @ ivf.centroids.T
    routed = np.argpartition(-routing_scores, probes - 1, axis=1)[:, :probes]
    best_scores = np.full((query_count, keep), -np.inf, dtype=np.float32)
    best_docs = np.full((query_count, keep), -1, dtype=np.int32)
    posting_reads = 0
    active_cells = 0
    started = time.perf_counter()

    query_gpu = torch.from_numpy(
        np.asarray(queries, dtype=np.float32).copy()
    ).to(device)
    with torch.inference_mode():
        for cell in range(len(ivf.centroids)):
            query_indices = np.flatnonzero(np.any(routed == cell, axis=1))
            if not len(query_indices):
                continue
            first, last = int(ivf.offsets[cell]), int(ivf.offsets[cell + 1])
            cell_documents = np.asarray(ivf.postings[first:last], dtype=np.int64)
            if not len(cell_documents):
                continue
            active_cells += 1
            posting_reads += len(cell_documents) * len(query_indices)
            rows_gpu = torch.from_numpy(
                np.asarray(embeddings[cell_documents], dtype=np.float32)
            ).to(device)
            values = query_gpu[query_indices] @ rows_gpu.T
            k = min(output_depth, len(cell_documents))
            cell_scores, local = torch.topk(values, k=k, dim=1)
            cell_scores = cell_scores.cpu().numpy()
            cell_docs = cell_documents[local.cpu().numpy()].astype(np.int32)

            merged_scores = np.concatenate(
                (best_scores[query_indices], cell_scores), axis=1
            )
            merged_docs = np.concatenate((best_docs[query_indices], cell_docs), axis=1)
            positions = np.argpartition(merged_scores, -keep, axis=1)[:, -keep:]
            best_scores[query_indices] = np.take_along_axis(
                merged_scores, positions, axis=1
            )
            best_docs[query_indices] = np.take_along_axis(
                merged_docs, positions, axis=1
            )
            del rows_gpu, values, cell_scores, local
            if cell == 0 or cell + 1 == len(ivf.centroids) or (cell + 1) % 128 == 0:
                print(
                    f"[plaintext IVF] {cell + 1:,}/{len(ivf.centroids):,} cells",
                    flush=True,
                )

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    scoring_seconds = time.perf_counter() - started
    # Reuse the audited duplicate-removal routine, then add an explicit
    # score/document-id tie break to make the plaintext reference deterministic.
    rankings = compact_unique_clouds(
        best_scores[None, :, :], best_docs[None, :, :], output_depth
    )[0]
    for query_index, row in enumerate(rankings):
        valid = row[row >= 0]
        scores = np.asarray(
            embeddings[valid], dtype=np.float32
        ) @ np.asarray(queries[query_index], dtype=np.float32)
        order = np.lexsort((valid, -scores))
        rankings[query_index, : len(valid)] = valid[order]
    return rankings, {
        "execution_order": "cell-major exact plaintext scoring",
        "queries": query_count,
        "probes": probes,
        "output_depth": output_depth,
        "active_cells": active_cells,
        "posting_reads_query_equivalent": int(posting_reads),
        "scoring_seconds": scoring_seconds,
        "throughput_queries_per_second": query_count / max(scoring_seconds, 1e-12),
    }


def mean_metric(
    rankings: np.ndarray,
    doc_ids: list[str],
    query_ids: list[str],
    qrels: dict[str, dict[str, int]],
) -> np.ndarray:
    return metric_rows(rankings, doc_ids, query_ids, qrels, 10)["ndcg"]


def load_selection(dataset: str) -> tuple[dict[str, object], dict[str, object]]:
    payload = json.loads(FUSION_RESULTS[dataset].read_text(encoding="utf-8"))
    row = payload["datasets"][0]
    return row["selection"], row["held_out"]


def run_dataset(
    dataset: str,
    *,
    output_root: Path,
    device: str,
    probes: int,
) -> dict[str, object]:
    config = CONFIGS[dataset]
    cache_dir = ROOT / "cache/full_semantic" / CACHE_NAMES[dataset]
    output_root.mkdir(parents=True, exist_ok=True)
    plain_cache = output_root / f"{dataset}_semantic_plain.rankings.npz"

    query_ids, query_texts, qrels, _ = load_queries_qrels(
        ROOT / "data" / dataset, str(config["split"])
    )
    doc_ids = read_ids(cache_dir / "doc_ids.txt")
    queries = np.load(cache_dir / QUERY_EMBEDDINGS[dataset], mmap_mode="r")
    embeddings = np.load(cache_dir / "corpus_embeddings.npy", mmap_mode="r")
    if len(queries) != len(query_ids):
        raise ValueError(
            f"{dataset}: query embedding count {len(queries)} != qrels {len(query_ids)}"
        )

    if plain_cache.exists():
        cached = np.load(plain_cache)
        semantic_plain = np.asarray(cached["semantic_plain"], dtype=np.int32)
        if semantic_plain.shape != (len(query_ids), 100):
            raise ValueError(f"stale plaintext semantic cache: {semantic_plain.shape}")
        timing = json.loads(str(cached["metadata"].item()))
        print(f"[{dataset}] loaded {plain_cache}", flush=True)
    else:
        semantic_plain, timing = cell_major_plaintext(
            ivf=IvfFiles.load(cache_dir),
            queries=np.asarray(queries, dtype=np.float32),
            embeddings=embeddings,
            probes=probes,
            output_depth=100,
            device=device,
        )
        np.savez_compressed(
            plain_cache,
            semantic_plain=semantic_plain,
            metadata=json.dumps(timing),
        )

    semantic_dpe = np.asarray(np.load(config["semantic"])["semantic"], dtype=np.int32)
    lexical_file = np.load(config["lexical"])
    lexical_plain = np.asarray(lexical_file["lexical_plaintext"], dtype=np.int32)
    lexical_dpe = np.asarray(lexical_file["lexical_dpe"], dtype=np.int32)
    if semantic_dpe.shape[1] != len(query_ids) or lexical_dpe.shape[1] != len(query_ids):
        raise ValueError(f"{dataset}: DPE ranking/query count mismatch")

    selection, held_out = load_selection(dataset)
    holdout = split_mask(query_ids, "outer")
    trace_path, trace_key = TRACES[dataset]
    trace = json.loads(trace_path.read_text(encoding="utf-8"))[trace_key]
    model = joblib.load(selection["model_path"])
    semantic_weight = float(selection["semantic_backoff_weight"])

    # Plaintext query features are evaluated from the plaintext path pair.  The
    # model and backoff coefficient remain frozen from calibration.
    plain_features = query_features(
        query_texts, semantic_plain[None, :, :], lexical_plain[:, :100], trace
    )
    plain_learned = learned_rankings(
        model, semantic_plain, lexical_plain, plain_features, mask=holdout
    )
    dual_plain = (
        plain_learned
        if semantic_weight == 0.0
        else mix_rankings(semantic_plain, plain_learned, semantic_weight)
    )

    dpe_cache = (
        ROOT
        / "results/budgeted_lexical_dpe"
        / f"duetrank_{dataset}_outer.rankings.npz"
    )
    if dpe_cache.exists():
        dual_dpe = np.asarray(np.load(dpe_cache)["duetrank"], dtype=np.int32)
    else:
        dpe_features = query_features(
            query_texts, semantic_dpe, lexical_dpe[0, :, :100], trace
        )
        generated = []
        for repeat in range(len(semantic_dpe)):
            learned = learned_rankings(
                model,
                semantic_dpe[repeat],
                lexical_dpe[repeat],
                dpe_features,
                mask=holdout,
            )
            generated.append(
                learned
                if semantic_weight == 0.0
                else mix_rankings(semantic_dpe[repeat], learned, semantic_weight)
            )
        dual_dpe = np.stack(generated)
        np.savez_compressed(dpe_cache, duetrank=dual_dpe, outer_holdout=holdout)

    semantic_plain_rows = mean_metric(
        semantic_plain, doc_ids, query_ids, qrels
    )[None, :]
    lexical_plain_rows = mean_metric(lexical_plain, doc_ids, query_ids, qrels)[None, :]
    dual_plain_rows = mean_metric(dual_plain, doc_ids, query_ids, qrels)[None, :]
    semantic_dpe_rows = np.stack(
        [mean_metric(row, doc_ids, query_ids, qrels) for row in semantic_dpe]
    )
    lexical_dpe_rows = np.stack(
        [mean_metric(row, doc_ids, query_ids, qrels) for row in lexical_dpe]
    )
    dual_dpe_rows = np.stack(
        [mean_metric(row, doc_ids, query_ids, qrels) for row in dual_dpe]
    )

    stored_dpe = float(held_out["semantic_backoff_ndcg_at_10"])
    audited_dpe = float(np.mean(dual_dpe_rows[:, holdout]))
    if not np.isclose(stored_dpe, audited_dpe, atol=1e-10):
        raise AssertionError(
            f"{dataset}: stored DuetRank audit failed: {stored_dpe} != {audited_dpe}"
        )

    plain = {
        "semantic": float(np.mean(semantic_plain_rows[:, holdout])),
        "lexical": float(np.mean(lexical_plain_rows[:, holdout])),
        "dual": float(np.mean(dual_plain_rows[:, holdout])),
    }
    dpe = {
        "semantic": float(np.mean(semantic_dpe_rows[:, holdout])),
        "lexical": float(np.mean(lexical_dpe_rows[:, holdout])),
        "dual": audited_dpe,
    }
    best_plain_rows = (
        semantic_plain_rows
        if plain["semantic"] >= plain["lexical"]
        else lexical_plain_rows
    )
    best_dpe_rows = (
        semantic_dpe_rows if dpe["semantic"] >= dpe["lexical"] else lexical_dpe_rows
    )
    repeated_dual_plain = np.repeat(dual_plain_rows, len(dual_dpe_rows), axis=0)
    expected = {
        "plain_dual_exceeds_both_singles": plain["dual"]
        > max(plain["semantic"], plain["lexical"]),
        "dpe_dual_exceeds_both_singles": dpe["dual"]
        > max(dpe["semantic"], dpe["lexical"]),
        "dpe_dual_below_plain_dual": dpe["dual"] < plain["dual"],
    }
    result = {
        "dataset": dataset,
        "documents": int(config["documents"]),
        "outer_holdout_queries": int(np.sum(holdout)),
        "dpe_repeats": int(len(semantic_dpe)),
        "plaintext": plain,
        "dpe": dpe,
        "dual_plaintext_retained_by_dpe": dpe["dual"] / max(plain["dual"], 1e-12),
        "dual_plain_minus_dpe": paired_bootstrap(
            dual_dpe_rows, repeated_dual_plain, holdout
        ),
        "plain_dual_vs_best_plain_single": paired_bootstrap(
            best_plain_rows, dual_plain_rows, holdout
        ),
        "dpe_dual_vs_best_dpe_single": paired_bootstrap(
            best_dpe_rows, dual_dpe_rows, holdout
        ),
        "expected_relationship": expected,
        "semantic_plaintext_timing": timing,
        "frozen_model": str(selection["model_path"]),
        "semantic_backoff_weight": semantic_weight,
        "ranking_artifacts": {
            "semantic_plaintext": str(plain_cache),
            "semantic_dpe": str(config["semantic"]),
            "lexical": str(config["lexical"]),
            "dual_dpe": str(dpe_cache),
        },
    }
    np.savez_compressed(
        output_root / f"{dataset}_paired.rankings.npz",
        semantic_plain=semantic_plain,
        lexical_plain=lexical_plain,
        dual_plain=dual_plain,
        semantic_dpe=semantic_dpe,
        lexical_dpe=lexical_dpe,
        dual_dpe=dual_dpe,
        outer_holdout=holdout,
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+", choices=CONFIGS, default=list(CONFIGS)
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--probes", type=int, default=128)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results/plaintext_duetrank",
    )
    args = parser.parse_args()
    results = [
        run_dataset(
            dataset,
            output_root=args.output_root,
            device=args.device,
            probes=args.probes,
        )
        for dataset in args.datasets
    ]
    all_expected = all(
        all(row["expected_relationship"].values()) for row in results
    )
    output = {
        "experiment": "paired complete-corpus plaintext/DPE path utility",
        "metric": "outer-holdout nDCG@10",
        "fairness_controls": [
            "identical corpus and mean-residual IVF routing",
            "identical cumulative-posting lexical candidates",
            "identical SHA-256 outer holdout",
            "frozen target-domain DuetRank model and semantic backoff",
            "no selection on the reporting holdout",
        ],
        "all_datasets_match_expected_relationship": all_expected,
        "datasets": results,
    }
    summary_path = args.output_root / "summary.json"
    atomic_json(summary_path, output)
    print(json.dumps(output, indent=2), flush=True)
    print(f"saved {summary_path}", flush=True)


if __name__ == "__main__":
    main()
