"""Evaluate the semantic utility of the paper's optional Gaussian mechanism.

This uses full-scan cosine search after clipping unit embeddings, adding the
specified Gaussian noise, and normalizing as post-processing.  Full scan is an
upper bound on indexed semantic utility because it removes candidate loss.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from dueter_common import evaluate, load_beir, normalize_rows, top_indices


ROOT = Path(__file__).resolve().parent


def run_dataset(
    name: str,
    data_dir: Path,
    corpus_embeddings: Path,
    query_embeddings: Path,
    sigmas: list[float],
    repeats: int,
    delta: float,
    clip: float,
    seed: int,
) -> dict[str, object]:
    corpus_ids, _, query_ids, _, qrels = load_beir(data_dir)
    documents = normalize_rows(np.load(corpus_embeddings).astype(np.float32))
    queries = normalize_rows(np.load(query_embeddings).astype(np.float32))
    if len(corpus_ids) != len(documents) or len(query_ids) != len(queries):
        raise ValueError(f"embedding/data length mismatch for {name}")

    clipped = queries * np.minimum(1.0, clip / np.maximum(np.linalg.norm(queries, axis=1), 1e-12))[:, None]
    classical_numerator = 2.0 * clip * math.sqrt(2.0 * math.log(1.25 / delta))
    rng = np.random.default_rng(seed)
    rows = []
    for sigma in sigmas:
        trials = []
        for _ in range(repeats):
            released = clipped.copy()
            if sigma > 0:
                released += rng.normal(0.0, sigma, released.shape).astype(np.float32)
            released = normalize_rows(released)
            scores = released @ documents.T
            rankings = {
                qid: top_indices(scores[qi], 100).tolist()
                for qi, qid in enumerate(query_ids)
            }
            trials.append(evaluate(rankings, query_ids, corpus_ids, qrels))
        metrics = {
            key: {
                "mean": float(np.mean([trial[key] for trial in trials])),
                "std": float(np.std([trial[key] for trial in trials], ddof=1)) if repeats > 1 else 0.0,
            }
            for key in trials[0]
        }
        epsilon = None if sigma == 0 else classical_numerator / sigma
        certified = epsilon is not None and epsilon <= 1.0
        rows.append(
            {
                "sigma": sigma,
                "classical_epsilon": epsilon if certified else None,
                "classical_bound_applicable": certified,
                "uncertified_epsilon_value_for_reference": epsilon if not certified else None,
                "metrics": metrics,
            }
        )
        print(
            f"{name:8s} sigma={sigma:6.3f} "
            f"eps={epsilon if certified else 'uncertified'} "
            f"nDCG={metrics['nDCG@10']['mean']:.4f} "
            f"R100={metrics['Recall@100']['mean']:.4f}"
        )
    return {
        "dataset": name,
        "documents": len(documents),
        "queries": len(queries),
        "clip": clip,
        "delta": delta,
        "repeats": repeats,
        "classical_gaussian_numerator": classical_numerator,
        "full_scan_upper_bound": True,
        "results": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigmas", nargs="+", type=float, default=[0.0, 0.015, 0.1, 0.5, 1.0, 2.0, 5.0, 9.69, 19.38])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "dp_utility.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    datasets = [
        (
            "SciFact",
            ROOT / "data" / "scifact",
            ROOT / "cache" / "granite_gpu" / "scifact-st-a7972eb5022e_corpus_embeddings.npy",
            ROOT / "cache" / "granite_gpu" / "scifact-st-a7972eb5022e_query_embeddings.npy",
        ),
        (
            "NFCorpus",
            ROOT / "data" / "nfcorpus",
            ROOT / "cache" / "granite_gpu" / "nfcorpus-st-a7972eb5022e_corpus_embeddings.npy",
            ROOT / "cache" / "granite_gpu" / "nfcorpus-st-a7972eb5022e_query_embeddings.npy",
        ),
    ]
    output = {
        "mechanism": "Clip to C, add iid Gaussian noise, normalize as post-processing, full-scan cosine",
        "datasets": [
            run_dataset(*dataset, args.sigmas, args.repeats, args.delta, args.clip, args.seed + index)
            for index, dataset in enumerate(datasets)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")
