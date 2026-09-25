"""Evaluate standard unweighted RRF on the final Plain/CCADPE rankings.

The experiment reuses the exact-BM25 paired ranking artifacts and untouched
SHA-256 outer holdouts used by the final DuetRank figure.  RRF has no fitted
parameters: both paths receive equal weight and the rank constant is c=60.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if (ROOT / ".deps").exists():
    sys.path.insert(0, str(ROOT / ".deps"))

import numpy as np

from benchmark_budgeted_lexical_candidates import fuse_all
from benchmark_million_semantic_hybrid import atomic_json, load_queries_qrels, read_ids
from evaluate_duetrank_recall import metric_rows


CONFIG = {
    "nq": {
        "label": "NQ",
        "split": "test",
        "cache": "nq_2681468",
    },
    "hotpotqa": {
        "label": "HotpotQA",
        "split": "test",
        "cache": "hotpotqa_5233329",
    },
    "msmarco": {
        "label": "MS MARCO",
        "split": "dev",
        "cache": "msmarco_8841823",
    },
}


def bootstrap_difference(
    baseline: np.ndarray,
    system: np.ndarray,
    samples: int = 5000,
) -> dict[str, float]:
    """Paired query bootstrap after averaging randomized CCADPE repeats."""
    differences = np.asarray(system - baseline, dtype=np.float64)
    rng = np.random.default_rng(20260922)
    bootstrap = np.empty(samples, dtype=np.float64)
    chunk = 250
    for first in range(0, samples, chunk):
        last = min(first + chunk, samples)
        indices = rng.integers(
            0, len(differences), size=(last - first, len(differences))
        )
        bootstrap[first:last] = np.mean(differences[indices], axis=1)
    return {
        "mean": float(np.mean(differences)),
        "ci95_low": float(np.percentile(bootstrap, 2.5)),
        "ci95_high": float(np.percentile(bootstrap, 97.5)),
        "two_sided_bootstrap_p": float(
            min(
                1.0,
                2.0
                * min(
                    (1 + np.sum(bootstrap <= 0.0)) / (samples + 1),
                    (1 + np.sum(bootstrap >= 0.0)) / (samples + 1),
                ),
            )
        ),
    }


def evaluate_rows(
    rankings: np.ndarray,
    doc_ids: list[str],
    query_ids: list[str],
    qrels: dict[str, dict[str, int]],
) -> dict[str, np.ndarray]:
    if rankings.ndim == 2:
        rankings = rankings[None, :, :]
    rows = [metric_rows(run, doc_ids, query_ids, qrels, 10) for run in rankings]
    return {
        metric: np.stack([row[metric] for row in rows])
        for metric in ("ndcg", "recall", "hit")
    }


def summarize(
    metric: dict[str, np.ndarray], holdout: np.ndarray
) -> dict[str, float]:
    return {
        "ndcg_at_10": float(np.mean(metric["ndcg"][:, holdout])),
        "recall_at_10": float(np.mean(metric["recall"][:, holdout])),
        "hit_at_10": float(np.mean(metric["hit"][:, holdout])),
    }


def run_dataset(dataset: str, output_root: Path) -> dict[str, object]:
    cfg = CONFIG[dataset]
    input_path = (
        ROOT
        / "results"
        / "revised_path_utility"
        / f"{dataset}_paired.rankings.npz"
    )
    payload = np.load(input_path)
    holdout = np.asarray(payload["outer_holdout"], dtype=bool)
    query_ids, _, qrels, _ = load_queries_qrels(ROOT / "data" / dataset, cfg["split"])
    doc_ids = read_ids(ROOT / "cache" / "full_semantic" / cfg["cache"] / "doc_ids.txt")

    semantic_plain = np.asarray(payload["semantic_plain"], dtype=np.int32)
    lexical_plain_repeats = np.asarray(payload["lexical_plain"], dtype=np.int32)
    lexical_plain = (
        lexical_plain_repeats[0]
        if lexical_plain_repeats.ndim == 3
        else lexical_plain_repeats
    )
    duetrank_plain = np.asarray(payload["dual_plain"], dtype=np.int32)
    rrf_plain = fuse_all(semantic_plain, lexical_plain, 100, 0.5)

    semantic_ccadpe = np.asarray(payload["semantic_ccadpe"], dtype=np.int32)
    lexical_ccadpe = np.asarray(payload["lexical_ccadpe"], dtype=np.int32)
    duetrank_ccadpe = np.asarray(payload["dual_ccadpe"], dtype=np.int32)
    rrf_ccadpe = np.stack(
        [
            fuse_all(
                semantic_ccadpe[r],
                lexical_ccadpe[r % len(lexical_ccadpe)],
                100,
                0.5,
            )
            for r in range(semantic_ccadpe.shape[0])
        ]
    )

    executions = {}
    rankings_by_execution = {
        "plain": {
            "semantic": semantic_plain,
            "lexical": lexical_plain,
            "rrf": rrf_plain,
            "duetrank": duetrank_plain,
        },
        "ccadpe": {
            "semantic": semantic_ccadpe,
            "lexical": lexical_ccadpe,
            "rrf": rrf_ccadpe,
            "duetrank": duetrank_ccadpe,
        },
    }
    for execution, rankings in rankings_by_execution.items():
        metrics = {
            name: evaluate_rows(run, doc_ids, query_ids, qrels)
            for name, run in rankings.items()
        }
        ndcg_per_query = {
            name: np.mean(values["ndcg"], axis=0)[holdout]
            for name, values in metrics.items()
        }
        executions[execution] = {
            "metrics": {
                name: summarize(values, holdout) for name, values in metrics.items()
            },
            "paired_differences": {
                "rrf_minus_semantic": bootstrap_difference(
                    ndcg_per_query["semantic"], ndcg_per_query["rrf"]
                ),
                "duetrank_minus_rrf": bootstrap_difference(
                    ndcg_per_query["rrf"], ndcg_per_query["duetrank"]
                ),
            },
        }

    ranking_path = output_root / f"{dataset}.rankings.npz"
    temporary_path = output_root / f".{dataset}.rankings.tmp.npz"
    np.savez_compressed(
        temporary_path,
        rrf_plain=rrf_plain,
        rrf_ccadpe=rrf_ccadpe,
        outer_holdout=holdout,
    )
    temporary_path.replace(ranking_path)
    return {
        "dataset": cfg["label"],
        "outer_holdout_queries": int(np.sum(holdout)),
        "rrf": {
            "constant": 60,
            "semantic_weight": 0.5,
            "lexical_weight": 0.5,
            "selection": "fixed a priori; no calibration or test tuning",
        },
        "executions": executions,
    }


def main() -> None:
    output_root = ROOT / "results" / "standard_rrf_final"
    output_root.mkdir(parents=True, exist_ok=True)
    results = [run_dataset(dataset, output_root) for dataset in CONFIG]
    report = {
        "experiment": "standard unweighted RRF on revised exact-BM25 candidate rankings",
        "evaluation": "same SHA-256 outer holdouts as the final DuetRank figure",
        "bootstrap_samples": 5000,
        "datasets": results,
    }
    output_path = output_root / "results.json"
    atomic_json(output_path, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
