"""Complete-corpus DuetRank calibration-size and seed robustness experiment.

The outer SHA-256 holdout, path rankings, query features, model capacity, and
semantic-backoff coefficient are fixed.  Only the subset sampled from the
original calibration half and the learner random seed change.  This isolates
whether DuetRank's held-out gain depends on a lucky calibration sample.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if (ROOT / ".deps").exists():
    sys.path.insert(0, str(ROOT / ".deps"))

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from benchmark_million_semantic_hybrid import atomic_json, load_queries_qrels, read_ids
from evaluate_duetrank_recall import metric_rows
from experiment_adaptive_fusion import query_features
from experiment_candidate_rank_fusion import (
    document_feature_matrix,
    mix_rankings,
    rank_maps,
    split_mask,
    training_matrix,
)
from experiment_fixed_rrf_weight_sweep import CONFIGS


FRACTIONS = (0.10, 0.25, 0.50, 0.75, 1.00)
SEEDS = (20260826, 20260901, 20260902, 20260903, 20260904,
         20260905, 20260906, 20260907, 20260908, 20260909)
SELECTION_RESULTS = {
    "nq": ROOT / "results/budgeted_lexical_dpe/learned_fusion_nq_full_d500_o300.json",
    "hotpotqa": ROOT / "results/budgeted_lexical_dpe/learned_fusion_hotpotqa_full.json",
    "msmarco": ROOT
    / "results/budgeted_lexical_dpe/learned_fusion_msmarco_dev_full_b250k_d500_o300.json",
}
TRACES = {
    "nq": (ROOT / "results/budgeted_lexical/nq_full.trace.json", "50000"),
    "hotpotqa": (
        ROOT / "results/budgeted_lexical/hotpotqa_full.trace.json",
        "50000",
    ),
    "msmarco": (
        ROOT / "results/budgeted_lexical/msmarco_dev_full.trace.json",
        "250000",
    ),
}


def prepare_outer_candidates(
    semantic: np.ndarray,
    lexical: np.ndarray,
    query_features: np.ndarray,
    outer: np.ndarray,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    prepared = []
    for query_index in np.flatnonzero(outer):
        semantic_positions = rank_maps(semantic[query_index], 100)
        lexical_positions = rank_maps(lexical[query_index], 1000)
        candidates = np.asarray(
            list(set(semantic_positions).union(lexical_positions)), dtype=np.int32
        )
        matrix = document_feature_matrix(
            candidates.tolist(),
            semantic_positions,
            lexical_positions,
            query_features[query_index],
        ).astype(np.float32)
        prepared.append((int(query_index), candidates, matrix))
    return prepared


def learned_rankings_precomputed(
    model: HistGradientBoostingClassifier,
    semantic: np.ndarray,
    prepared: list[tuple[int, np.ndarray, np.ndarray]],
    depth: int = 100,
    row_batch: int = 200_000,
) -> np.ndarray:
    output = np.asarray(semantic[:, :depth], dtype=np.int32).copy()

    def score_batch(batch: list[tuple[int, np.ndarray, np.ndarray]]) -> None:
        matrix = np.concatenate([item[2] for item in batch], axis=0)
        probabilities = model.predict_proba(matrix)[:, 1]
        offset = 0
        for query_index, candidates, features in batch:
            count = len(features)
            scores = probabilities[offset : offset + count]
            offset += count
            order = sorted(
                zip(candidates.tolist(), scores.tolist()),
                key=lambda item: (-item[1], item[0]),
            )[:depth]
            output[query_index, : len(order)] = [document for document, _ in order]

    pending: list[tuple[int, np.ndarray, np.ndarray]] = []
    pending_rows = 0
    for item in prepared:
        if pending and pending_rows + len(item[2]) > row_batch:
            score_batch(pending)
            pending = []
            pending_rows = 0
        pending.append(item)
        pending_rows += len(item[2])
    if pending:
        score_batch(pending)
    return output


def load_selection(dataset: str) -> dict[str, object]:
    payload = json.loads(SELECTION_RESULTS[dataset].read_text(encoding="utf-8"))
    return payload["datasets"][0]


def deterministic_subset(
    calibration_indices: np.ndarray, fraction: float, seed: int, size: int
) -> np.ndarray:
    if fraction >= 1.0:
        chosen = calibration_indices
    else:
        count = max(1, int(round(fraction * len(calibration_indices))))
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(calibration_indices, size=count, replace=False))
    mask = np.zeros(size, dtype=bool)
    mask[chosen] = True
    return mask


def summarize_runs(
    fraction: float,
    calibration_queries: int,
    semantic_score: float,
    primary_score: float,
    runs: list[dict[str, object]],
) -> dict[str, object]:
    scores = np.asarray([run["outer_ndcg_at_10"] for run in runs], dtype=np.float64)
    gains = scores - semantic_score
    primary_gain = primary_score - semantic_score
    return {
        "calibration_fraction": fraction,
        "calibration_queries": (
            calibration_queries
            if fraction >= 1.0
            else int(round(fraction * calibration_queries))
        ),
        "seeds": len(runs),
        "outer_ndcg_at_10_mean": float(np.mean(scores)),
        "outer_ndcg_at_10_sd": float(np.std(scores)),
        "outer_ndcg_at_10_min": float(np.min(scores)),
        "outer_ndcg_at_10_max": float(np.max(scores)),
        "mean_gain_vs_semantic": float(np.mean(gains)),
        "minimum_gain_vs_semantic": float(np.min(gains)),
        "positive_gain_seed_fraction": float(np.mean(gains > 0.0)),
        "mean_primary_gain_retained": (
            float(np.mean(gains) / primary_gain) if primary_gain > 0 else None
        ),
        "runs": runs,
    }


def run_dataset(
    dataset: str,
    fractions: tuple[float, ...],
    seeds: tuple[int, ...],
    output_dir: Path,
) -> dict[str, object]:
    started = time.perf_counter()
    config = CONFIGS[dataset]
    selection_payload = load_selection(dataset)
    selection = selection_payload["selection"]
    query_ids, query_texts, qrels, _ = load_queries_qrels(
        ROOT / "data" / dataset, str(config["split"])
    )
    cache_dir = ROOT / "cache/full_semantic" / f"{dataset}_{config['documents']}"
    doc_ids = read_ids(cache_dir / "doc_ids.txt")
    doc_to_index = {doc_id: index for index, doc_id in enumerate(doc_ids)}
    semantic_repeats = np.asarray(np.load(config["semantic"])["semantic"], dtype=np.int32)
    lexical_repeats = np.asarray(np.load(config["lexical"])["lexical_dpe"], dtype=np.int32)
    semantic = semantic_repeats[0]
    lexical = lexical_repeats[0]
    trace_path, trace_key = TRACES[dataset]
    trace = json.loads(trace_path.read_text(encoding="utf-8"))[trace_key]
    features = query_features(query_texts, semantic_repeats, lexical[:, :100], trace)
    outer = split_mask(query_ids, "outer")
    calibration = ~outer
    calibration_indices = np.flatnonzero(calibration)
    outer_candidates = prepare_outer_candidates(semantic, lexical, features, outer)
    leaves = int(selection["max_leaf_nodes"])
    backoff = float(selection["semantic_backoff_weight"])
    semantic_rows = metric_rows(semantic, doc_ids, query_ids, qrels, 10)["ndcg"]
    semantic_score = float(np.mean(semantic_rows[outer]))
    primary_score = float(selection_payload["held_out"]["semantic_backoff_ndcg_at_10"])

    fraction_rows: list[dict[str, object]] = []
    for fraction in fractions:
        runs: list[dict[str, object]] = []
        for seed in seeds:
            subset = deterministic_subset(
                calibration_indices, fraction, seed, len(query_ids)
            )
            train_started = time.perf_counter()
            matrix, labels, weights = training_matrix(
                semantic,
                lexical,
                features,
                subset,
                query_ids,
                qrels,
                doc_to_index,
            )
            model = HistGradientBoostingClassifier(
                max_iter=120,
                learning_rate=0.06,
                max_leaf_nodes=leaves,
                min_samples_leaf=40,
                l2_regularization=1.0,
                random_state=seed,
            )
            model.fit(matrix, labels, sample_weight=weights)
            train_seconds = time.perf_counter() - train_started
            infer_started = time.perf_counter()
            learned = learned_rankings_precomputed(model, semantic, outer_candidates)
            mixed = learned if backoff == 0.0 else mix_rankings(semantic, learned, backoff)
            ndcg = metric_rows(mixed, doc_ids, query_ids, qrels, 10)["ndcg"]
            inference_seconds = time.perf_counter() - infer_started
            score = float(np.mean(ndcg[outer]))
            runs.append(
                {
                    "seed": seed,
                    "training_queries": int(np.sum(subset)),
                    "training_rows": int(len(matrix)),
                    "training_positives": int(np.sum(labels)),
                    "outer_ndcg_at_10": score,
                    "gain_vs_semantic": score - semantic_score,
                    "training_seconds": train_seconds,
                    "outer_inference_seconds": inference_seconds,
                }
            )
            print(
                f"[{dataset}] fraction={fraction:.2f} seed={seed} "
                f"nDCG@10={score:.6f} gain={score-semantic_score:+.6f}",
                flush=True,
            )
            del matrix, labels, weights, model, learned, mixed, ndcg
            gc.collect()
        fraction_rows.append(
            summarize_runs(
                fraction,
                len(calibration_indices),
                semantic_score,
                primary_score,
                runs,
            )
        )
        atomic_json(
            output_dir / f"{dataset}_partial.json",
            {
                "status": "running",
                "dataset": dataset,
                "completed_fractions": [row["calibration_fraction"] for row in fraction_rows],
                "fraction_results": fraction_rows,
            },
        )

    original_seed_row = next(
        run
        for row in fraction_rows
        if row["calibration_fraction"] == 1.0
        for run in row["runs"]
        if int(run["seed"]) == 20260826
    )
    return {
        "dataset": dataset,
        "documents": int(config["documents"]),
        "queries": len(query_ids),
        "calibration_queries": int(np.sum(calibration)),
        "outer_holdout_queries": int(np.sum(outer)),
        "fixed_max_leaf_nodes": leaves,
        "fixed_semantic_backoff_weight": backoff,
        "semantic_outer_ndcg_at_10": semantic_score,
        "primary_duetrank_outer_ndcg_at_10": primary_score,
        "original_seed_full_calibration_ndcg_at_10": original_seed_row[
            "outer_ndcg_at_10"
        ],
        "original_seed_reproduction_absolute_error": abs(
            float(original_seed_row["outer_ndcg_at_10"]) - primary_score
        ),
        "fraction_results": fraction_rows,
        "total_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+", choices=tuple(CONFIGS), default=list(CONFIGS)
    )
    parser.add_argument(
        "--fractions", default=",".join(str(value) for value in FRACTIONS)
    )
    parser.add_argument("--seeds", default=",".join(str(value) for value in SEEDS))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "duetrank_calibration_robustness",
    )
    args = parser.parse_args()
    fractions = tuple(float(value) for value in args.fractions.split(","))
    seeds = tuple(int(value) for value in args.seeds.split(","))
    if tuple(sorted(set(fractions))) != fractions or not all(
        0.0 < value <= 1.0 for value in fractions
    ):
        raise ValueError("fractions must be unique, ascending, and in (0,1]")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    if 1.0 not in fractions or 20260826 not in seeds:
        raise ValueError("the full calibration and original seed are required for audit")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    datasets = []
    for dataset in args.datasets:
        result = run_dataset(dataset, fractions, seeds, args.output_dir)
        datasets.append(result)
        atomic_json(args.output_dir / f"{dataset}_results.json", result)
        atomic_json(
            args.output_dir / "results.json",
            {
                "status": "running",
                "experiment": "DuetRank calibration-size and random-seed robustness",
                "datasets": datasets,
            },
        )
    output = {
        "status": "passed",
        "experiment": "DuetRank calibration-size and random-seed robustness",
        "fractions": fractions,
        "seeds": seeds,
        "retrieval_repeat_policy": (
            "first stored DPE repeat fixed for both training and evaluation to isolate "
            "calibration randomness; the primary five-repeat score is retained as an audit"
        ),
        "datasets": datasets,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn_model": "HistGradientBoostingClassifier",
        },
    }
    atomic_json(args.output_dir / "results.json", output)
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
