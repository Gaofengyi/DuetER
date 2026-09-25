"""Calibration-size robustness of DuetRank with exact client BM25 rankings."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if (ROOT / ".deps").exists():
    sys.path.insert(0, str(ROOT / ".deps"))

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from benchmark_million_semantic_hybrid import atomic_json, load_queries_qrels, read_ids
from experiment_adaptive_fusion import ndcg_rows, query_features
from experiment_candidate_rank_fusion import (
    FULL_DOCUMENTS,
    learned_rankings,
    mix_rankings,
    paired_bootstrap,
    split_mask,
    training_matrix,
)


CONFIG = {
    "nq": {
        "split": "test",
        "result": "nq",
        "trace": "nq_full.trace.json",
        "budget": "50000",
    },
    "hotpotqa": {
        "split": "test",
        "result": "hotpotqa",
        "trace": "hotpotqa_full.trace.json",
        "budget": "50000",
    },
    "msmarco": {
        "split": "dev",
        "result": "msmarco_dev",
        "trace": "msmarco_dev_full.trace.json",
        "budget": "250000",
    },
}
FRACTIONS = (0.25, 0.50, 1.00)
COMPARTMENT_ROOT = ROOT / "results" / "dual_compartment_full_p256"
EXACT_ROOT = ROOT / "results" / "exact_bm25_client_rerank"
DESTINATION = ROOT / "results" / "exact_bm25_calibration_robustness" / "summary.json"


def nested_calibration_mask(
    query_ids: list[str], calibration: np.ndarray, fraction: float
) -> np.ndarray:
    indices = np.flatnonzero(calibration)
    ordered = sorted(
        indices,
        key=lambda index: hashlib.sha256(
            f"calibration-size:{query_ids[index]}".encode("utf-8")
        ).digest(),
    )
    count = max(1, int(round(len(indices) * fraction)))
    mask = np.zeros(len(query_ids), dtype=bool)
    mask[np.asarray(ordered[:count], dtype=np.int64)] = True
    return mask


def run_dataset(dataset: str) -> dict[str, object]:
    cfg = CONFIG[dataset]
    split = str(cfg["split"])
    result_dir = COMPARTMENT_ROOT / str(cfg["result"])
    exact_dir = EXACT_ROOT / str(cfg["result"])
    query_ids, query_texts, qrels, _ = load_queries_qrels(
        ROOT / "data" / dataset, split
    )
    documents = FULL_DOCUMENTS[dataset]
    doc_ids = read_ids(
        ROOT / "cache" / "full_semantic" / f"{dataset}_{documents}" / "doc_ids.txt"
    )
    doc_to_index = {doc_id: index for index, doc_id in enumerate(doc_ids)}
    semantic_repeats = np.asarray(
        np.load(result_dir / "semantic_compartment.rankings.npz")["semantic"],
        dtype=np.int32,
    )
    lexical_repeats = np.asarray(
        np.load(exact_dir / "lexical_exact_bm25.rankings.npz")["lexical_dpe"],
        dtype=np.int32,
    )
    lexical = lexical_repeats[0]
    trace = json.loads(
        (ROOT / "results" / "budgeted_lexical" / str(cfg["trace"])).read_text(
            encoding="utf-8"
        )
    )[str(cfg["budget"])]
    q_features = query_features(
        query_texts, semantic_repeats, lexical[:, :100], trace
    )
    outer = split_mask(query_ids, "outer")
    calibration = ~outer
    final_payload = json.loads((exact_dir / "results.json").read_text(encoding="utf-8"))
    final_selection = final_payload["duetrank"]["selection"]
    leaves = int(final_selection["max_leaf_nodes"])
    semantic_weight = float(final_selection["semantic_backoff_weight"])

    semantic_values = np.asarray(
        [
            ndcg_rows(semantic, doc_to_index, query_ids, qrels)
            for semantic in semantic_repeats
        ]
    )
    rows: list[dict[str, object]] = []
    for fraction in FRACTIONS:
        train_mask = nested_calibration_mask(query_ids, calibration, fraction)
        print(
            f"[{dataset}] calibration={fraction:.0%} ({int(np.sum(train_mask))} queries)",
            flush=True,
        )
        matrix, labels, sample_weights = training_matrix(
            semantic_repeats[0],
            lexical,
            q_features,
            train_mask,
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
            random_state=20260826,
        )
        started = time.perf_counter()
        model.fit(matrix, labels, sample_weight=sample_weights)
        training_seconds = time.perf_counter() - started

        mixed_values = []
        for repeat_index, semantic in enumerate(semantic_repeats):
            lexical_repeat = lexical_repeats[repeat_index % len(lexical_repeats)]
            learned = learned_rankings(
                model, semantic, lexical_repeat, q_features, mask=outer
            )
            mixed = (
                learned
                if semantic_weight == 0.0
                else mix_rankings(semantic, learned, semantic_weight)
            )
            mixed_values.append(ndcg_rows(mixed, doc_to_index, query_ids, qrels))
        mixed_values_array = np.asarray(mixed_values)
        significance = paired_bootstrap(semantic_values, mixed_values_array, outer)
        rows.append(
            {
                "calibration_fraction": fraction,
                "calibration_queries": int(np.sum(train_mask)),
                "training_rows": int(len(matrix)),
                "training_positives": int(np.sum(labels)),
                "training_seconds": training_seconds,
                "max_leaf_nodes": leaves,
                "semantic_backoff_weight": semantic_weight,
                "outer_queries": int(np.sum(outer)),
                "semantic_nDCG@10": float(np.mean(semantic_values[:, outer])),
                "DuetRank_nDCG@10": float(
                    np.mean(mixed_values_array[:, outer])
                ),
                "gain_vs_semantic": significance,
            }
        )
        atomic_json(
            DESTINATION.with_name(f"{dataset}.json"),
            {"dataset": dataset, "rows": rows},
        )
        del matrix, labels, sample_weights, model
    return {
        "documents": documents,
        "queries": len(query_ids),
        "split": split,
        "fixed_max_leaf_nodes": leaves,
        "fixed_semantic_backoff_weight": semantic_weight,
        "rows": rows,
    }


def main() -> None:
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "DuetRank calibration robustness with candidate-local exact BM25",
        "fractions": list(FRACTIONS),
        "nested_deterministic_subsets": True,
        "datasets": {},
    }
    for dataset in CONFIG:
        report["datasets"][dataset] = run_dataset(dataset)
        atomic_json(DESTINATION, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
