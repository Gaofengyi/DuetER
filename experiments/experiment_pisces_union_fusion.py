"""Evaluate the exact set-union aggregation used by the Pisces artifact.

Pisces reports relevant-document coverage of semantic@k union lexical@k.  The
union is unordered and can contain 2k records, so nDCG is deliberately not
reported.  Semantic@2k is included as an equal-maximum-output-size control.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmark_million_semantic_hybrid import atomic_json, load_queries_qrels, read_ids
from experiment_candidate_rank_fusion import paired_bootstrap, split_mask
from experiment_fixed_rrf_weight_sweep import CONFIGS, ROOT


def returned_ids(row: np.ndarray, depth: int, doc_ids: list[str]) -> set[str]:
    return {doc_ids[int(value)] for value in row[:depth] if int(value) >= 0}


def metric_rows(
    semantic: np.ndarray,
    lexical: np.ndarray,
    depth: int,
    query_ids: list[str],
    qrels: dict[str, dict[str, int]],
    doc_ids: list[str],
) -> dict[str, np.ndarray]:
    names = (
        "semantic_recall",
        "lexical_recall",
        "union_recall",
        "semantic_2k_recall",
        "semantic_hit",
        "lexical_hit",
        "union_hit",
        "semantic_2k_hit",
        "union_size",
        "semantic_2k_size",
        "union_precision",
        "lexical_rescue",
        "additional_relevant",
    )
    rows = {name: np.zeros(len(query_ids), dtype=np.float64) for name in names}
    for query_index, query_id in enumerate(query_ids):
        relevant = {
            doc_id for doc_id, gain in qrels[query_id].items() if int(gain) > 0
        }
        sem = returned_ids(semantic[query_index], depth, doc_ids)
        lex = returned_ids(lexical[query_index], depth, doc_ids)
        sem_2k = returned_ids(semantic[query_index], 2 * depth, doc_ids)
        union = sem | lex
        sem_rel = relevant & sem
        lex_rel = relevant & lex
        union_rel = relevant & union
        sem_2k_rel = relevant & sem_2k
        denominator = max(len(relevant), 1)
        rows["semantic_recall"][query_index] = len(sem_rel) / denominator
        rows["lexical_recall"][query_index] = len(lex_rel) / denominator
        rows["union_recall"][query_index] = len(union_rel) / denominator
        rows["semantic_2k_recall"][query_index] = len(sem_2k_rel) / denominator
        rows["semantic_hit"][query_index] = bool(sem_rel)
        rows["lexical_hit"][query_index] = bool(lex_rel)
        rows["union_hit"][query_index] = bool(union_rel)
        rows["semantic_2k_hit"][query_index] = bool(sem_2k_rel)
        rows["union_size"][query_index] = len(union)
        rows["semantic_2k_size"][query_index] = len(sem_2k)
        rows["union_precision"][query_index] = len(union_rel) / max(len(union), 1)
        rows["lexical_rescue"][query_index] = bool(union_rel) and not bool(sem_rel)
        rows["additional_relevant"][query_index] = len(union_rel - sem_rel)
    return rows


def summarize(stacked: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, float | int]:
    result: dict[str, float | int] = {"queries": int(np.sum(mask))}
    for name, values in stacked.items():
        result[name] = float(np.mean(values[:, mask]))
    result["union_recall_gain_vs_semantic_k"] = (
        float(result["union_recall"]) - float(result["semantic_recall"])
    )
    result["union_recall_gain_vs_semantic_2k"] = (
        float(result["union_recall"]) - float(result["semantic_2k_recall"])
    )
    return result


def run_dataset(dataset: str) -> dict[str, object]:
    config = CONFIGS[dataset]
    split = str(config["split"])
    query_ids, _, qrels, _ = load_queries_qrels(ROOT / "data" / dataset, split)
    doc_ids = read_ids(
        ROOT / "cache/full_semantic" / f"{dataset}_{config['documents']}" / "doc_ids.txt"
    )
    semantic = np.asarray(np.load(config["semantic"])["semantic"], dtype=np.int32)
    lexical = np.asarray(np.load(config["lexical"])["lexical_dpe"], dtype=np.int32)
    holdout = split_mask(query_ids, "outer")
    depth_results = {}
    for depth in (5, 10):
        repeats = [
            metric_rows(
                semantic[repeat],
                lexical[repeat],
                depth,
                query_ids,
                qrels,
                doc_ids,
            )
            for repeat in range(len(semantic))
        ]
        stacked = {
            name: np.stack([repeat[name] for repeat in repeats]) for name in repeats[0]
        }
        depth_results[str(depth)] = {
            "all_queries": summarize(stacked, np.ones(len(query_ids), dtype=bool)),
            "outer_holdout": summarize(stacked, holdout),
            "outer_union_vs_semantic_k": paired_bootstrap(
                stacked["semantic_recall"], stacked["union_recall"], holdout
            ),
            "outer_union_vs_equal_budget_semantic_2k": paired_bootstrap(
                stacked["semantic_2k_recall"], stacked["union_recall"], holdout
            ),
        }
    return {
        "dataset": dataset,
        "split": split,
        "documents": int(config["documents"]),
        "queries": len(query_ids),
        "repeats": len(semantic),
        "depths": depth_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/budgeted_lexical_dpe/pisces_union_fusion.json",
    )
    args = parser.parse_args()
    output = {
        "method": "Pisces artifact: relevant documents in semantic@k union lexical@k",
        "source": "baselines/pisces_official_reference/script/similarity_plain.py: bm_top{k} | sim_top{k}",
        "comparability_warning": "The union is unordered and returns up to 2k documents; nDCG is undefined. semantic@2k is the equal-budget control.",
        "datasets": [run_dataset(dataset) for dataset in args.datasets],
    }
    atomic_json(args.output, output)
    compact = {}
    for dataset in output["datasets"]:
        compact[dataset["dataset"]] = dataset["depths"]["10"]["outer_holdout"]
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
