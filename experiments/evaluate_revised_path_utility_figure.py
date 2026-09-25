"""Paired Plain/CCADPE outer-holdout utility for the revised lexical setup."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if (ROOT / ".deps").exists():
    sys.path.insert(0, str(ROOT / ".deps"))
os.environ.setdefault("DUETER_FORCE_CPU_DEPS", "1")

import joblib
import numpy as np

from benchmark_million_semantic_hybrid import atomic_json, load_queries_qrels, read_ids
from evaluate_duetrank_recall import metric_rows
from experiment_adaptive_fusion import query_features
from experiment_candidate_rank_fusion import learned_rankings, mix_rankings, split_mask


CONFIG = {
    "nq": {
        "label": "NQ",
        "split": "test",
        "documents": 2_681_468,
        "semantic_cache": "nq_2681468",
        "semantic_result": "nq",
        "revised_result": "nq",
        "trace": "nq_full.trace.json",
        "budget": "50000",
        "model": "nq_test_full_b50000_duetrank_revised_d8192_rc32_b16_d8192-r32-b16.joblib",
        "semantic_backoff": 0.5,
        "expected_dual": 0.5207032354392801,
        "dpe_lexical": "exact_bm25_rankings.npz",
    },
    "hotpotqa": {
        "label": "HotpotQA",
        "split": "test",
        "documents": 5_233_329,
        "semantic_cache": "hotpotqa_5233329",
        "semantic_result": "hotpotqa",
        "revised_result": "hotpotqa",
        "trace": "hotpotqa_full.trace.json",
        "budget": "50000",
        "model": "hotpotqa_test_full_b50000_duetrank_revised_d8192_rc32_b16_l64_d8192-r32-b16-l64.joblib",
        "semantic_backoff": 0.0,
        "expected_dual": 0.6833913394015391,
        "dpe_lexical": "selected_l64_exact_bm25_rankings.npz",
    },
    "msmarco": {
        "label": "MS MARCO",
        "split": "dev",
        "documents": 8_841_823,
        "semantic_cache": "msmarco_8841823",
        "semantic_result": "msmarco_dev",
        "revised_result": "msmarco_dev",
        "trace": "msmarco_dev_full.trace.json",
        "budget": "250000",
        "model": "msmarco_dev_full_b250000_duetrank_revised_d8192_rc32_b16_d8192-r32-b16.joblib",
        "semantic_backoff": 0.0,
        "expected_dual": 0.3015123568086712,
        "dpe_lexical": "exact_bm25_rankings.npz",
    },
}


def ndcg_rows(
    rankings: np.ndarray,
    doc_ids: list[str],
    query_ids: list[str],
    qrels: dict[str, dict[str, int]],
) -> np.ndarray:
    return metric_rows(rankings, doc_ids, query_ids, qrels, 10)["ndcg"]


def fused(
    model: object,
    semantic: np.ndarray,
    lexical: np.ndarray,
    features: np.ndarray,
    holdout: np.ndarray,
    semantic_backoff: float,
) -> np.ndarray:
    learned = learned_rankings(model, semantic, lexical, features, mask=holdout)
    if semantic_backoff == 0.0:
        return learned
    return mix_rankings(semantic, learned, semantic_backoff)


def run_dataset(dataset: str, output_root: Path) -> dict[str, object]:
    cfg = CONFIG[dataset]
    query_ids, query_texts, qrels, _ = load_queries_qrels(
        ROOT / "data" / dataset, str(cfg["split"])
    )
    doc_ids = read_ids(
        ROOT / "cache" / "full_semantic" / str(cfg["semantic_cache"]) / "doc_ids.txt"
    )
    holdout = split_mask(query_ids, "outer")

    plain_semantic = np.asarray(
        np.load(ROOT / "results" / "plaintext_duetrank" / f"{dataset}_semantic_plain.rankings.npz")[
            "semantic_plain"
        ],
        dtype=np.int32,
    )
    dpe_semantic = np.asarray(
        np.load(
            ROOT
            / "results"
            / "dual_compartment_full_p256"
            / str(cfg["semantic_result"])
            / "semantic_compartment.rankings.npz"
        )["semantic"],
        dtype=np.int32,
    )
    plain_lexical = np.asarray(
        np.load(
            ROOT
            / "results"
            / "revised_d8192_rc32_b16_delta"
            / str(cfg["revised_result"])
            / "plaintext_exact_bm25_rankings.npz"
        )["lexical"],
        dtype=np.int32,
    )
    dpe_lexical = np.asarray(
        np.load(
            ROOT
            / "results"
            / "revised_d8192_rc32_b16"
            / str(cfg["revised_result"])
            / str(cfg["dpe_lexical"])
        )["lexical_dpe"],
        dtype=np.int32,
    )

    trace = json.loads(
        (ROOT / "results" / "budgeted_lexical" / str(cfg["trace"])).read_text(
            encoding="utf-8"
        )
    )[str(cfg["budget"])]
    model = joblib.load(
        ROOT / "results" / "budgeted_lexical_dpe" / str(cfg["model"])
    )
    semantic_backoff = float(cfg["semantic_backoff"])

    plain_features = query_features(
        query_texts, plain_semantic[None, :, :], plain_lexical[0, :, :100], trace
    )
    plain_dual = fused(
        model,
        plain_semantic,
        plain_lexical[0],
        plain_features,
        holdout,
        semantic_backoff,
    )

    dpe_features = query_features(
        query_texts, dpe_semantic, dpe_lexical[0, :, :100], trace
    )
    dpe_dual = np.stack(
        [
            fused(
                model,
                semantic,
                dpe_lexical[index % len(dpe_lexical)],
                dpe_features,
                holdout,
                semantic_backoff,
            )
            for index, semantic in enumerate(dpe_semantic)
        ]
    )

    plain = {
        "semantic": float(np.mean(ndcg_rows(plain_semantic, doc_ids, query_ids, qrels)[holdout])),
        "lexical": float(np.mean(ndcg_rows(plain_lexical[0], doc_ids, query_ids, qrels)[holdout])),
        "dual": float(np.mean(ndcg_rows(plain_dual, doc_ids, query_ids, qrels)[holdout])),
    }
    ccadpe = {
        "semantic": float(
            np.mean(
                np.stack(
                    [ndcg_rows(row, doc_ids, query_ids, qrels) for row in dpe_semantic]
                )[:, holdout]
            )
        ),
        "lexical": float(
            np.mean(
                np.stack(
                    [ndcg_rows(row, doc_ids, query_ids, qrels) for row in dpe_lexical]
                )[:, holdout]
            )
        ),
        "dual": float(
            np.mean(
                np.stack(
                    [ndcg_rows(row, doc_ids, query_ids, qrels) for row in dpe_dual]
                )[:, holdout]
            )
        ),
    }
    if not np.isclose(ccadpe["dual"], float(cfg["expected_dual"]), atol=1e-10):
        raise AssertionError(
            f"{dataset}: recomputed revised DuetRank {ccadpe['dual']} "
            f"!= expected {cfg['expected_dual']}"
        )

    np.savez_compressed(
        output_root / f"{dataset}_paired.rankings.npz",
        semantic_plain=plain_semantic,
        lexical_plain=plain_lexical,
        dual_plain=plain_dual,
        semantic_ccadpe=dpe_semantic,
        lexical_ccadpe=dpe_lexical,
        dual_ccadpe=dpe_dual,
        outer_holdout=holdout,
    )
    return {
        "dataset": str(cfg["label"]),
        "outer_holdout_queries": int(np.sum(holdout)),
        "plain": plain,
        "ccadpe": ccadpe,
        "revised_lexical_parameters": {
            "hash_dimension": 1024,
            "d_l": 8192,
            "r_c_l": 32,
            "K_l_cmp": 16,
            "t_l": 64,
            "returned_depth": 300,
        },
        "controls": {
            "candidate_pool": "same cumulative-posting pool within each lexical Plain/CCADPE pair",
            "scoring": "exact cosine and candidate-local exact BM25",
            "fusion": "same frozen revised DuetRank model and semantic backoff",
            "reporting": "same SHA-256 outer holdout",
        },
    }


def main() -> None:
    output_root = ROOT / "results" / "revised_path_utility"
    output_root.mkdir(parents=True, exist_ok=True)
    rows = [run_dataset(dataset, output_root) for dataset in CONFIG]
    summary = {
        "experiment": "paired outer-holdout path utility for revised lexical CCADPE",
        "metric": "outer-holdout nDCG@10",
        "datasets": rows,
    }
    atomic_json(output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
