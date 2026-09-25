"""Paired Plain/CCADPE utility audit for the final exact-BM25 system."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if (ROOT / ".deps").exists():
    sys.path.insert(0, str(ROOT / ".deps"))

import joblib
import numpy as np

from benchmark_million_semantic_hybrid import atomic_json, load_queries_qrels, read_ids
from evaluate_duetrank_recall import TRACES, metric_rows
from experiment_adaptive_fusion import query_features
from experiment_candidate_rank_fusion import learned_rankings, mix_rankings, split_mask


CONFIG = {
    "nq": {
        "label": "NQ",
        "split": "test",
        "cache": "nq_2681468",
        "plain_lexical": "nq_full_d500_o300.rankings.npz",
        "exact": "nq",
        "compartment": "nq",
    },
    "hotpotqa": {
        "label": "HotpotQA",
        "split": "test",
        "cache": "hotpotqa_5233329",
        "plain_lexical": "hotpotqa_full_d200.rankings.npz",
        "exact": "hotpotqa",
        "compartment": "hotpotqa",
    },
    "msmarco": {
        "label": "MS MARCO",
        "split": "dev",
        "cache": "msmarco_8841823",
        "plain_lexical": "msmarco_dev_full_b250k_d500_o300.rankings.npz",
        "exact": "msmarco_dev",
        "compartment": "msmarco_dev",
    },
}


def ndcg_rows(
    rankings: np.ndarray,
    doc_ids: list[str],
    query_ids: list[str],
    qrels: dict[str, dict[str, int]],
) -> np.ndarray:
    return metric_rows(rankings, doc_ids, query_ids, qrels, 10)["ndcg"]


def run_dataset(dataset: str, output_root: Path) -> dict[str, object]:
    cfg = CONFIG[dataset]
    exact_root = ROOT / "results" / "exact_bm25_client_rerank" / str(cfg["exact"])
    exact_report = json.loads((exact_root / "results.json").read_text(encoding="utf-8"))
    selection = exact_report["duetrank"]["selection"]
    held_out = exact_report["duetrank"]["held_out"]

    query_ids, query_texts, qrels, _ = load_queries_qrels(
        ROOT / "data" / dataset, str(cfg["split"])
    )
    doc_ids = read_ids(
        ROOT / "cache" / "full_semantic" / str(cfg["cache"]) / "doc_ids.txt"
    )
    holdout = split_mask(query_ids, "outer")

    plain_semantic = np.asarray(
        np.load(ROOT / "results" / "plaintext_duetrank" / f"{dataset}_semantic_plain.rankings.npz")[
            "semantic_plain"
        ],
        dtype=np.int32,
    )
    plain_lexical = np.asarray(
        np.load(
            ROOT
            / "results"
            / "budgeted_lexical_dpe"
            / str(cfg["plain_lexical"])
        )["lexical_plaintext"],
        dtype=np.int32,
    )
    dpe_semantic = np.asarray(
        np.load(
            ROOT
            / "results"
            / "dual_compartment_full_p256"
            / str(cfg["compartment"])
            / "semantic_compartment.rankings.npz"
        )["semantic"],
        dtype=np.int32,
    )
    dpe_lexical = np.asarray(
        np.load(exact_root / "lexical_exact_bm25.rankings.npz")["lexical_dpe"],
        dtype=np.int32,
    )
    dpe_dual = np.asarray(
        np.load(exact_root / "duetrank.rankings.npz")["duetrank"], dtype=np.int32
    )

    trace_path, trace_key = TRACES[dataset]
    trace = json.loads(trace_path.read_text(encoding="utf-8"))[trace_key]
    model = joblib.load(selection["model_path"])
    semantic_backoff = float(selection["semantic_backoff_weight"])
    features = query_features(
        query_texts, plain_semantic[None, :, :], plain_lexical[:, :100], trace
    )
    learned = learned_rankings(
        model, plain_semantic, plain_lexical, features, mask=holdout
    )
    plain_dual = (
        learned
        if semantic_backoff == 0.0
        else mix_rankings(plain_semantic, learned, semantic_backoff)
    )

    plain_rows = {
        "semantic": ndcg_rows(plain_semantic, doc_ids, query_ids, qrels)[None, :],
        "lexical": ndcg_rows(plain_lexical, doc_ids, query_ids, qrels)[None, :],
        "dual": ndcg_rows(plain_dual, doc_ids, query_ids, qrels)[None, :],
    }
    dpe_rows = {
        "semantic": np.stack(
            [ndcg_rows(row, doc_ids, query_ids, qrels) for row in dpe_semantic]
        ),
        "lexical": np.stack(
            [ndcg_rows(row, doc_ids, query_ids, qrels) for row in dpe_lexical]
        ),
        "dual": np.stack(
            [ndcg_rows(row, doc_ids, query_ids, qrels) for row in dpe_dual]
        ),
    }
    plain = {
        key: float(np.mean(values[:, holdout])) for key, values in plain_rows.items()
    }
    ccadpe = {
        key: float(np.mean(values[:, holdout])) for key, values in dpe_rows.items()
    }
    if not np.isclose(
        ccadpe["dual"], float(held_out["semantic_backoff_ndcg_at_10"]), atol=1e-10
    ):
        raise AssertionError(f"{dataset}: stored final DuetRank result is inconsistent")

    result = {
        "dataset": str(cfg["label"]),
        "outer_holdout_queries": int(np.sum(holdout)),
        "plain": plain,
        "ccadpe": ccadpe,
        "controls": {
            "candidate_planners": "identical semantic IVF and lexical posting planners",
            "scoring": "exact cosine and candidate-local standard BM25",
            "fusion": "same frozen exact-BM25 DuetRank model and semantic backoff",
            "reporting": "same untouched SHA-256 outer holdout",
        },
    }
    ranking_path = output_root / f"{dataset}_paired.rankings.npz"
    temporary_path = output_root / f".{dataset}_paired.rankings.tmp.npz"
    np.savez_compressed(
        temporary_path,
        semantic_plain=plain_semantic,
        lexical_plain=plain_lexical,
        dual_plain=plain_dual,
        semantic_ccadpe=dpe_semantic,
        lexical_ccadpe=dpe_lexical,
        dual_ccadpe=dpe_dual,
        outer_holdout=holdout,
    )
    temporary_path.replace(ranking_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=tuple(CONFIG), default=list(CONFIG))
    args = parser.parse_args()
    output_root = ROOT / "results" / "exact_bm25_plaintext_duetrank"
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.json"
    previous: dict[str, dict[str, object]] = {}
    if summary_path.exists():
        old = json.loads(summary_path.read_text(encoding="utf-8"))
        previous = {str(row["dataset"]): row for row in old.get("datasets", [])}
    for dataset in args.datasets:
        row = run_dataset(dataset, output_root)
        previous[str(row["dataset"])] = row
    results = [previous[str(CONFIG[name]["label"])] for name in CONFIG]
    summary = {
        "experiment": "paired Plain/CCADPE utility for final exact-BM25 DuetRank",
        "metric": "outer-holdout nDCG@10",
        "datasets": results,
    }
    atomic_json(summary_path, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
