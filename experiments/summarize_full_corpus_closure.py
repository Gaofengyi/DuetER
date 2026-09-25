"""Create the audited summary for complete-corpus lexical-DPE/DuetRank runs."""

from __future__ import annotations

import json
from pathlib import Path

from benchmark_million_semantic_hybrid import atomic_json


ROOT = Path(__file__).resolve().parent


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def fusion(relative: str) -> dict:
    return load(relative)["datasets"][0]


def main() -> None:
    ms_test = load("results/budgeted_lexical_dpe/msmarco_full_d200.json")
    ms_test_fusion = fusion(
        "results/budgeted_lexical_dpe/learned_fusion_msmarco_full.json"
    )
    nq = load("results/budgeted_lexical_dpe/nq_full_d500_o300.json")
    nq_fusion = fusion(
        "results/budgeted_lexical_dpe/learned_fusion_nq_full_d500_o300.json"
    )
    hotpot = load("results/budgeted_lexical_dpe/hotpotqa_full_d200.json")
    hotpot_fusion = fusion(
        "results/budgeted_lexical_dpe/learned_fusion_hotpotqa_full.json"
    )
    ms_dev_semantic = load("results/msmarco_dev_full_semantic/results.json")
    ms_dev_planner = load("results/budgeted_lexical/msmarco_dev_full.json")
    ms_dev = load(
        "results/budgeted_lexical_dpe/msmarco_dev_full_b250k_d500_o300.json"
    )
    ms_dev_fusion = fusion(
        "results/budgeted_lexical_dpe/"
        "learned_fusion_msmarco_dev_full_b250k_d500_o300.json"
    )

    def row(run: dict, learned: dict) -> dict:
        held = learned["held_out"]
        return {
            "dataset": run["dataset"],
            "split": run.get("split", "test"),
            "documents": run["documents"],
            "queries": run["queries"],
            "posting_budget": run["configuration"].get("budget_key", 50_000),
            "cloud_depth": run["configuration"]["cloud_depth"],
            "client_lexical_depth": run["configuration"]["output_depth"],
            "materialized_candidate_rows": run["materialization"][
                "materialized_candidate_rows"
            ],
            "lexical_plaintext_ndcg_at_10": run["lexical_plaintext_metrics"][
                "nDCG@10"
            ],
            "lexical_dpe_ndcg_at_10": run["lexical_dpe_mean"]["nDCG@10"],
            "heldout_queries": held["queries"],
            "heldout_semantic_ndcg_at_10": held["semantic_ndcg_at_10"],
            "heldout_calibrated_fixed_rrf_ndcg_at_10": held[
                "calibrated_fixed_rrf_ndcg_at_10"
            ],
            "heldout_duetrank_ndcg_at_10": held["semantic_backoff_ndcg_at_10"],
            "duetrank_vs_semantic": held["backoff_vs_semantic"],
            "duetrank_vs_fixed": held["backoff_vs_calibrated_fixed_rrf"],
            "client_inference_ms": learned["selection"][
                "client_inference_mean_ms_per_query"
            ],
            "dpe_online_steady_mean_ms": run.get("online_steady_mean_ms"),
            "dpe_online_steady_p95_ms": run.get("online_steady_p95_ms"),
        }

    output = {
        "complete_corpus_actual_lexical_dpe_duetrank": [
            row(ms_test, ms_test_fusion),
            row(nq, nq_fusion),
            row(hotpot, hotpot_fusion),
            row(ms_dev, ms_dev_fusion),
        ],
        "msmarco_dev_semantic_cell_major": {
            "documents": ms_dev_semantic["documents"],
            "queries": ms_dev_semantic["queries"],
            "semantic_dpe_mean": ms_dev_semantic["semantic_dpe_mean"],
            "timing": ms_dev_semantic["timing"],
        },
        "msmarco_dev_planner": {
            "semantic_dpe_mean": ms_dev_planner["semantic_dpe_mean"],
            "budget_50k_cap_1000": next(
                row
                for row in ms_dev_planner["sweep"]
                if row["posting_budget"] == 50_000 and row["candidate_cap"] == 1000
            ),
            "budget_250k_cap_1000": next(
                row
                for row in ms_dev_planner["sweep"]
                if row["posting_budget"] == 250_000 and row["candidate_cap"] == 1000
            ),
        },
        "scope_warning": (
            "Every reported complete-corpus query workload uses the full released "
            "HMAC index. Lexical coordinates are materialized for the union of every "
            "candidate row touched by that workload; results equal full-table row "
            "reads, but deployment-wide offline DPE storage/build cost is not measured."
        ),
    }
    destination = ROOT / "results" / "budgeted_lexical_dpe" / "FINAL_FULL_SUMMARY.json"
    atomic_json(destination, output)
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
