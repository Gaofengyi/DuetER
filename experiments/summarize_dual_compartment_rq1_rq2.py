"""Consolidate the audited dual-path Compartment-DPE RQ1/RQ2 runs.

NQ and HotpotQA use their complete test workloads.  MS MARCO uses the 6,980
query dev workload for fusion and latency because its public test qrels contain
only 43 evaluated queries.  The script only reads experiment artifacts and
writes a compact machine-readable and human-readable audit bundle.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULT_ROOT = ROOT / "results" / "dual_compartment_full_p256"
RUNS = {
    "NQ": RESULT_ROOT / "nq" / "results.json",
    "HotpotQA": RESULT_ROOT / "hotpotqa" / "results.json",
    "MS MARCO": RESULT_ROOT / "msmarco_dev" / "results.json",
}


def load_row(label: str, path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    utility = payload["rq1_utility"]
    efficiency = payload["rq2_efficiency"]
    semantic = utility["selected_semantic"]
    lexical = utility["selected_lexical"]
    held_out = utility["duetrank"]["held_out"]
    gain = held_out["backoff_vs_semantic"]
    matched = utility["matched_budget_recall"]
    return {
        "dataset": label,
        "documents": payload["documents"],
        "queries": payload["queries"],
        "split": payload["configuration"].get("split", "test"),
        "projection_dimension": payload["configuration"]["projection_dimension"],
        "semantic_reference": payload["configuration"].get(
            "semantic_retention_reference", "exact dense"
        ),
        "semantic_top_per_cell": utility["selected_semantic_top_per_cell"],
        "lexical_top_per_cell": utility["selected_lexical_top_per_cell"],
        "semantic_ndcg10": semantic["metrics"]["nDCG@10"],
        "semantic_ndcg10_retention": semantic["ndcg_retention_vs_exact_dense"],
        "semantic_recall100": semantic["metrics"]["Recall@100"],
        "semantic_recall100_retention": semantic[
            "recall100_retention_vs_exact_dense"
        ],
        "lexical_ndcg10": lexical["metrics"]["nDCG@10"],
        "lexical_ndcg10_retention": lexical["ndcg_retention_vs_budget_bm25"],
        "lexical_recall100": lexical["metrics"]["Recall@100"],
        "lexical_recall100_retention": lexical[
            "recall100_retention_vs_budget_bm25"
        ],
        "heldout_semantic_ndcg10": held_out["semantic_ndcg_at_10"],
        "heldout_dual_ndcg10": held_out["semantic_backoff_ndcg_at_10"],
        "heldout_dual_gain": gain["mean_absolute_gain"],
        "heldout_dual_gain_ci95_low": gain["ci95_low"],
        "heldout_dual_gain_ci95_high": gain["ci95_high"],
        "semantic_recall20": matched["semantic_recall_at_20"],
        "dual_recall20": matched["duetrank_recall_at_20"],
        "top10_path_union_recall": matched["top10_path_union_recall"],
        "dual_recall_at_union_cardinality": matched[
            "duetrank_recall_at_union_cardinality"
        ],
        "server_batched_ms": efficiency["server_online_mean_ms_batched"],
        "server_interactive_ms": efficiency[
            "server_online_mean_ms_interactive_sample"
        ],
        "communication_mib": efficiency["communication"]["total_mib"],
        "estimated_storage_gib": payload["setup"][
            "estimated_full_server_storage_gib"
        ],
        "lexical_lookup_ms": efficiency["lexical_candidate_lookup_mean_ms"],
        "semantic_client_rerank_ms": efficiency["semantic"][
            "client_rerank_ms_per_query_repeat"
        ],
        "duetrank_client_ms": utility["duetrank"]["selection"][
            "client_inference_mean_ms_per_query"
        ],
        "source": str(path),
    }


def fmt(value: object, digits: int = 5) -> str:
    return f"{float(value):.{digits}f}"


def main() -> None:
    rows = [load_row(label, path) for label, path in RUNS.items()]
    output = {
        "experiment": "full-corpus dual-path Compartment-DPE RQ1/RQ2 audit",
        "operating_point": {
            "projection_dimension": 256,
            "semantic_cells": 2048,
            "semantic_probes": 128,
            "lexical_cells": 64,
            "selection_gate": "smallest local depth with >=98% nDCG@10 and >=95% Recall@100 retention; best-utility fallback if no point satisfies both",
        },
        "scope_notes": [
            "NQ and HotpotQA use complete test workloads; MS MARCO uses all 6,980 dev queries because only 43 test queries have public judgments.",
            "The MS MARCO semantic retention denominator is the audited pre-compartment global-DPE IVF output, not an infeasible 8.84M-by-6,980 exact full scan.",
            "Semantic compartment coordinates are fully materialized. Lexical online execution materializes every record touched by the complete query workload; full-deployment lexical storage is estimated at one local ciphertext per corpus record.",
            "Batched server time measures cell-major throughput. Interactive server time is a query-major five-query sample and should not be conflated with batched throughput.",
            "NQ and HotpotQA path metrics average three fixed-seed executions; the full MS MARCO dev run uses one deterministic seed, while confidence intervals bootstrap held-out queries.",
        ],
        "datasets": rows,
    }
    (RESULT_ROOT / "summary.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    with (RESULT_ROOT / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Dual-path Compartment-DPE: RQ1/RQ2 audit",
        "",
        "## RQ1 utility",
        "",
        "| Data | Split/Q | Local depth S/L | S nDCG ret. | L nDCG ret. | S→Dual nDCG@10 | Dual R@20 | Top-10 union R |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['split']}/{row['queries']:,} | "
            f"{row['semantic_top_per_cell']}/{row['lexical_top_per_cell']} | "
            f"{fmt(row['semantic_ndcg10_retention'], 4)} | "
            f"{fmt(row['lexical_ndcg10_retention'], 4)} | "
            f"{fmt(row['heldout_semantic_ndcg10'])}→{fmt(row['heldout_dual_ndcg10'])} | "
            f"{fmt(row['dual_recall20'])} | {fmt(row['top10_path_union_recall'])} |"
        )
    lines.extend(
        [
            "",
            "## RQ2 cost",
            "",
            "| Data | Batched server ms | Interactive server ms | Comm. MiB | Est. storage GiB | Client rerank+fusion ms |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        client = float(row["semantic_client_rerank_ms"]) + float(
            row["duetrank_client_ms"]
        )
        lines.append(
            f"| {row['dataset']} | {fmt(row['server_batched_ms'], 2)} | "
            f"{fmt(row['server_interactive_ms'], 2)} | "
            f"{fmt(row['communication_mib'], 2)} | "
            f"{fmt(row['estimated_storage_gib'], 2)} | {fmt(client, 2)} |"
        )
    lines.extend(["", "## Scope", ""])
    lines.extend(f"- {note}" for note in output["scope_notes"])
    (RESULT_ROOT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
