"""Assemble the completed revised-parameter reruns and final analysis."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if (ROOT / ".deps").exists():
    sys.path.insert(0, str(ROOT / ".deps"))
os.environ.setdefault("DUETER_FORCE_CPU_DEPS", "1")

import numpy as np

from benchmark_million_semantic_hybrid import atomic_json
from experiment_candidate_rank_fusion import run_dataset as run_duetrank
from rerun_revised_main import communication

SELECTED = {"nq": ("nq", 64), "hotpotqa": ("hotpotqa", 64), "msmarco": ("msmarco_dev", 64)}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    root = ROOT / "results" / "revised_d8192_rc32_b16"
    nq_main = load_json(root / "nq" / "results.json")
    ms_main = load_json(root / "msmarco_dev" / "results.json")

    # HotpotQA's original budget-preserving L=32 point failed the utility
    # criterion; promote the completed L=64 ablation point and retrain fusion.
    hot_rank_file = ROOT / "results" / "revised_d8192_rc32_b_sweep" / "hotpotqa" / "rankings.npz"
    hot_rankings = np.load(hot_rank_file)["lexical_dpe"][0:1]
    hot_selected = root / "hotpotqa" / "selected_l64_exact_bm25_rankings.npz"
    np.savez_compressed(hot_selected, lexical_dpe=hot_rankings, lexical=hot_rankings)
    hot_fusion = run_duetrank(
        "hotpotqa", lexical_source="revised_d8192_rc32_b16_l64", scale="full",
        lexical_rankings_path=hot_selected, split="test",
        semantic_rankings_path=ROOT / "results" / "dual_compartment_full_p256" / "hotpotqa" / "semantic_compartment.rankings.npz",
        trace_path=ROOT / "results" / "budgeted_lexical" / "hotpotqa_full.trace.json",
        budget_key=50_000, model_tag="d8192-r32-b16-l64",
    )

    b_reports = {
        dataset: load_json(ROOT / "results" / "revised_d8192_rc32_b_sweep" / label / "results.json")
        for dataset, (label, _) in SELECTED.items()
    }
    selected_rows = {dataset: report["configuration_results"][0] for dataset, report in b_reports.items()}
    old_reports = {
        "nq": load_json(ROOT / "results" / "exact_bm25_client_rerank" / "nq" / "results.json"),
        "hotpotqa": load_json(ROOT / "results" / "exact_bm25_client_rerank" / "hotpotqa" / "results.json"),
        "msmarco": load_json(ROOT / "results" / "exact_bm25_client_rerank" / "msmarco_dev" / "results.json"),
    }
    fusions = {"nq": nq_main["duetrank"], "hotpotqa": hot_fusion, "msmarco": ms_main["duetrank"]}

    # Exact HotpotQA payload accounting for the selected L=64 ranking.
    payload_root = ROOT / "cache" / "exact_bm25_client_rerank" / "hotpotqa_d8192_rc32_b_sweep_rerun_v3"
    payload_rows = np.load(payload_root / "rows.npy", mmap_mode="r")
    payload_lengths = np.load(payload_root / "token_lengths.npy", mmap_mode="r")
    hot_comm = communication(64, payload_lengths, payload_rows, hot_rankings)
    hot_total = (
        128 * (16 + 256 * 2) + 128 * 16 * 860
        + hot_comm["lexical_total_bytes"] + 48
    )
    hot_comm.update({"semantic_upload_bytes": 128 * (16 + 256 * 2), "semantic_padded_records": 2048, "combined_total_bytes": int(hot_total), "combined_total_mib": float(hot_total / 1024**2)})
    communications = {"nq": nq_main["communication"], "hotpotqa": hot_comm, "msmarco": ms_main["communication"]}

    semantic_labels = {"nq": "nq", "hotpotqa": "hotpotqa", "msmarco": "msmarco_dev"}
    batch_efficiency = {}
    for dataset, label in semantic_labels.items():
        old_dual = load_json(ROOT / "results" / "dual_compartment_full_p256" / label / "results.json")
        rq2 = old_dual["rq2_efficiency"]
        lexical_ms = float(selected_rows[dataset]["server_timing"]["mean_ms"])
        revised = float(rq2["lexical_candidate_lookup_mean_ms"]) + float(rq2["semantic"]["batch_server_ms_per_query_repeat"]) + lexical_ms
        batch_efficiency[dataset] = {
            "candidate_lookup_ms": rq2["lexical_candidate_lookup_mean_ms"],
            "semantic_server_ms": rq2["semantic"]["batch_server_ms_per_query_repeat"],
            "revised_lexical_refinement_ms": lexical_ms,
            "revised_server_online_ms": revised,
            "paper_server_online_ms": rq2["server_online_mean_ms_batched"],
            "relative_change": revised / float(rq2["server_online_mean_ms_batched"]) - 1.0,
        }

    storage = {}
    old_cache = {"nq": "nq_2681468", "hotpotqa": "hotpotqa_5233329", "msmarco": "msmarco_8841823_dev"}
    new_cache = {
        "nq": ROOT / "cache" / "lexical_d8192_rc32_b_sweep" / "nq_2681468_b16",
        "hotpotqa": ROOT / "cache" / "revised_d8192_rc32_b16" / "hotpotqa",
        "msmarco": ROOT / "cache" / "revised_d8192_rc32_b16" / "msmarco_dev",
    }
    for dataset in SELECTED:
        old_bytes = (ROOT / "cache" / "dual_compartment_full_p256" / old_cache[dataset] / "lexical_compartment_fp16.npy").stat().st_size
        new_bytes = (new_cache[dataset] / "lexical_compartment_fp16.npy").stat().st_size
        storage[dataset] = {"paper_bytes": old_bytes, "revised_bytes": new_bytes, "ratio": new_bytes / old_bytes}

    attacks = {}
    for dataset in SELECTED:
        report = load_json(ROOT / "results" / "security" / "revised_stitching_d8192_rc32_b16" / f"{dataset}.json")
        configuration = report["configuration"]
        attacks[dataset] = {
            "graph": configuration["graph"],
            "queries_16": next(row for row in configuration["query_sweep"] if row["queries"] == 16),
            "queries_512": next(row for row in configuration["query_sweep"] if row["queries"] == 512),
        }
    seed_sweep = load_json(ROOT / "results" / "security" / "selector_graph_seed_sweep_d8192_rc32" / "results.json")
    seed_summary = [{key: value for key, value in row.items() if key != "per_seed"} for row in seed_sweep["configurations"]]

    selected = {}
    for dataset in SELECTED:
        new_metrics = selected_rows[dataset]["exact_bm25_metrics"]
        old_metrics = old_reports[dataset]["lexical_metrics_exact_bm25"]
        old_fusion = old_reports[dataset]["duetrank"]
        selected[dataset] = {
            "parameters": {"d_l": 8192, "r_c_l": 32, "B": 16, "local_depth": 64},
            "paper_lexical_metrics": old_metrics,
            "revised_lexical_metrics": new_metrics,
            "retention": {key: new_metrics[key] / old_metrics[key] for key in new_metrics},
            "paper_duetrank_held_out": old_fusion["held_out"],
            "revised_duetrank_held_out": fusions[dataset]["held_out"],
            "communication": communications[dataset],
            "batch_efficiency": batch_efficiency[dataset],
            "storage": storage[dataset],
        }

    output = {
        "recommended_lexical_parameters": {"hash_dimension": 1024, "d_l": 8192, "r_c_l": 32, "B": 16, "local_depth": {"nq": 64, "hotpotqa": 64, "msmarco": 64}, "returned_depth": 300, "beta": 0.1, "scale": 3.0},
        "semantic_parameters": "unchanged from the paper (d_s=512, r_c,s=256, 2048 cells, nprobe=128)",
        "selected_results": selected,
        "stitching_attack": attacks,
        "selector_seed_sweep": seed_summary,
        "interpretation": {
            "utility": "L=64 is the smallest tested common depth retaining approximately all paper lexical quality on all three datasets; HotpotQA L=32 is not acceptable.",
            "security": "The parameters reduce overlap-graph connectivity but do not prevent synchronized-query correlation or recovery of every repeated coordinate.",
            "claim_limit": "Results support 'limits cross-compartment stitching scope', not 'breaks global geometry' or elimination of anchor alignment.",
        },
    }
    atomic_json(root / "final_summary.json", output)

    def pct(value: float) -> str:
        return f"{100.0 * value:.2f}%"

    lines = [
        "# Revised-parameter rerun summary", "",
        "Selected lexical setting: `d_l=8192`, `r_c,l=32`, `B=16`, and `L=64` for all three datasets. Semantic parameters are unchanged.", "",
        "## Retrieval and fusion", "",
        "| Dataset | lexical nDCG@10 old → new | lexical R@100 old → new | retained nDCG | DuetRank held-out nDCG old → new |", "|---|---:|---:|---:|---:|",
    ]
    for dataset in ("nq", "hotpotqa", "msmarco"):
        row = selected[dataset]
        old, new = row["paper_lexical_metrics"], row["revised_lexical_metrics"]
        old_d, new_d = row["paper_duetrank_held_out"], row["revised_duetrank_held_out"]
        lines.append(f"| {dataset} | {old['nDCG@10']:.4f} → {new['nDCG@10']:.4f} | {old['Recall@100']:.4f} → {new['Recall@100']:.4f} | {pct(row['retention']['nDCG@10'])} | {old_d['semantic_backoff_ndcg_at_10']:.4f} → {new_d['semantic_backoff_ndcg_at_10']:.4f} |")
    lines += ["", "## Stitching attack at 512 synchronized queries", "", "| Dataset | overlap P/R | distance Pearson | kNN R@10 | graph components | largest component |", "|---|---:|---:|---:|---:|---:|"]
    for dataset in ("nq", "hotpotqa", "msmarco"):
        attack = attacks[dataset]
        q = attack["queries_512"]
        graph = attack["graph"]
        lines.append(f"| {dataset} | {q['precision']:.3f}/{q['recall']:.3f} | {q['distance_pearson']:.4f} | {q['knn_recall_at_10']:.4f} | {graph['connected_components']} | {graph['largest_component_cells']}/16 |")
    lines += ["", "## Main conclusion", "", "The rerun rejects the strong statement that the construction *breaks* global geometric coherence. The revised parameters materially fragment the stitching graph (B=16 is fully connected in only 4% of 100 selector seeds), but shared coordinates remain perfectly identifiable once enough synchronized queries are observed. The defensible statement is that the configuration **reduces the coverage of a stitched frame**.", ""]
    (root / "ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
