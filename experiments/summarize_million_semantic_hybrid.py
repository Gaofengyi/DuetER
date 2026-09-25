"""Validate and summarize the completed one-million-document experiments."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path("results/million_semantic_hybrid")
DATASETS = ("msmarco", "nq", "hotpotqa")
DISPLAY = {"msmarco": "MS MARCO", "nq": "NQ", "hotpotqa": "HotpotQA"}
EXPECTED_DEPTHS = {
    "exact_dense": 100,
    "ivf_exact": 100,
    "lexical": 200,
    "exact_hybrid": 100,
    "ivf_exact_hybrid": 100,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric(metrics: dict[str, float], name: str) -> float:
    return float(metrics[name])


def validate_rankings(path: Path, queries: int, repeats: int, documents: int) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    with np.load(path) as archive:
        expected_keys = set(EXPECTED_DEPTHS) | {"semantic", "hybrid"}
        if set(archive.files) != expected_keys:
            raise ValueError(f"{path}: unexpected arrays {archive.files}")
        for name, depth in EXPECTED_DEPTHS.items():
            array = archive[name]
            expected = (queries, depth)
            if array.shape != expected:
                raise ValueError(f"{path}:{name} has {array.shape}, expected {expected}")
            shapes[name] = list(array.shape)
        for name in ("semantic", "hybrid"):
            array = archive[name]
            expected = (repeats, queries, 100)
            if array.shape != expected:
                raise ValueError(f"{path}:{name} has {array.shape}, expected {expected}")
            shapes[name] = list(array.shape)
        for name in archive.files:
            array = archive[name]
            if array.dtype != np.int32:
                raise ValueError(f"{path}:{name} dtype {array.dtype}, expected int32")
            if array.size and (int(array.min()) < -1 or int(array.max()) >= documents):
                raise ValueError(f"{path}:{name} contains an out-of-range row index")
            # A lexical query can have fewer than the fixed 200 slots.  In that
            # case -1 is the sole permitted sentinel and it must form a suffix.
            flat = array.reshape(-1, array.shape[-1])
            if np.any((flat[:, :-1] == -1) & (flat[:, 1:] != -1)):
                raise ValueError(f"{path}:{name} contains non-trailing padding")
    return shapes


def main() -> None:
    rows: list[dict[str, object]] = []
    manifest: dict[str, object] = {"schema": 1, "files": {}, "ranking_shapes": {}}
    for dataset in DATASETS:
        directory = ROOT / f"{dataset}_1000000"
        result_path = directory / "results.json"
        ranking_path = directory / "rankings.npz"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        documents = int(result["documents"])
        queries = int(result["queries"])
        repeats = int(result["configuration"]["repeats"])
        if documents != 1_000_000 or repeats != 5:
            raise ValueError(f"{dataset}: expected 1,000,000 documents and five repeats")
        shapes = validate_rankings(ranking_path, queries, repeats, documents)
        selected = int(result["configuration"]["selected_probes"])
        selected_sweep = next(x for x in result["probe_sweep"] if int(x["nprobe"]) == selected)
        online_means = [float(x["semantic_online"]["online_mean_ms"]) for x in result["repeats"]]
        online_p95s = [float(x["semantic_online"]["online_p95_ms"]) for x in result["repeats"]]
        dense = result["reference_metrics"]["dense_exact"]
        lexical = result["reference_metrics"]["keyed_lexical_p2"]
        exact_hybrid = result["reference_metrics"]["hybrid_dense_exact_plus_keyed_lexical_p2"]
        semantic = result["semantic_dpe_mean"]
        hybrid = result["hybrid_dpe_plus_keyed_lexical_mean"]
        rows.append(
            {
                "dataset": DISPLAY[dataset],
                "documents": documents,
                "queries": queries,
                "selected_probes": selected,
                "candidate_mean": float(selected_sweep["candidate_mean"]),
                "candidate_fraction": float(selected_sweep["candidate_fraction_mean"]),
                "dense_top100_coverage": float(selected_sweep["dense_top100_coverage_mean"]),
                "relevant_coverage": float(selected_sweep["relevant_document_coverage_mean"]),
                "index_lookup_mean_ms": float(selected_sweep["lookup_mean_ms"]),
                "semantic_online_mean_ms": float(np.mean(online_means)),
                "semantic_online_p95_ms": float(np.mean(online_p95s)),
                "dense_exact_ndcg10": metric(dense, "nDCG@10"),
                "dense_exact_recall100": metric(dense, "Recall@100"),
                "semantic_dpe_ndcg10": metric(semantic, "nDCG@10"),
                "semantic_dpe_recall100": metric(semantic, "Recall@100"),
                "semantic_ndcg_retention": float(result["retention"]["semantic_ndcg_vs_dense"]),
                "semantic_recall100_retention": float(result["retention"]["semantic_recall100_vs_dense"]),
                "lexical_ndcg10": metric(lexical, "nDCG@10"),
                "exact_hybrid_ndcg10": metric(exact_hybrid, "nDCG@10"),
                "exact_hybrid_recall100": metric(exact_hybrid, "Recall@100"),
                "dpe_hybrid_ndcg10": metric(hybrid, "nDCG@10"),
                "dpe_hybrid_recall100": metric(hybrid, "Recall@100"),
                "hybrid_ndcg_retention": float(result["retention"]["hybrid_ndcg_vs_exact_semantic_same_lexical"]),
                "hybrid_recall100_retention": float(result["retention"]["hybrid_recall100_vs_exact_semantic_same_lexical"]),
                "corpus_encoding_seconds": float(result["encoding"]["corpus_encoding_seconds"]),
                "semantic_index_build_seconds": float(result["semantic_index_setup"]["build_seconds"]),
                "dpe_database_build_seconds": float(result["semantic_dpe_setup"]["build_seconds"]),
                "semantic_index_server_mib": float(result["semantic_index_setup"]["server_estimated_bytes"]) / 2**20,
                "dpe_ciphertext_mib": float(result["semantic_dpe_setup"]["ciphertext_bytes"]) / 2**20,
            }
        )
        manifest["files"][str(result_path.as_posix())] = sha256(result_path)
        manifest["files"][str(ranking_path.as_posix())] = sha256(ranking_path)
        manifest["ranking_shapes"][dataset] = shapes

    ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = ROOT / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report_lines = [
        "# One-million-document semantic and hybrid retrieval",
        "",
        "All rows are actual RTX 4060/Windows executions over declared relevance-preserving",
        "one-million-document BEIR subsets. They are scale stress tests, not canonical",
        "full-corpus effectiveness runs. The semantic path uses the Granite encoder,",
        "HMAC-labeled two-assignment residual IVF, stored conditional-DPE coordinates,",
        "server top-200 refinement, and exact client top-100 reranking. The lexical path",
        "is the disk HMAC-FTS5 exact-BM25 candidate implementation, not lexical DPE.",
        "Hybrid results use client-side RRF.",
        "",
        "## Main results",
        "",
        "| Dataset | Q | probes | candidate % | dense cov. | rel. cov. | exact dense nDCG/R100 | DPE semantic nDCG/R100 | semantic retention | exact hybrid nDCG/R100 | DPE hybrid nDCG/R100 | hybrid retention | online mean/p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report_lines.append(
            f"| {row['dataset']} | {row['queries']:,} | {row['selected_probes']} | "
            f"{100*row['candidate_fraction']:.2f} | {row['dense_top100_coverage']:.4f} | "
            f"{row['relevant_coverage']:.4f} | {row['dense_exact_ndcg10']:.4f}/{row['dense_exact_recall100']:.4f} | "
            f"{row['semantic_dpe_ndcg10']:.4f}/{row['semantic_dpe_recall100']:.4f} | "
            f"{100*row['semantic_ndcg_retention']:.2f}%/{100*row['semantic_recall100_retention']:.2f}% | "
            f"{row['exact_hybrid_ndcg10']:.4f}/{row['exact_hybrid_recall100']:.4f} | "
            f"{row['dpe_hybrid_ndcg10']:.4f}/{row['dpe_hybrid_recall100']:.4f} | "
            f"{100*row['hybrid_ndcg_retention']:.2f}%/{100*row['hybrid_recall100_retention']:.2f}% | "
            f"{row['semantic_online_mean_ms']:.2f}/{row['semantic_online_p95_ms']:.2f} |"
        )
    report_lines += [
        "",
        "## Interpretation",
        "",
        "The fixed sweep selected 128 probes for every dataset. NQ and HotpotQA satisfy",
        "both 0.95 candidate-coverage gates. MS MARCO reaches 0.9705 exact-dense top-100",
        "coverage but only 0.9201 judged-relevant coverage, so it uses the prespecified",
        "maximum rather than claiming the target was met. Across datasets the server",
        "examines 9.18%--9.91% of the million-vector corpus, and semantic nDCG retention",
        "is 97.25%--99.15%. DPE and exact candidate refinement produce effectively the",
        "same utility here; the observed loss is dominated by candidate generation.",
        "Hybrid fusion is not uniformly better than dense retrieval on these subsets,",
        "because the fast two-term lexical branch has low recall and RRF can demote strong",
        "dense results. Its value here is evaluating the executed dual-path architecture,",
        "not demonstrating universal effectiveness gains.",
        "",
        "## Reproducibility",
        "",
        "`manifest.json` contains SHA-256 hashes and validated ranking-array shapes.",
        "Each `results.json` records the model, subset policy, GPU/software environment,",
        "build costs, probe sweep, five repeats, and the cryptographic-scope warning.",
    ]
    (ROOT / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    manifest_path = ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": rows, "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
