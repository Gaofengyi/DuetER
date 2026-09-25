"""Validate and summarize executed complete-corpus semantic/hybrid runs."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path("results/full_semantic_hybrid")
DATASETS = ("msmarco", "nq", "hotpotqa")
COUNTS = {"msmarco": 8_841_823, "nq": 2_681_468, "hotpotqa": 5_233_329}
DISPLAY = {"msmarco": "MS MARCO", "nq": "NQ", "hotpotqa": "HotpotQA"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_rankings(path: Path, queries: int, documents: int, repeats: int) -> dict[str, list[int]]:
    expected = {
        "exact_dense": (queries, 100),
        "lexical": (queries, 200),
        "exact_hybrid": (queries, 100),
        "semantic": (repeats, queries, 100),
        "hybrid": (repeats, queries, 100),
    }
    shapes: dict[str, list[int]] = {}
    with np.load(path) as archive:
        if set(archive.files) != set(expected):
            raise ValueError(f"{path}: unexpected arrays {archive.files}")
        for name, shape in expected.items():
            array = archive[name]
            if array.shape != shape or array.dtype != np.int32:
                raise ValueError(f"{path}:{name} is {array.shape}/{array.dtype}, expected {shape}/int32")
            if array.size and (int(array.min()) < -1 or int(array.max()) >= documents):
                raise ValueError(f"{path}:{name} contains an invalid document row")
            flat = array.reshape(-1, array.shape[-1])
            if np.any((flat[:, :-1] == -1) & (flat[:, 1:] != -1)):
                raise ValueError(f"{path}:{name} contains non-trailing padding")
            shapes[name] = list(shape)
    return shapes


def main() -> None:
    rows: list[dict[str, object]] = []
    manifest: dict[str, object] = {"schema": 1, "files": {}, "ranking_shapes": {}}
    for dataset in DATASETS:
        directory = ROOT / f"{dataset}_{COUNTS[dataset]}"
        result_path = directory / "results.json"
        ranking_path = directory / "rankings.npz"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        documents = int(result["documents"])
        queries = int(result["queries"])
        repeats = int(result["configuration"]["repeats"])
        if documents != COUNTS[dataset] or repeats != 5:
            raise ValueError(f"{dataset}: wrong corpus size or repeat count")
        if result["subset_policy"] != "complete corpus in released order":
            raise ValueError(f"{dataset}: result is not labeled as the complete corpus")
        shapes = validate_rankings(ranking_path, queries, documents, repeats)
        selected = int(result["configuration"]["selected_probes"])
        sweep = next(x for x in result["probe_sweep"] if int(x["nprobe"]) == selected)
        lat_mean = [float(x["semantic_online"]["online_mean_ms"]) for x in result["repeats"]]
        lat_p95 = [float(x["semantic_online"]["online_p95_ms"]) for x in result["repeats"]]
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
                "candidate_mean": float(sweep["candidate_mean"]),
                "candidate_fraction": float(sweep["candidate_fraction_mean"]),
                "dense_top100_coverage": float(sweep["dense_top100_coverage_mean"]),
                "relevant_coverage": float(sweep["relevant_document_coverage_mean"]),
                "dense_ndcg10": float(dense["nDCG@10"]),
                "dense_recall100": float(dense["Recall@100"]),
                "lexical_ndcg10": float(lexical["nDCG@10"]),
                "lexical_recall100": float(lexical["Recall@100"]),
                "semantic_ndcg10": float(semantic["nDCG@10"]),
                "semantic_recall100": float(semantic["Recall@100"]),
                "semantic_ndcg_retention": float(result["retention"]["semantic_ndcg_vs_dense"]),
                "semantic_recall100_retention": float(result["retention"]["semantic_recall100_vs_dense"]),
                "exact_hybrid_ndcg10": float(exact_hybrid["nDCG@10"]),
                "exact_hybrid_recall100": float(exact_hybrid["Recall@100"]),
                "dpe_hybrid_ndcg10": float(hybrid["nDCG@10"]),
                "dpe_hybrid_recall100": float(hybrid["Recall@100"]),
                "hybrid_ndcg_retention": float(result["retention"]["hybrid_ndcg_vs_exact_semantic_same_lexical"]),
                "hybrid_recall100_retention": float(result["retention"]["hybrid_recall100_vs_exact_semantic_same_lexical"]),
                "online_mean_ms": float(np.mean(lat_mean)),
                "online_p95_ms": float(np.mean(lat_p95)),
                "encoding_seconds": float(result["encoding"]["corpus_encoding_seconds"]),
                "index_build_seconds": float(result["semantic_index_setup"]["build_seconds"]),
                "dpe_build_seconds": float(result["semantic_dpe_setup"]["build_seconds"]),
                "exact_dense_seconds": float(result["exact_dense_seconds"]),
            }
        )
        manifest["files"][result_path.as_posix()] = sha256(result_path)
        manifest["files"][ranking_path.as_posix()] = sha256(ranking_path)
        manifest["ranking_shapes"][dataset] = shapes

    csv_path = ROOT / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Complete-corpus semantic and hybrid retrieval",
        "",
        "Every row is an actual execution over the complete BEIR corpus in original JSONL",
        "order: 8,841,823 MS MARCO passages, 2,681,468 NQ documents, and 5,233,329",
        "HotpotQA documents. Exact dense top-100 is streamed over every vector on the GPU.",
        "The semantic path uses HMAC-labeled residual IVF, stored conditional-DPE",
        "coordinates, server top-200, and exact client reranking. Hybrid uses the independently",
        "built full-corpus HMAC-FTS5 two-probe lexical branch and client RRF.",
        "",
        "| Dataset | documents | Q | probes | candidate % | dense/rel. coverage | exact dense nDCG/R100 | DPE semantic nDCG/R100 | retention nDCG/R100 | lexical nDCG/R100 | exact hybrid nDCG/R100 | DPE hybrid nDCG/R100 | online mean/p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['documents']:,} | {row['queries']:,} | {row['selected_probes']} | "
            f"{100*row['candidate_fraction']:.2f} | {row['dense_top100_coverage']:.4f}/{row['relevant_coverage']:.4f} | "
            f"{row['dense_ndcg10']:.4f}/{row['dense_recall100']:.4f} | "
            f"{row['semantic_ndcg10']:.4f}/{row['semantic_recall100']:.4f} | "
            f"{100*row['semantic_ndcg_retention']:.2f}%/{100*row['semantic_recall100_retention']:.2f}% | "
            f"{row['lexical_ndcg10']:.4f}/{row['lexical_recall100']:.4f} | "
            f"{row['exact_hybrid_ndcg10']:.4f}/{row['exact_hybrid_recall100']:.4f} | "
            f"{row['dpe_hybrid_ndcg10']:.4f}/{row['dpe_hybrid_recall100']:.4f} | "
            f"{row['online_mean_ms']:.2f}/{row['online_p95_ms']:.2f} |"
        )
    lines += [
        "",
        "The latency is an out-of-core storage-tier measurement: each query materializes its",
        "immutable candidate ciphertext rows once, and that cost is divided equally among five",
        "independent DPE query-noise scores. It must not be compared directly with the earlier",
        "one-million-vector GPU-resident timing without this qualification.",
        "NumPy PCG64 models bounded DPE noise; it is not a production PRF.",
    ]
    (ROOT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest_path = ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": rows, "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
