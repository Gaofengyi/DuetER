"""Validate and summarize the three executed million-document lexical-DPE runs."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path("results/million_lexical_dpe")
DATASETS = ("msmarco", "nq", "hotpotqa")
DISPLAY = {"msmarco": "MS MARCO", "nq": "NQ", "hotpotqa": "HotpotQA"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_rankings(path: Path, queries: int, documents: int, repeats: int) -> dict[str, list[int]]:
    expected = {
        "plaintext_sketch_reference": (queries, 100),
        "fts5_two_probe_reference": (queries, 100),
        "lexical_dpe": (repeats, queries, 100),
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
        directory = ROOT / f"{dataset}_1000000_h1024"
        result_path = directory / "results.json"
        ranking_path = directory / "rankings.npz"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        documents = int(result["documents"])
        queries = int(result["queries"])
        repeats = int(result["configuration"]["repeats"])
        if documents != 1_000_000 or repeats != 5:
            raise ValueError(f"{dataset}: expected 1,000,000 documents and five repeats")
        shapes = validate_rankings(ranking_path, queries, documents, repeats)
        latency_mean = [float(x["online_mean_ms"]) for x in result["repeats"]]
        latency_p95 = [float(x["online_p95_ms"]) for x in result["repeats"]]
        plain = result["reference_plaintext_sketch_same_candidates"]
        dpe = result["lexical_dpe_mean"]
        rows.append(
            {
                "dataset": DISPLAY[dataset],
                "documents": documents,
                "queries": queries,
                "candidate_mean": float(result["candidate_generation"]["candidate_mean"]),
                "candidate_p95": float(result["candidate_generation"]["candidate_p95"]),
                "relevant_coverage": float(result["candidate_generation"]["relevant_document_coverage_mean"]),
                "plaintext_ndcg10": float(plain["nDCG@10"]),
                "plaintext_recall100": float(plain["Recall@100"]),
                "dpe_ndcg10": float(dpe["nDCG@10"]),
                "dpe_recall100": float(dpe["Recall@100"]),
                "ndcg_retention": float(result["retention"]["ndcg"]),
                "recall100_retention": float(result["retention"]["recall100"]),
                "online_mean_ms": float(np.mean(latency_mean)),
                "online_p95_ms": float(np.mean(latency_p95)),
                "sketch_build_seconds": float(result["sketch_setup"]["build_seconds_last_invocation"]),
                "dpe_build_seconds": float(result["dpe_setup"]["build_seconds_last_invocation"]),
                "sketch_gib": float(result["sketch_setup"]["bytes"]) / 2**30,
                "ciphertext_gib": float(result["dpe_setup"]["ciphertext_bytes"]) / 2**30,
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
        "# One-million-document lexical DPE",
        "",
        "These are measured executions over three relevance-preserving million-document subsets.",
        "The server candidate stage is an unordered union of two rare HMAC postings (not FTS5",
        "BM25 ordering), followed by stored 2048-dimensional conditional-DPE ciphertext L2",
        "top-200 and exact client BM25 top-100 reranking. The primary plaintext reference uses",
        "the identical candidates and signed-hash MIPS sketch; five DPE query-noise repeats are run.",
        "",
        "| Dataset | Q | candidates mean/p95 | rel. coverage | plaintext nDCG/R100 | DPE nDCG/R100 | retention nDCG/R100 | online mean/p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['queries']:,} | {row['candidate_mean']:.1f}/{row['candidate_p95']:.1f} | "
            f"{row['relevant_coverage']:.4f} | {row['plaintext_ndcg10']:.4f}/{row['plaintext_recall100']:.4f} | "
            f"{row['dpe_ndcg10']:.4f}/{row['dpe_recall100']:.4f} | "
            f"{100*row['ndcg_retention']:.2f}%/{100*row['recall100_retention']:.2f}% | "
            f"{row['online_mean_ms']:.2f}/{row['online_p95_ms']:.2f} |"
        )
    lines += [
        "",
        "The experiment demonstrates that the conditional-DPE L2 stage itself preserves the",
        "same-candidate lexical ranking almost exactly at this noise setting. End-to-end recall",
        "remains bounded by the deliberately lightweight two-posting candidate generator; DPE",
        "retention must therefore not be interpreted as global BM25-order preservation.",
        "The bounded-noise prototype uses NumPy PCG64 and is not a production PRF implementation.",
    ]
    (ROOT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest_path = ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": rows, "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
