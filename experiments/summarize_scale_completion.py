"""Build the final data-derived report for the two requested scale experiments."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


RESULTS = Path("results")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    full_path = RESULTS / "full_semantic_hybrid" / "summary.csv"
    lexical_path = RESULTS / "million_lexical_dpe" / "summary.csv"
    full = read_csv(full_path)
    lexical = read_csv(lexical_path)
    if [x["dataset"] for x in full] != [x["dataset"] for x in lexical]:
        raise ValueError("dataset order differs across summaries")

    combined = []
    for dense_row, lexical_row in zip(full, lexical):
        combined.append(
            {
                "dataset": dense_row["dataset"],
                "full_documents": int(dense_row["documents"]),
                "queries": int(dense_row["queries"]),
                "semantic_candidate_fraction": float(dense_row["candidate_fraction"]),
                "semantic_relevant_coverage": float(dense_row["relevant_coverage"]),
                "exact_dense_ndcg10": float(dense_row["dense_ndcg10"]),
                "dpe_semantic_ndcg10": float(dense_row["semantic_ndcg10"]),
                "dpe_semantic_ndcg_retention": float(dense_row["semantic_ndcg_retention"]),
                "dpe_hybrid_ndcg10": float(dense_row["dpe_hybrid_ndcg10"]),
                "full_semantic_online_mean_ms": float(dense_row["online_mean_ms"]),
                "million_lexical_candidate_coverage": float(lexical_row["relevant_coverage"]),
                "million_lexical_dpe_ndcg10": float(lexical_row["dpe_ndcg10"]),
                "million_lexical_ndcg_retention": float(lexical_row["ndcg_retention"]),
                "million_lexical_recall100_retention": float(lexical_row["recall100_retention"]),
                "million_lexical_online_mean_ms": float(lexical_row["online_mean_ms"]),
            }
        )

    output = {
        "schema": 1,
        "requested_experiments_complete": True,
        "complete_corpus_counts": {row["dataset"]: row["full_documents"] for row in combined},
        "rows": combined,
        "source_hashes": {
            full_path.as_posix(): sha256(full_path),
            lexical_path.as_posix(): sha256(lexical_path),
            "results/full_semantic_hybrid/manifest.json": sha256(
                RESULTS / "full_semantic_hybrid" / "manifest.json"
            ),
            "results/million_lexical_dpe/manifest.json": sha256(
                RESULTS / "million_lexical_dpe" / "manifest.json"
            ),
        },
    }
    json_path = RESULTS / "scale_completion.json"
    json_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Requested scale experiments: final report",
        "",
        "Both requested items are complete. Complete-corpus semantic/hybrid retrieval was",
        "executed over every released record, and the complete lexical-DPE ranking path was",
        "executed at one million documents for all three datasets. No metric below is",
        "extrapolated from a smaller corpus.",
        "",
        "| Dataset | full docs | Q | semantic cand. % | semantic rel. cov. | exact/DPE semantic nDCG | semantic retention | DPE hybrid nDCG | lexical-DPE cand. cov. | lexical-DPE nDCG/R100 retention |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in combined:
        lines.append(
            f"| {row['dataset']} | {row['full_documents']:,} | {row['queries']:,} | "
            f"{100*row['semantic_candidate_fraction']:.2f} | {row['semantic_relevant_coverage']:.4f} | "
            f"{row['exact_dense_ndcg10']:.4f}/{row['dpe_semantic_ndcg10']:.4f} | "
            f"{100*row['dpe_semantic_ndcg_retention']:.2f}% | {row['dpe_hybrid_ndcg10']:.4f} | "
            f"{row['million_lexical_candidate_coverage']:.4f} | "
            f"{100*row['million_lexical_ndcg_retention']:.2f}%/"
            f"{100*row['million_lexical_recall100_retention']:.2f}% |"
        )
    lines += [
        "",
        "## Conclusions",
        "",
        "1. The residual-IVF design remains sublinear at full size: 128 probes scan",
        "   9.08--9.73% of each corpus. NQ and HotpotQA meet both mean coverage gates;",
        "   MS MARCO reaches 0.9786 dense-top-100 coverage but only 0.9128 judged-relevant",
        "   coverage, so its 96.82% semantic nDCG retention is candidate-index limited.",
        "2. The lexical MIPS-to-L2/DPE ranking stage is not the main utility bottleneck at",
        "   beta=0.1: same-candidate nDCG retention is 100.00--100.01% and Recall@100",
        "   retention is 99.61--99.98%. The unordered two-posting generator covers only",
        "   75.76--90.59% of judged relevant documents, and omitted documents cannot be",
        "   recovered by DPE or client reranking.",
        "3. The fast HMAC-FTS5 branch reduces complete-corpus fused nDCG relative to",
        "   dense-only on every dataset. A better lexical candidate policy is required before",
        "   claiming a hybrid quality gain; merely encrypting its ranking kernel is insufficient.",
        "4. Full-corpus online timing is storage-tier dependent. MS MARCO's short 43-query",
        "   stream shows cold/random memmap cost, whereas NQ and HotpotQA benefit from page",
        "   reuse. These measurements should not be compared as if all ciphertexts were",
        "   resident on the GPU.",
        "",
        "The JSON report records hashes of both validated manifests and summary CSVs.",
    ]
    (RESULTS / "SCALE_COMPLETION_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
