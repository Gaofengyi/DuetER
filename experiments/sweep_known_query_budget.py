"""Sweep the global known-query recovery budget for the CCADPE attack table."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from compartment_dpe import CompartmentDPEIndex
from run_compartment_security import global_known_query_recovery, primary_document_view
from run_experiment import normalize_rows
from semantic_candidate_index import KeyedResidualSphericalIVF


ROOT = Path(__file__).resolve().parent


def main() -> None:
    seed = 20260824
    documents = normalize_rows(
        np.load(
            ROOT
            / "cache"
            / "granite_gpu"
            / "scifact-st-a7972eb5022e_corpus_embeddings.npy"
        ).astype(np.float32)
    )
    routing = KeyedResidualSphericalIVF(
        b"DuetDPE-document-compartment-routing-v1",
        n_clusters=512,
        doc_assignments=2,
        nprobe=16,
        centered=True,
        seed=seed + 200,
    )
    routing.build(documents)
    index = CompartmentDPEIndex(
        key=b"DuetDPE-document-compartment-coordinate-v1",
        projection_dim=64,
        beta=0.10,
        scale=3.0,
        seed=seed + 201,
    )
    index.build(documents, routing)
    document_cells, document_cipher = primary_document_view(index, documents)
    queries = normalize_rows(
        np.concatenate(
            [
                np.load(
                    ROOT
                    / "cache"
                    / "granite_gpu"
                    / "scifact-st-a7972eb5022e_query_embeddings.npy"
                ).astype(np.float32),
                np.load(
                    ROOT
                    / "cache"
                    / "granite_gpu"
                    / "nfcorpus-st-a7972eb5022e_query_embeddings.npy"
                ).astype(np.float32),
            ],
            axis=0,
        )
    )
    rows = []
    for budget in (32, 64, 128, 256, 384, 512, 623):
        result = global_known_query_recovery(
            index,
            documents,
            document_cells,
            document_cipher,
            queries,
            known_queries=budget,
            document_sample=1000,
            repeats=10,
            seed=seed + 206,
        )
        result["excess_over_mean_baseline"] = (
            result["recovered_cosine_all_targets"]
            - result["known_query_mean_baseline_cosine"]
        )
        rows.append(result)
        print(
            budget,
            f"cos={result['recovered_cosine_all_targets']:.4f}",
            f"baseline={result['known_query_mean_baseline_cosine']:.4f}",
            f"excess={result['excess_over_mean_baseline']:+.4f}",
            f"coverage={result['target_compartment_coverage']:.4f}",
            flush=True,
        )
    output = ROOT / "results" / "known_query_budget_sweep.json"
    output.write_text(json.dumps({"results": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
