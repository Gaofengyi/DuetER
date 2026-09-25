"""Measure known-query recovery against the executed semantic DPE prototype.

The attacker observes known plaintext query embeddings X and their DPE query
coordinates C, estimates the effective padded transform with least squares,
and applies its pseudoinverse to static encrypted document coordinates.  This
is an auxiliary-information stress test, not a cryptographic security proof.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_experiment import ConditionalDPE, normalize_rows


ROOT = Path(__file__).resolve().parent


def cosine_summary(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    reference_n = normalize_rows(reference)
    estimate_n = normalize_rows(estimate)
    cosine = np.sum(reference_n * estimate_n, axis=1)
    l2 = np.linalg.norm(reference - estimate, axis=1)
    return {
        "mean_cosine": float(np.mean(cosine)),
        "median_cosine": float(np.median(cosine)),
        "p05_cosine": float(np.quantile(cosine, 0.05)),
        "mean_l2": float(np.mean(l2)),
        "p95_l2": float(np.quantile(l2, 0.95)),
    }


def encrypt_known_queries(
    dpe: ConditionalDPE,
    queries: np.ndarray,
    repeats: int,
    dp_sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    observations = []
    for _ in range(repeats):
        released = queries
        if dp_sigma > 0:
            released = normalize_rows(
                queries + rng.normal(0.0, dp_sigma, queries.shape).astype(np.float32)
            )
        observations.append(dpe.encrypt_queries(released).astype(np.float64))
    return np.mean(observations, axis=0)


def run(args: argparse.Namespace) -> dict[str, object]:
    query_files = [ROOT / path for path in args.query_files]
    document_file = ROOT / args.document_file
    query_blocks = [np.load(path).astype(np.float32) for path in query_files]
    queries = normalize_rows(np.concatenate(query_blocks, axis=0))
    documents = normalize_rows(np.load(document_file).astype(np.float32))

    rng = np.random.default_rng(args.seed)
    queries = queries[rng.permutation(len(queries))]
    if args.document_sample < len(documents):
        documents = documents[rng.choice(len(documents), args.document_sample, replace=False)]

    dpe = ConditionalDPE(queries.shape[1], args.beta, args.scale, args.seed + 1)
    identity = np.eye(queries.shape[1], dtype=np.float32)
    true_map = args.scale * dpe.transform(identity).astype(np.float64)
    encrypted_documents = dpe.encrypt_database(documents).astype(np.float64)

    rows: list[dict[str, object]] = []
    for dp_sigma in args.dp_sigmas:
        for repeats in args.repeats:
            all_observations = encrypt_known_queries(
                dpe, queries, repeats, dp_sigma, rng
            )
            for requested_m in args.known_counts:
                m = min(requested_m, len(queries))
                known = queries[:m].astype(np.float64)
                observed = all_observations[:m]
                singular = np.linalg.svd(known, compute_uv=False)
                rank = int(np.linalg.matrix_rank(known))
                least_squares_map, _, _, _ = np.linalg.lstsq(known, observed, rcond=None)
                # The executed map has orthogonal rows with public experiment
                # scale s.  Project the unconstrained estimate onto that model
                # before attempting inversion (rectangular Procrustes/polar step).
                left, _, right_t = np.linalg.svd(least_squares_map, full_matrices=False)
                estimate_map = args.scale * (left @ right_t)
                relative_map_error = float(
                    np.linalg.norm(estimate_map - true_map, ord="fro")
                    / np.linalg.norm(true_map, ord="fro")
                )
                identifiable = rank == known.shape[1]
                recovery = None
                if identifiable:
                    recovered_documents = (
                        encrypted_documents @ estimate_map.T / (args.scale**2)
                    )
                    recovery = cosine_summary(
                        documents.astype(np.float64), recovered_documents
                    )
                row = {
                    "known_queries": m,
                    "rank": rank,
                    "input_dimension": int(known.shape[1]),
                    "smallest_reported_singular_value": float(singular[-1]),
                    "condition_number_nonzero_subspace": float(singular[0] / singular[-1]),
                    "query_repeats": repeats,
                    "dp_sigma": dp_sigma,
                    "relative_effective_transform_error": relative_map_error,
                    "full_effective_map_identifiable": identifiable,
                    "document_recovery": recovery,
                }
                rows.append(row)
                cosine_text = "n/a" if recovery is None else f"{recovery['mean_cosine']:.4f}"
                print(
                    f"sigma={dp_sigma:.3f} repeats={repeats:2d} m={m:3d} "
                    f"rank={rank:3d} map={relative_map_error:.4f} "
                    f"cos={cosine_text}"
                )

    return {
        "purpose": "Known-query least-squares recovery stress test",
        "configuration": {
            "query_files": [str(path.relative_to(ROOT)) for path in query_files],
            "document_file": str(document_file.relative_to(ROOT)),
            "available_known_queries": len(queries),
            "document_sample": len(documents),
            "input_dimension": queries.shape[1],
            "ciphertext_work_dimension": dpe.work_dim,
            "beta": args.beta,
            "scale": args.scale,
            "seed": args.seed,
        },
        "caveat": (
            "The test grants exact query embeddings and averages independent "
            "releases when repeats > 1; it estimates the effective transform, "
            "not the secret key representation."
        ),
        "results": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query-files",
        nargs="+",
        default=[
            "cache/granite_gpu/scifact-st-a7972eb5022e_query_embeddings.npy",
            "cache/granite_gpu/nfcorpus-st-a7972eb5022e_query_embeddings.npy",
        ],
    )
    parser.add_argument(
        "--document-file",
        default="cache/granite_gpu/scifact-st-a7972eb5022e_corpus_embeddings.npy",
    )
    parser.add_argument("--known-counts", nargs="+", type=int, default=[64, 128, 256, 384, 512, 623])
    parser.add_argument("--repeats", nargs="+", type=int, default=[1, 4, 16])
    parser.add_argument("--dp-sigmas", nargs="+", type=float, default=[0.0, 0.015])
    parser.add_argument("--document-sample", type=int, default=1000)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "known_pair_recovery.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    result = run(parsed)
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    parsed.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {parsed.output}")
