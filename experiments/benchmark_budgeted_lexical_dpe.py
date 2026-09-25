"""Run stored lexical DPE over the redesigned budget-aware candidate sets."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from pathlib import Path

import numpy as np

from benchmark_budgeted_lexical_candidates import ROOT, fuse_all, load_semantic_rankings
from benchmark_million_lexical_dpe import (
    build_ciphertexts,
    build_plain_sketches,
    document_vector,
    encrypt_query_matrix,
    exact_bm25_for_rows,
    query_mips,
)
from benchmark_million_semantic_hybrid import (
    atomic_json,
    ball_noise,
    dpe_transform,
    evaluate,
    load_queries_qrels,
    read_ids,
)
from experiment_candidate_rank_fusion import split_mask
from dueter_common import tokenize


FULL_DOCUMENTS = {
    "msmarco": 8_841_823,
    "nq": 2_681_468,
    "hotpotqa": 5_233_329,
}


def _candidate_digest(rows: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(rows, dtype=np.int32).tobytes()).hexdigest()


def build_candidate_scoped_cache(
    *,
    data_dir: Path,
    cache_dir: Path,
    candidates: np.ndarray,
    documents: int,
    document_frequency: dict[str, int],
    average_length: float,
    dimension: int,
    k1: float,
    b: float,
    beta: float,
    scale: float,
    seed: int,
    block: int,
) -> dict[str, object]:
    """Materialize exact DPE rows for the union touched by the query workload.

    Full-corpus candidate generation is unchanged.  This cache optimization is
    result-equivalent to reading the same rows from a deployment-wide DPE table,
    but deliberately does not measure deployment-wide offline storage/build cost.
    """

    cache_dir.mkdir(parents=True, exist_ok=True)
    unique_rows = np.unique(candidates[candidates >= 0]).astype(np.int32)
    digest = _candidate_digest(unique_rows)
    metadata_path = cache_dir / "candidate_scope.json"
    rows_path = cache_dir / "candidate_rows.npy"
    sketches_path = cache_dir / "lexical_sketch_fp16.npy"
    offsets_path = cache_dir / "candidate_offsets.npy"
    cipher_path = cache_dir / "lexical_cipher_fp16.npy"
    norms_path = cache_dir / "lexical_cipher_norms.npy"
    key_path = cache_dir / "lexical_dpe_key.npz"
    required = (
        rows_path,
        sketches_path,
        offsets_path,
        cipher_path,
        norms_path,
        key_path,
    )
    if metadata_path.exists() and all(path.exists() for path in required):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("candidate_digest") != digest:
            raise RuntimeError("candidate-scoped cache was built for different rows")
        return metadata

    if any(path.exists() for path in required) or metadata_path.exists():
        raise RuntimeError(
            f"incomplete candidate cache at {cache_dir}; use a fresh cache directory"
        )
    np.save(rows_path, unique_rows)
    sketches = np.lib.format.open_memmap(
        sketches_path, mode="w+", dtype=np.float16, shape=(len(unique_rows), dimension)
    )
    offsets = np.lib.format.open_memmap(
        offsets_path, mode="w+", dtype=np.int64, shape=(len(unique_rows),)
    )
    started = time.perf_counter()
    pointer = 0
    maximum_norm = 0.0
    with (data_dir / "corpus.jsonl").open("rb") as stream:
        for row_index in range(documents):
            offset = stream.tell()
            line = stream.readline()
            if not line:
                raise RuntimeError(f"corpus ended at row {row_index:,}/{documents:,}")
            if pointer >= len(unique_rows) or row_index != int(unique_rows[pointer]):
                continue
            row = json.loads(line)
            text = f"{row.get('title', '')} {row.get('text', '')}".strip()
            vector, _ = document_vector(
                text,
                document_frequency,
                documents,
                average_length,
                dimension,
                k1,
                b,
            )
            stored = vector.astype(np.float16)
            sketches[pointer] = stored
            offsets[pointer] = offset
            maximum_norm = max(
                maximum_norm, float(np.linalg.vector_norm(stored.astype(np.float32)))
            )
            pointer += 1
            if pointer % 50_000 == 0 or pointer == len(unique_rows):
                sketches.flush()
                offsets.flush()
                print(
                    f"[candidate-cache] {pointer:,}/{len(unique_rows):,} rows; "
                    f"scanned {row_index + 1:,}/{documents:,}",
                    flush=True,
                )
    if pointer != len(unique_rows):
        raise RuntimeError(f"materialized {pointer:,}/{len(unique_rows):,} candidate rows")

    work_dimension = 1 << dimension.bit_length()
    rng = np.random.default_rng(seed + 2)
    sign1 = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), work_dimension)
    sign2 = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), work_dimension)
    permutation = rng.permutation(work_dimension)
    np.savez(key_path, sign1=sign1, sign2=sign2, permutation=permutation)
    cipher = np.lib.format.open_memmap(
        cipher_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(unique_rows), work_dimension),
    )
    norms = np.lib.format.open_memmap(
        norms_path, mode="w+", dtype=np.float32, shape=(len(unique_rows),)
    )
    maximum_norm = max(maximum_norm, 1e-12)
    noise_rng = np.random.default_rng(seed + 2 + 1009)
    radius = 3.0 * scale * beta / 8.0
    for first in range(0, len(unique_rows), block):
        last = min(first + block, len(unique_rows))
        base = np.asarray(sketches[first:last], dtype=np.float32) / maximum_norm
        mips = np.zeros((last - first, dimension + 1), dtype=np.float32)
        mips[:, :dimension] = base
        mips[:, dimension] = np.sqrt(
            np.maximum(0.0, 1.0 - np.sum(base * base, axis=1))
        )
        encrypted = scale * dpe_transform(mips, sign1, sign2, permutation)
        encrypted += ball_noise(noise_rng, len(mips), work_dimension, radius)
        stored = encrypted.astype(np.float16)
        cipher[first:last] = stored
        norms[first:last] = np.sum(stored.astype(np.float32) ** 2, axis=1)
        if last % (block * 20) == 0 or last == len(unique_rows):
            cipher.flush()
            norms.flush()
            print(f"[candidate-dpe] {last:,}/{len(unique_rows):,}", flush=True)
    metadata = {
        "documents": documents,
        "materialized_candidate_rows": len(unique_rows),
        "candidate_digest": digest,
        "hash_dimension": dimension,
        "work_dimension": work_dimension,
        "maximum_sketch_norm": maximum_norm,
        "beta": beta,
        "scale": scale,
        "database_noise_radius": radius,
        "build_seconds": time.perf_counter() - started,
        "materialization_scope": (
            "union of cap-limited candidates actually accessed by the complete-corpus "
            "query workload; result-equivalent to full-table row reads, excluding "
            "deployment-wide offline storage/build cost"
        ),
    }
    atomic_json(metadata_path, metadata)
    return metadata


def run(args: argparse.Namespace) -> dict[str, object]:
    import torch

    dataset = args.dataset
    data_dir = ROOT / "data" / dataset
    documents = 1_000_000 if args.scale == "million" else FULL_DOCUMENTS[dataset]
    suffix = "1000000" if args.scale == "million" else "full"
    if args.scale == "million":
        cache_dir = (
            ROOT
            / "cache"
            / "million_lexical_dpe"
            / f"{dataset}_1000000_h{args.dimension}"
        )
        semantic_cache = ROOT / "cache" / "million_semantic" / f"{dataset}_1000000"
    else:
        cache_label = args.cache_tag or (
            dataset if args.split == "test" else f"{dataset}_{args.split}"
        )
        cache_dir = args.cache_root / f"{cache_label}_full_h{args.dimension}"
        semantic_cache = ROOT / "cache" / "full_semantic" / f"{dataset}_{documents}"
    doc_ids = read_ids(semantic_cache / "doc_ids.txt")
    if len(doc_ids) != documents:
        raise RuntimeError(f"expected {documents:,} document IDs, found {len(doc_ids):,}")
    query_ids, query_texts, qrels, _ = load_queries_qrels(data_dir, args.split)
    db_path = ROOT / "results" / "keyed_fts5" / f"{dataset}_{suffix}.sqlite3"
    connection = sqlite3.connect(db_path)
    document_frequency = {
        str(term): int(df) for term, df in connection.execute("SELECT term, doc FROM vocab")
    }
    token_count = int(connection.execute("SELECT sum(cnt) FROM vocab").fetchone()[0])
    connection.close()
    average_length = token_count / len(doc_ids)
    candidate_path = (
        args.candidate_rankings
        if args.candidate_rankings is not None
        else ROOT / "results" / "budgeted_lexical" / f"{dataset}_{args.scale}.raw.npz"
    )
    candidates_all = np.load(candidate_path)[f"budget_{args.budget_key}"][:, : args.candidate_cap]
    materialization = None
    if args.scale == "full":
        materialization = build_candidate_scoped_cache(
            data_dir=data_dir,
            cache_dir=cache_dir,
            candidates=candidates_all,
            documents=documents,
            document_frequency=document_frequency,
            average_length=average_length,
            dimension=args.dimension,
            k1=args.k1,
            b=args.b,
            beta=args.beta,
            scale=args.scale_factor,
            seed=args.seed,
            block=args.dpe_block,
        )
    query_plain = np.stack(
        [
            query_mips(text, document_frequency, len(doc_ids), args.dimension)
            for text in query_texts
        ]
    )
    query_ciphers = np.stack(
        [
            encrypt_query_matrix(
                query_plain,
                cache_dir / "lexical_dpe_key.npz",
                args.beta,
                args.scale_factor,
                args.seed + 2027 + repeat * 100003,
            )
            for repeat in range(args.repeats)
        ]
    )
    cipher = np.load(cache_dir / "lexical_cipher_fp16.npy", mmap_mode="r")
    norms = np.load(cache_dir / "lexical_cipher_norms.npy", mmap_mode="r")
    sketches = np.load(cache_dir / "lexical_sketch_fp16.npy", mmap_mode="r")
    candidate_rows = None
    if args.scale == "full":
        candidate_rows = np.load(cache_dir / "candidate_rows.npy", mmap_mode="r")
        offsets = np.load(cache_dir / "candidate_offsets.npy", mmap_mode="r")
    else:
        offsets = np.load(cache_dir / "corpus_offsets.npy", mmap_mode="r")
    plaintext = np.full((len(query_ids), args.output_depth), -1, dtype=np.int32)
    dpe = np.full(
        (args.repeats, len(query_ids), args.output_depth), -1, dtype=np.int32
    )
    latencies = [[] for _ in range(args.repeats)]
    candidate_counts = []
    with (data_dir / "corpus.jsonl").open("rb") as corpus_stream, torch.inference_mode():
        for query_index, text in enumerate(query_texts):
            candidates = np.asarray(
                [int(value) for value in candidates_all[query_index] if int(value) >= 0],
                dtype=np.int64,
            )
            candidate_counts.append(len(candidates))
            if len(candidates):
                if candidate_rows is None:
                    local_candidates = candidates
                else:
                    local_candidates = np.searchsorted(candidate_rows, candidates)
                    if np.any(candidate_rows[local_candidates] != candidates):
                        raise RuntimeError("candidate row missing from scoped DPE cache")
                load_started = time.perf_counter()
                rows_gpu = torch.from_numpy(
                    np.asarray(cipher[local_candidates], dtype=np.float16)
                ).to(args.device)
                norms_gpu = torch.from_numpy(
                    np.asarray(norms[local_candidates], dtype=np.float32)
                ).to(args.device)
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize()
                shared_ms = (time.perf_counter() - load_started) * 1000.0
                plain_scores = (
                    np.asarray(sketches[local_candidates], dtype=np.float32)
                    @ query_plain[query_index, : args.dimension]
                )
                depth = min(args.cloud_depth, len(candidates))
                plain_local = np.argpartition(-plain_scores, depth - 1)[:depth]
                plain_cloud = candidates[plain_local]
                clouds = []
                for repeat in range(args.repeats):
                    started = time.perf_counter()
                    query_gpu = torch.from_numpy(query_ciphers[repeat, query_index]).to(
                        device=args.device, dtype=torch.float16
                    )
                    scores = 2.0 * torch.mv(rows_gpu, query_gpu).float() - norms_gpu
                    local = torch.topk(scores, k=depth).indices.cpu().numpy()
                    if args.device.startswith("cuda"):
                        torch.cuda.synchronize()
                    clouds.append(candidates[local])
                    latencies[repeat].append(
                        (time.perf_counter() - started) * 1000.0
                        + shared_ms / args.repeats
                    )
                unique_cloud = np.unique(np.concatenate([plain_cloud, *clouds]))
                exact_rows = (
                    unique_cloud
                    if candidate_rows is None
                    else np.searchsorted(candidate_rows, unique_cloud)
                )
                exact_scores = exact_bm25_for_rows(
                    stream=corpus_stream,
                    offsets=offsets,
                    row_indices=exact_rows,
                    query_terms=set(tokenize(text)),
                    document_frequency=document_frequency,
                    documents=len(doc_ids),
                    average_length=average_length,
                    k1=args.k1,
                    b=args.b,
                )
                lookup = dict(zip(map(int, unique_cloud), map(float, exact_scores)))
                plain_order = np.argsort(
                    -np.asarray([lookup[int(row)] for row in plain_cloud]), kind="stable"
                )[: args.output_depth]
                plaintext[query_index, : len(plain_order)] = plain_cloud[plain_order]
                for repeat, cloud in enumerate(clouds):
                    order = np.argsort(
                        -np.asarray([lookup[int(row)] for row in cloud]), kind="stable"
                    )[: args.output_depth]
                    dpe[repeat, query_index, : len(order)] = cloud[order]
            if (
                query_index == 0
                or query_index + 1 == len(query_ids)
                or (query_index + 1) % 250 == 0
            ):
                print(
                    f"[{dataset} budgeted lexical-DPE] {query_index + 1:,}/{len(query_ids):,}",
                    flush=True,
                )

    semantic = (
        load_semantic_rankings(dataset, args.scale)[0]
        if args.semantic_rankings is None
        else np.asarray(np.load(args.semantic_rankings)["semantic"], dtype=np.int32)
    )
    if semantic.shape[1] != len(query_ids):
        raise RuntimeError("semantic ranking/query mismatch")
    semantic_metrics = [evaluate(row, doc_ids, query_ids, qrels) for row in semantic]
    lexical_plain_metrics = evaluate(plaintext, doc_ids, query_ids, qrels)
    lexical_dpe_metrics = [evaluate(row, doc_ids, query_ids, qrels) for row in dpe]
    weights = args.semantic_weights
    fused = {
        str(weight): [
            evaluate(
                fuse_all(semantic[repeat], dpe[repeat], 100, weight),
                doc_ids,
                query_ids,
                qrels,
            )
            for repeat in range(args.repeats)
        ]
        for weight in weights
    }
    holdout = split_mask(query_ids, "outer")
    doc_to_index = {doc_id: index for index, doc_id in enumerate(doc_ids)}
    from experiment_adaptive_fusion import ndcg_rows

    semantic_rows = np.asarray(
        [ndcg_rows(row, doc_to_index, query_ids, qrels) for row in semantic]
    )
    fused_rows = {
        str(weight): np.asarray(
            [
                ndcg_rows(
                    fuse_all(semantic[repeat], dpe[repeat], 100, weight),
                    doc_to_index,
                    query_ids,
                    qrels,
                )
                for repeat in range(args.repeats)
            ]
        )
        for weight in weights
    }
    flat_latencies = [value for repeat in latencies for value in repeat]
    steady_latencies = [value for repeat in latencies for value in repeat[1:]]
    report = {
        "dataset": dataset,
        "split": args.split,
        "documents": len(doc_ids),
        "queries": len(query_ids),
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "materialization": materialization,
        "candidate_mean": float(np.mean(candidate_counts)),
        "candidate_p95": float(np.percentile(candidate_counts, 95)),
        "lexical_plaintext_metrics": lexical_plain_metrics,
        "lexical_dpe_mean": {
            metric: float(np.mean([row[metric] for row in lexical_dpe_metrics]))
            for metric in lexical_dpe_metrics[0]
        },
        "semantic_mean": {
            metric: float(np.mean([row[metric] for row in semantic_metrics]))
            for metric in semantic_metrics[0]
        },
        "hybrid_mean_by_weight": {
            weight: {
                metric: float(np.mean([row[metric] for row in rows]))
                for metric in rows[0]
            }
            for weight, rows in fused.items()
        },
        "outer_holdout": {
            "queries": int(np.sum(holdout)),
            "semantic_ndcg_at_10": float(np.mean(semantic_rows[:, holdout])),
            "hybrid_ndcg_at_10_by_weight": {
                weight: float(np.mean(rows[:, holdout]))
                for weight, rows in fused_rows.items()
            },
        },
        "online_mean_ms": float(np.mean(flat_latencies)),
        "online_median_ms": float(np.median(flat_latencies)),
        "online_p95_ms": float(np.percentile(flat_latencies, 95)),
        "online_max_ms": float(np.max(flat_latencies)),
        "online_steady_mean_ms": float(np.mean(steady_latencies)),
        "online_steady_p95_ms": float(np.percentile(steady_latencies, 95)),
        "scope": (
            f"actual stored lexical-DPE scoring over the budget-{args.budget_key} "
            "cap-1000 "
            "candidates, followed by exact un-hashed client BM25 reranking; "
            + (
                "complete-corpus index/query workload with candidate-scoped coordinate "
                "materialization (deployment-wide offline DPE storage is not timed)"
                if args.scale == "full"
                else "one-million-document relevance-preserving subset"
            )
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)
    np.savez_compressed(
        args.output.with_suffix(".rankings.npz"),
        lexical_plaintext=plaintext,
        lexical_dpe=dpe,
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("msmarco", "nq", "hotpotqa"), required=True)
    parser.add_argument("--scale", choices=("million", "full"), default="million")
    parser.add_argument("--split", default="test")
    parser.add_argument("--candidate-rankings", type=Path, default=None)
    parser.add_argument("--semantic-rankings", type=Path, default=None)
    parser.add_argument("--cache-tag", default=None)
    parser.add_argument("--candidate-cap", type=int, default=1000)
    parser.add_argument("--budget-key", type=int, default=50_000)
    parser.add_argument("--cloud-depth", type=int, default=200)
    parser.add_argument("--output-depth", type=int, default=100)
    parser.add_argument("--dimension", type=int, default=1024)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--scale-factor", type=float, default=3.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=20250308)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dpe-block", type=int, default=2048)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "cache" / "full_candidate_lexical_dpe",
    )
    parser.add_argument(
        "--semantic-weights", nargs="+", type=float, default=[0.8, 0.9, 0.95, 0.98, 1.0]
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    print(json.dumps(run(parsed), indent=2, default=str))
