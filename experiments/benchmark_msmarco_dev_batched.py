"""Full-corpus MS MARCO dev semantic-DPE with cell-major batched execution.

The online protocol is unchanged: each query probes the same keyed IVF cells,
the server computes the same stored-coordinate DPE score, and the client exactly
reranks the same DPE top-200.  This evaluator changes only the execution order:
each immutable cell is loaded once for all dev queries that probe it.  Reported
timing is therefore throughput timing, not single-query latency.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT_FOR_DEPS = Path(__file__).resolve().parent
if sys.version_info >= (3, 12) and (ROOT_FOR_DEPS / ".deps").exists():
    sys.path.insert(0, str(ROOT_FOR_DEPS / ".deps"))
    if (ROOT_FOR_DEPS / ".gpu_deps").exists():
        sys.path.insert(0, str(ROOT_FOR_DEPS / ".gpu_deps"))

import numpy as np
import torch  # load the CUDA build before legacy modules can prepend .deps

from benchmark_million_semantic_hybrid import (
    ROOT,
    IvfFiles,
    atomic_json,
    encrypt_queries,
    evaluate,
    load_queries_qrels,
    read_ids,
    semantic_dpe_retrieval_repeats,
)


def compact_unique_clouds(
    scores: np.ndarray, documents: np.ndarray, depth: int
) -> np.ndarray:
    repeats, queries, _ = scores.shape
    output = np.full((repeats, queries, depth), -1, dtype=np.int32)
    for repeat in range(repeats):
        for query in range(queries):
            order = np.argsort(-scores[repeat, query], kind="stable")
            seen: set[int] = set()
            selected: list[int] = []
            for position in order:
                document = int(documents[repeat, query, position])
                if document < 0 or document in seen:
                    continue
                seen.add(document)
                selected.append(document)
                if len(selected) == depth:
                    break
            output[repeat, query, : len(selected)] = selected
    return output


def cell_major_dpe(
    *,
    ivf: IvfFiles,
    queries: np.ndarray,
    query_ciphers: np.ndarray,
    cipher: np.ndarray,
    norms: np.ndarray,
    probes: int,
    cloud_depth: int,
    output_depth: int,
    embeddings: np.ndarray,
    device: str,
    exact_chunk: int,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    repeats, query_count, _ = query_ciphers.shape
    keep = cloud_depth * 2  # documents have two assignments; preserves top-depth uniques
    residual = np.asarray(queries, dtype=np.float32) - ivf.mean
    residual /= np.maximum(np.linalg.norm(residual, axis=1, keepdims=True), 1e-12)
    routing_scores = residual @ ivf.centroids.T
    routed = np.argpartition(-routing_scores, probes - 1, axis=1)[:, :probes]
    best_scores = np.full((repeats, query_count, keep), -np.inf, dtype=np.float32)
    best_docs = np.full((repeats, query_count, keep), -1, dtype=np.int32)
    posting_reads = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for cell in range(len(ivf.centroids)):
            query_indices = np.flatnonzero(np.any(routed == cell, axis=1))
            if not len(query_indices):
                continue
            first, last = int(ivf.offsets[cell]), int(ivf.offsets[cell + 1])
            cell_documents = np.asarray(ivf.postings[first:last], dtype=np.int64)
            if not len(cell_documents):
                continue
            posting_reads += len(cell_documents) * len(query_indices)
            rows_gpu = torch.from_numpy(
                np.asarray(cipher[cell_documents], dtype=np.float16)
            ).to(device)
            norms_gpu = torch.from_numpy(
                np.asarray(norms[cell_documents], dtype=np.float32)
            ).to(device)
            selected_queries = np.asarray(
                query_ciphers[:, query_indices], dtype=np.float16
            ).reshape(-1, query_ciphers.shape[-1])
            query_gpu = torch.from_numpy(selected_queries).to(device)
            values = 2.0 * torch.matmul(query_gpu, rows_gpu.T).float() - norms_gpu
            k = min(cloud_depth, len(cell_documents))
            cell_scores, local = torch.topk(values, k=k, dim=1)
            cell_scores = cell_scores.cpu().numpy().reshape(
                repeats, len(query_indices), k
            )
            cell_docs = cell_documents[local.cpu().numpy()].astype(np.int32).reshape(
                repeats, len(query_indices), k
            )
            merged_scores = np.concatenate(
                (best_scores[:, query_indices], cell_scores), axis=2
            )
            merged_docs = np.concatenate((best_docs[:, query_indices], cell_docs), axis=2)
            positions = np.argpartition(merged_scores, -keep, axis=2)[:, :, -keep:]
            best_scores[:, query_indices] = np.take_along_axis(
                merged_scores, positions, axis=2
            )
            best_docs[:, query_indices] = np.take_along_axis(
                merged_docs, positions, axis=2
            )
            if cell == 0 or cell + 1 == len(ivf.centroids) or (cell + 1) % 128 == 0:
                print(
                    f"[cell-major semantic-DPE] {cell + 1:,}/{len(ivf.centroids):,}",
                    flush=True,
                )
    dpe_seconds = time.perf_counter() - started
    clouds = compact_unique_clouds(best_scores, best_docs, cloud_depth)

    output = np.full((repeats, query_count, output_depth), -1, dtype=np.int32)
    exact_started = time.perf_counter()
    for repeat in range(repeats):
        for first in range(0, query_count, exact_chunk):
            last = min(first + exact_chunk, query_count)
            block = clouds[repeat, first:last]
            valid = np.maximum(block, 0)
            rows = np.asarray(embeddings[valid.reshape(-1)], dtype=np.float32).reshape(
                len(block), cloud_depth, -1
            )
            scores = np.einsum(
                "bkd,bd->bk", rows, np.asarray(queries[first:last], dtype=np.float32)
            )
            scores[block < 0] = -np.inf
            positions = np.argsort(-scores, axis=1, kind="stable")[:, :output_depth]
            output[repeat, first:last] = np.take_along_axis(block, positions, axis=1)
        print(f"[exact client rerank] repeat {repeat + 1}/{repeats}", flush=True)
    exact_seconds = time.perf_counter() - exact_started
    return output, {
        "execution_order": "cell-major batched evaluation; protocol scores unchanged",
        "probes": probes,
        "cloud_depth": cloud_depth,
        "posting_reads_query_equivalent": posting_reads,
        "dpe_cell_scoring_seconds": dpe_seconds,
        "exact_client_rerank_seconds": exact_seconds,
        "total_seconds": dpe_seconds + exact_seconds,
        "throughput_queries_per_second_all_repeats": (
            repeats * query_count / max(dpe_seconds + exact_seconds, 1e-12)
        ),
        "latency_warning": "throughput execution; do not interpret as independent online latency",
    }


def encode_queries(
    query_texts: list[str], path: Path, model_path: Path, device: str, batch_size: int
) -> tuple[np.ndarray, float]:
    if path.exists():
        return np.load(path), 0.0
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(model_path), device=device)
    started = time.perf_counter()
    queries = model.encode(
        query_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    seconds = time.perf_counter() - started
    np.save(path, queries)
    return queries, seconds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--probes", type=int, default=128)
    parser.add_argument("--cloud-depth", type=int, default=200)
    parser.add_argument("--output-depth", type=int, default=100)
    parser.add_argument("--exact-chunk", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--verify-query-major", action="store_true")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "cache" / "models" / "granite-embedding-small-english-r2",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "msmarco_dev_full_semantic",
    )
    args = parser.parse_args()

    data_dir = ROOT / "data" / "msmarco"
    cache_dir = ROOT / "cache" / "full_semantic" / "msmarco_8841823"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    query_ids, query_texts, qrels, relevant_ids = load_queries_qrels(data_dir, "dev")
    if args.max_queries is not None:
        query_ids = query_ids[: args.max_queries]
        query_texts = query_texts[: args.max_queries]
        qrels = {query_id: qrels[query_id] for query_id in query_ids}
        relevant_ids = {
            doc_id
            for query_id in query_ids
            for doc_id, gain in qrels[query_id].items()
            if gain > 0
        }
    doc_ids = read_ids(cache_dir / "doc_ids.txt")
    doc_index = {doc_id: index for index, doc_id in enumerate(doc_ids)}
    missing = relevant_ids.difference(doc_index)
    if missing:
        raise RuntimeError(f"full corpus omits {len(missing)} dev-relevant documents")
    queries, encoding_seconds = encode_queries(
        query_texts,
        cache_dir
        / (
            "query_embeddings_dev.npy"
            if args.max_queries is None
            else f"query_embeddings_dev_{args.max_queries}.npy"
        ),
        args.model,
        args.device,
        args.batch_size,
    )
    ivf = IvfFiles.load(cache_dir)
    embeddings = np.load(cache_dir / "corpus_embeddings.npy", mmap_mode="r")
    cipher = np.load(cache_dir / "semantic_cipher_fp16.npy", mmap_mode="r")
    norms = np.load(cache_dir / "semantic_cipher_norms.npy", mmap_mode="r")
    query_ciphers = np.stack(
        [
            encrypt_queries(
                queries,
                cache_dir / "semantic_dpe_key.npz",
                args.beta,
                args.scale,
                args.seed + 2027 + repeat * 100003,
            )
            for repeat in range(args.repeats)
        ]
    )
    rankings, timing = cell_major_dpe(
        ivf=ivf,
        queries=queries,
        query_ciphers=query_ciphers,
        cipher=cipher,
        norms=norms,
        probes=args.probes,
        cloud_depth=args.cloud_depth,
        output_depth=args.output_depth,
        embeddings=embeddings,
        device=args.device,
        exact_chunk=args.exact_chunk,
    )
    equivalence = None
    if args.verify_query_major:
        query_major, _ = semantic_dpe_retrieval_repeats(
            ivf=ivf,
            queries=queries,
            query_ciphers=query_ciphers,
            probes=args.probes,
            candidate_depth=args.cloud_depth,
            output_depth=args.output_depth,
            embeddings=embeddings,
            cipher=cipher,
            norms=norms,
            device=args.device,
        )
        overlaps = []
        for first, second in zip(
            rankings.reshape(-1, args.output_depth),
            query_major.reshape(-1, args.output_depth),
        ):
            overlaps.append(len(set(map(int, first)).intersection(map(int, second))) / args.output_depth)
        equivalence = {
            "exact_ranking_fraction": float(
                np.mean(np.all(rankings == query_major, axis=2))
            ),
            "mean_top100_set_overlap": float(np.mean(overlaps)),
            "minimum_top100_set_overlap": float(np.min(overlaps)),
        }
    metrics = [evaluate(row, doc_ids, query_ids, qrels) for row in rankings]
    mean = {
        metric: float(np.mean([row[metric] for row in metrics])) for metric in metrics[0]
    }
    output = {
        "dataset": "msmarco",
        "split": "dev",
        "documents": len(doc_ids),
        "queries": len(query_ids),
        "qrels": sum(len(row) for row in qrels.values()),
        "configuration": vars(args)
        | {"model": str(args.model), "output_dir": str(args.output_dir)},
        "query_encoding_seconds": encoding_seconds,
        "semantic_dpe_mean": mean,
        "repeat_metrics": metrics,
        "timing": timing,
        "query_major_equivalence_check": equivalence,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "scope": (
            "official BEIR MS MARCO dev qrels over the complete 8,841,823-passage "
            "corpus; stored semantic DPE coordinates and exact client reranking"
        ),
    }
    atomic_json(args.output_dir / "results.json", output)
    np.savez_compressed(args.output_dir / "rankings.npz", semantic=rankings)
    print(json.dumps(output, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
