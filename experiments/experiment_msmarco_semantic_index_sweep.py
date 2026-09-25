"""Complete-corpus MS MARCO semantic-index parameter expansion.

The experiment varies only the number of residual-IVF probes.  A single
cell-major pass scores the union required by the largest operating point and
maintains independent top-200 DPE accumulators for every smaller probe prefix.
Thus ciphertext rows are physically loaded once, while every operating point
has exactly the candidate set, DPE refinement, and exact client reranking that
an isolated execution would produce.
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
)
from benchmark_msmarco_dev_batched import compact_unique_clouds, encode_queries


DEFAULT_PROBES = (16, 32, 64, 128, 256)


def nested_routes(
    queries: np.ndarray, ivf: IvfFiles, maximum_probes: int
) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    residual = np.asarray(queries, dtype=np.float32) - ivf.mean
    residual /= np.maximum(np.linalg.norm(residual, axis=1, keepdims=True), 1e-12)
    scores = residual @ ivf.centroids.T
    selected = np.argpartition(-scores, maximum_probes - 1, axis=1)[
        :, :maximum_probes
    ]
    selected_scores = np.take_along_axis(scores, selected, axis=1)
    order = np.argsort(-selected_scores, axis=1, kind="stable")
    routed = np.take_along_axis(selected, order, axis=1).astype(np.int32)
    return routed, time.perf_counter() - started


def exact_candidate_statistics(
    *,
    ivf: IvfFiles,
    routed: np.ndarray,
    probe_values: tuple[int, ...],
    assignments_path: Path,
    relevant_rows: list[np.ndarray],
    document_count: int,
) -> dict[int, dict[str, float]]:
    """Count unique candidates without materializing every posting-list union.

    Each document has two distinct IVF assignments.  The sum of selected
    posting sizes counts a document twice exactly when both assigned cells are
    selected.  A 2,048 x 2,048 co-assignment table therefore gives the exact
    inclusion-exclusion correction for every query prefix.
    """

    assignments = np.load(assignments_path, mmap_mode="r")
    if assignments.shape != (document_count, 2):
        raise RuntimeError(f"unexpected assignment shape: {assignments.shape}")
    if np.any(assignments[:, 0] == assignments[:, 1]):
        raise RuntimeError("candidate-count formula requires distinct assignments")
    clusters = len(ivf.centroids)
    flat_pairs = (
        np.asarray(assignments[:, 0], dtype=np.int64) * clusters
        + np.asarray(assignments[:, 1], dtype=np.int64)
    )
    pair_counts = np.bincount(flat_pairs, minlength=clusters * clusters).reshape(
        clusters, clusters
    )
    posting_sizes = np.diff(ivf.offsets).astype(np.int64)
    counts = {probe: [] for probe in probe_values}
    relevant_coverage = {probe: [] for probe in probe_values}
    posting_reads = {probe: [] for probe in probe_values}
    targets = set(probe_values)

    for query_index, route in enumerate(routed):
        selected: list[int] = []
        posting_total = 0
        duplicate_total = 0
        for rank, cell_value in enumerate(route, start=1):
            cell = int(cell_value)
            if selected:
                previous = np.asarray(selected, dtype=np.int64)
                duplicate_total += int(pair_counts[cell, previous].sum())
                duplicate_total += int(pair_counts[previous, cell].sum())
            duplicate_total += int(pair_counts[cell, cell])
            selected.append(cell)
            posting_total += int(posting_sizes[cell])
            if rank not in targets:
                continue
            counts[rank].append(posting_total - duplicate_total)
            posting_reads[rank].append(posting_total)
            truth = relevant_rows[query_index]
            if len(truth):
                truth_assignments = np.asarray(assignments[truth], dtype=np.int32)
                chosen = np.asarray(selected, dtype=np.int32)
                # MS MARCO has almost always one relevant passage, but retain
                # the general per-query relevant-document fraction.
                covered_each = np.isin(truth_assignments[:, 0], chosen) | np.isin(
                    truth_assignments[:, 1], chosen
                )
                relevant_coverage[rank].append(float(np.mean(covered_each)))
            else:
                relevant_coverage[rank].append(1.0)

    output: dict[int, dict[str, float]] = {}
    for probe in probe_values:
        values = np.asarray(counts[probe], dtype=np.float64)
        reads = np.asarray(posting_reads[probe], dtype=np.float64)
        output[probe] = {
            "candidate_mean": float(np.mean(values)),
            "candidate_p95": float(np.percentile(values, 95)),
            "candidate_fraction_mean": float(np.mean(values) / document_count),
            "posting_entries_read_mean": float(np.mean(reads)),
            "relevant_document_coverage_mean": float(
                np.mean(relevant_coverage[probe])
            ),
        }
    return output


def verify_candidate_counts(
    ivf: IvfFiles,
    queries: np.ndarray,
    routed: np.ndarray,
    probe_values: tuple[int, ...],
    statistics: dict[int, dict[str, float]],
) -> None:
    # Verify route nesting and exact unions on a deterministic small sample.
    del statistics  # the per-query arrays are intentionally not retained
    for query_index in range(min(8, len(queries))):
        for probes in probe_values:
            direct, _ = ivf.candidates(queries[query_index], probes)
            expected_cells = set(map(int, routed[query_index, :probes]))
            residual = np.asarray(queries[query_index], dtype=np.float32) - ivf.mean
            residual /= max(float(np.linalg.norm(residual)), 1e-12)
            direct_cells = set(
                map(
                    int,
                    np.argpartition(-(ivf.centroids @ residual), probes - 1)[:probes],
                )
            )
            if direct_cells != expected_cells:
                raise RuntimeError("nested route differs from direct top-k route")
            if len(direct) == 0:
                raise RuntimeError("unexpected empty candidate union")


def client_rerank(
    *,
    clouds: np.ndarray,
    embeddings: np.ndarray,
    queries: np.ndarray,
    output_depth: int,
    exact_chunk: int,
) -> tuple[np.ndarray, float]:
    repeats, query_count, cloud_depth = clouds.shape
    output = np.full((repeats, query_count, output_depth), -1, dtype=np.int32)
    started = time.perf_counter()
    for repeat in range(repeats):
        for first in range(0, query_count, exact_chunk):
            last = min(first + exact_chunk, query_count)
            block = clouds[repeat, first:last]
            valid = np.maximum(block, 0)
            rows = np.asarray(
                embeddings[valid.reshape(-1)], dtype=np.float32
            ).reshape(len(block), cloud_depth, -1)
            scores = np.einsum(
                "bkd,bd->bk", rows, np.asarray(queries[first:last], dtype=np.float32)
            )
            scores[block < 0] = -np.inf
            positions = np.argsort(-scores, axis=1, kind="stable")[:, :output_depth]
            output[repeat, first:last] = np.take_along_axis(block, positions, axis=1)
        print(f"[client rerank] repeat {repeat + 1}/{repeats}", flush=True)
    return output, time.perf_counter() - started


def cell_major_probe_sweep(
    *,
    ivf: IvfFiles,
    routed: np.ndarray,
    query_ciphers: np.ndarray,
    cipher: np.ndarray,
    norms: np.ndarray,
    probe_values: tuple[int, ...],
    cloud_depth: int,
    device: str,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict[int, int], float]:
    repeats, query_count, _ = query_ciphers.shape
    keep = cloud_depth * 2
    best = {
        probes: (
            np.full((repeats, query_count, keep), -np.inf, dtype=np.float32),
            np.full((repeats, query_count, keep), -1, dtype=np.int32),
        )
        for probes in probe_values
    }
    posting_reads = {probes: 0 for probes in probe_values}
    started = time.perf_counter()
    with torch.inference_mode():
        for cell in range(len(ivf.centroids)):
            query_indices, route_ranks = np.where(routed == cell)
            if not len(query_indices):
                continue
            order = np.argsort(query_indices, kind="stable")
            query_indices = query_indices[order]
            route_ranks = route_ranks[order]
            first, last = int(ivf.offsets[cell]), int(ivf.offsets[cell + 1])
            cell_documents = np.asarray(ivf.postings[first:last], dtype=np.int64)
            if not len(cell_documents):
                continue
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
            cell_scores_np = cell_scores.cpu().numpy().reshape(
                repeats, len(query_indices), k
            )
            cell_docs_np = cell_documents[local.cpu().numpy()].astype(np.int32).reshape(
                repeats, len(query_indices), k
            )

            for probes in probe_values:
                eligible = np.flatnonzero(route_ranks < probes)
                if not len(eligible):
                    continue
                target_queries = query_indices[eligible]
                posting_reads[probes] += len(cell_documents) * len(eligible)
                best_scores, best_docs = best[probes]
                merged_scores = np.concatenate(
                    (best_scores[:, target_queries], cell_scores_np[:, eligible]), axis=2
                )
                merged_docs = np.concatenate(
                    (best_docs[:, target_queries], cell_docs_np[:, eligible]), axis=2
                )
                positions = np.argpartition(merged_scores, -keep, axis=2)[
                    :, :, -keep:
                ]
                best_scores[:, target_queries] = np.take_along_axis(
                    merged_scores, positions, axis=2
                )
                best_docs[:, target_queries] = np.take_along_axis(
                    merged_docs, positions, axis=2
                )
            if cell == 0 or cell + 1 == len(ivf.centroids) or (cell + 1) % 128 == 0:
                print(f"[shared DPE sweep] {cell + 1:,}/{len(ivf.centroids):,}", flush=True)
    return best, posting_reads, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probes", default="16,32,64,128,256")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cloud-depth", type=int, default=200)
    parser.add_argument("--output-depth", type=int, default=100)
    parser.add_argument("--exact-chunk", type=int, default=128)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "cache" / "models" / "granite-embedding-small-english-r2",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "msmarco_semantic_probe_sweep",
    )
    args = parser.parse_args()
    probe_values = tuple(int(value) for value in args.probes.split(","))
    if tuple(sorted(set(probe_values))) != probe_values:
        raise ValueError("probes must be unique and ascending")
    if probe_values != DEFAULT_PROBES:
        print(f"[warning] non-default probe sweep: {probe_values}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = ROOT / "data" / "msmarco"
    cache_dir = ROOT / "cache" / "full_semantic" / "msmarco_8841823"
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
    relevant_rows = [
        np.asarray(
            [doc_index[doc_id] for doc_id, gain in qrels[query_id].items() if gain > 0],
            dtype=np.int64,
        )
        for query_id in query_ids
    ]
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
        128,
    )
    ivf = IvfFiles.load(cache_dir)
    if probe_values[-1] > len(ivf.centroids):
        raise ValueError("maximum probes exceeds the number of IVF cells")
    routed, routing_seconds = nested_routes(queries, ivf, probe_values[-1])
    statistics = exact_candidate_statistics(
        ivf=ivf,
        routed=routed,
        probe_values=probe_values,
        assignments_path=cache_dir / "semantic_assignments.npy",
        relevant_rows=relevant_rows,
        document_count=len(doc_ids),
    )
    verify_candidate_counts(ivf, queries, routed, probe_values, statistics)

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
    best, posting_reads, shared_dpe_seconds = cell_major_probe_sweep(
        ivf=ivf,
        routed=routed,
        query_ciphers=query_ciphers,
        cipher=cipher,
        norms=norms,
        probe_values=probe_values,
        cloud_depth=args.cloud_depth,
        device=args.device,
    )

    rows: list[dict[str, object]] = []
    for probes in probe_values:
        best_scores, best_docs = best.pop(probes)
        clouds = compact_unique_clouds(best_scores, best_docs, args.cloud_depth)
        del best_scores, best_docs
        rankings, rerank_seconds = client_rerank(
            clouds=clouds,
            embeddings=embeddings,
            queries=queries,
            output_depth=args.output_depth,
            exact_chunk=args.exact_chunk,
        )
        del clouds
        repeat_metrics = [
            evaluate(ranking, doc_ids, query_ids, qrels) for ranking in rankings
        ]
        mean = {
            metric: float(np.mean([row[metric] for row in repeat_metrics]))
            for metric in repeat_metrics[0]
        }
        sd = {
            metric: float(np.std([row[metric] for row in repeat_metrics]))
            for metric in repeat_metrics[0]
        }
        np.savez_compressed(
            args.output_dir / f"rankings_nprobe_{probes}.npz", semantic=rankings
        )
        row = {
            "nprobe": probes,
            **statistics[probes],
            "posting_entries_read_query_equivalent": int(posting_reads[probes]),
            "semantic_dpe_mean": mean,
            "semantic_dpe_sd": sd,
            "client_exact_rerank_seconds": rerank_seconds,
            "repeat_metrics": repeat_metrics,
        }
        rows.append(row)
        partial = {
            "status": "running",
            "completed_probes": [item["nprobe"] for item in rows],
            "rows": rows,
        }
        atomic_json(args.output_dir / "partial_results.json", partial)
        print(
            f"[completed] nprobe={probes} nDCG@10={mean['nDCG@10']:.6f} "
            f"Recall@100={mean['Recall@100']:.6f}",
            flush=True,
        )

    baseline_path = ROOT / "results" / "msmarco_dev_full_semantic" / "rankings.npz"
    baseline_validation = None
    if baseline_path.exists() and 128 in probe_values:
        baseline = np.load(baseline_path)["semantic"]
        current = np.load(args.output_dir / "rankings_nprobe_128.npz")["semantic"]
        intersections = []
        for first, second in zip(
            baseline.reshape(-1, args.output_depth),
            current.reshape(-1, args.output_depth),
        ):
            intersections.append(
                len(set(map(int, first)).intersection(map(int, second)))
                / args.output_depth
            )
        baseline_validation = {
            "exact_ranking_fraction": float(np.mean(np.all(baseline == current, axis=2))),
            "mean_top100_set_overlap": float(np.mean(intersections)),
            "minimum_top100_set_overlap": float(np.min(intersections)),
        }

    output = {
        "status": "passed",
        "experiment": "MS MARCO complete-corpus residual-IVF probe expansion",
        "dataset": "msmarco",
        "split": "dev",
        "documents": len(doc_ids),
        "queries": len(query_ids),
        "qrels": sum(len(row) for row in qrels.values()),
        "configuration": {
            "clusters": len(ivf.centroids),
            "assignments": 2,
            "probe_values": probe_values,
            "repeats": args.repeats,
            "cloud_depth": args.cloud_depth,
            "output_depth": args.output_depth,
            "beta": args.beta,
            "scale": args.scale,
            "seed": args.seed,
            "device": args.device,
            "model": str(args.model),
            "execution": (
                "shared cell-major ciphertext scan with independent per-prefix "
                "DPE top-200 accumulators and exact client reranking"
            ),
        },
        "query_encoding_seconds": encoding_seconds,
        "routing_seconds": routing_seconds,
        "shared_dpe_sweep_seconds": shared_dpe_seconds,
        "rows": rows,
        "baseline_nprobe_128_validation": baseline_validation,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "scope": (
            "official BEIR MS MARCO dev qrels over all 8,841,823 passages; "
            "actual stored float16 DPE coordinates and exact client reranking"
        ),
    }
    atomic_json(args.output_dir / "results.json", output)
    print(json.dumps(output, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
