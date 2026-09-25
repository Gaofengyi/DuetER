"""Complete-corpus return-depth/communication/quality ablation for DuetDPE.

The server-side DPE score is computed once up to the largest tested response
depth.  Every smaller operating point is a prefix of that actual DPE order and
is independently reranked with plaintext-equivalent client scores.  This is not
post-hoc truncation of the final client ranking.

For each operating point, the DuetRank architecture and semantic-backoff choice
selected by the original nested experiment are frozen, while model parameters
are refit on the same calibration half.  The outer SHA-256 holdout remains
untouched until reporting.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

ROOT_FOR_DEPS = Path(__file__).resolve().parent
if sys.version_info >= (3, 12) and (ROOT_FOR_DEPS / ".deps").exists():
    sys.path.insert(0, str(ROOT_FOR_DEPS / ".deps"))
    if (ROOT_FOR_DEPS / ".gpu_deps").exists():
        sys.path.insert(0, str(ROOT_FOR_DEPS / ".gpu_deps"))

import joblib
import numpy as np
import torch  # import before legacy modules that prepend the CPU-only .deps path
from sklearn.ensemble import HistGradientBoostingClassifier

from benchmark_budgeted_lexical_dpe import FULL_DOCUMENTS
from benchmark_duetdpe_communication import (
    serialize_request,
    serialize_response_depths,
    verify_response,
)
from benchmark_million_lexical_dpe import (
    encrypt_query_matrix,
    exact_bm25_for_rows,
    query_mips,
)
from benchmark_million_semantic_hybrid import (
    ROOT,
    IvfFiles,
    atomic_json,
    encrypt_queries,
    load_queries_qrels,
    read_ids,
)
from benchmark_msmarco_dev_batched import compact_unique_clouds
from evaluate_duetrank_recall import metric_rows
from experiment_adaptive_fusion import query_features
from experiment_candidate_rank_fusion import (
    document_feature_matrix,
    learned_rankings,
    mix_rankings,
    rank_maps,
    split_mask,
)
from dueter_common import tokenize


SEMANTIC_DEPTHS = (50, 100, 200)
LEXICAL_DEPTHS = (100, 200, 300, 500)
OPERATING_POINTS = ((50, 100), (100, 200), (200, 200), (200, 300), (200, 500))
REPEATS = 5
PROBES = 128
SEMANTIC_SEED = 20260822
LEXICAL_SEED = 20250308


CONFIGS: dict[str, dict[str, object]] = {
    "nq": {
        "split": "test",
        "documents": 2_681_468,
        "budget": 50_000,
        "candidate": ROOT / "results/budgeted_lexical/nq_full.raw.npz",
        "trace": ROOT / "results/budgeted_lexical/nq_full.trace.json",
        "lexical_cache": ROOT / "cache/full_candidate_lexical_dpe/nq_full_h1024",
        "semantic_source": ROOT / "results/full_semantic_hybrid/nq_2681468/rankings.npz",
        "lexical_source": ROOT / "results/budgeted_lexical_dpe/nq_full_d500_o300.rankings.npz",
        "final_lexical_response": 500,
        "lexical_client_cap": 300,
        "selection": ROOT / "results/budgeted_lexical_dpe/learned_fusion_nq_full_d500_o300.json",
    },
    "hotpotqa": {
        "split": "test",
        "documents": 5_233_329,
        "budget": 50_000,
        "candidate": ROOT / "results/budgeted_lexical/hotpotqa_full.raw.npz",
        "trace": ROOT / "results/budgeted_lexical/hotpotqa_full.trace.json",
        "lexical_cache": ROOT / "cache/full_candidate_lexical_dpe/hotpotqa_full_h1024",
        "semantic_source": ROOT / "results/full_semantic_hybrid/hotpotqa_5233329/rankings.npz",
        "lexical_source": ROOT / "results/budgeted_lexical_dpe/hotpotqa_full_d200.rankings.npz",
        "final_lexical_response": 200,
        "lexical_client_cap": 100,
        "selection": ROOT / "results/budgeted_lexical_dpe/learned_fusion_hotpotqa_full.json",
    },
    "msmarco": {
        "split": "dev",
        "documents": 8_841_823,
        "budget": 250_000,
        "candidate": ROOT / "results/budgeted_lexical/msmarco_dev_full.raw.npz",
        "trace": ROOT / "results/budgeted_lexical/msmarco_dev_full.trace.json",
        "lexical_cache": ROOT / "cache/full_candidate_lexical_dpe/msmarco_dev_b250k_full_h1024",
        "semantic_source": ROOT / "results/msmarco_dev_full_semantic/rankings.npz",
        "lexical_source": ROOT / "results/budgeted_lexical_dpe/msmarco_dev_full_b250k_d500_o300.rankings.npz",
        "final_lexical_response": 500,
        "lexical_client_cap": 300,
        "selection": ROOT / "results/budgeted_lexical_dpe/learned_fusion_msmarco_dev_full_b250k_d500_o300.json",
    },
}


RESULT_ROOT = ROOT / "results/return_depth_ablation"


def semantic_queries(dataset: str, cache_dir: Path) -> np.ndarray:
    suffix = "_dev" if dataset == "msmarco" else ""
    return np.load(cache_dir / f"query_embeddings{suffix}.npy")


def semantic_multi_depth(dataset: str, device: str, force: bool) -> Path:
    """Cell-major DPE scoring followed by exact reranking of every DPE prefix."""
    import torch

    config = CONFIGS[dataset]
    output_path = RESULT_ROOT / f"{dataset}_semantic_depths.npz"
    if output_path.exists() and not force:
        return output_path
    documents = int(config["documents"])
    split = str(config["split"])
    data_dir = ROOT / "data" / dataset
    cache_dir = ROOT / "cache/full_semantic" / f"{dataset}_{documents}"
    query_ids, _, _, _ = load_queries_qrels(data_dir, split)
    queries = semantic_queries(dataset, cache_dir)
    if len(queries) != len(query_ids):
        raise RuntimeError(f"{dataset}: semantic query count mismatch")
    ivf = IvfFiles.load(cache_dir)
    embeddings = np.load(cache_dir / "corpus_embeddings.npy", mmap_mode="r")
    cipher = np.load(cache_dir / "semantic_cipher_fp16.npy", mmap_mode="r")
    norms = np.load(cache_dir / "semantic_cipher_norms.npy", mmap_mode="r")
    query_ciphers = np.stack(
        [
            encrypt_queries(
                queries,
                cache_dir / "semantic_dpe_key.npz",
                0.1,
                3.0,
                SEMANTIC_SEED + 2027 + repeat * 100003,
            )
            for repeat in range(REPEATS)
        ]
    )

    maximum = max(SEMANTIC_DEPTHS)
    keep = maximum * 2  # two IVF assignments per document
    residual = np.asarray(queries, dtype=np.float32) - ivf.mean
    residual /= np.maximum(np.linalg.norm(residual, axis=1, keepdims=True), 1e-12)
    routing = residual @ ivf.centroids.T
    routed = np.argpartition(-routing, PROBES - 1, axis=1)[:, :PROBES]
    best_scores = np.full((REPEATS, len(queries), keep), -np.inf, dtype=np.float32)
    best_docs = np.full((REPEATS, len(queries), keep), -1, dtype=np.int32)
    started = time.perf_counter()
    with torch.inference_mode():
        for cell in range(len(ivf.centroids)):
            query_indices = np.flatnonzero(np.any(routed == cell, axis=1))
            if not len(query_indices):
                continue
            first, last = int(ivf.offsets[cell]), int(ivf.offsets[cell + 1])
            documents_in_cell = np.asarray(ivf.postings[first:last], dtype=np.int64)
            if not len(documents_in_cell):
                continue
            rows_gpu = torch.from_numpy(
                np.asarray(cipher[documents_in_cell], dtype=np.float16)
            ).to(device)
            norms_gpu = torch.from_numpy(
                np.asarray(norms[documents_in_cell], dtype=np.float32)
            ).to(device)
            selected = np.asarray(query_ciphers[:, query_indices], dtype=np.float16)
            query_gpu = torch.from_numpy(
                selected.reshape(-1, query_ciphers.shape[-1])
            ).to(device)
            values = 2.0 * torch.matmul(query_gpu, rows_gpu.T).float() - norms_gpu
            k = min(maximum, len(documents_in_cell))
            cell_scores, local = torch.topk(values, k=k, dim=1)
            cell_scores = cell_scores.cpu().numpy().reshape(REPEATS, len(query_indices), k)
            cell_docs = documents_in_cell[local.cpu().numpy()].astype(np.int32).reshape(
                REPEATS, len(query_indices), k
            )
            merged_scores = np.concatenate((best_scores[:, query_indices], cell_scores), axis=2)
            merged_docs = np.concatenate((best_docs[:, query_indices], cell_docs), axis=2)
            positions = np.argpartition(merged_scores, -keep, axis=2)[:, :, -keep:]
            best_scores[:, query_indices] = np.take_along_axis(
                merged_scores, positions, axis=2
            )
            best_docs[:, query_indices] = np.take_along_axis(
                merged_docs, positions, axis=2
            )
            if cell == 0 or cell + 1 == len(ivf.centroids) or (cell + 1) % 128 == 0:
                print(f"[{dataset} semantic] cell {cell + 1:,}/{len(ivf.centroids):,}", flush=True)
    clouds = compact_unique_clouds(best_scores, best_docs, maximum)
    del best_scores, best_docs

    outputs = {
        depth: np.full((REPEATS, len(queries), 100), -1, dtype=np.int32)
        for depth in SEMANTIC_DEPTHS
    }
    for repeat in range(REPEATS):
        for first in range(0, len(queries), 128):
            last = min(first + 128, len(queries))
            block = clouds[repeat, first:last]
            valid = np.maximum(block, 0)
            rows = np.asarray(embeddings[valid.reshape(-1)], dtype=np.float32).reshape(
                len(block), maximum, -1
            )
            scores = np.einsum(
                "bkd,bd->bk", rows, np.asarray(queries[first:last], dtype=np.float32)
            )
            scores[block < 0] = -np.inf
            for depth in SEMANTIC_DEPTHS:
                retained = min(depth, 100)
                order = np.argsort(-scores[:, :depth], axis=1, kind="stable")[:, :retained]
                outputs[depth][repeat, first:last, :retained] = np.take_along_axis(
                    block[:, :depth], order, axis=1
                )
        print(f"[{dataset} semantic exact] repeat {repeat + 1}/{REPEATS}", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        **{f"semantic_{depth}": value for depth, value in outputs.items()},
    )
    print(f"[{dataset} semantic] {time.perf_counter() - started:.2f}s", flush=True)
    return output_path


def lexical_multi_depth(dataset: str, device: str, force: bool) -> Path:
    """Score each 1,000-row candidate set once and rerank DPE prefixes exactly."""
    import torch

    config = CONFIGS[dataset]
    output_path = RESULT_ROOT / f"{dataset}_lexical_depths.npz"
    if output_path.exists() and not force:
        return output_path
    split = str(config["split"])
    documents = int(config["documents"])
    budget = int(config["budget"])
    client_cap = int(config["lexical_client_cap"])
    data_dir = ROOT / "data" / dataset
    semantic_cache = ROOT / "cache/full_semantic" / f"{dataset}_{documents}"
    cache_dir = Path(config["lexical_cache"])
    doc_ids = read_ids(semantic_cache / "doc_ids.txt")
    query_ids, query_texts, _, _ = load_queries_qrels(data_dir, split)
    connection = sqlite3.connect(ROOT / "results/keyed_fts5" / f"{dataset}_full.sqlite3")
    document_frequency = {
        str(term): int(df) for term, df in connection.execute("SELECT term, doc FROM vocab")
    }
    token_count = int(connection.execute("SELECT sum(cnt) FROM vocab").fetchone()[0])
    connection.close()
    average_length = token_count / len(doc_ids)
    candidates_all = np.load(Path(config["candidate"]))[f"budget_{budget}"][:, :1000]
    query_plain = np.stack(
        [query_mips(text, document_frequency, len(doc_ids), 1024) for text in query_texts]
    )
    query_ciphers = np.stack(
        [
            encrypt_query_matrix(
                query_plain,
                cache_dir / "lexical_dpe_key.npz",
                0.1,
                3.0,
                LEXICAL_SEED + 2027 + repeat * 100003,
            )
            for repeat in range(REPEATS)
        ]
    )
    cipher = np.load(cache_dir / "lexical_cipher_fp16.npy", mmap_mode="r")
    norms = np.load(cache_dir / "lexical_cipher_norms.npy", mmap_mode="r")
    candidate_rows = np.load(cache_dir / "candidate_rows.npy", mmap_mode="r")
    offsets = np.load(cache_dir / "candidate_offsets.npy", mmap_mode="r")
    outputs = {
        depth: np.full((REPEATS, len(query_ids), client_cap), -1, dtype=np.int32)
        for depth in LEXICAL_DEPTHS
    }
    maximum = max(LEXICAL_DEPTHS)
    started = time.perf_counter()
    with (data_dir / "corpus.jsonl").open("rb") as corpus_stream, torch.inference_mode():
        for query_index, text in enumerate(query_texts):
            candidates = np.asarray(
                [int(value) for value in candidates_all[query_index] if int(value) >= 0],
                dtype=np.int64,
            )
            if len(candidates):
                local_candidates = np.searchsorted(candidate_rows, candidates)
                if np.any(candidate_rows[local_candidates] != candidates):
                    raise RuntimeError(f"{dataset}: candidate row missing from lexical cache")
                rows_gpu = torch.from_numpy(
                    np.asarray(cipher[local_candidates], dtype=np.float16)
                ).to(device)
                norms_gpu = torch.from_numpy(
                    np.asarray(norms[local_candidates], dtype=np.float32)
                ).to(device)
                depth_max = min(maximum, len(candidates))
                clouds: list[np.ndarray] = []
                for repeat in range(REPEATS):
                    query_gpu = torch.from_numpy(query_ciphers[repeat, query_index]).to(
                        device=device, dtype=torch.float16
                    )
                    scores = 2.0 * torch.mv(rows_gpu, query_gpu).float() - norms_gpu
                    local = torch.topk(scores, k=depth_max).indices.cpu().numpy()
                    clouds.append(candidates[local])
                unique_cloud = np.unique(np.concatenate(clouds))
                exact_rows = np.searchsorted(candidate_rows, unique_cloud)
                exact_scores = exact_bm25_for_rows(
                    stream=corpus_stream,
                    offsets=offsets,
                    row_indices=exact_rows,
                    query_terms=set(tokenize(text)),
                    document_frequency=document_frequency,
                    documents=len(doc_ids),
                    average_length=average_length,
                    k1=1.2,
                    b=0.75,
                )
                lookup = dict(zip(map(int, unique_cloud), map(float, exact_scores)))
                for repeat, cloud in enumerate(clouds):
                    for depth in LEXICAL_DEPTHS:
                        prefix = cloud[: min(depth, len(cloud))]
                        retained = min(client_cap, len(prefix))
                        order = np.argsort(
                            -np.asarray([lookup[int(row)] for row in prefix]), kind="stable"
                        )[:retained]
                        outputs[depth][repeat, query_index, :retained] = prefix[order]
            if query_index == 0 or query_index + 1 == len(query_ids) or (query_index + 1) % 250 == 0:
                print(f"[{dataset} lexical] {query_index + 1:,}/{len(query_ids):,}", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        **{f"lexical_{depth}": value for depth, value in outputs.items()},
    )
    print(f"[{dataset} lexical] {time.perf_counter() - started:.2f}s", flush=True)
    return output_path


def depth_training_matrix(
    semantic: np.ndarray,
    lexical: np.ndarray,
    query_feature_rows: np.ndarray,
    train_mask: np.ndarray,
    query_ids: list[str],
    qrels: dict[str, dict[str, int]],
    doc_to_index: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The published sampler, parameterized by the candidates actually returned."""
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    rng = np.random.default_rng(20260826)
    for query_index in np.flatnonzero(train_mask):
        semantic_positions = rank_maps(semantic[query_index], 100)
        lexical_positions = rank_maps(lexical[query_index], lexical.shape[1])
        relevant = {
            doc_to_index[doc_id]
            for doc_id, gain in qrels[query_ids[query_index]].items()
            if gain > 0 and doc_id in doc_to_index
        }
        candidates = set(semantic_positions).union(lexical_positions)
        positives = candidates.intersection(relevant)
        hard_negatives = set(list(semantic_positions)[:50]).union(
            list(lexical_positions)[:100]
        ) - relevant
        remaining = np.asarray(list(candidates - positives - hard_negatives), dtype=np.int64)
        if len(remaining) > 100:
            remaining = rng.choice(remaining, size=100, replace=False)
        negatives = hard_negatives.union(int(value) for value in remaining)
        selected = list(positives) + list(negatives)
        if not selected:
            continue
        selected_labels = np.asarray(
            [int(document in positives) for document in selected], dtype=np.int8
        )
        features.append(
            document_feature_matrix(
                selected,
                semantic_positions,
                lexical_positions,
                query_feature_rows[query_index],
            )
        )
        labels.append(selected_labels)
        positive_weight = 0.5 / max(len(positives), 1)
        negative_weight = 0.5 / max(len(negatives), 1)
        weights.append(np.where(selected_labels > 0, positive_weight, negative_weight))
    return (
        np.concatenate(features).astype(np.float32),
        np.concatenate(labels),
        np.concatenate(weights),
    )


def selection_for(dataset: str) -> dict[str, object]:
    payload = json.loads(Path(CONFIGS[dataset]["selection"]).read_text(encoding="utf-8"))
    return payload["datasets"][0]["selection"]


def communication_bytes(semantic_depth: int, lexical_depth: int, budget: int) -> int:
    request = serialize_request(budget, lexical_depth, semantic_depth)
    response, key = serialize_response_depths(semantic_depth, lexical_depth)
    verify_response(response, key)
    return len(request) + len(response)


def evaluate_dataset(dataset: str) -> dict[str, object]:
    config = CONFIGS[dataset]
    split = str(config["split"])
    documents = int(config["documents"])
    budget = int(config["budget"])
    query_ids, query_texts, qrels, _ = load_queries_qrels(ROOT / "data" / dataset, split)
    doc_ids = read_ids(ROOT / "cache/full_semantic" / f"{dataset}_{documents}" / "doc_ids.txt")
    doc_to_index = {doc_id: index for index, doc_id in enumerate(doc_ids)}
    outer = split_mask(query_ids, "outer")
    calibration = ~outer
    trace = json.loads(Path(config["trace"]).read_text(encoding="utf-8"))[str(budget)]
    selection = selection_for(dataset)
    leaves = int(selection["max_leaf_nodes"])
    backoff = float(selection["semantic_backoff_weight"])
    semantic_archive = np.load(RESULT_ROOT / f"{dataset}_semantic_depths.npz")
    lexical_archive = np.load(RESULT_ROOT / f"{dataset}_lexical_depths.npz")
    rows: list[dict[str, object]] = []
    for semantic_depth, lexical_depth in OPERATING_POINTS:
        semantic = np.asarray(
            semantic_archive[f"semantic_{semantic_depth}"], dtype=np.int32
        )
        lexical = np.asarray(
            lexical_archive[f"lexical_{lexical_depth}"], dtype=np.int32
        )
        q_features = query_features(query_texts, semantic, lexical[0], trace)
        matrix, labels, sample_weights = depth_training_matrix(
            semantic[0],
            lexical[0],
            q_features,
            calibration,
            query_ids,
            qrels,
            doc_to_index,
        )
        model = HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.06,
            max_leaf_nodes=leaves,
            min_samples_leaf=40,
            l2_regularization=1.0,
            random_state=20260826,
        )
        model.fit(matrix, labels, sample_weight=sample_weights)
        model_path = RESULT_ROOT / f"{dataset}_s{semantic_depth}_l{lexical_depth}.joblib"
        joblib.dump(model, model_path)
        ndcg_values: list[np.ndarray] = []
        recall_values: list[np.ndarray] = []
        inference_ms: list[float] = []
        for repeat in range(REPEATS):
            started = time.perf_counter()
            learned = learned_rankings(
                model, semantic[repeat], lexical[repeat], q_features, depth=100, mask=outer
            )
            ranking = learned if backoff == 0.0 else mix_rankings(
                semantic[repeat], learned, backoff
            )
            inference_ms.append(
                (time.perf_counter() - started) * 1000.0 / max(int(np.sum(outer)), 1)
            )
            ndcg_values.append(metric_rows(ranking, doc_ids, query_ids, qrels, 10)["ndcg"])
            recall_values.append(metric_rows(ranking, doc_ids, query_ids, qrels, 20)["recall"])
        total_bytes = communication_bytes(semantic_depth, lexical_depth, budget)
        rows.append(
            {
                "semantic_response_depth": semantic_depth,
                "lexical_response_depth": lexical_depth,
                "communication_bytes": total_bytes,
                "communication_mib": total_bytes / (1024**2),
                "outer_holdout_queries": int(np.sum(outer)),
                "ndcg_at_10": float(np.mean(np.asarray(ndcg_values)[:, outer])),
                "recall_at_20": float(np.mean(np.asarray(recall_values)[:, outer])),
                "client_duetrank_ms": float(np.mean(inference_ms)),
                "training_rows": int(len(matrix)),
                "model_bytes": int(model_path.stat().st_size),
            }
        )
        print(json.dumps({"dataset": dataset, **rows[-1]}), flush=True)
        del matrix, labels, sample_weights, model

    semantic_reference = np.asarray(np.load(Path(config["semantic_source"]))["semantic"])
    semantic_current = np.asarray(semantic_archive["semantic_200"])
    lexical_reference = np.asarray(np.load(Path(config["lexical_source"]))["lexical_dpe"])
    final_lexical_depth = int(config["final_lexical_response"])
    lexical_current = np.asarray(lexical_archive[f"lexical_{final_lexical_depth}"])
    semantic_compared = semantic_current[:, :, : semantic_reference.shape[2]]
    lexical_compared = lexical_current[:, :, : lexical_reference.shape[2]]

    def top10_overlap(first: np.ndarray, second: np.ndarray) -> float:
        values = []
        for left_repeat, right_repeat in zip(first, second):
            for left, right in zip(left_repeat, right_repeat):
                left_set = {int(value) for value in left[:10] if int(value) >= 0}
                right_set = {int(value) for value in right[:10] if int(value) >= 0}
                denominator = max(len(left_set), len(right_set), 1)
                values.append(len(left_set.intersection(right_set)) / denominator)
        return float(np.mean(values))

    validation = {
        "semantic_exact_row_fraction": float(
            np.mean(np.all(semantic_compared == semantic_reference, axis=2))
        ),
        "semantic_top10_overlap": top10_overlap(semantic_compared, semantic_reference),
        "lexical_exact_row_fraction": float(
            np.mean(np.all(lexical_compared == lexical_reference, axis=2))
        ),
        "lexical_top10_overlap": top10_overlap(lexical_compared, lexical_reference),
    }
    semantic_valid = (
        validation["semantic_exact_row_fraction"] == 1.0
        or validation["semantic_top10_overlap"] >= 0.999
    )
    lexical_valid = (
        validation["lexical_exact_row_fraction"] == 1.0
        or validation["lexical_top10_overlap"] >= 0.999
    )
    if not semantic_valid or not lexical_valid:
        raise RuntimeError(f"{dataset}: depth ablation failed top-10 validation: {validation}")
    return {
        "dataset": dataset,
        "documents": documents,
        "queries": len(query_ids),
        "split": split,
        "outer_holdout_queries": int(np.sum(outer)),
        "frozen_architecture": {
            "max_leaf_nodes": leaves,
            "semantic_backoff_weight": backoff,
            "calibration_split": "same non-outer SHA-256 half as the primary experiment",
        },
        "validation": validation,
        "operating_points": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=tuple(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--phase", choices=("all", "retrieval", "evaluation", "combine"), default="all"
    )
    args = parser.parse_args()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.phase in ("all", "retrieval"):
        for dataset in args.datasets:
            semantic_multi_depth(dataset, args.device, args.force)
            lexical_multi_depth(dataset, args.device, args.force)
    if args.phase in ("all", "evaluation"):
        dataset_reports = []
        for dataset in args.datasets:
            dataset_report = evaluate_dataset(dataset)
            atomic_json(RESULT_ROOT / f"{dataset}_results.json", dataset_report)
            dataset_reports.append(dataset_report)
        report = {
            "experiment": "complete-corpus return-depth--communication--quality ablation",
            "status": "passed",
            "semantic_depths": list(SEMANTIC_DEPTHS),
            "lexical_depths": list(LEXICAL_DEPTHS),
            "operating_points": [list(point) for point in OPERATING_POINTS],
            "protocol_boundary": (
                "serialized two-path retrieval request and AEAD candidate response; "
                "transport framing and final evidence ciphertext are excluded"
            ),
            "method": (
                "actual stored-coordinate DPE prefixes are independently client-reranked; "
                "DuetRank architecture/backoff is frozen and parameters are refit only on "
                "the original calibration half"
            ),
            "datasets": dataset_reports,
        }
        atomic_json(RESULT_ROOT / "results.json", report)
        print(json.dumps(report, indent=2), flush=True)
    elif args.phase == "combine":
        dataset_reports = [
            json.loads((RESULT_ROOT / f"{dataset}_results.json").read_text(encoding="utf-8"))
            for dataset in args.datasets
        ]
        report = {
            "experiment": "complete-corpus return-depth--communication--quality ablation",
            "status": "passed",
            "semantic_depths": list(SEMANTIC_DEPTHS),
            "lexical_depths": list(LEXICAL_DEPTHS),
            "operating_points": [list(point) for point in OPERATING_POINTS],
            "protocol_boundary": (
                "serialized two-path retrieval request and AEAD candidate response; "
                "transport framing and final evidence ciphertext are excluded"
            ),
            "method": (
                "actual stored-coordinate DPE prefixes are independently client-reranked; "
                "DuetRank architecture/backoff is frozen and parameters are refit only on "
                "the original calibration half"
            ),
            "datasets": dataset_reports,
        }
        atomic_json(RESULT_ROOT / "results.json", report)
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
