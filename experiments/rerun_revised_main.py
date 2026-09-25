"""Re-run the revised lexical CCADPE operating point on all paper datasets.

Fixed lexical parameters: hashed input 1024, padded DPE d=8192, local r_c=32,
B=16.  The per-cell depths preserve the paper's original total return budget:
64 (NQ), 32 (HotpotQA), and 64 (MS MARCO dev).
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if (ROOT / ".deps").exists():
    sys.path.insert(0, str(ROOT / ".deps"))
os.environ.setdefault("DUETER_FORCE_CPU_DEPS", "1")

import numpy as np

from benchmark_dual_compartment_full import (
    KEY_LEXICAL,
    cell_transform,
    metric_mean,
    relevant_coverage,
)
from benchmark_exact_bm25_client_rerank import (
    build_candidate_term_matrix,
    exact_bm25_rerank,
    load_document_frequencies,
    load_document_lengths,
    query_vocabulary,
    selected_document_rows,
)
from benchmark_million_lexical_dpe import query_mips
from benchmark_million_semantic_hybrid import (
    LEXICAL_KEY,
    atomic_json,
    load_queries_qrels,
    read_ids,
)
from evaluate_lexical_d8192_rc32 import (
    BASE_SEED,
    BETA,
    GLOBAL_PERMUTATION,
    GLOBAL_SIGN1,
    GLOBAL_SIGN2,
    HASH_DIMENSION,
    NOISE_RADIUS,
    PROJECTION_DIMENSION,
    SCALE,
    WORK_DIMENSION,
    selected_ball_noise,
    selected_dpe_matrix,
)
from evaluate_lexical_d8192_rc32_b_sweep import build_local_cache, retrieve
from experiment_candidate_rank_fusion import run_dataset as run_duetrank
from dueter_common import tokenize


CELLS = 16
CONFIG = {
    "nq": {
        "data": "nq", "split": "test", "documents": 2_681_468,
        "source": "nq_full_h1024", "raw": "nq_full.raw.npz",
        "budget": 50_000, "local_depth": 64, "result": "nq",
        "fts": "nq_full.sqlite3", "old_local_depth": 16,
    },
    "hotpotqa": {
        "data": "hotpotqa", "split": "test", "documents": 5_233_329,
        "source": "hotpotqa_full_h1024", "raw": "hotpotqa_full.raw.npz",
        "budget": 50_000, "local_depth": 32, "result": "hotpotqa",
        "fts": "hotpotqa_full.sqlite3", "old_local_depth": 8,
    },
    "msmarco": {
        "data": "msmarco", "split": "dev", "documents": 8_841_823,
        "source": "msmarco_dev_b250k_full_h1024", "raw": "msmarco_dev_full.raw.npz",
        "budget": 250_000, "local_depth": 64, "result": "msmarco_dev",
        "fts": "msmarco_full.sqlite3", "old_local_depth": 16,
    },
}


def keyed_term(value: str) -> str:
    return hmac.new(LEXICAL_KEY, value.encode("utf-8"), hashlib.sha256).hexdigest()[:32]


def lexical_document_frequencies(query_texts: list[str], database: Path) -> dict[str, int]:
    tokens = sorted({keyed_term(term) for text in query_texts for term in dict.fromkeys(tokenize(text))})
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    result: dict[str, int] = {}
    for start in range(0, len(tokens), 800):
        batch = tokens[start:start + 800]
        placeholders = ",".join("?" for _ in batch)
        for token, frequency in connection.execute(
            f"SELECT term, doc FROM vocab WHERE term IN ({placeholders})", batch
        ):
            result[str(token)] = int(frequency)
    connection.close()
    return result


def build_local_queries(query_plain: np.ndarray, transforms: list) -> np.ndarray:
    union = np.unique(np.concatenate([item.coordinates for item in transforms])).astype(np.int32)
    matrix = selected_dpe_matrix(
        union, GLOBAL_SIGN1, GLOBAL_SIGN2, GLOBAL_PERMUTATION, query_plain.shape[1]
    )
    started = time.perf_counter()
    selected = SCALE * (query_plain @ matrix.T)
    selected += selected_ball_noise(
        np.random.default_rng(BASE_SEED + 2027), len(query_plain), len(union)
    )
    lookup = {int(value): index for index, value in enumerate(union)}
    output = np.empty((len(query_plain), len(transforms), PROJECTION_DIMENSION), dtype=np.float32)
    for cell, transform in enumerate(transforms):
        positions = np.asarray([lookup[int(value)] for value in transform.coordinates], dtype=np.int32)
        values = (
            selected[:, positions] * transform.signs * transform.multiplier
            + transform.translation
        )
        output[:, cell] = values.astype(np.float16).astype(np.float32)
    return output, {
        "mean_ms_per_query": (time.perf_counter() - started) * 1000.0 / len(query_plain),
        "unique_visible_latent_coordinates": int(len(union)),
    }


def communication(local_depth: int, token_lengths: np.ndarray, selected_rows: np.ndarray,
                  exact: np.ndarray) -> dict[str, float | int]:
    # Semantic path is unchanged: 128 probes, r_c,s=256, 16 returned per cell
    # for NQ and HotpotQA and 32 for MS MARCO (patched by the caller below).
    positions = np.searchsorted(selected_rows, exact[exact >= 0])
    plaintext = 4 + 4 * np.asarray(token_lengths[positions], dtype=np.int64)
    bucketed = ((plaintext + 255) // 256) * 256
    average_bucket = float(np.mean(bucketed))
    lexical_records = CELLS * local_depth
    lexical_upload = CELLS * (16 + PROJECTION_DIMENSION * 2)
    lexical_record_bytes = 92.0 + average_bucket
    return {
        "lexical_upload_bytes": int(lexical_upload),
        "lexical_padded_records": int(lexical_records),
        "mean_plaintext_payload_bytes": float(np.mean(plaintext)),
        "mean_padded_payload_bytes": average_bucket,
        "lexical_download_bytes": int(round(lexical_records * lexical_record_bytes)),
        "lexical_total_bytes": int(round(lexical_upload + lexical_records * lexical_record_bytes)),
        "lexical_total_mib": float((lexical_upload + lexical_records * lexical_record_bytes) / 1024**2),
    }


def previous_metrics(name: str) -> dict[str, float] | None:
    path = ROOT / "results" / "exact_bm25_client_rerank" / name / "results.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    return (
        report.get("lexical_metrics_exact_bm25")
        or report.get("lexical_exact_bm25_metrics")
        or report.get("exact_bm25_metrics")
    )


def run_one(dataset: str) -> dict[str, object]:
    cfg = CONFIG[dataset]
    started = time.perf_counter()
    destination = ROOT / "results" / "revised_d8192_rc32_b16" / cfg["result"]
    destination.mkdir(parents=True, exist_ok=True)
    source = ROOT / "cache" / "full_candidate_lexical_dpe" / cfg["source"]
    cache = ROOT / "cache" / "revised_d8192_rc32_b16" / cfg["result"]
    # Reuse the already-audited NQ B=16 cache from the preliminary sweep.
    if dataset == "nq":
        old = ROOT / "cache" / "lexical_d8192_rc32_b_sweep" / "nq_2681468_b16"
        if old.exists():
            cache = old
    database = ROOT / "results" / "keyed_fts5" / cfg["fts"]
    query_ids, query_texts, qrels, _ = load_queries_qrels(
        ROOT / "data" / cfg["data"], cfg["split"]
    )
    print(f"[{dataset}] queries={len(query_ids):,}; preparing transforms", flush=True)
    transforms = [cell_transform(KEY_LEXICAL, cell, WORK_DIMENSION, PROJECTION_DIMENSION) for cell in range(CELLS)]
    matrices = [
        selected_dpe_matrix(item.coordinates, GLOBAL_SIGN1, GLOBAL_SIGN2, GLOBAL_PERMUTATION, HASH_DIMENSION + 1)
        for item in transforms
    ]
    setup = build_local_cache(source, cache, CELLS, transforms, matrices)
    df = lexical_document_frequencies(query_texts, database)
    query_plain = np.stack([
        query_mips(text, df, int(cfg["documents"]), HASH_DIMENSION) for text in query_texts
    ])
    local_queries, query_timing = build_local_queries(query_plain, transforms)
    raw = np.asarray(
        np.load(ROOT / "results" / "budgeted_lexical" / cfg["raw"])[f"budget_{cfg['budget']}"][:, :1000],
        dtype=np.int32,
    )
    cloud, server_timing = retrieve(source, cache, raw, local_queries, int(cfg["local_depth"]))
    np.savez_compressed(destination / "cloud_candidates.npz", lexical=cloud)

    rankings = cloud[None, :, :]
    rows = selected_document_rows(rankings)
    terms, term_to_index = query_vocabulary(query_texts)
    exact_root = ROOT / "cache" / "exact_bm25_client_rerank" / f"{cfg['result']}_d8192_rc32_b16_l{cfg['local_depth']}"
    full_lengths = load_document_lengths(
        database,
        ROOT / "cache" / "exact_bm25_client_rerank" / cfg["result"] / "full_doc_lengths_v2.npy",
    )
    exact_df = load_document_frequencies(database, terms)
    token_lengths, indptr, indices, counts = build_candidate_term_matrix(
        ROOT / "data" / cfg["data"] / "corpus.jsonl", rows, term_to_index, exact_root
    )
    exact, exact_timing = exact_bm25_rerank(
        rankings, rows, token_lengths, indptr, indices, counts,
        query_texts, term_to_index, exact_df, len(full_lengths), float(np.mean(full_lengths)),
    )
    exact_path = destination / "exact_bm25_rankings.npz"
    np.savez_compressed(exact_path, lexical_dpe=exact, lexical=exact)
    doc_ids = read_ids(ROOT / "cache" / "full_semantic" / f"{cfg['data']}_{cfg['documents']}" / "doc_ids.txt")
    cloud_metrics = metric_mean(rankings, doc_ids, query_ids, qrels)
    exact_metrics = metric_mean(exact, doc_ids, query_ids, qrels)
    coverage = relevant_coverage(cloud, doc_ids, query_ids, qrels)
    baseline = previous_metrics(str(cfg["result"]))

    print(f"[{dataset}] fitting/evaluating DuetRank", flush=True)
    fusion = run_duetrank(
        str(cfg["data"]), lexical_source="revised_d8192_rc32_b16", scale="full",
        lexical_rankings_path=exact_path, split=str(cfg["split"]),
        semantic_rankings_path=ROOT / "results" / "dual_compartment_full_p256" / cfg["result"] / "semantic_compartment.rankings.npz",
        trace_path=ROOT / "results" / "budgeted_lexical" / str(cfg["raw"]).replace(".raw.npz", ".trace.json"),
        budget_key=int(cfg["budget"]), model_tag="d8192-r32-b16",
    )
    comm = communication(int(cfg["local_depth"]), token_lengths, rows, exact)
    semantic_local_depth = 32 if dataset == "msmarco" else 16
    semantic_records = 128 * semantic_local_depth
    semantic_upload = 128 * (16 + 256 * 2)
    semantic_record_bytes = 16 + 12 + (384 * 2 + 16) + 12 + (20 + 16)
    total_bytes = semantic_upload + semantic_records * semantic_record_bytes + comm["lexical_total_bytes"] + 48
    comm.update({
        "semantic_upload_bytes": semantic_upload,
        "semantic_padded_records": semantic_records,
        "combined_total_bytes": int(total_bytes),
        "combined_total_mib": float(total_bytes / 1024**2),
    })
    retention = None if baseline is None else {
        key: float(exact_metrics[key] / baseline[key]) for key in exact_metrics if key in baseline
    }
    report = {
        "experiment": "revised lexical main operating point",
        "dataset": dataset,
        "documents": int(cfg["documents"]), "queries": len(query_ids),
        "parameters": {
            "hash_dimension": HASH_DIMENSION, "dpe_work_dimension": WORK_DIMENSION,
            "lexical_projection_dimension": PROJECTION_DIMENSION, "partitions": CELLS,
            "local_depth_per_partition": int(cfg["local_depth"]),
            "maximum_local_return_budget": CELLS * int(cfg["local_depth"]),
            "posting_budget": int(cfg["budget"]), "returned_depth": 300,
            "beta": BETA, "scale": SCALE, "seed": BASE_SEED,
        },
        "cache_setup": setup, "query_preparation_timing": query_timing,
        "server_timing": server_timing, "exact_bm25_latency": exact_timing,
        "cloud_candidate_metrics": cloud_metrics,
        "cloud_relevant_candidate_coverage": coverage,
        "exact_bm25_metrics": exact_metrics,
        "paper_baseline_exact_bm25_metrics": baseline,
        "retention_vs_paper_configuration": retention,
        "candidate_documents_materialized": int(len(rows)),
        "communication": comm, "duetrank": fusion,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(destination / "results.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["all", *CONFIG], default="all")
    args = parser.parse_args()
    names = list(CONFIG) if args.dataset == "all" else [args.dataset]
    output = {name: run_one(name) for name in names}
    if len(output) > 1:
        atomic_json(ROOT / "results" / "revised_d8192_rc32_b16" / "summary.json", output)


if __name__ == "__main__":
    main()
