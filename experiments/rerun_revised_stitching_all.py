"""Synchronized-query stitching attack on the revised B=16 lexical setting."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if (ROOT / ".deps").exists():
    sys.path.insert(0, str(ROOT / ".deps"))
os.environ.setdefault("DUETER_FORCE_CPU_DEPS", "1")

import numpy as np

from attack_lexical_stitching_d8192_rc32_b_sweep import run_configuration
from benchmark_million_lexical_dpe import query_mips
from benchmark_million_semantic_hybrid import atomic_json, ball_noise, dpe_transform, load_queries_qrels
from evaluate_lexical_d8192_rc32 import (
    BASE_SEED, GLOBAL_PERMUTATION, GLOBAL_SIGN1, GLOBAL_SIGN2,
    HASH_DIMENSION, NOISE_RADIUS, SCALE,
)
from rerun_revised_main import CONFIG, CELLS, lexical_document_frequencies


def run(dataset: str) -> dict[str, object]:
    cfg = CONFIG[dataset]
    started = time.perf_counter()
    _, texts, _, _ = load_queries_qrels(ROOT / "data" / cfg["data"], cfg["split"])
    texts = texts[:512]
    database = ROOT / "results" / "keyed_fts5" / cfg["fts"]
    df = lexical_document_frequencies(texts, database)
    plain = np.stack([query_mips(text, df, int(cfg["documents"]), HASH_DIMENSION) for text in texts])
    global_queries = SCALE * dpe_transform(plain, GLOBAL_SIGN1, GLOBAL_SIGN2, GLOBAL_PERMUTATION)
    global_queries += ball_noise(
        np.random.default_rng(BASE_SEED + 2027), len(plain), global_queries.shape[1], NOISE_RADIUS
    )
    configuration = run_configuration(global_queries, CELLS)
    report = {
        "experiment": "revised lexical synchronized-query stitching",
        "dataset": dataset,
        "threat_view": "server-visible fp16 local query coordinates and synchronization only; secret selectors used solely for scoring",
        "configuration": configuration,
        "elapsed_seconds": time.perf_counter() - started,
    }
    destination = ROOT / "results" / "security" / "revised_stitching_d8192_rc32_b16"
    destination.mkdir(parents=True, exist_ok=True)
    atomic_json(destination / f"{dataset}.json", report)
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(CONFIG), required=True)
    args = parser.parse_args()
    run(args.dataset)


if __name__ == "__main__":
    main()
