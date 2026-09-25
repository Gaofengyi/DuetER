#!/usr/bin/env python3
"""Prepare a deterministic FiQA test-query sample for Pisces and DuetER."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import sentence_transformers
import torch
import transformers
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


TOKENIZER_NAME = "bert-base-uncased"
MODEL_NAME = "ibm-granite/granite-embedding-small-english-r2"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    with path.open("wb") as handle:
        for row in rows:
            encoded = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
            handle.write(encoded)
            digest.update(encoded)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument(
        "--exclude-manifest", type=Path,
        help="Manifest whose query_ids are excluded, enabling disjoint extensions.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    qrels: dict[str, dict[str, float]] = defaultdict(dict)
    with args.qrels.open(encoding="utf-8") as handle:
        next(handle, None)
        for line in handle:
            query_id, document_id, score = line.rstrip("\n").split("\t")
            if float(score) > 0:
                qrels[str(query_id)][str(document_id)] = float(score)
    excluded: set[str] = set()
    if args.exclude_manifest:
        excluded = set(map(str, json.loads(args.exclude_manifest.read_text(encoding="utf-8"))["query_ids"]))
    eligible = np.asarray(sorted(set(qrels) - excluded), dtype=object)
    if args.sample_size > len(eligible):
        raise ValueError(f"requested {args.sample_size} queries from only {len(eligible)} judged queries")
    rng = np.random.default_rng(args.seed)
    selected = set(map(str, rng.choice(eligible, size=args.sample_size, replace=False)))

    query_map = {str(row["_id"]): str(row["text"]) for row in read_jsonl(args.queries)}
    missing = selected - query_map.keys()
    if missing:
        raise ValueError(f"missing {len(missing)} sampled query texts")
    selected_ids = sorted(selected, key=lambda value: int(value))
    raw_rows = [{"id": int(query_id), "input": query_map[query_id]} for query_id in selected_ids]

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    bm25_rows = []
    for row in raw_rows:
        token_ids = tokenizer(row["input"], add_special_tokens=True, truncation=False)["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=True)
        bm25_rows.append({
            **row,
            "query_bert_input_ids": token_ids,
            "query_bert_tokens": tokens,
            "query_bert_token_counts": dict(Counter(tokens)),
        })

    model = SentenceTransformer(MODEL_NAME, device=args.device)
    embeddings = np.asarray(
        model.encode([row["input"] for row in raw_rows], batch_size=args.batch_size,
                     show_progress_bar=True, normalize_embeddings=True),
        dtype=np.float32,
    )
    sim_rows = [
        {**row, "query_embeddings_norm": embedding.tolist()}
        for row, embedding in zip(raw_rows, embeddings, strict=True)
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    hashes = {
        "bm25_query.jsonl": write_jsonl(args.output_dir / "bm25_query.jsonl", bm25_rows),
        "sim_query.jsonl": write_jsonl(args.output_dir / "sim_query.jsonl", sim_rows),
    }
    with (args.output_dir / "qrels.tsv").open("w", encoding="utf-8", newline="") as handle:
        handle.write("query-id\tcorpus-id\tscore\n")
        for query_id in selected_ids:
            for document_id, score in sorted(qrels[query_id].items()):
                handle.write(f"{query_id}\t{document_id}\t{score:g}\n")
    manifest = {
        "dataset": "BEIR FiQA test",
        "sampling": "uniform without replacement over judged test queries after exclusions",
        "sample_size": len(selected_ids),
        "seed": args.seed,
        "excluded_query_count": len(excluded),
        "query_ids": selected_ids,
        "tokenizer": TOKENIZER_NAME,
        "embedding_model": MODEL_NAME,
        "device": args.device,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "sha256": hashes,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: manifest[k] for k in ("dataset", "sample_size", "seed", "device", "cuda_device")}, indent=2))


if __name__ == "__main__":
    main()
