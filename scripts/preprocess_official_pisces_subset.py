#!/usr/bin/env python3
"""Prepare a deterministic CLAPNQ subset for the official Pisces benchmarks.

The upstream preprocessor batches variable-length strings with
``return_tensors="np"`` but without padding, which recent Transformers releases
reject.  This compatibility driver performs the same BERT tokenization one
record at a time and uses the same Granite embedding model and L2
normalization.  It does not modify the Pisces cryptographic benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import Counter
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


def read_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
                if len(records) == limit:
                    break
    if len(records) != limit:
        raise ValueError(f"{path} has only {len(records)} records; need {limit}")
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    with path.open("wb") as handle:
        for record in records:
            encoded = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
            handle.write(encoded)
            digest.update(encoded)
    return digest.hexdigest()


def add_corpus_tokens(records: list[dict[str, Any]], tokenizer: Any) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        token_ids = tokenizer(item["content"], add_special_tokens=True, truncation=False)["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=True)
        item["bert_input_ids"] = token_ids
        item["bert_tokens"] = tokens
        item["bert_token_counts"] = dict(Counter(tokens))
        output.append(item)
    return output


def add_query_tokens(records: list[dict[str, Any]], tokenizer: Any) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        token_ids = tokenizer(item["input"], add_special_tokens=True, truncation=False)["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=True)
        item["query_bert_input_ids"] = token_ids
        item["query_bert_tokens"] = tokens
        item["query_bert_token_counts"] = dict(Counter(tokens))
        output.append(item)
    return output


def normalized_embeddings(model: SentenceTransformer, texts: list[str], batch_size: int) -> np.ndarray:
    embeddings = np.asarray(
        model.encode(texts, show_progress_bar=True, batch_size=batch_size),
        dtype=np.float32,
    )
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Granite emitted a zero-norm embedding")
    return embeddings / norms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--corpus-limit", type=int, default=256)
    parser.add_argument("--query-limit", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    corpus = read_jsonl(args.corpus, args.corpus_limit)
    queries = read_jsonl(args.queries, args.query_limit)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    bm25_corpus = add_corpus_tokens(corpus, tokenizer)
    bm25_queries = add_query_tokens(queries, tokenizer)

    model = SentenceTransformer(MODEL_NAME, device=args.device)
    corpus_embeddings = normalized_embeddings(model, [row["content"] for row in corpus], args.batch_size)
    query_embeddings = normalized_embeddings(model, [row["input"] for row in queries], args.batch_size)

    sim_corpus: list[dict[str, Any]] = []
    for record, embedding in zip(corpus, corpus_embeddings, strict=True):
        item = dict(record)
        item["context_embeddings_norm"] = embedding.tolist()
        sim_corpus.append(item)

    sim_queries: list[dict[str, Any]] = []
    for record, embedding in zip(queries, query_embeddings, strict=True):
        item = dict(record)
        item["query_embeddings_norm"] = embedding.tolist()
        sim_queries.append(item)

    outputs = {
        "bm25_corpus.jsonl": bm25_corpus,
        "bm25_query.jsonl": bm25_queries,
        "sim_corpus.jsonl": sim_corpus,
        "sim_query.jsonl": sim_queries,
    }
    hashes = {name: write_jsonl(args.output_dir / name, rows) for name, rows in outputs.items()}
    manifest = {
        "source_corpus": str(args.corpus.resolve()),
        "source_queries": str(args.queries.resolve()),
        "corpus_count": len(corpus),
        "query_count": len(queries),
        "tokenizer": TOKENIZER_NAME,
        "embedding_model": MODEL_NAME,
        "embedding_dimension": int(corpus_embeddings.shape[1]),
        "device": args.device,
        "cuda_available": torch.cuda.is_available(),
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
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
