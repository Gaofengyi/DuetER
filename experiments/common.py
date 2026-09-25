"""Shared utilities required by the released DuetER experiments."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norms, eps)).astype(np.float32)


def top_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, scores.shape[0])
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    if k == scores.shape[0]:
        return np.argsort(-scores, kind="stable")
    chosen = np.argpartition(-scores, k - 1)[:k]
    return chosen[np.argsort(-scores[chosen], kind="stable")]


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def load_beir(
    data_dir: Path, split: str = "test"
) -> tuple[list[str], list[str], list[str], list[str], dict[str, dict[str, int]]]:
    corpus_rows = load_jsonl(data_dir / "corpus.jsonl")
    query_rows = load_jsonl(data_dir / "queries.jsonl")
    corpus_ids = [str(row["_id"]) for row in corpus_rows]
    corpus_texts = [
        f"{row.get('title', '')}. {row.get('text', '')}".strip()
        for row in corpus_rows
    ]
    query_lookup = {str(row["_id"]): row["text"] for row in query_rows}
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with (data_dir / "qrels" / f"{split}.tsv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            qrels[str(row["query-id"])][str(row["corpus-id"])] = int(row["score"])
    query_ids = [qid for qid in query_lookup if qid in qrels]
    query_texts = [query_lookup[qid] for qid in query_ids]
    return corpus_ids, corpus_texts, query_ids, query_texts, dict(qrels)


def fwht_batch(x: np.ndarray) -> np.ndarray:
    """Normalized Walsh--Hadamard transform over the final dimension."""
    y = np.asarray(x, dtype=np.float32).copy()
    n = y.shape[1]
    if n & (n - 1):
        raise ValueError("FWHT dimension must be a power of two")
    h = 1
    while h < n:
        y = y.reshape(y.shape[0], -1, 2 * h)
        left = y[:, :, :h].copy()
        right = y[:, :, h : 2 * h].copy()
        y[:, :, :h] = left + right
        y[:, :, h : 2 * h] = left - right
        y = y.reshape(y.shape[0], n)
        h *= 2
    return y / math.sqrt(n)


def cooccurrence_link_attack(
    semantic_results: list[np.ndarray],
    lexical_results: list[np.ndarray],
    sem_alias: np.ndarray,
    lex_alias: np.ndarray,
    attack_depth: int,
) -> dict[str, float | int]:
    counts: dict[int, Counter[int]] = defaultdict(Counter)
    for semantic_docs, lexical_docs in zip(semantic_results, lexical_results):
        for semantic_alias in sem_alias[semantic_docs[:attack_depth]]:
            counts[int(semantic_alias)].update(
                int(value) for value in lex_alias[lexical_docs[:attack_depth]]
            )
    inverse_semantic = {int(alias): idx for idx, alias in enumerate(sem_alias)}
    correct = 0
    for semantic_alias, counter in counts.items():
        if counter:
            predicted = counter.most_common(1)[0][0]
            correct += int(
                predicted == int(lex_alias[inverse_semantic[semantic_alias]])
            )
    evaluated = sum(bool(counter) for counter in counts.values())
    return {
        "evaluated_semantic_aliases": evaluated,
        "top1_link_accuracy": correct / evaluated if evaluated else 0.0,
    }
