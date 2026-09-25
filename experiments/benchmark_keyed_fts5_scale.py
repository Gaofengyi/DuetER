"""Disk-backed million-document benchmark for the keyed lexical index.

This is an optimized exact-weight realization of the collision-free keyed term
index.  It HMACs every lexical term before inserting it into a contentless
SQLite FTS5 index.  The MIPS-to-L2 residual coordinate is absent because it is
orthogonal to every query and cannot affect lexical MIPS order.  Unlike the
in-memory prototype's 16-bit candidate weights, FTS5 evaluates BM25 directly;
the distinction is recorded in the output and must not be hidden in the paper.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import math
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_experiment import tokenize


ROOT = Path(__file__).resolve().parent
KEY = b"DuetDPE-keyed-vocabulary-tail-v2"


def load_queries_and_qrels(data_dir: Path) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    queries: dict[str, str] = {}
    with (data_dir / "queries.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            queries[str(row["_id"])] = str(row["text"])
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with (data_dir / "qrels" / "test.tsv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            qrels[str(row["query-id"])][str(row["corpus-id"])] = int(row["score"])
    return queries, dict(qrels)


class KeyedTerms:
    def __init__(self) -> None:
        self.cache: dict[str, str] = {}

    def term(self, value: str) -> str:
        encoded = self.cache.get(value)
        if encoded is None:
            encoded = hmac.new(KEY, value.encode("utf-8"), hashlib.sha256).hexdigest()[:32]
            self.cache[value] = encoded
        return encoded

    def text(self, value: str) -> str:
        return " ".join(self.term(token) for token in tokenize(value))

    def match_or(self, value: str) -> str:
        unique = list(dict.fromkeys(self.term(token) for token in tokenize(value)))
        return " OR ".join(unique)


def dcg(relevances: list[int]) -> float:
    return sum((2.0**rel - 1.0) / math.log2(rank + 2.0) for rank, rel in enumerate(relevances))


def evaluate_query(ranked: list[str], truth: dict[str, int], cap: int) -> tuple[float, float, float]:
    observed = [truth.get(doc_id, 0) for doc_id in ranked[:10]]
    ideal = sorted(truth.values(), reverse=True)[:10]
    denom = dcg(ideal)
    ndcg = dcg(observed) / denom if denom else 0.0
    relevant = {doc_id for doc_id, rel in truth.items() if rel > 0}
    recall = len(relevant.intersection(ranked[:100])) / len(relevant) if relevant else 0.0
    candidate_recall = len(relevant.intersection(ranked[:cap])) / len(relevant) if relevant else 0.0
    return ndcg, recall, candidate_recall


def build_index(
    data_dir: Path,
    db_path: Path,
    max_docs: int | None,
    relevant_ids: set[str],
    batch_size: int,
) -> dict[str, float | int | str]:
    if db_path.exists():
        raise FileExistsError(f"refusing to overwrite existing index: {db_path}")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=MEMORY;
        PRAGMA locking_mode=EXCLUSIVE;
        PRAGMA cache_size=-262144;
        CREATE TABLE docmap(rowid INTEGER PRIMARY KEY, doc_id TEXT UNIQUE NOT NULL);
        CREATE VIRTUAL TABLE postings USING fts5(body, content='', tokenize='unicode61');
        """
    )
    keyed = KeyedTerms()
    selected_relevant = 0
    selected_fillers = 0
    filler_budget = None if max_docs is None else max(0, max_docs - len(relevant_ids))
    rows: list[tuple[int, str, str]] = []
    total_tokens = 0
    started = time.perf_counter()
    next_rowid = 1
    with (data_dir / "corpus.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            doc_id = str(record["_id"])
            is_relevant = doc_id in relevant_ids
            if not is_relevant and filler_budget is not None and selected_fillers >= filler_budget:
                continue
            title = str(record.get("title", ""))
            body = f"{title} {record.get('text', '')}".strip()
            encrypted = keyed.text(body)
            total_tokens += 0 if not encrypted else encrypted.count(" ") + 1
            rows.append((next_rowid, doc_id, encrypted))
            next_rowid += 1
            if is_relevant:
                selected_relevant += 1
            else:
                selected_fillers += 1
            if len(rows) >= batch_size:
                connection.executemany("INSERT INTO docmap(rowid, doc_id) VALUES (?, ?)", ((r, d) for r, d, _ in rows))
                connection.executemany("INSERT INTO postings(rowid, body) VALUES (?, ?)", ((r, b) for r, _, b in rows))
                connection.commit()
                rows.clear()
                if (next_rowid - 1) % 100_000 < batch_size:
                    print(f"[fts5-build] {next_rowid - 1:,} documents", flush=True)
    if rows:
        connection.executemany("INSERT INTO docmap(rowid, doc_id) VALUES (?, ?)", ((r, d) for r, d, _ in rows))
        connection.executemany("INSERT INTO postings(rowid, body) VALUES (?, ?)", ((r, b) for r, _, b in rows))
        connection.commit()
    connection.execute("INSERT INTO postings(postings) VALUES ('optimize')")
    connection.execute("CREATE VIRTUAL TABLE vocab USING fts5vocab(postings, 'row')")
    connection.commit()
    elapsed = time.perf_counter() - started
    count = next_rowid - 1
    connection.close()
    return {
        "documents": count,
        "selected_relevant_documents": selected_relevant,
        "selected_filler_documents": selected_fillers,
        "tokens": total_tokens,
        "unique_plaintext_terms_seen_client_side": len(keyed.cache),
        "build_seconds": elapsed,
        "database_bytes": db_path.stat().st_size,
        "subset_policy": "all test-qrel documents plus earliest non-qrel corpus records" if max_docs else "complete corpus",
    }


def run_queries(
    db_path: Path,
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    cap: int,
    repeats: int,
    max_probe_terms: int,
) -> dict[str, object]:
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA cache_size=-262144")
    connection.execute("CREATE VIRTUAL TABLE IF NOT EXISTS vocab USING fts5vocab(postings, 'row')")
    connection.commit()
    keyed = KeyedTerms()
    query_ids = [qid for qid in queries if qid in qrels]
    latency_by_repeat: list[list[float]] = [[] for _ in range(repeats)]
    ndcg, recall, candidate_recall = [], [], []
    probe_counts, estimated_posting_reads = [], []
    for qid in query_ids:
        all_terms = list(dict.fromkeys(keyed.term(token) for token in tokenize(queries[qid])))
        frequencies = {
            str(term): int(doc_frequency)
            for term, doc_frequency in connection.execute(
                f"SELECT term, doc FROM vocab WHERE term IN ({','.join('?' for _ in all_terms)})",
                all_terms,
            ).fetchall()
        } if all_terms else {}
        matched_terms = sorted((term for term in all_terms if term in frequencies), key=lambda term: frequencies[term])
        probe_terms = matched_terms[:max_probe_terms] if max_probe_terms > 0 else matched_terms
        expression = " OR ".join(probe_terms)
        probe_counts.append(len(probe_terms))
        estimated_posting_reads.append(sum(frequencies[term] for term in probe_terms))
        if not expression:
            ranked: list[str] = []
        else:
            ranked = []
            for repeat in range(repeats):
                started = time.perf_counter()
                hits = connection.execute(
                    "SELECT d.doc_id FROM postings p JOIN docmap d ON d.rowid=p.rowid "
                    "WHERE postings MATCH ? ORDER BY bm25(postings) LIMIT ?",
                    (expression, cap),
                ).fetchall()
                latency_by_repeat[repeat].append((time.perf_counter() - started) * 1000.0)
                ranked = [str(row[0]) for row in hits]
        q_ndcg, q_recall, q_candidate_recall = evaluate_query(ranked, qrels[qid], cap)
        ndcg.append(q_ndcg)
        recall.append(q_recall)
        candidate_recall.append(q_candidate_recall)
        if len(ndcg) == 1 or len(ndcg) == len(query_ids) or len(ndcg) % 500 == 0:
            print(f"[fts5-query] {len(ndcg):,}/{len(query_ids):,}", flush=True)
    connection.close()
    repeat_means = [float(np.mean(row)) for row in latency_by_repeat]
    flat = [value for row in latency_by_repeat for value in row]
    return {
        "queries": len(query_ids),
        "cap": cap,
        "repeats": repeats,
        "max_rarest_probe_terms": max_probe_terms,
        "probe_terms_mean": float(np.mean(probe_counts)),
        "estimated_posting_entries_read_mean": float(np.mean(estimated_posting_reads)),
        "ndcg_at_10": float(np.mean(ndcg)),
        "recall_at_100": float(np.mean(recall)),
        "relevant_candidate_recall_at_cap": float(np.mean(candidate_recall)),
        "candidate_latency_repeat_mean_ms": repeat_means,
        "candidate_latency_mean_ms": float(np.mean(flat)),
        "candidate_latency_p50_ms": float(np.percentile(flat, 50)),
        "candidate_latency_p95_ms": float(np.percentile(flat, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--max-docs", type=int, default=1_000_000, help="0 indexes the complete corpus")
    parser.add_argument("--cap", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-probe-terms", type=int, default=0, help="0 probes every matched term")
    parser.add_argument("--reuse-index", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results" / "keyed_fts5")
    args = parser.parse_args()
    data_dir = ROOT / "data" / args.dataset
    queries, qrels = load_queries_and_qrels(data_dir)
    relevant_ids = {doc_id for truth in qrels.values() for doc_id, rel in truth.items() if rel > 0}
    max_docs = args.max_docs or None
    suffix = "full" if max_docs is None else str(max_docs)
    db_path = args.results_dir / f"{args.dataset}_{suffix}.sqlite3"
    base_output = args.results_dir / f"{args.dataset}_{suffix}.json"
    if args.reuse_index:
        if not db_path.is_file() or not base_output.is_file():
            raise FileNotFoundError("--reuse-index requires the existing database and base result JSON")
        build = json.loads(base_output.read_text(encoding="utf-8"))["build"]
    else:
        build = build_index(data_dir, db_path, max_docs, relevant_ids, args.batch_size)
    query = run_queries(db_path, queries, qrels, args.cap, args.repeats, args.max_probe_terms)
    result = {
        "dataset": args.dataset,
        "index_variant": "disk-backed collision-free HMAC term index with exact FTS5 BM25 weights; MIPS-to-L2 residual omitted",
        "build": build,
        "query": query,
        "leakage": "keyed term equality, posting lengths, search repetition, access pattern, result size (before padding)",
    }
    probe_suffix = "" if args.max_probe_terms == 0 else f"_p{args.max_probe_terms}"
    output = args.results_dir / f"{args.dataset}_{suffix}{probe_suffix}.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
