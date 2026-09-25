"""DPE margin, quantization, and dummy-access ablations on BEIR SciFact."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from lexical_mips_index import KeyedBM25TailIndex
from dueter_common import (
    BM25Index,
    ConditionalDPE,
    evaluate,
    load_beir,
    mips_document_transform,
    mips_query_transform,
    neighbor_overlap,
    normalize_rows,
    rrf,
    tokenize,
    top_indices,
)


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "scifact"
CACHE = ROOT / "cache"
RESULTS = ROOT / "results" / "ablations"
SEED = 20260822


def dpe_margin_sweep(
    corpus_ids: list[str], corpus_texts: list[str], query_ids: list[str],
    query_texts: list[str], qrels: dict[str, dict[str, int]], bm25: BM25Index,
    corpus_embedding_path: Path, query_embedding_path: Path,
) -> list[dict]:
    dense_docs = normalize_rows(np.load(corpus_embedding_path))
    dense_queries = normalize_rows(np.load(query_embedding_path))
    hashed = bm25.hashed_document_vectors(1024, b"DuetDPE-Lexical-v1")
    lexical_docs, _ = mips_document_transform(hashed)
    lexical_queries = np.stack([
        mips_query_transform(bm25.hashed_query_vector(query, 1024, b"DuetDPE-Lexical-v1"))
        for query in query_texts
    ])
    rows = []
    for beta in (0.0, 0.05, 0.10, 0.20):
        sem = ConditionalDPE(dense_docs.shape[1], beta, 3.0, SEED + 1)
        lex = ConditionalDPE(lexical_docs.shape[1], beta, 3.0, SEED + 2)
        start = time.perf_counter()
        sem_db, lex_db = sem.encrypt_database(dense_docs), lex.encrypt_database(lexical_docs)
        sem_q, lex_q = sem.encrypt_queries(dense_queries), lex.encrypt_queries(lexical_queries)
        setup_seconds = time.perf_counter() - start
        rankings = {"semantic": {}, "lexical": {}, "dual": {}}
        for qi, (qid, query) in enumerate(zip(query_ids, query_texts)):
            sem_candidates = top_indices(-np.linalg.norm(sem_db - sem_q[qi], axis=1), 200)
            lex_candidates = top_indices(-np.linalg.norm(lex_db - lex_q[qi], axis=1), 200)
            sem_rank = sem_candidates[top_indices(dense_docs[sem_candidates] @ dense_queries[qi], 100)]
            bm25_scores = bm25.score(query)
            lex_rank = lex_candidates[top_indices(bm25_scores[lex_candidates], 100)]
            rankings["semantic"][qid] = sem_rank.tolist()
            rankings["lexical"][qid] = lex_rank.tolist()
            rankings["dual"][qid] = rrf(sem_rank, lex_rank, 100, 0.5)
        rows.append({
            "beta": beta,
            "setup_and_query_encrypt_seconds": setup_seconds,
            "semantic": evaluate(rankings["semantic"], query_ids, corpus_ids, qrels),
            "lexical": evaluate(rankings["lexical"], query_ids, corpus_ids, qrels),
            "dual": evaluate(rankings["dual"], query_ids, corpus_ids, qrels),
            "semantic_db_top10_neighbor_overlap": neighbor_overlap(dense_docs, sem_db, 50, 10, SEED + 31),
            "lexical_db_top10_neighbor_overlap": neighbor_overlap(lexical_docs, lex_db, 50, 10, SEED + 32),
        })
    return rows


def index_sweeps(bm25: BM25Index, query_texts: list[str]) -> dict:
    quantization = []
    for bits in (8, 12, 16):
        index = KeyedBM25TailIndex(b"DuetDPE-keyed-vocabulary-tail-v2", bits)
        index.build(bm25.postings, bm25.idf, bm25.doc_len, bm25.avgdl, bm25.n_docs, bm25.k1, bm25.b)
        for cap in (100, 200, 400):
            coverage = []
            for query in query_texts:
                scores = bm25.score(query)
                positive = top_indices(scores, min(100, int(np.count_nonzero(scores > 0))))
                candidates, _ = index.candidates(tokenize(query), cap)
                coverage.append(len(set(map(int, positive)).intersection(map(int, candidates))) / len(positive))
            quantization.append({"bits": bits, "cap": cap, "positive_top100_coverage": float(np.mean(coverage))})

    base = KeyedBM25TailIndex(b"DuetDPE-keyed-vocabulary-tail-v2", 16)
    base.build(bm25.postings, bm25.idf, bm25.doc_len, bm25.avgdl, bm25.n_docs, bm25.k1, bm25.b)
    vocabulary = np.asarray(list(bm25.idf), dtype=object)
    dummy_rows = []
    for dummy_count in (0, 2, 4, 8, 16):
        coverage, sizes, reads, jaccards = [], [], [], []
        rng = np.random.default_rng(SEED + dummy_count)
        for query in query_texts:
            terms = tokenize(query)
            base_candidates, _ = base.candidates(terms, 1600)
            dummies = [str(x) for x in rng.choice(vocabulary, dummy_count, replace=False)] if dummy_count else []
            candidates, trace = base.candidates(terms + dummies, 1600)
            exact_scores = bm25.score(query)
            positive = top_indices(exact_scores, min(100, int(np.count_nonzero(exact_scores > 0))))
            coverage.append(len(set(map(int, positive)).intersection(map(int, candidates))) / len(positive))
            sizes.append(len(candidates))
            reads.append(trace["posting_entries_read"])
            union = set(map(int, base_candidates)).union(map(int, candidates))
            jaccards.append(len(set(map(int, base_candidates)).intersection(map(int, candidates))) / len(union))
        dummy_rows.append({
            "dummy_tokens": dummy_count,
            "positive_top100_coverage": float(np.mean(coverage)),
            "candidate_count_mean": float(np.mean(sizes)),
            "posting_entries_read_mean": float(np.mean(reads)),
            "candidate_set_jaccard_with_unpadded_mean": float(np.mean(jaccards)),
        })
    return {"quantization_and_cap": quantization, "dummy_access_padding": dummy_rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-cache-dir", type=Path, default=CACHE)
    parser.add_argument("--cache-tag", default="scifact-st-5f4291fb760b")
    parser.add_argument("--model-label", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--output-name", default="scifact")
    args = parser.parse_args()
    corpus_ids, corpus_texts, query_ids, query_texts, qrels = load_beir(DATA)
    bm25 = BM25Index(corpus_texts)
    corpus_embedding_path = args.embedding_cache_dir / f"{args.cache_tag}_corpus_embeddings.npy"
    query_embedding_path = args.embedding_cache_dir / f"{args.cache_tag}_query_embeddings.npy"
    output = {
        "dataset": "scifact",
        "dense_model": args.model_label,
        "dpe_margin": dpe_margin_sweep(
            corpus_ids, corpus_texts, query_ids, query_texts, qrels, bm25,
            corpus_embedding_path, query_embedding_path,
        ),
        "index": index_sweeps(bm25, query_texts),
        "warning": "Dummy-token padding is an engineering access-pattern defense, not a proven DP mechanism.",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{args.output_name}.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
