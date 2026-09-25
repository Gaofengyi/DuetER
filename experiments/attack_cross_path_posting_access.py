"""Cross-path linkage from synchronized posting/access-pattern traces.

The attacker uses only the concrete server view declared by DuetDPE: opaque
posting membership (PostInc), the postings touched by every query (Acc), and
cross-path query timing (Time).  Candidate/response identities, scores, ranks,
plaintext terms, and DPE coordinates are not attack features.

The contentless FTS5 prototype does not expose a convenient materialized
term--document table.  For evaluation we query FTS5's ``fts5vocab(instance)``
view, which is exactly the opaque posting incidence already visible to the
server.  HMAC labels are never inverted.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEPS = ROOT / ".deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))

import numpy as np
from scipy import sparse

from attack_cross_path_cooccurrence import (
    CONFIGS as BASE_CONFIGS,
    expected_matching_metrics,
    random_baseline,
    route_semantic_queries,
    sparse_cosine,
    summarize,
)
from benchmark_budgeted_lexical_candidates import keyed_term, select_terms_under_budget
from benchmark_million_semantic_hybrid import load_queries_qrels
from run_experiment import tokenize


CONFIGS = {
    "nq": {
        **BASE_CONFIGS["nq"],
        "data": ROOT / "data/nq",
        "split": "test",
        "database": ROOT / "results/keyed_fts5/nq_full.sqlite3",
        "budget": 50_000,
    },
    "hotpotqa": {
        **BASE_CONFIGS["hotpotqa"],
        "data": ROOT / "data/hotpotqa",
        "split": "test",
        "database": ROOT / "results/keyed_fts5/hotpotqa_full.sqlite3",
        "budget": 50_000,
    },
    "msmarco": {
        **BASE_CONFIGS["msmarco"],
        "data": ROOT / "data/msmarco",
        "split": "dev",
        "database": ROOT / "results/keyed_fts5/msmarco_full.sqlite3",
        "budget": 250_000,
    },
}


def semantic_access_traces(
    documents: np.ndarray,
    assignments: np.ndarray,
    routed: np.ndarray,
    cells: int,
    corpus_size: int,
) -> dict[str, sparse.csr_matrix]:
    """Build document-by-query traces from IVF PostInc and Acc only."""
    route_mask = np.zeros((len(routed), cells), dtype=bool)
    route_mask[np.arange(len(routed))[:, None], routed] = True
    doc_cells = np.asarray(assignments[documents], dtype=np.int64)
    posting_sizes = np.bincount(np.asarray(assignments).reshape(-1), minlength=cells)
    posting_weight = np.log1p(corpus_size / np.maximum(posting_sizes, 1)).astype(np.float32)

    count = np.zeros((len(documents), len(routed)), dtype=np.float32)
    rarity = np.zeros_like(count)
    for slot in range(doc_cells.shape[1]):
        hit = route_mask[:, doc_cells[:, slot]].T
        count += hit
        rarity += hit * posting_weight[doc_cells[:, slot], None]
    binary = (count > 0).astype(np.float32)
    return {
        "binary": sparse.csr_matrix(binary),
        "multiplicity": sparse.csr_matrix(count),
        "rarity": sparse.csr_matrix(rarity),
    }


def selected_lexical_postings(
    database: Path,
    query_texts: list[str],
    budget: int,
    corpus_size: int,
) -> tuple[list[list[str]], dict[str, int]]:
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA cache_size=-262144")
    connection.execute("CREATE VIRTUAL TABLE IF NOT EXISTS vocab USING fts5vocab(postings, 'row')")
    selected_by_query: list[list[str]] = []
    all_df: dict[str, int] = {}
    for query in query_texts:
        terms = list(dict.fromkeys(keyed_term(term) for term in tokenize(query)))
        if terms:
            placeholders = ",".join("?" for _ in terms)
            frequencies = {
                str(term): int(df)
                for term, df in connection.execute(
                    f"SELECT term, doc FROM vocab WHERE term IN ({placeholders})", terms
                ).fetchall()
            }
        else:
            frequencies = {}
        selected = select_terms_under_budget(
            frequencies, budget, alpha=0.5, document_count=corpus_size
        )
        selected_by_query.append(selected)
        all_df.update({term: frequencies[term] for term in selected})
    connection.close()
    return selected_by_query, all_df


def lexical_access_traces(
    database: Path,
    documents: np.ndarray,
    selected_by_query: list[list[str]],
    document_frequency: dict[str, int],
    corpus_size: int,
) -> dict[str, sparse.csr_matrix]:
    """Build exact HMAC-posting access traces for a sampled document pool."""
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA cache_size=-262144")
    connection.execute("CREATE VIRTUAL TABLE IF NOT EXISTS attack_vocab USING fts5vocab(postings, 'instance')")
    connection.execute("DROP TABLE IF EXISTS temp.attack_pool")
    connection.execute("CREATE TEMP TABLE attack_pool(rowid INTEGER PRIMARY KEY, local_id INTEGER NOT NULL)")
    connection.executemany(
        "INSERT INTO attack_pool(rowid, local_id) VALUES (?, ?)",
        ((int(doc) + 1, index) for index, doc in enumerate(documents)),
    )
    connection.commit()

    term_members: dict[str, np.ndarray] = {}
    unique_terms = sorted({term for selected in selected_by_query for term in selected})
    for index, term in enumerate(unique_terms):
        rows = connection.execute(
            "SELECT p.local_id FROM attack_vocab AS v "
            "JOIN attack_pool AS p ON p.rowid=v.doc WHERE v.term=?",
            (term,),
        ).fetchall()
        term_members[term] = np.asarray([row[0] for row in rows], dtype=np.int32)
        if index == 0 or (index + 1) % 1000 == 0 or index + 1 == len(unique_terms):
            print(f"  lexical postings {index + 1:,}/{len(unique_terms):,}", flush=True)
    connection.close()

    rows_out: list[np.ndarray] = []
    columns_out: list[np.ndarray] = []
    count_values: list[np.ndarray] = []
    rarity_values: list[np.ndarray] = []
    for query_index, selected in enumerate(selected_by_query):
        for term in selected:
            members = term_members[term]
            if not len(members):
                continue
            rows_out.append(members)
            columns_out.append(np.full(len(members), query_index, dtype=np.int32))
            count_values.append(np.ones(len(members), dtype=np.float32))
            weight = math.log1p(corpus_size / max(document_frequency[term], 1))
            rarity_values.append(np.full(len(members), weight, dtype=np.float32))
    shape = (len(documents), len(selected_by_query))
    if not rows_out:
        empty = sparse.csr_matrix(shape, dtype=np.float32)
        return {"binary": empty, "multiplicity": empty, "rarity": empty}
    row = np.concatenate(rows_out)
    column = np.concatenate(columns_out)
    count = sparse.coo_matrix((np.concatenate(count_values), (row, column)), shape=shape).tocsr()
    rarity = sparse.coo_matrix((np.concatenate(rarity_values), (row, column)), shape=shape).tocsr()
    count.sum_duplicates()
    rarity.sum_duplicates()
    binary = count.copy()
    binary.data[:] = 1.0
    return {"binary": binary, "multiplicity": count, "rarity": rarity}


def evaluate(
    semantic: dict[str, sparse.csr_matrix],
    lexical: dict[str, sparse.csr_matrix],
    row_indices: np.ndarray,
    query_order: np.ndarray,
    query_counts: list[int],
    alias_permutation: np.ndarray,
    rng: np.random.Generator,
) -> dict[tuple[int, str], dict[str, float]]:
    output: dict[tuple[int, str], dict[str, float]] = {}
    for requested in query_counts:
        count = min(requested, len(query_order))
        columns = query_order[:count]
        for method in ("binary", "multiplicity", "rarity"):
            first = semantic[method][row_indices][:, columns]
            second = lexical[method][row_indices][:, columns]
            similarity = sparse_cosine(first, second)
            metrics = expected_matching_metrics(similarity[:, alias_permutation], alias_permutation)
            metrics["semantic_coverage"] = float(np.mean(np.asarray(first.getnnz(axis=1)) > 0))
            metrics["lexical_coverage"] = float(np.mean(np.asarray(second.getnnz(axis=1)) > 0))
            output[(count, method)] = metrics

        # Negative control: destroy cross-path Time while preserving both path marginals.
        shuffled = columns[rng.permutation(len(columns))]
        first = semantic["rarity"][row_indices][:, columns]
        second = lexical["rarity"][row_indices][:, shuffled]
        similarity = sparse_cosine(first, second)
        metrics = expected_matching_metrics(similarity[:, alias_permutation], alias_permutation)
        metrics["semantic_coverage"] = float(np.mean(np.asarray(first.getnnz(axis=1)) > 0))
        metrics["lexical_coverage"] = float(np.mean(np.asarray(second.getnnz(axis=1)) > 0))
        output[(count, "rarity_time_shuffled_control")] = metrics
    return output


def run_dataset(name: str, args: argparse.Namespace) -> dict[str, object]:
    config = CONFIGS[name]
    started = time.perf_counter()
    cache_dir = Path(config["semantic_cache"])
    routed = route_semantic_queries(cache_dir, str(config["query_embeddings"]), args.probes)
    assignments = np.load(cache_dir / "semantic_assignments.npy", mmap_mode="r")
    cells = int(np.load(cache_dir / "semantic_centroids.npy", mmap_mode="r").shape[0])
    _, query_texts, _, _ = load_queries_qrels(Path(config["data"]), str(config["split"]))
    if len(query_texts) != len(routed):
        raise ValueError(f"{name}: lexical/semantic query order mismatch")

    selected_by_query, df = selected_lexical_postings(
        Path(config["database"]), query_texts, int(config["budget"]), int(config["documents"])
    )
    lexical_candidates = np.asarray(
        np.load(config["lexical_candidates"])[str(config["lexical_candidate_key"])], dtype=np.int32
    )
    eligible = np.unique(lexical_candidates[lexical_candidates >= 0]).astype(np.int32)
    if len(eligible) < args.pool_size:
        raise ValueError(f"{name}: insufficient posting-observed documents")

    trial_documents: list[np.ndarray] = []
    seeds: list[int] = []
    for repeat in range(args.repeats):
        seed = args.seed + 10007 * repeat + 101 * len(name)
        seeds.append(seed)
        rng = np.random.default_rng(seed)
        trial_documents.append(rng.choice(eligible, args.pool_size, replace=False).astype(np.int32))
    union_documents = np.unique(np.concatenate(trial_documents)).astype(np.int32)
    local = {int(doc): index for index, doc in enumerate(union_documents)}
    semantic = semantic_access_traces(
        union_documents, assignments, routed, cells, int(config["documents"])
    )
    lexical = lexical_access_traces(
        Path(config["database"]), union_documents, selected_by_query, df, int(config["documents"])
    )

    query_counts = sorted(set(min(value, len(routed)) for value in args.query_counts + [len(routed)]))
    collected: dict[tuple[int, str], list[dict[str, float]]] = {}
    for repeat, documents in enumerate(trial_documents):
        rng = np.random.default_rng(seeds[repeat])
        row_indices = np.asarray([local[int(doc)] for doc in documents], dtype=np.int32)
        query_order = rng.permutation(len(routed))
        alias_permutation = rng.permutation(args.pool_size)
        trial = evaluate(
            semantic, lexical, row_indices, query_order, query_counts, alias_permutation, rng
        )
        for key, metrics in trial.items():
            collected.setdefault(key, []).append(metrics)
        final = trial[(len(routed), "rarity")]
        print(
            f"{name:9s} repeat={repeat + 1}/{args.repeats} queries={len(routed)} "
            f"top1={final['top1']:.4f} top10={final['top10']:.4f} "
            f"mrr={final['mrr']:.4f}", flush=True
        )

    rows = []
    for (count, method), trials in sorted(collected.items()):
        rows.append({"queries": count, "method": method, "summary": summarize(trials), "trials": trials})
    return {
        "dataset": name,
        "documents": int(config["documents"]),
        "queries": len(routed),
        "posting_budget": int(config["budget"]),
        "semantic_probes": args.probes,
        "cohort": "documents observed in at least one lexical top-1000 candidate list",
        "eligible_documents_lower_bound": int(len(eligible)),
        "eligible_fraction_lower_bound": float(len(eligible) / int(config["documents"])),
        "pool_size": args.pool_size,
        "random_baseline": random_baseline(args.pool_size),
        "results": rows,
        "seconds": float(time.perf_counter() - started),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=sorted(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--query-counts", nargs="+", type=int, default=[10, 50, 100, 500, 1000])
    parser.add_argument("--pool-size", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--probes", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--output", type=Path, default=ROOT / "results/cross_path_posting_access.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = {
        "attack": "cross-path posting/access-pattern linkage",
        "attacker_view": "PostInc + Acc + Time only; no candidate/output ranks, plaintext, or DPE geometry",
        "configuration": vars(args) | {"output": str(args.output)},
        "datasets": [run_dataset(name, args) for name in args.datasets],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
