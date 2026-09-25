#!/usr/bin/env python3
"""Convert a BEIR corpus and query file to the raw Pisces JSONL schema.

The official Pisces preprocessor expects CLAPNQ-style ``content`` and
``input`` fields.  This converter changes field names only; it preserves the
document/query identifiers and the complete text of every selected record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_records(path: Path, records: Iterable[dict[str, Any]]) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    with path.open("wb") as handle:
        for record in records:
            encoded = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
            handle.write(encoded)
            digest.update(encoded)
            count += 1
    return count, digest.hexdigest()


def corpus_records(path: Path, limit: int) -> Iterable[dict[str, str]]:
    for index, record in enumerate(iter_jsonl(path)):
        if index == limit:
            break
        title = str(record.get("title", "")).strip()
        text = str(record.get("text", "")).strip()
        content = f"{title}\n{text}" if title else text
        yield {"context_id": str(record["_id"]), "content": content}


def query_records(path: Path, limit: int) -> Iterable[dict[str, str | int]]:
    for index, record in enumerate(iter_jsonl(path)):
        if index == limit:
            break
        query_id = str(record["_id"])
        # Pisces parses query IDs as JSON numbers (the CLAPNQ IDs are uint64),
        # whereas BEIR stores even numeric IDs as strings.
        if not query_id.isdecimal():
            raise ValueError(f"Pisces requires a numeric query ID, got {query_id!r}")
        yield {"id": int(query_id), "input": str(record["text"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--corpus-limit", type=int, required=True)
    parser.add_argument("--query-limit", type=int, default=3)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    corpus_count, corpus_hash = write_records(
        args.output_dir / "corpus.jsonl", corpus_records(args.corpus, args.corpus_limit)
    )
    query_count, query_hash = write_records(
        args.output_dir / "queries.jsonl", query_records(args.queries, args.query_limit)
    )
    if corpus_count != args.corpus_limit:
        raise ValueError(f"corpus has {corpus_count} records; requested {args.corpus_limit}")
    if query_count != args.query_limit:
        raise ValueError(f"query file has {query_count} records; requested {args.query_limit}")

    manifest = {
        "source_format": "BEIR",
        "source_corpus": str(args.corpus.resolve()),
        "source_queries": str(args.queries.resolve()),
        "corpus_count": corpus_count,
        "query_count": query_count,
        "conversion": "title + newline + text; field renaming only",
        "sha256": {"corpus.jsonl": corpus_hash, "queries.jsonl": query_hash},
    }
    (args.output_dir / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
