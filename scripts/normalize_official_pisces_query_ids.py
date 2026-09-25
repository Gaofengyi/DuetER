#!/usr/bin/env python3
"""Normalize numeric-string query IDs in generated Pisces JSONL artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def normalize(path: Path) -> str:
    normalized: list[bytes] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        query_id = str(record["id"])
        if not query_id.isdecimal():
            raise ValueError(f"Pisces requires a numeric query ID, got {query_id!r}")
        record["id"] = int(query_id)
        normalized.append((json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
    content = b"".join(normalized)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    hashes: dict[str, str] = {}
    for path in args.paths:
        digest = normalize(path)
        hashes[path.name] = digest
        print(f"{digest}  {path}")
    if args.manifest:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        manifest["sha256"].update(hashes)
        manifest["query_id_normalization"] = "numeric BEIR IDs encoded as JSON numbers for the Pisces uint64 parser"
        args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
