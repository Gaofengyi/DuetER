"""Fail on common reproducibility-release mistakes."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 25 * 1024 * 1024
SKIP_PARTS = {".git", ".venv", "__pycache__"}
TEXT_SUFFIXES = {
    ".cfg", ".csv", ".ini", ".json", ".md", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml"
}
FORBIDDEN = {
    "Windows user path": re.compile(r"[A-Za-z]:\\Users\\", re.IGNORECASE),
    "Linux home path": re.compile(r"/home/[A-Za-z0-9_.-]+/"),
    "GitHub token": re.compile(r"(?:ghp_|github_pat_)[A-Za-z0-9_]+"),
    "Hugging Face token": re.compile(r"hf_[A-Za-z0-9]{20,}"),
    "private key": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
}


def files() -> list[Path]:
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file() and not any(part in SKIP_PARTS for part in path.parts)
    ]


def main() -> None:
    failures: list[str] = []
    for path in files():
        relative = path.relative_to(ROOT)
        if path.stat().st_size > MAX_FILE_BYTES:
            failures.append(f"large file ({path.stat().st_size} bytes): {relative}")
        if path.suffix.lower() == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except Exception as error:  # pragma: no cover - diagnostic path
                failures.append(f"invalid JSON {relative}: {error}")
        if path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="replace")
            for label, pattern in FORBIDDEN.items():
                if pattern.search(text):
                    failures.append(f"{label}: {relative}")
    if failures:
        print("Release validation failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        raise SystemExit(1)
    print(f"Release validation passed for {len(files())} files.")


if __name__ == "__main__":
    main()

