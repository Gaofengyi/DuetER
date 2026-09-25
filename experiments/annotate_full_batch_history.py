"""Attach the observed resumable-encoding batch history to final provenance."""

from __future__ import annotations

import json
import os
from pathlib import Path


HISTORY = {
    "msmarco": [
        {"batch_size": 64, "first_document": 0, "last_document_exclusive": 798_720},
        {"batch_size": 128, "first_document": 798_720, "last_document_exclusive": 856_064},
        {"batch_size": 256, "first_document": 856_064, "last_document_exclusive": 8_841_823},
    ],
    "nq": [
        {"batch_size": 256, "first_document": 0, "last_document_exclusive": 32_768},
        {"batch_size": 64, "first_document": 32_768, "last_document_exclusive": 2_351_104},
        {"batch_size": 16, "first_document": 2_351_104, "last_document_exclusive": 2_681_468},
    ],
    "hotpotqa": [
        {"batch_size": 64, "first_document": 0, "last_document_exclusive": 5_233_329},
    ],
}
COUNTS = {"msmarco": 8_841_823, "nq": 2_681_468, "hotpotqa": 5_233_329}


def atomic_write(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    for dataset, history in HISTORY.items():
        if history[0]["first_document"] != 0 or history[-1]["last_document_exclusive"] != COUNTS[dataset]:
            raise ValueError(f"{dataset}: incomplete history")
        for left, right in zip(history, history[1:]):
            if left["last_document_exclusive"] != right["first_document"]:
                raise ValueError(f"{dataset}: discontinuous history")
        cache = Path("cache/full_semantic") / f"{dataset}_{COUNTS[dataset]}" / "encoding_manifest.json"
        result = Path("results/full_semantic_hybrid") / f"{dataset}_{COUNTS[dataset]}" / "results.json"
        cache_value = json.loads(cache.read_text(encoding="utf-8"))
        result_value = json.loads(result.read_text(encoding="utf-8"))
        cache_value["batch_size_history"] = history
        cache_value["batch_size_note"] = "resumable ranges; reductions followed observed CUDA OOM on longer documents"
        result_value["encoding"]["batch_size_history"] = history
        result_value["encoding"]["batch_size_note"] = cache_value["batch_size_note"]
        atomic_write(cache, cache_value)
        atomic_write(result, result_value)
        print(f"annotated {dataset}")


if __name__ == "__main__":
    main()
