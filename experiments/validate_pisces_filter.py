"""Validate the manual Pisces SimHash filter against plaintext dense top-100."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pisces_reimplementation import PiscesSimHashFilter
from run_experiment import load_scifact, normalize_rows, top_indices


ROOT = Path(__file__).resolve().parent


def main() -> None:
    _, _, query_ids, _, _ = load_scifact(ROOT / "data" / "scifact")
    docs = normalize_rows(np.load(ROOT / "cache" / "lsa256_corpus_embeddings.npy"))
    queries = normalize_rows(np.load(ROOT / "cache" / "lsa256_query_embeddings.npy"))
    index = PiscesSimHashFilter(docs.shape[1], seed=20260842)
    index.build(docs)
    counts, coverages, seeds, radii = [], [], [], []
    for qi in range(len(query_ids)):
        candidates, trace = index.candidates(queries[qi])
        plain = top_indices(docs @ queries[qi], 100)
        counts.append(len(candidates))
        coverages.append(len(set(map(int, plain)).intersection(map(int, candidates))) / 100.0)
        seeds.append(trace["fuzzy_seed_digests"])
        radii.append(trace["expanded_hamming_radius"])
    result = {
        "encoder": "offline LSA-256 control; Granite results are stored in scifact_granite_* runs",
        "simhash_bits": index.simhash_bits,
        "projection_num": index.projection_num,
        "projection_weight": index.projection_weight,
        "queries": len(query_ids),
        "candidate_mean": float(np.mean(counts)),
        "candidate_p95": float(np.percentile(counts, 95)),
        "dense_top100_coverage_mean": float(np.mean(coverages)),
        "fuzzy_seed_mean": float(np.mean(seeds)),
        "no_seed_queries": int(np.count_nonzero(np.asarray(seeds) == 0)),
        "radius_histogram": {
            str(radius): int(np.count_nonzero(np.asarray(radii) == radius))
            for radius in sorted(set(radii))
        },
    }
    output = ROOT / "results" / "pisces_filter_validation.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
