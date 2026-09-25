# Released result artifacts

`paper/` contains compact, human-readable outputs used to audit the manuscript.
Large rankings, embeddings, coordinates, posting databases, and trained
DuetRank models are excluded because they are generated intermediates.

- `paper/main/`: complete-corpus utility, exact-BM25, fixed-RRF, and efficiency;
- `paper/ablations/`: projection, probe-count, local-depth, and calibration runs;
- `paper/security/`: topic, neighborhood, anchor, known-query, alias, and
  cross-compartment stitching evaluations;
- `paper/baselines/`: non-executable controlled-FiQA and official-Pisces
  summaries, plus explicitly author-reported PRAG values. No external-system
  source or execution scripts are included.

JSON files retain full precision. CSV and Markdown files are convenience views.
The repository release check validates JSON syntax and rejects absolute local
paths or files larger than the configured artifact limit.
