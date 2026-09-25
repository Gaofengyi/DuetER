# Released result artifacts

`paper/` contains the compact, human-readable outputs reported in the manuscript.
Large rankings, embeddings, coordinates, posting databases, and trained
DuetRank models are excluded because they are generated intermediates.

- `paper/main/`: complete-corpus utility, fixed-RRF, and efficiency;
- `paper/ablations/`: probe-count, local-depth, and calibration runs;
- `paper/security/`: topic, neighborhood, anchor, known-query, and alias-linkage
  evaluations;

JSON files retain full precision.
The repository release check validates JSON syntax and rejects absolute local
paths or files larger than the configured artifact limit.
