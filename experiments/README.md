# Final DuetER experiment code

This directory contains only the implementation and experiments used by the
final DuetER system.

## Core modules

- `ccadpe.py`: independently keyed compartment-local coordinate transform;
- `semantic_index.py`: keyed mean-residual spherical IVF;
- `semantic_backend.py`: complete-corpus semantic data and ranking backend;
- `lexical_backend.py`: compressed lexical MIPS-to-L2 backend;
- `lexical_planner.py`: cumulative-posting candidate planner;
- `exact_bm25.py`: candidate-local native BM25 reranking;
- `duetrank_features.py`: query and ranking features;
- `duetrank.py`: client-side learned fusion;
- `common.py`: shared tokenization, transforms, and dataset loaders.

## Final experiment entry points

- `run_full_corpus.py`: complete-corpus utility and efficiency experiment;
- `evaluate_rrf.py`: fixed standard-RRF control;
- `run_security.py`: final topic, neighborhood, anchor, query-recovery, and
  alias-linkage evaluation;
- `ablate_probes.py`: semantic probe-count ablation;
- `ablate_local_depth.py`: compartment-local return-depth ablation;
- `ablate_calibration.py`: DuetRank calibration-size ablation;
- `plot_path_utility.py`: final path-utility figure;
- `download_beir.py`: verified public-dataset downloader.

Run the documented profiles through `python scripts/reproduce.py --profile
<quick|full|security|ablations>`. Generated caches and raw rankings are ignored;
compact final outputs are retained under `results/paper/`.
