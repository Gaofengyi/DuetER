# Experiment program index

The experiment code remains flat because several audited scripts import shared
modules by filename. Run commands from the repository root as shown below.

## Core implementation

- `compartment_dpe.py`: independently keyed compartment-local coordinate index;
- `semantic_candidate_index.py`: keyed mean-residual spherical IVF;
- `lexical_mips_index.py`: keyed BM25-tail candidate index and MIPS-to-L2 path;
- `benchmark_budgeted_lexical_candidates.py`: cumulative-posting lexical planner;
- `benchmark_exact_bm25_client_rerank.py`: candidate-local native BM25 reranking;
- `experiment_candidate_rank_fusion.py`: client-side DuetRank features/training;
- `benchmark_dual_compartment_full.py`: complete-corpus RQ1/RQ2 driver.

## Main evaluations

- `evaluate_standard_rrf_final.py`: standard fixed RRF control;
- `summarize_dual_compartment_rq1_rq2.py`: complete-corpus table summaries;
- `plot_dual_compartment_path_utility.py`: main path-utility figure;
- `plot_standard_rrf_figure.py`: figure including the fixed-RRF control.

## Security evaluations

- `run_compartment_security.py`: topic, neighborhood, anchor, known-query, and
  alias-linkage attacks;
- `attack_cross_compartment_stitching.py`: full-view compartment stitching;
- `attack_cross_path_cooccurrence.py`: candidate/return-set co-occurrence;
- `attack_cross_path_posting_access.py`: posting/access-pattern attack;
- `sweep_known_query_budget.py`: known-query recovery budget sweep.

## Ablations

- `ablate_compartment_nprobe.py`: semantic probe count;
- `ablate_compartment_larger_lexical_depth.py`: compartment-local depth;
- `ablate_duetrank_calibration_final.py`: DuetRank calibration size;
- `measure_compartment_projection.py`: projection distortion diagnostics.

Other files retain experiments used during index design, validation, and
reviewer-response audits. Outputs are written to ignored cache/result locations;
the compact final artifacts are under `../results/paper/`.

## Internal controls

Scripts containing `global`, `million`, or `lsh` in their name are retained for
DuetER ablation and provenance. They are not the final CCADPE operating point.
No external-system implementation or comparison-execution code is distributed.
