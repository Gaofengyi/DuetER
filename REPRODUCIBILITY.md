# Reproducibility guide

## Evaluated workloads

| Dataset | Corpus | Evaluated split |
|---|---:|---:|
| NQ | 2,681,468 documents | 3,452 test queries |
| HotpotQA | 5,233,329 documents | 7,405 test queries |
| MS MARCO | 8,841,823 documents | 6,980 dev queries |
| FiQA | 57,638 documents | 300 sampled test queries for the controlled comparison |

The main machine used an Intel Core i9-14900HX CPU and an NVIDIA GeForce RTX
4060 Laptop GPU. The controlled comparison used the same Ubuntu 20.04.5 VMware
guest with 16 vCPUs and 32 GiB RAM. Latency should be interpreted with the
execution mode and hardware recorded in each result artifact.

## Main operating point

- semantic index: 2,048 residual spherical-IVF compartments, two document
  assignments, 128 probes;
- compartment projection dimension: 256;
- lexical planner: cumulative posting budget 50,000 and candidate cap 1,000;
- semantic client output depth: 100;
- lexical client output depth: 300;
- CCADPE bounded-noise parameter: beta = 0.10;
- transform scale: 3.0;
- complete-corpus seed: 20260917;
- primary neural encoder: `ibm-granite/granite-embedding-small-english-r2`;
- lexical final scoring: candidate-local native BM25 on the client;
- fusion: client-side DuetRank, calibrated only on the designated calibration
  partition, with fixed RRF and semantic-only controls.

Parameters that vary by dataset, including the selected semantic and lexical
local depths, are stored in `results/paper/main/complete_corpus_summary.json`.

## Artifact-to-claim map

| Paper component | Released artifact |
|---|---|
| Complete-corpus selectivity and fidelity | `results/paper/main/complete_corpus_summary.json` |
| Semantic, lexical, fixed-RRF, and DuetRank utility | `results/paper/main/path_utility.json`, `standard_rrf.json` |
| Candidate-local exact-BM25 audit | `results/paper/main/exact_bm25_summary.json` |
| Final online/storage summary | `results/paper/main/final_system_summary.json` |
| Probe/local-depth/calibration ablations | `results/paper/ablations/` |
| Plain/global-DPE/CCADPE attacks | `results/paper/security/` |
| Controlled FiQA comparison | `results/paper/baselines/fiqa_test_q300_controlled_summary.json` |
| Official Pisces summaries | `results/paper/baselines/official_pisces_summary.{json,csv}` |
| PRAG literature reference values | `results/paper/baselines/prag_author_reported.json` |

## Reproduction profiles

`scripts/reproduce.py` defines the release profiles. Run with `--dry-run` first.

- `quick`: dataset-free tests plus a SciFact LSA command;
- `full`: complete-corpus CCADPE runs for NQ, HotpotQA, and MS MARCO;
- `security`: the concrete-view attack suite;
- `ablations`: semantic probe, local-depth, and calibration sensitivity;
- `all`: all of the above.

Full-corpus runs deliberately do not auto-download or silently build hundreds
of gigabytes of intermediates. Downloading and preprocessing are explicit steps,
and generated outputs go under ignored `results/generated/` or
`experiments/results/` directories.

## Determinism

All released programs expose or fix seeds. Exact retrieval metrics should be
reproducible from the same corpus revision, split, model snapshot, cached
embeddings, and package versions. Timing is not deterministic and should be
remeasured after warm-up on the target hardware. GPU kernels and library
versions can introduce small numerical differences near ranking ties.

## Comparison-result interpretation

The controlled FiQA result uses the same corpus, 300-query workload, output
policy, and Ubuntu guest for DuetER and the official Pisces execution.
Cross-paper PRAG values are retained only as context and are not a controlled
head-to-head measurement. External implementation and execution code are not
distributed in this repository.
