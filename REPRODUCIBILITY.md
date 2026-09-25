# Reproducibility guide

## Evaluated workloads

| Dataset | Corpus | Evaluated split |
|---|---:|---:|
| NQ | 2,681,468 documents | 3,452 test queries |
| HotpotQA | 5,233,329 documents | 7,405 test queries |
| MS MARCO | 8,841,823 documents | 6,980 dev queries |

The main machine used an Intel Core i9-14900HX CPU and an NVIDIA GeForce RTX
4060 Laptop GPU. Latency should be interpreted with the execution mode and
hardware recorded in each result artifact.

## Main operating point

- semantic index: 2,048 residual spherical-IVF compartments, two document
  assignments, 128 probes;
- semantic compartment projection dimension: 256;
- lexical representation: 1,024-D signed feature hash, 8,192-D DPE work
  space, 32-D local projection, and 16 compartments;
- lexical planner: cumulative posting budget 50,000 and candidate cap 1,000;
- semantic client output depth: 100;
- lexical client output depth: 300;
- lexical local return depth: 64;
- CCADPE bounded-noise parameter: beta = 0.10;
- transform scale: 3.0;
- complete-corpus seed: 20260917;
- primary neural encoder: `ibm-granite/granite-embedding-small-english-r2`;
- lexical final scoring: candidate-local native BM25 on the client;
- fusion: client-side DuetRank, calibrated only on the designated calibration
  partition, with fixed RRF and semantic-only controls.

Dataset-specific semantic local depths and the final lexical configuration are
stored in `results/paper/main/final_system_summary.json`.

## Artifact-to-claim map

| Paper component | Released artifact |
|---|---|
| Complete-corpus selectivity, fidelity, and cost | `results/paper/main/final_system_summary.json` |
| Semantic, lexical, fixed-RRF, and DuetRank utility | `results/paper/main/path_utility.json`, `standard_rrf.json` |
| Final online/storage summary | `results/paper/main/final_system_summary.json` |
| Probe/local-depth/calibration ablations | `results/paper/ablations/` |
| Plain/global-DPE/CCADPE attacks | `results/paper/security/` |

## Reproduction profiles

`scripts/reproduce.py` defines the release profiles. Run with `--dry-run` first.

- `quick`: dataset-free unit and release-validation tests;
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

External-system source, execution scripts, and comparison-only result artifacts
are intentionally excluded from this minimal DuetER release.
