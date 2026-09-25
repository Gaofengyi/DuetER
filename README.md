# DuetER

DuetER is a research prototype for efficient dual-path encrypted retrieval in
retrieval-augmented generation. It combines semantic residual-IVF retrieval and
a keyed lexical posting planner, ranks candidates in independently keyed
compartment-local coordinate systems, and performs exact path reranking and
query-adaptive fusion at the client.

This repository contains the implementation, evaluation scripts, compact
paper-result artifacts, and commands needed to reproduce the reported
experiments. It deliberately excludes corpora, model checkpoints, cached
embeddings, and generated indexes.

## Security scope

Compartment conditional approximate distance-preserving encryption (CCADPE) is
an efficiency and structured leakage-reduction mechanism, not a replacement for
HE, MPC, PIR, or ORAM. The cloud observes keyed compartment/posting accesses,
compartment sizes, candidate and timing traces, and coordinates and distance
orders inside each accessed compartment. Independent aliases remove direct
cross-path identifier equality; they do not hide access-pattern correlation.
The attack scripts in `experiments/` and audited outputs in
`results/paper/security/` quantify this operating point.

The NumPy pseudorandom generator is used for deterministic experiments. It is
not a production cryptographic PRF. A deployment must use a CSPRNG and proper
key derivation.

## Repository layout

```text
experiments/       Retrieval, ablation, security, and plotting programs
scripts/           Reproduction driver and official-Pisces helper scripts
tests/             Dataset-free smoke tests
results/paper/     Compact JSON/CSV/Markdown artifacts used by the paper
docs/              Index design and baseline-validation notes
```

## Installation

Python 3.11 was used for the release artifact.

```bash
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For neural encoders, install the PyTorch wheel matching the local CUDA driver,
then install the remaining neural dependencies:

```bash
python -m pip install -r requirements-neural.txt
```

The reported GPU runs used PyTorch 2.12.1 with CUDA 12.6. CPU-only execution is
supported but is not expected to reproduce GPU latency.

## Quick validation

The following checks do not download a dataset:

```bash
python -m unittest discover -s tests -v
python scripts/check_release.py
python scripts/reproduce.py --profile quick --dry-run
```

For a small end-to-end run using the public BEIR SciFact corpus:

```bash
python experiments/download_beir.py scifact
python experiments/run_experiment.py \
  --data-dir experiments/data/scifact \
  --dataset-name scifact \
  --dense-backend lsa \
  --results-dir results/generated/scifact_lsa
```

## Paper reproduction

The reproduction driver prints commands unless `--execute` is supplied. This
makes the exact workload visible before a potentially long or storage-intensive
run.

```bash
# Complete-corpus RQ1/RQ2 commands
python scripts/reproduce.py --profile full --dry-run

# Execute one complete-corpus dataset after data/model preparation
python experiments/benchmark_dual_compartment_full.py \
  --dataset nq --split test --projection-dimension 256 \
  --semantic-probes 128 --device cuda \
  --cache-root experiments/cache/dual_compartment_full \
  --output-root results/generated/dual_compartment_full

# Security evaluation (requires the cached embeddings named by --known-query-files)
python scripts/reproduce.py --profile security --dry-run

# Final-system ablations
python scripts/reproduce.py --profile ablations --dry-run
```

The full corpora contain 2.68M NQ, 5.23M HotpotQA, and 8.84M MS MARCO
documents. Complete reproduction requires substantial disk space for source
corpora, embeddings, compartment coordinates, and indexes. The released result
summaries allow table-level auditing without downloading those intermediates.
See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for dataset splits, main parameters,
hardware, seeds, and the mapping from paper claims to artifacts.

## Baselines

`experiments/pisces_reimplementation.py` is a functional reimplementation used
for protocol-output validation; its Python runtime is **not** an official
cryptographic Pisces measurement. Official Pisces execution is handled by the
scripts under `scripts/` against a separately obtained upstream checkout. The
released controlled FiQA summary and official logs-derived summaries are in
`results/paper/baselines/`. PRAG entries are explicitly labeled as
author-reported literature values.

Third-party source trees and binaries are not redistributed. Obtain each
baseline from its upstream repository and follow its license.

## Data and models

`experiments/download_beir.py` downloads BEIR archives and verifies their MD5
checksums. Dataset licenses and terms remain those of the original providers.
Neural runs expect a locally available sentence-transformers-compatible model;
the evaluated primary encoder is identified in `REPRODUCIBILITY.md`. No private
or personal data are included in this repository.

## Reproducibility and anonymity

The repository avoids machine-specific paths, credentials, generated caches,
and author metadata. Nevertheless, a URL under a personal GitHub account is not
anonymous. During double-blind review, use an anonymized supplementary ZIP or
an anonymous repository rather than linking the personal repository directly.

## License

The DuetER code in this repository is released under the MIT License. Dataset,
model, and third-party baseline licenses are not changed by this release.
