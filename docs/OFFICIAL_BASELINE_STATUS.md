# Official cryptographic baseline audit

Audit updated: 2026-08-26. This file separates official executable systems
from functional reimplementations and paper-only baselines. Emulator timing is
never substituted for an official protocol measurement.

## Pisces (Liang et al., 2026): executed

- Official repository: <https://github.com/ant-intl/Pisces>
- Audited revision: `a581a5e932501867871f9374c61e97db34775f7d`
- Local source snapshot: `baselines/pisces_official_reference`
- Execution source in Ubuntu: an isolated checkout of the official Pisces repository
- Guest: Ubuntu 20.04 under VMware, four vCPUs, 7.8 GiB RAM, GCC 9.4.0,
  Bazel 7.4.1. Neural preprocessing used the host RTX 4060 Laptop GPU; the
  cryptographic binaries ran on guest CPUs.

Both released benchmark binaries were built and executed with the released
LPSI/FPSI, SPU, garbled-circuit TopK, and SUDA PIR code. The lexical executable
uses the repository's in-process `Mock2PCLink(false)`; the semantic executable
uses two localhost BRPC parties. These are real protocol executions, but they
are not a cross-machine WAN experiment.

Two minimal compiler-compatibility patches are retained in `baselines/` and in
the evidence archive:

- `pisces_yacl_gcc9_compat.patch` computes the width of
  `unsigned __int128` using `sizeof` when GCC 9 reports an unspecialized
  `numeric_limits`, and removes the GCC-11-only
  `-Wno-mismatched-new-delete` option from extracted YACL configuration.
- `pisces_spulib_gcc9_compat.patch` removes external SPU's global `-Werror`.

Neither patch changes a Pisces protocol or experiment parameter. Before/after
hashes and complete build logs are in `results/official_pisces_vm/evidence`.
The successful binaries have SHA-256 values:

- BM25: `99c24d2b7eebd5013a5c2902044d5d3ab155e7035e208c27475c22f9fa1368d9`
- Similarity: `01fe5e6d2213b92ac9402ff6277b48a5603dba299b95e20b2f831bf0d8ecd2c5`

### Executed benchmark

The released CLAPNQ answerable development corpus contains 1,990 contexts.
Documents and the first three released queries were tokenized with
`bert-base-uncased` and embedded with the official paper's
`ibm-granite/granite-embedding-small-english-r2` model. The manifest records
file hashes, package versions, GPU, counts, and embedding dimension. We ran
`K=10` and five complete repeats at 1,990 documents, for 15 observations per
path. A 256-document one-repeat run is retained as a smoke/scale point.

| Documents | Path | Repeats | Query observations | Mean +/- sd (s) | p95 (s) | Upload (MiB) | Download (MiB) | Plaintext Top-10 recall | Exact Top-1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | BM25 | 1 | 3 | 2.533 +/- 0.082 | 2.626 | 20.220 | 7.827 | 0.933 | 1.000 |
| 256 | similarity | 1 | 3 | 3.027 +/- 0.049 | 3.068 | 21.268 | 9.145 | 0.933 | 1.000 |
| 1,990 | BM25 | 5 | 15 | 2.541 +/- 0.085 | 2.741 | 22.458 | 23.967 | 0.980 | 1.000 |
| 1,990 | similarity | 5 | 15 | 3.149 +/- 0.101 | 3.330 | 24.830 | 26.455 | 0.800 | 1.000 |

All 30 full-subset path executions exited successfully and preserved exact
plaintext Top-1. The BM25 path produced 1, 2, and 1 distinct rankings for the
three queries; three observations replaced one boundary Top-10 item. The
randomized similarity path produced 2, 5, and 2 distinct rankings. Its FPSI
filter supplied 1,220 candidates on average (p95 1,493), and candidate variation
changed lower-ranked Top-10 items. SUDA contributed about 2.47--2.50
seconds/query. Mean process wall time for three queries was 8.10 seconds (BM25)
and 10.67 seconds (similarity), with mean peak RSS of 471 and 608 MiB.

Quality was independently checked against exact plaintext BM25 and cosine
rankings computed from the identical preprocessed records. The BM25 reference
uses the constants and IDF formula in the official C++ source.

The released unit-test targets were also rebuilt and executed on 2026-08-25.
`//pisces/bm25:bm25_test` passed all 7 tests, including the 100,000-instance
case, in 62.675 seconds; its peak RSS was 2,524,308 KiB. The standard Bazel
target `//pisces/similarity:similarity_test` passed 1/1 in 1.3 seconds and its
direct run used 1,271,532 KiB peak RSS. The latter checks 10,000 dot products
at dimension 1,024 against plaintext with absolute error below 0.001. Because
GitHub raw registry endpoints were intermittent, dependency resolution used
shallow local mirrors of the official SecretFlow and Bazel Central registries;
the LLVM archive was the registry-specified commit with verified SHA-256
`f27ca1bd652f820ed87eeec00a218b3a87469052027860e8747e92f5cba11391`.
This changes dependency transport only, not source, test parameters, or
protocol code. GoogleTest XML, the full similarity log, binary hashes, and the
local-registry Bazel configuration are retained in the evidence directory.

Primary artifacts:

- `results/official_pisces_vm/official_pisces_summary.{md,csv,json}`
- `results/official_pisces_vm/preprocessed_clapnq_{256,1990}/manifest.json`
- `results/official_pisces_vm/evidence/`
- `results/official_pisces_vm/official_unit_tests_20260825.md`
- `results/official_pisces_vm/pisces_official_unit_tests_20260825.tgz`
- `results/official_pisces_vm/pisces_official_evidence_20260825.tgz`

### Larger-scale official FiQA execution

The exact audited binaries above were copied to a second Ubuntu 20.04 VMware
guest and verified by SHA-256 before execution. This guest exposes 16 vCPUs on
an Intel Core i9-14900HX and 31.4 GiB RAM. The complete BEIR FiQA corpus was
converted to the official JSONL schema without changing text, tokenized with
`bert-base-uncased`, and embedded with the same 384-dimensional Granite R2
model. Deterministic prefixes of 4,096, 8,192, 16,384, and 32,768 documents and
the complete 57,638-document corpus were each run five times with the same
three queries and `K=10`.

All 50 path/scale executions exited successfully, giving 15 online query
observations per row. At 57,638 documents, BM25 used 5.279 seconds and
130.95/677.42 MiB client upload/download per query; similarity used 9.688
seconds and 191.68/795.38 MiB. Exact-plaintext Recall@10 was 1.000 for BM25
at every scale and 0.960 for full-corpus similarity. Mean whole-process wall
time (offline setup plus three queries) was 34.47/56.14 seconds, and mean peak
RSS was 3,419/4,761 MiB. The full-corpus similarity FPSI stage retained 47,590
candidates on average (82.6%), explaining both its strong recall and high
communication.

Scale artifacts:

- `results/official_pisces_vm/official_pisces_fiqa_scale_summary.{md,csv,json}`
- `results/official_pisces_vm/preprocessed_fiqa_57638/manifest.json`
- `results/official_pisces_vm/fiqa_scale_evidence/` (50 JSONL, 50 logs, 50 GNU-time files)
- `results/official_pisces_vm/pisces_fiqa_scale_results.tgz`
  (SHA-256 `aef5cb49a20b3389ccc914578743ba7cf45657ffbc5808166da83cb8e90b90e2`)

## PRAG (Li et al., 2026; arXiv:2604.26525): paper-reported baseline

- Paper record: <https://arxiv.org/abs/2604.26525>
- Repository identified by the v2 paper (redirects to the canonical location):
  <https://github.com/BDS-SDU/PRAG>
- The repository has not been locally executed. The manuscript instead labels
  the paper's 100K-TriviaQA CKKS numbers as author-reported: PRAG-I/PRAG-II
  retrieval 1.29/7.91 s, communication 4.1175/78.5478 MB, and Recall@10
  72.45%/74.45%. These values are not used for matched-workload speedups.

## Distinct MPC-PRAG (South et al., 2024): not substituted

- Official repository: <https://github.com/tobinsouth/prag>
- Audited revision: `b910e13bd3e9371929d12b9e4126d8d41032ffec`
- Local source: `baselines/prag_2024_official`
- This archived CrypTen/MPC project is not the 2026 homomorphic PRAG-I/II
  baseline cited by the manuscript and is not labeled as such.
