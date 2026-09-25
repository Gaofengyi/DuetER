# Dual-path Compartment-DPE: RQ1/RQ2 audit

## RQ1 utility

| Data | Split/Q | Local depth S/L | S nDCG ret. | L nDCG ret. | S→Dual nDCG@10 | Dual R@20 | Top-10 union R |
|---|---:|---:|---:|---:|---:|---:|---:|
| NQ | test/3,452 | 16/16 | 0.9942 | 0.9965 | 0.51513→0.51932 | 0.82349 | 0.77434 |
| HotpotQA | test/7,405 | 16/8 | 0.9881 | 0.9826 | 0.64062→0.66219 | 0.73761 | 0.71649 |
| MS MARCO | dev/6,980 | 32/16 | 0.9996 | 0.9950 | 0.28349→0.29692 | 0.62181 | 0.57237 |

## RQ2 cost

| Data | Batched server ms | Interactive server ms | Comm. MiB | Est. storage GiB | Client rerank+fusion ms |
|---|---:|---:|---:|---:|---:|
| NQ | 67.38 | 235.91 | 3.87 | 4.75 | 20.82 |
| HotpotQA | 48.58 | 340.39 | 2.82 | 8.80 | 26.00 |
| MS MARCO | 99.21 | 1022.97 | 5.55 | 14.98 | 64.13 |

## Scope

- NQ and HotpotQA use complete test workloads; MS MARCO uses all 6,980 dev queries because only 43 test queries have public judgments.
- The MS MARCO semantic retention denominator is the audited pre-compartment global-DPE IVF output, not an infeasible 8.84M-by-6,980 exact full scan.
- Semantic compartment coordinates are fully materialized. Lexical online execution materializes every record touched by the complete query workload; full-deployment lexical storage is estimated at one local ciphertext per corpus record.
- Batched server time measures cell-major throughput. Interactive server time is a query-major five-query sample and should not be conflated with batched throughput.
- NQ and HotpotQA path metrics average three fixed-seed executions; the full MS MARCO dev run uses one deterministic seed, while confidence intervals bootstrap held-out queries.
