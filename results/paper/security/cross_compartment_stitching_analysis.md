# Cross-compartment stitching supplement (NQ)

## Setup

- Construction: the current CCADPE implementation in
  `run_full_corpus.py`.
- Threat view: only the synchronized compartment-local query coordinates and
  query/cell co-access timing visible to the honest-but-curious cloud.
- Wire precision: every local query coordinate is quantized to `float16` before
  the attack.
- Ground-truth selectors/signs/translations are not used by the attack. They
  are read only after matching to compute precision, recall, sign accuracy,
  and translation error.
- Dataset: NQ (3,452 semantic queries); fixed seed 20260917.
- Matcher: mean-center each coordinate time series, retrieve candidate pairs
  with random-hyperplane LSH, and accept a pair when absolute Pearson
  correlation is at least 0.9999. Semantic cell-pair matrices are searched
  exhaustively.

## Lexical path: all 64 partitions

The implementation has work dimension 2,048 and samples 256 coordinates in
each of 64 partitions. The expected pairwise overlap is therefore
`256^2 / 2048 = 32`; the observed mean is 32.213. Across all 64 selectors,
all 2,048 latent coordinates occur at least once, with eight views per latent
coordinate on average.

| Queries | Match precision | Match recall | Sign accuracy | Translation MAE | Distance Pearson | Calibrated distance NRMSE | kNN R@10 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 77.233% | 100% | 100% | 7.61e-5 | 0.992414 | 1.155% | 100%* |
| 16 | 85.093% | 100% | 100% | 5.38e-5 | 0.994870 | 1.105% | 90.625% |
| 32 | 93.023% | 100% | 100% | 3.80e-5 | 0.997529 | 0.809% | 90.625% |
| 64 | 98.470% | 100% | 100% | 2.67e-5 | 0.999328 | 0.400% | 90.469% |
| 128 | 99.877% | 100% | 100% | 1.87e-5 | 0.999952 | 0.115% | 93.594% |

`*` With eight queries, R@10 reduces to recall over the other seven queries.

At 128 queries, the attack recovers all 64,942 true cross-partition coordinate
pairs and admits only 80 false pairs. It reduces 16,384 observed local
coordinates to 2,046 inferred components (the ground truth is 2,048; the two
missing components are caused by false merges). The recovered representation
therefore restores essentially the full global query geometry.

## Semantic path: actual 128-probe routing

The experiment uses all 3,452 NQ queries and their actual IVF top-128 cell
sets. Among the 32 most frequently accessed cells, the 12 highest-co-access
pairs have 429--459 synchronized queries. Their selector overlaps range from
126 to 138 coordinates. Every true overlap is recovered, with no false match:
aggregate precision and recall are both 100%, relative-sign accuracy is 100%,
and translation-difference MAE is approximately 1e-5.

For cells 53 and 449, the posting lists contain 1,108 copies of the same
documents. Using only the query-derived 135-coordinate alignment, the attack
tests 100 randomly sampled replicas against all 3,162 records in cell 449.
All 100 true replicas rank first (top-1 and top-10 linkage are both 100%;
median rank 1). This directly shows that independent aliases do not prevent
content-based replica linkage after the coordinate frames are aligned.

## Interpretation

For two local coordinates selecting the same latent coordinate k,

`c_{t,b,j} = alpha s_{b,j} u_t[k] + t_{b,j}`.

Mean centering over the synchronized transcript removes the fixed translation,
so the two traces have correlation exactly `s_{b,j}s_{b',j'}` before
quantization. Path-level query noise is already part of the shared `u_t[k]`
and therefore does not reduce this correlation. The float16 experiment shows
that wire quantization is also insufficient to disrupt it.

The current results do not support an unconditional statement that CCADPE
"breaks global geometric coherence." A defensible statement is narrower:
CCADPE removes direct one-shot comparability before transcript-based alignment,
but the current shared-intermediate construction does not protect against
synchronized multi-query stitching. Table 5 should add this attack and report
the negative result. The security theorem should either include the derived
leakage or the construction must be redesigned so that compartments do not
expose deterministic multi-view functions of the same query intermediate.

## Scope

The lexical experiment covers all 64 partitions requested by the reviewer.
The semantic experiment tests the most highly co-accessed region rather than
every one of the 2,048 choose 2 cell pairs. Replica linkage is evaluated on
100 randomly sampled shared documents. These limits do not weaken the
existence result, but broader datasets and all-cell graph coverage would be
appropriate for a final camera-ready robustness study.
