# Pisces manual reimplementation validation

The implementation in `pisces_reimplementation.py` is an independent research
reimplementation. The official Apache-2.0 source is downloaded under
`baselines/pisces_official_reference` only for parameter and control-flow
verification; the Python module is not a translation or copy of the C++ code.

## Checked against the released end-to-end implementation

- 32-bit SimHash (`Digest<uint32_t>`), not 64-bit;
- 160 projections and projection weight 16;
- the released code uses `digest | mask`, so mask bits are discarded and the
  complement bits form the exact-match projection;
- threshold-two reconstruction: at least two matching projections produce a
  fuzzy-PSI seed digest;
- server-side Hamming expansion starts at radius 5 and grows through radius 10
  while the selected set is below 15% of the corpus;
- fine scoring uses cosine/dot-product over selected embeddings;
- multi-instance labeled PSI uses a reusable keyed store, one ideal OPRF result
  per unique query token, document-index-bound KDF keys, encrypted frequency
  labels, fixed-point BM25, secret shares, and private top-k semantics.
- the released BM25 uses $k_1=1.5$, merges duplicate query tokens through an
  unordered map, uses 18 SPU fractional bits, and shifts scores by 8 bits before
  Top-K; the experiment includes a matching plaintext adapter baseline.

## Automated checks

- slow step-by-step labeled-PSI decoding equals the accelerated equivalent path;
- an identical vector survives SimHash filtering;
- additive uint64 score shares reconstruct exactly;
- official end-to-end defaults are asserted;
- the SciFact filter validation records candidate size, fuzzy seeds, expansion
  radii, and plaintext dense top-100 coverage in
  `results/pisces_filter_validation.json`. That standalone diagnostic uses the
  offline LSA-256 control embeddings; the paper's 4,513.5-candidate figure is
  the separate Granite run under `results/scifact_granite_*`. The encoder label
  is now embedded in the validation JSON to prevent these results from being
  conflated.

## Security and timing boundary

The Python code models ideal functionality with HMAC, a direct dictionary, and
local arithmetic. It does not instantiate blind OPRF, an oblivious OKVS, additive
HE, garbled circuits, PIR-to-share, or a two-party network runtime. Retrieval
outputs can be evaluated; emulator latency cannot be compared with the official
Pisces cryptographic latency.
