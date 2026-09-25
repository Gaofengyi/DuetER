# Redesign of the semantic candidate index

## Why the previous index failed

The legacy index applies 24-table Gaussian p-stable LSH directly to unit-normalized
Granite embeddings. Granite has a strong common direction, so many documents obtain
the same or adjacent quantized projection codes. Unioning nine probes per table returns
5,177.2/5,183 SciFact documents and 3,629.4/3,633 NFCorpus documents on average. It is
therefore an approximate full scan rather than a selective candidate index.

## Keyed residual spherical IVF (KRS-IVF)

1. Normalize each original embedding `v_i`. These original vectors remain the inputs
   to DPE and exact client reranking.
2. Compute the public-to-authorized-client corpus mean `mu` and the unit residual
   `z_i = normalize(v_i - mu)`. This removes the encoder-wide common direction only
   for candidate partitioning.
3. Train `K=256` spherical cells with mini-batch k-means over `z_i`, then normalize
   every centroid.
4. Assign each document to its `r=4` nearest centroids. Multi-assignment prevents a
   hard cell boundary from dropping a true neighbor.
5. Replace every cell number `j` by a 128-bit label
   `HMAC(K_bucket, "semantic-residual-ivf-v1" || j)`. The cloud stores only keyed
   labels and document aliases; the owner/client retains `mu` and the centroids.
6. Residualize a query and choose its `nprobe=32` nearest centroids locally. Send the
   corresponding HMAC labels, padded with dummy labels if a fixed request shape is
   required.
7. The cloud unions the accessed postings, ranks only those candidates by encrypted
   DPE distance, and returns the best 200 per path.
8. The client reranks returned semantic vectors with the original, uncentered cosine
   query and fuses them with the independently indexed lexical path.

Client probing costs `O(Kd)` and the server stores `rN` posting entries. The redesign
does not claim obliviousness: the cloud still sees keyed-label equality, posting sizes,
access repetition, and candidate-set overlap. HMAC hides the semantic identity of a
cell, not the access pattern.

## Tests

The 24-point sweep varies centering, `K in {128,256,512}`, document assignments
`r in {1,2,4}`, and `nprobe in {4,8,16,32}`. Selection uses downstream retrieval,
not arbitrary reproduction of the dense top-100: minimize mean candidates subject to
at least 98% nDCG@10 retention, 95% Recall@100 retention, and a 40% mean candidate
fraction. NFCorpus is the binding dataset, producing the shared `256/4/32` setting;
the same setting is then run unchanged on SciFact.

| Dataset | Legacy LSH candidates | KRS-IVF candidates | Reduction | Top-100 coverage | Semantic DPE nDCG@10 | Full-scan nDCG@10 | Retention |
|---|---:|---:|---:|---:|---:|---:|---:|
| SciFact | 5,177.2 | 1,605.5 | 69.0% | 96.93% | .7484 | .7497 | 99.84% |
| NFCorpus | 3,629.4 | 1,157.0 | 68.1% | 89.06% | .3678 | .3736 | 98.46% |

SciFact relevant-document candidate coverage is 99.33%; NFCorpus's lower 66.93%
reflects its many distributed relevance judgments, but Recall@100 still retains 96.0%
of the dense/full-scan value. In the dual path, the redesigned system reaches .7090
nDCG@10 on SciFact and .3578 on NFCorpus, compared with .6977 and .3541 for dual
full-scan DPE. Candidate filtering can improve fusion by removing low-ranked lexical
sketch collisions, so the dual result need not be below the full-scan result.

The unit suite contains 13 passing tests, including identical-vector recovery,
multi-assignment posting counts, and key separation. Full result files and parameter
sweeps are listed in the project README.
