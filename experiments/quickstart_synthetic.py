"""Dataset-free dual-path DuetER smoke example."""

from __future__ import annotations

import json

import numpy as np

from ccadpe import CompartmentDPEIndex
from common import (
    BM25Index,
    mips_document_transform,
    mips_query_transform,
    normalize_rows,
    rrf,
)
from semantic_index import KeyedResidualSphericalIVF


def build_path(
    documents: np.ndarray,
    query: np.ndarray,
    *,
    routing_key: bytes,
    coordinate_key: bytes,
    projection_dimension: int,
) -> tuple[np.ndarray, dict[str, int]]:
    routing = KeyedResidualSphericalIVF(
        key=routing_key,
        n_clusters=4,
        doc_assignments=2,
        nprobe=2,
        seed=17,
    )
    routing.build(documents)
    index = CompartmentDPEIndex(
        key=coordinate_key,
        projection_dim=projection_dimension,
        beta=0.10,
        scale=3.0,
        seed=23,
    )
    index.build(documents, routing)
    candidates, trace = index.candidates(
        query, nprobe=2, top_per_cell=3, nonce=0
    )
    return candidates, {
        "probed_compartments": trace.probed_cells,
        "ciphertext_distances": trace.ciphertext_distances,
        "unique_candidates": trace.unique_candidates,
    }


def main() -> None:
    texts = [
        "encrypted hybrid retrieval for private rag",
        "semantic vector search over a document collection",
        "exact lexical matching with bm25",
        "weather forecast and climate observations",
        "database query optimization and indexes",
        "privacy leakage from geometric embeddings",
        "client side reranking and rank fusion",
        "homomorphic encryption for secure computation",
    ]
    query_text = "private encrypted lexical semantic retrieval"
    rng = np.random.default_rng(11)
    semantic_documents = normalize_rows(
        rng.normal(size=(len(texts), 12)).astype(np.float32)
    )
    semantic_query = normalize_rows(
        (semantic_documents[0] + 0.05 * rng.normal(size=12)).astype(np.float32)[None, :]
    )[0]

    semantic_candidates, semantic_trace = build_path(
        semantic_documents,
        semantic_query,
        routing_key=b"semantic-routing",
        coordinate_key=b"semantic-coordinates",
        projection_dimension=6,
    )

    bm25 = BM25Index(texts)
    lexical_documents = bm25.hashed_document_vectors(16, b"lexical-sketch")
    lexical_query = bm25.hashed_query_vector(query_text, 16, b"lexical-sketch")
    transformed_documents, _ = mips_document_transform(lexical_documents)
    transformed_query = mips_query_transform(lexical_query)
    lexical_candidates, lexical_trace = build_path(
        transformed_documents,
        transformed_query,
        routing_key=b"lexical-routing",
        coordinate_key=b"lexical-coordinates",
        projection_dimension=8,
    )

    semantic_rank = semantic_candidates[
        np.argsort(-(semantic_documents[semantic_candidates] @ semantic_query))
    ]
    lexical_scores = bm25.score(query_text)
    lexical_rank = lexical_candidates[
        np.argsort(-lexical_scores[lexical_candidates])
    ]
    fused = rrf(semantic_rank, lexical_rank, depth=6, alpha=0.5)[:5]
    print(
        json.dumps(
            {
                "semantic_trace": semantic_trace,
                "lexical_trace": lexical_trace,
                "semantic_top": [texts[int(index)] for index in semantic_rank[:3]],
                "lexical_top": [texts[int(index)] for index in lexical_rank[:3]],
                "dual_top": [texts[int(index)] for index in fused],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

