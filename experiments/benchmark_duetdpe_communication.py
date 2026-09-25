"""Measure DuetDPE retrieval-protocol communication in a concrete wire format.

The benchmark serializes the two encrypted query vectors, fixed-shape HMAC
probe arrays, and the server response needed for exact client reranking.  The
response carries an unlinkable alias, an AES-GCM-wrapped exact feature vector,
and an AES-GCM join capsule for every returned path candidate.  Final evidence
passage ciphertexts are deliberately excluded, matching the retrieval-only
boundary used by the Pisces and PRAG reference measurements in the paper.

This is a byte-serialization benchmark, not a network-throughput benchmark.
It executes AES-128-GCM and verifies a full serialize/parse/decrypt round trip.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "duetdpe_communication.json"

LABEL_BYTES = 16
ALIAS_BYTES = 16
NONCE_BYTES = 12
CAPSULE_PLAINTEXT_BYTES = 20  # uint64 id, uint64 location, uint32 version
SEMANTIC_WORK_DIMENSION = 512
LEXICAL_WORK_DIMENSION = 2048
SEMANTIC_FEATURE_DIMENSION = 384
LEXICAL_FEATURE_DIMENSION = 1024
SEMANTIC_PROBE_SLOTS = 128
LEXICAL_PROBE_SLOTS = 32
SEMANTIC_RESPONSE_DEPTH = 200

REQUEST_HEADER = struct.Struct("!4sHHHHIIII")
RESPONSE_HEADER = struct.Struct("!4sHHII")


WORKLOADS = {
    "nq": {
        "documents": 2_681_468,
        "queries": 3_452,
        "posting_budget": 50_000,
        "lexical_response_depth": 500,
        "trace": ROOT / "results" / "budgeted_lexical" / "nq_full.trace.json",
        "trace_key": "50000",
    },
    "hotpotqa": {
        "documents": 5_233_329,
        "queries": 7_405,
        "posting_budget": 50_000,
        "lexical_response_depth": 200,
        "trace": ROOT / "results" / "budgeted_lexical" / "hotpotqa_full.trace.json",
        "trace_key": "50000",
    },
    "msmarco": {
        "documents": 8_841_823,
        "queries": 6_980,
        "posting_budget": 250_000,
        "lexical_response_depth": 500,
        "trace": ROOT / "results" / "budgeted_lexical" / "msmarco_dev_full.trace.json",
        "trace_key": "250000",
    },
}


def serialize_request(
    posting_budget: int,
    lexical_depth: int,
    semantic_depth: int = SEMANTIC_RESPONSE_DEPTH,
) -> bytes:
    header = REQUEST_HEADER.pack(
        b"DDQ1",
        1,
        0,
        SEMANTIC_PROBE_SLOTS,
        LEXICAL_PROBE_SLOTS,
        semantic_depth,
        lexical_depth,
        posting_budget,
        1_000,
    )
    semantic_labels = bytes(SEMANTIC_PROBE_SLOTS * LABEL_BYTES)
    semantic_cipher = np.zeros(SEMANTIC_WORK_DIMENSION, dtype=">f4").tobytes()
    lexical_labels = bytes(LEXICAL_PROBE_SLOTS * LABEL_BYTES)
    lexical_weights = np.zeros(LEXICAL_PROBE_SLOTS, dtype=">f4").tobytes()
    lexical_cipher = np.zeros(LEXICAL_WORK_DIMENSION, dtype=">f4").tobytes()
    return b"".join(
        (
            header,
            semantic_labels,
            semantic_cipher,
            lexical_labels,
            lexical_weights,
            lexical_cipher,
        )
    )


def nonce(counter: int) -> bytes:
    return counter.to_bytes(NONCE_BYTES, "big")


def encrypted_record(
    *, aes: AESGCM, path: int, row: int, feature_dimension: int
) -> bytes:
    alias = (path.to_bytes(1, "big") + row.to_bytes(15, "big"))[-ALIAS_BYTES:]
    vector_plaintext = np.zeros(feature_dimension, dtype=">f2").tobytes()
    vector_nonce = nonce(path * 1_000_000 + row * 2)
    vector_ciphertext = aes.encrypt(vector_nonce, vector_plaintext, alias)
    capsule_plaintext = struct.pack("!QQI", row, row * 4096, 1)
    capsule_nonce = nonce(path * 1_000_000 + row * 2 + 1)
    capsule_ciphertext = aes.encrypt(capsule_nonce, capsule_plaintext, alias)
    return b"".join(
        (
            alias,
            vector_nonce,
            vector_ciphertext,
            capsule_nonce,
            capsule_ciphertext,
        )
    )


def serialize_response_depths(
    semantic_depth: int, lexical_depth: int
) -> tuple[bytes, bytes]:
    key = bytes(range(16))
    aes = AESGCM(key)
    parts = [
        RESPONSE_HEADER.pack(
            b"DDR1", 1, 0, semantic_depth, lexical_depth
        )
    ]
    for row in range(semantic_depth):
        parts.append(
            encrypted_record(
                aes=aes, path=1, row=row, feature_dimension=SEMANTIC_FEATURE_DIMENSION
            )
        )
    for row in range(lexical_depth):
        parts.append(
            encrypted_record(
                aes=aes, path=2, row=row, feature_dimension=LEXICAL_FEATURE_DIMENSION
            )
        )
    return b"".join(parts), key


def serialize_response(lexical_depth: int) -> tuple[bytes, bytes]:
    """Serialize the paper's default 200-candidate semantic response."""
    return serialize_response_depths(SEMANTIC_RESPONSE_DEPTH, lexical_depth)


def verify_response(payload: bytes, key: bytes) -> None:
    aes = AESGCM(key)
    magic, version, flags, semantic_depth, lexical_depth = RESPONSE_HEADER.unpack_from(
        payload, 0
    )
    assert magic == b"DDR1" and version == 1 and flags == 0
    offset = RESPONSE_HEADER.size
    for path, count, dimension in (
        (1, semantic_depth, SEMANTIC_FEATURE_DIMENSION),
        (2, lexical_depth, LEXICAL_FEATURE_DIMENSION),
    ):
        vector_ciphertext_bytes = dimension * 2 + 16
        capsule_ciphertext_bytes = CAPSULE_PLAINTEXT_BYTES + 16
        for row in range(count):
            alias = payload[offset : offset + ALIAS_BYTES]
            offset += ALIAS_BYTES
            vector_nonce = payload[offset : offset + NONCE_BYTES]
            offset += NONCE_BYTES
            vector_ciphertext = payload[
                offset : offset + vector_ciphertext_bytes
            ]
            offset += vector_ciphertext_bytes
            capsule_nonce = payload[offset : offset + NONCE_BYTES]
            offset += NONCE_BYTES
            capsule_ciphertext = payload[
                offset : offset + capsule_ciphertext_bytes
            ]
            offset += capsule_ciphertext_bytes
            assert len(aes.decrypt(vector_nonce, vector_ciphertext, alias)) == dimension * 2
            capsule = aes.decrypt(capsule_nonce, capsule_ciphertext, alias)
            identifier, location, version_value = struct.unpack("!QQI", capsule)
            assert identifier == row
            assert location == row * 4096
            assert version_value == 1
            assert alias[0] == path
    assert offset == len(payload)


def trace_statistics(path: Path, key: str) -> dict[str, float | int]:
    trace = json.loads(path.read_text(encoding="utf-8"))[key]
    terms = np.asarray(trace["terms"], dtype=np.int32)
    return {
        "queries": int(len(terms)),
        "selected_terms_mean": float(np.mean(terms)),
        "selected_terms_max": int(np.max(terms)),
        "queries_exceeding_32_slots": int(np.count_nonzero(terms > LEXICAL_PROBE_SLOTS)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RESULTS)
    args = parser.parse_args()

    rows = []
    for dataset, configuration in WORKLOADS.items():
        lexical_depth = int(configuration["lexical_response_depth"])
        request = serialize_request(int(configuration["posting_budget"]), lexical_depth)
        response, key = serialize_response(lexical_depth)
        verify_response(response, key)
        trace = trace_statistics(
            Path(configuration["trace"]), str(configuration["trace_key"])
        )
        assert trace["queries"] == configuration["queries"]
        assert trace["queries_exceeding_32_slots"] == 0
        total = len(request) + len(response)
        rows.append(
            {
                "dataset": dataset,
                "documents": configuration["documents"],
                "queries": configuration["queries"],
                "posting_budget": configuration["posting_budget"],
                "semantic_response_depth": SEMANTIC_RESPONSE_DEPTH,
                "lexical_response_depth": lexical_depth,
                "upload_bytes": len(request),
                "download_bytes": len(response),
                "total_bytes": total,
                "upload_kib": len(request) / 1024,
                "download_mib": len(response) / (1024**2),
                "total_mib": total / (1024**2),
                **trace,
            }
        )

    output = {
        "experiment": "DuetDPE serialized retrieval-protocol communication",
        "status": "passed",
        "wire_format": {
            "query_cipher_dtype": "float32",
            "returned_exact_feature_dtype": "float16 inside AES-128-GCM",
            "semantic_work_dimension": SEMANTIC_WORK_DIMENSION,
            "lexical_work_dimension": LEXICAL_WORK_DIMENSION,
            "semantic_feature_dimension": SEMANTIC_FEATURE_DIMENSION,
            "lexical_feature_dimension": LEXICAL_FEATURE_DIMENSION,
            "semantic_probe_slots": SEMANTIC_PROBE_SLOTS,
            "lexical_probe_slots": LEXICAL_PROBE_SLOTS,
            "hmac_label_bytes": LABEL_BYTES,
            "alias_bytes": ALIAS_BYTES,
            "aead_nonce_bytes": NONCE_BYTES,
            "aead_tag_bytes": 16,
            "join_capsule_plaintext_bytes": CAPSULE_PLAINTEXT_BYTES,
        },
        "boundary": {
            "included": "two query ciphertexts, fixed HMAC probes and lexical weights, returned aliases, AEAD exact feature vectors, and AEAD join capsules",
            "excluded": "query encoding, transport framing such as TLS/IP, and final evidence-passage ciphertext fetch",
            "note": "Bytes are deterministic at the fixed padded request and response depths; AES-GCM round trips were executed and verified.",
        },
        "workloads": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
