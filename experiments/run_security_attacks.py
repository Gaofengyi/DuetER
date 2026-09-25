"""Execute the previously missing leakage attacks for DuetDPE.

Attacks:
1. topic inference from visible semantic query coordinates on 20 Newsgroups;
2. known-record graph/fingerprint alignment on SciFact document ciphertexts;
3. neighborhood reconstruction (kNN overlap and clustering agreement).

The existing cross-view co-occurrence attack remains in run_experiment.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEPS = ROOT / ".deps"
GPU_DEPS = ROOT / ".gpu_deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))
if os.environ.get("DUETDPE_GPU_DEPS") == "1" and GPU_DEPS.exists():
    sys.path.insert(0, str(GPU_DEPS))

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from run_experiment import ConditionalDPE, neighbor_overlap, normalize_rows


def mean_summary(values: list[float]) -> dict[str, float | list[float]]:
    array = np.asarray(values, dtype=np.float64)
    half = 1.96 * float(np.std(array, ddof=1)) / np.sqrt(len(array)) if len(array) > 1 else 0.0
    mean = float(np.mean(array))
    return {
        "mean": mean,
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "normal_ci95_low": max(0.0, mean - half),
        "normal_ci95_high": min(1.0, mean + half),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "trials": [float(value) for value in array],
    }


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    half = z * np.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total)) / denominator
    return float(center - half), float(center + half)


def balanced_subset(data: list[str], labels: np.ndarray, per_class: int, seed: int) -> tuple[list[str], np.ndarray]:
    rng = np.random.default_rng(seed)
    selected = []
    for label in np.unique(labels):
        candidates = np.flatnonzero(labels == label)
        selected.extend(rng.choice(candidates, min(per_class, len(candidates)), replace=False))
    selected = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected)
    return [data[index] for index in selected], labels[selected]


def encode_20ng(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data_home = ROOT / "data" / "20newsgroups"
    train = fetch_20newsgroups(
        data_home=data_home,
        subset="train",
        remove=("headers", "footers", "quotes"),
    )
    test = fetch_20newsgroups(
        data_home=data_home,
        subset="test",
        remove=("headers", "footers", "quotes"),
    )
    train_text, train_y = balanced_subset(train.data, np.asarray(train.target), args.topic_train_per_class, args.seed)
    test_text, test_y = balanced_subset(test.data, np.asarray(test.target), args.topic_test_per_class, args.seed + 1)
    cache = ROOT / "cache" / "security_20ng"
    cache.mkdir(parents=True, exist_ok=True)
    train_path = cache / f"granite_train_{len(train_text)}.npy"
    test_path = cache / f"granite_test_{len(test_text)}.npy"
    if train_path.exists() and test_path.exists():
        return np.load(train_path), train_y, np.load(test_path), test_y

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        str(ROOT / "cache" / "models" / "granite-embedding-small-english-r2"),
        device=args.device,
        local_files_only=True,
    )
    # 20 Newsgroups messages can be far longer than BEIR passages.  Fixing the
    # public truncation length keeps the attack reproducible on an 8-GB GPU and
    # prevents a few long messages from exhausting attention memory.
    model.max_seq_length = args.max_seq_length
    train_x = model.encode(
        train_text,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    test_x = model.encode(
        test_text,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    np.save(train_path, train_x)
    np.save(test_path, test_x)
    return train_x, train_y, test_x, test_y


def topic_inference(args: argparse.Namespace) -> dict[str, object]:
    train_x, train_y, test_x, test_y = encode_20ng(args)
    dpe = ConditionalDPE(train_x.shape[1], args.beta, args.scale, args.seed + 100)
    rng = np.random.default_rng(args.seed + 101)
    train_dp = normalize_rows(train_x + rng.normal(0.0, args.dp_sigma, train_x.shape).astype(np.float32))
    test_dp = normalize_rows(test_x + rng.normal(0.0, args.dp_sigma, test_x.shape).astype(np.float32))
    views = {
        "plaintext_embedding": (train_x, test_x),
        "dpe_coordinate": (dpe.encrypt_queries(train_x), dpe.encrypt_queries(test_x)),
        "heuristic_dp_plus_dpe_coordinate": (dpe.encrypt_queries(train_dp), dpe.encrypt_queries(test_dp)),
    }
    results = {}
    for name, (train_view, test_view) in views.items():
        classifier = LogisticRegression(max_iter=1000, solver="lbfgs", n_jobs=1)
        classifier.fit(train_view, train_y)
        predictions = classifier.predict(test_view)
        successes = int(np.count_nonzero(predictions == test_y))
        accuracy = accuracy_score(test_y, predictions)
        low, high = wilson_interval(successes, len(test_y))
        results[name] = {"accuracy": float(accuracy), "wilson_ci95": [low, high]}
        print(f"topic {name:32s} accuracy={accuracy:.4f}")
    return {
        "dataset": "20 Newsgroups",
        "classes": int(len(np.unique(train_y))),
        "train_queries": len(train_y),
        "test_queries": len(test_y),
        "random_accuracy": float(1.0 / len(np.unique(train_y))),
        "results": results,
    }


def known_record_alignment(
    plain: np.ndarray,
    encrypted: np.ndarray,
    seed_counts: list[int],
    target_count: int,
    seed: int,
    repeats: int,
) -> list[dict[str, object]]:
    max_seeds = max(seed_counts)
    rows = []
    for count in seed_counts:
        accuracies, scales = [], []
        for repeat in range(repeats):
            rng = np.random.default_rng(seed + repeat)
            order = rng.permutation(len(plain))
            seed_pool = order[:max_seeds]
            targets = order[max_seeds : max_seeds + target_count]
            seeds = seed_pool[:count]
            plain_seed_pair = cdist(plain[seeds], plain[seeds])
            enc_seed_pair = cdist(encrypted[seeds], encrypted[seeds])
            mask = plain_seed_pair > 1e-8
            scale_hat = float(np.median(enc_seed_pair[mask] / plain_seed_pair[mask]))
            plain_fp = cdist(plain[targets], plain[seeds])
            encrypted_fp = cdist(encrypted[targets], encrypted[seeds]) / scale_hat
            cost = cdist(plain_fp, encrypted_fp)
            public_rows, cipher_columns = linear_sum_assignment(cost)
            accuracies.append(float(np.mean(public_rows == cipher_columns)))
            scales.append(scale_hat)
        summary = mean_summary(accuracies)
        rows.append(
            {
                "known_seed_records": count,
                "unknown_target_records": target_count,
                "repeats": repeats,
                "estimated_scale_mean": float(np.mean(scales)),
                "global_alignment_accuracy": summary,
            }
        )
        print(f"known-record seeds={count:3d} targets={target_count} accuracy={summary['mean']:.4f}")
    return rows


def document_attacks(args: argparse.Namespace) -> dict[str, object]:
    plain = normalize_rows(
        np.load(ROOT / "cache" / "granite_gpu" / "scifact-st-a7972eb5022e_corpus_embeddings.npy").astype(np.float32)
    )
    dpe = ConditionalDPE(plain.shape[1], args.beta, args.scale, args.seed + 200)
    encrypted = dpe.encrypt_database(plain)
    known_record = known_record_alignment(
        plain,
        encrypted,
        args.known_seed_counts,
        args.known_target_count,
        args.seed + 201,
        args.attack_repeats,
    )

    overlaps, aris, nmis = [], [], []
    for repeat in range(args.attack_repeats):
        repeat_seed = args.seed + 202 + repeat
        plain_clusters = KMeans(n_clusters=args.clusters, n_init=10, random_state=repeat_seed).fit_predict(plain)
        encrypted_clusters = KMeans(n_clusters=args.clusters, n_init=10, random_state=repeat_seed).fit_predict(encrypted)
        overlaps.append(neighbor_overlap(plain, encrypted, args.neighbor_sample, args.neighbor_k, repeat_seed))
        aris.append(float(adjusted_rand_score(plain_clusters, encrypted_clusters)))
        nmis.append(float(normalized_mutual_info_score(plain_clusters, encrypted_clusters)))
    reconstruction = {
        "sampled_knn_recall": mean_summary(overlaps),
        "adjusted_rand_index": mean_summary(aris),
        "normalized_mutual_information": mean_summary(nmis),
        "sample_size": min(args.neighbor_sample, len(plain)),
        "neighbors": args.neighbor_k,
        "clusters": args.clusters,
        "repeats": args.attack_repeats,
    }
    print(
        "neighborhood "
        f"recall={reconstruction['sampled_knn_recall']['mean']:.4f} "
        f"ARI={reconstruction['adjusted_rand_index']['mean']:.4f} "
        f"NMI={reconstruction['normalized_mutual_information']['mean']:.4f}"
    )
    return {
        "dataset": "BEIR/SciFact Granite documents",
        "documents": len(plain),
        "known_record_expansion": known_record,
        "neighborhood_reconstruction": reconstruction,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=3.0)
    parser.add_argument("--dp-sigma", type=float, default=0.015)
    parser.add_argument("--topic-train-per-class", type=int, default=150)
    parser.add_argument("--topic-test-per-class", type=int, default=75)
    parser.add_argument("--known-seed-counts", nargs="+", type=int, default=[2, 4, 8, 16, 32, 64])
    parser.add_argument("--known-target-count", type=int, default=300)
    parser.add_argument("--neighbor-sample", type=int, default=512)
    parser.add_argument("--neighbor-k", type=int, default=10)
    parser.add_argument("--clusters", type=int, default=20)
    parser.add_argument("--attack-repeats", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "security_attacks.json")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    report = {
        "configuration": {
            "beta": parsed.beta,
            "scale": parsed.scale,
            "heuristic_dp_sigma": parsed.dp_sigma,
            "seed": parsed.seed,
        },
        "topic_inference": topic_inference(parsed),
        "document_attacks": document_attacks(parsed),
        "cross_view_linkage": {
            "source": "primary Granite run results.json",
            "note": "Executed by run_experiment.py; paper reports 3.04% top-1 with independent aliases and 100% with shared identifiers.",
        },
    }
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    parsed.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {parsed.output}")
