"""Print or execute the documented DuetER reproduction profiles."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def command(*parts: str) -> list[str]:
    return [PYTHON, *parts]


PROFILES: dict[str, list[list[str]]] = {
    "quick": [
        command("-m", "unittest", "discover", "-s", "tests", "-v"),
        command("experiments/download_beir.py", "scifact"),
        command(
            "experiments/run_experiment.py",
            "--data-dir", "experiments/data/scifact",
            "--dataset-name", "scifact",
            "--dense-backend", "lsa",
            "--results-dir", "results/generated/scifact_lsa",
        ),
    ],
    "full": [
        command(
            "experiments/benchmark_dual_compartment_full.py",
            "--dataset", dataset,
            "--split", "dev" if dataset == "msmarco" else "test",
            "--projection-dimension", "256",
            "--semantic-probes", "128",
            "--device", "cuda",
            "--cache-root", "experiments/cache/dual_compartment_full",
            "--output-root", "results/generated/dual_compartment_full",
        )
        for dataset in ("nq", "hotpotqa", "msmarco")
    ],
    "security": [
        command(
            "experiments/run_compartment_security.py",
            "--projection-dim", "256",
            "--beta", "0.10",
            "--scale", "3.0",
            "--output", "results/generated/compartment_security.json",
        ),
        command("experiments/attack_cross_compartment_stitching.py"),
    ],
    "ablations": [
        command("experiments/ablate_compartment_nprobe.py"),
        command("experiments/ablate_compartment_larger_lexical_depth.py"),
        command("experiments/ablate_duetrank_calibration_final.py"),
        command("experiments/measure_compartment_projection.py"),
    ],
}
PROFILES["all"] = (
    PROFILES["quick"] + PROFILES["full"] + PROFILES["security"] + PROFILES["ablations"]
)


def render(parts: list[str]) -> str:
    return subprocess.list2cmdline(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="quick")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="run commands sequentially")
    mode.add_argument("--dry-run", action="store_true", help="print commands only (default)")
    args = parser.parse_args()

    for index, parts in enumerate(PROFILES[args.profile], start=1):
        print(f"[{index}/{len(PROFILES[args.profile])}] {render(parts)}", flush=True)
        if args.execute:
            subprocess.run(parts, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

