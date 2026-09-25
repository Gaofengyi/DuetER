"""Sweep topic leakage over compartment counts using the fixed 20NG cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_compartment_security import topic_attack


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--beta", type=float, default=0.10)
    parser.add_argument("--scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "compartment_topic_sweep.json")
    args = parser.parse_args()
    rows = []
    for cells in args.cells:
        print(f"topic sweep cells={cells}", flush=True)
        local = argparse.Namespace(
            topic_train_per_class=150,
            topic_test_per_class=75,
            topic_cells=cells,
            projection_dim=args.projection_dim,
            beta=args.beta,
            scale=args.scale,
            seed=args.seed,
        )
        rows.append({"cells": cells, "attack": topic_attack(local)})
    report = {
        "configuration": {
            "projection_dimension": args.projection_dim,
            "beta": args.beta,
            "scale": args.scale,
            "seed": args.seed,
        },
        "sweep": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

