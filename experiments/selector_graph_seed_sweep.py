"""Multi-key-seed audit of the cross-compartment overlap graph."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if (ROOT / ".deps").exists():
    sys.path.insert(0, str(ROOT / ".deps"))

import numpy as np

from benchmark_dual_compartment_full import KEY_LEXICAL, cell_transform
from benchmark_million_semantic_hybrid import atomic_json
from attack_lexical_stitching_d8192_rc32_b_sweep import graph_metrics

D = 8192
R = 32
SEEDS = 100


def run(cells: int) -> dict[str, object]:
    rows = []
    for seed in range(SEEDS):
        key = KEY_LEXICAL + b"-selector-seed-" + seed.to_bytes(4, "big")
        coordinates = np.stack([
            cell_transform(key, cell, D, R).coordinates for cell in range(cells)
        ])
        graph = graph_metrics(coordinates)
        graph["seed"] = seed
        rows.append(graph)
    sizes = np.asarray([row["largest_component_cells"] for row in rows])
    components = np.asarray([row["connected_components"] for row in rows])
    isolated = np.asarray([row["isolated_cells"] for row in rows])
    edges = np.asarray([row["cell_pairs_with_overlap"] for row in rows])
    return {
        "partitions": cells,
        "seeds": SEEDS,
        "connected_probability": float(np.mean(components == 1)),
        "largest_component_mean": float(np.mean(sizes)),
        "largest_component_p05": float(np.percentile(sizes, 5)),
        "largest_component_min": int(np.min(sizes)),
        "largest_component_fraction_mean": float(np.mean(sizes / cells)),
        "connected_components_mean": float(np.mean(components)),
        "isolated_cells_mean": float(np.mean(isolated)),
        "overlap_edges_mean": float(np.mean(edges)),
        "overlap_edges_p05_p95": [float(np.percentile(edges, 5)), float(np.percentile(edges, 95))],
        "per_seed": rows,
    }


def main() -> None:
    report = {
        "experiment": "selector-key overlap-graph seed sweep",
        "parameters": {"work_dimension": D, "projection_dimension": R, "seeds": SEEDS},
        "configurations": [run(cells) for cells in (16, 32, 64)],
    }
    destination = ROOT / "results" / "security" / "selector_graph_seed_sweep_d8192_rc32"
    destination.mkdir(parents=True, exist_ok=True)
    atomic_json(destination / "results.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
