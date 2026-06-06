#!/usr/bin/env python3
"""Plot JSON from benchmarks/bench_versions.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def make_plot(payload: dict, path: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed (uv sync --extra viz) - skipping plot.")
        return

    runs = payload["runs"]
    backends = sorted({b for run in runs.values() for b in run["results"]})
    if not backends:
        print("No plottable backend results.")
        return

    fig, axes = plt.subplots(1, len(backends), figsize=(5.2 * len(backends), 4.2), squeeze=False)
    for i, backend in enumerate(backends):
        ax = axes[0][i]
        for label, run in runs.items():
            rows = run["results"].get(backend, [])
            if not rows:
                continue
            xs = [r["n_qubits"] for r in rows]
            ys = [r["ms"] for r in rows]
            ax.plot(xs, ys, marker="o", label=label)
        ax.set_title(backend)
        ax.set_xlabel("qubits")
        ax.set_ylabel("runtime (ms)")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"Chart -> {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("json")
    p.add_argument("--out", default="benchmarks/data/version_regression.png")
    args = p.parse_args()
    make_plot(json.loads(Path(args.json).read_text()), args.out)


if __name__ == "__main__":
    main()
