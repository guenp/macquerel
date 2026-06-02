"""
Plot CPU vs MLX backend benchmark results across commits.

Reads every bench_backends JSON file in benchmarks/data/ (named
``<commit>-<label>.json``) and draws time-vs-qubits curves: one CPU
reference line plus one MLX line per commit, so the effect of each
optimization is visible at a glance. A second panel shows MLX speedup
over CPU.

Requires the ``viz`` extra (matplotlib): ``uv sync --extra viz``.

Usage:
    uv run python benchmarks/plot_results.py
    uv run python benchmarks/plot_results.py --data-dir benchmarks/data --out benchmarks/data/benchmark.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _load(path: Path) -> dict:
    data = json.loads(path.read_text())
    results = sorted(data["results"], key=lambda r: r["n_qubits"])
    commit, _, label = path.stem.partition("-")
    return {"commit": commit, "label": label or path.stem, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot backend benchmark results")
    parser.add_argument("--data-dir", default="benchmarks/data")
    parser.add_argument("--out", default="benchmarks/data/benchmark.png")
    args = parser.parse_args()

    files = sorted(Path(args.data_dir).glob("*.json"))
    if not files:
        raise SystemExit(f"No benchmark JSON files found in {args.data_dir}")
    runs = [_load(f) for f in files]

    # Order slowest -> fastest (by mean MLX time) so the color ramp tracks the
    # optimization progression regardless of commit-hash filename ordering.
    def _mean_mlx(run: dict) -> float:
        vals = [r["mlx_ms"] for r in run["results"] if r["mlx_ms"]]
        return sum(vals) / len(vals) if vals else 0.0

    runs.sort(key=_mean_mlx, reverse=True)

    fig, (ax_time, ax_speed) = plt.subplots(1, 2, figsize=(13, 5))

    # CPU reference: take the most recent run's CPU numbers (they're ~stable).
    ref = runs[-1]
    qubits = [r["n_qubits"] for r in ref["results"]]
    cpu_ms = [r["cpu_ms"] for r in ref["results"]]
    ax_time.plot(qubits, cpu_ms, "k--o", linewidth=2, label="CPU (NumPy)")

    cmap = plt.get_cmap("viridis")
    for i, run in enumerate(runs):
        color = cmap(i / max(1, len(runs) - 1))
        q = [r["n_qubits"] for r in run["results"]]
        mlx = [r["mlx_ms"] for r in run["results"]]
        name = f"MLX {run['label']} ({run['commit']})"
        ax_time.plot(q, mlx, "-o", color=color, label=name)

        speed = [r["cpu_ms"] / r["mlx_ms"] for r in run["results"] if r["mlx_ms"]]
        sq = [r["n_qubits"] for r in run["results"] if r["mlx_ms"]]
        ax_speed.plot(sq, speed, "-o", color=color, label=name)

    ax_time.set_yscale("log")
    ax_time.set_xlabel("qubits")
    ax_time.set_ylabel("time (ms, log scale)")
    ax_time.set_title("Circuit runtime: CPU vs MLX (depth-50 random circuit)")
    ax_time.grid(True, which="both", alpha=0.3)
    ax_time.legend(fontsize=8)

    ax_speed.axhline(1.0, color="k", linestyle="--", linewidth=1, label="parity (CPU = MLX)")
    ax_speed.set_xlabel("qubits")
    ax_speed.set_ylabel("MLX speedup over CPU (×)")
    ax_speed.set_title("MLX speedup over CPU (>1 = MLX faster)")
    ax_speed.grid(True, alpha=0.3)
    ax_speed.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
