"""Render benchmark-3.png: CPU vs MLX vs Metal circuit time across qubit counts.

Tells the Step 15 story: Metal is the only backend past 30q (MLX's int32
ceiling), and stays in-place where MLX double-buffers, so it also pulls far
ahead of MLX at 26-30q. Reads the JSON written by benchmarks/bench_metal.py.

Usage:
    uv run python benchmarks/plot_metal.py benchmarks/data/<commit>-metal.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else _latest_metal_json()
    data = json.loads(src.read_text())
    results = data["results"]
    depth = data["config"]["depth"]

    qs = [r["n_qubits"] for r in results]

    def series(key):
        xs = [r["n_qubits"] for r in results if r[key] is not None]
        ys = [r[key] for r in results if r[key] is not None]
        return xs, ys

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for key, label, color, marker in [
        ("cpu_ms", "CPU (NumPy)", "#888888", "o"),
        ("mlx_ms", "MLX", "#1f77b4", "s"),
        ("metal_ms", "Metal (this work)", "#d62728", "D"),
    ]:
        xs, ys = series(key)
        if xs:
            ax.plot(xs, ys, marker=marker, color=color, label=label, linewidth=2, markersize=6)

    # MLX int32 ceiling: it cannot allocate >=2**31 amplitudes (>30q).
    ax.axvspan(30.5, max(qs) + 0.5, color="#d62728", alpha=0.07)
    ax.axvline(30.5, color="#d62728", ls="--", lw=1, alpha=0.6)
    ymax = max(r["metal_ms"] for r in results if r["metal_ms"] is not None)
    ax.text(31, ymax * 0.5, "Metal-only\n(MLX int32\nceiling: 30q)",
            color="#d62728", fontsize=9, ha="center", va="center")

    ax.set_yscale("log")
    ax.set_xlabel("qubits")
    ax.set_ylabel(f"circuit time (ms, log) — depth {depth}")
    ax.set_title("macquerel backends: CPU vs MLX vs Metal")
    ax.set_xticks(qs)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="upper left")
    fig.tight_layout()

    out = Path(__file__).parent / "data" / "benchmark-3.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


def _latest_metal_json() -> Path:
    data_dir = Path(__file__).parent / "data"
    cands = sorted(data_dir.glob("*-metal.json"))
    if not cands:
        raise SystemExit("no *-metal.json in benchmarks/data; run bench_metal.py first")
    return cands[-1]


if __name__ == "__main__":
    main()
