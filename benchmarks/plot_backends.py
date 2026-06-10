"""
plot_backends.py — CPU vs MLX vs Metal comparison charts for the docs.

Reads the per-backend result files produced by ``bench_statevector.py``
(``benchmarks/data/large/macquerel-{cpu,mlx,metal}.json``) and renders the two
charts embedded in ``docs/backends.md``:

- ``backend_runtimes.png`` — absolute runtime vs qubit count, one panel per
  circuit, one line per backend (log y). Shows the three regimes and the
  CPU→GPU crossover around 16q.
- ``backend_relative.png`` — each backend's runtime divided by the fastest
  backend at that (circuit, n) cell. A line at 1.0 is the winner; the gap
  above it is the cost of picking the wrong backend. Makes the per-circuit
  spread visible (e.g. MLX is near-parity on QAOA but far behind on QFT).

Usage:
    uv run python benchmarks/plot_backends.py
    uv run python benchmarks/plot_backends.py --data benchmarks/data/large --out docs/assets

Requires the ``viz`` extra (matplotlib): ``uv sync --extra viz``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BACKENDS = ["cpu", "mlx", "metal"]
COLORS = {"cpu": "#555555", "mlx": "#1f77b4", "metal": "#d62728"}
CIRCUITS = ["ghz", "qft", "random", "qaoa"]


def load(data_dir: Path) -> dict[str, dict[str, dict[int, float]]]:
    """results[backend][circuit][n] = seconds."""
    results: dict[str, dict[str, dict[int, float]]] = {}
    for b in BACKENDS:
        path = data_dir / f"macquerel-{b}.json"
        if not path.exists():
            continue
        raw = json.loads(path.read_text())["results"]
        results[b] = {
            circ: {int(n): t for series in by_backend.values() for n, t in series}
            for circ, by_backend in raw.items()
        }
    return results


def plot_runtimes(results: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, circ in zip(axes.flat, CIRCUITS, strict=True):
        for b in BACKENDS:
            by_n = results.get(b, {}).get(circ, {})
            ns = sorted(by_n)
            ax.plot(
                ns,
                [by_n[n] * 1e3 for n in ns],
                marker="o",
                ms=4,
                color=COLORS[b],
                label=b,
            )
        ax.set_title(circ)
        ax.set_xlabel("qubits")
        ax.set_ylabel("circuit time (ms)")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.suptitle(
        "Backend runtimes by circuit — CPU wins ≤16q, Metal everywhere above",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    print(f"Chart -> {path}")


def plot_relative(results: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, circ in zip(axes.flat, CIRCUITS, strict=True):
        ns = sorted({n for b in BACKENDS for n in results.get(b, {}).get(circ, {})})
        for b in BACKENDS:
            by_n = results.get(b, {}).get(circ, {})
            xs, ys = [], []
            for n in ns:
                if n not in by_n:
                    continue
                fastest = min(
                    results[o][circ][n] for o in BACKENDS if n in results.get(o, {}).get(circ, {})
                )
                xs.append(n)
                ys.append(by_n[n] / fastest)
            ax.plot(xs, ys, marker="o", ms=4, color=COLORS[b], label=b)
        ax.axhline(1.0, color="black", lw=0.8, alpha=0.5)
        ax.set_title(circ)
        ax.set_xlabel("qubits")
        ax.set_ylabel("time / fastest backend")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.suptitle(
        "Runtime relative to the fastest backend per cell — "
        "1.0 = winner; note MLX's near-parity on QAOA vs its QFT gap",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    print(f"Chart -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU/MLX/Metal docs charts")
    parser.add_argument("--data", default="benchmarks/data/large", metavar="DIR")
    parser.add_argument("--out", default="docs/assets", metavar="DIR")
    args = parser.parse_args()

    try:
        import matplotlib

        matplotlib.use("Agg")
    except ImportError:
        raise SystemExit("matplotlib not installed (uv sync --extra viz).") from None

    results = load(Path(args.data))
    if not results:
        raise SystemExit(f"No macquerel-*.json files in {args.data}.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plot_runtimes(results, out / "backend_runtimes.png")
    plot_relative(results, out / "backend_relative.png")


if __name__ == "__main__":
    main()
