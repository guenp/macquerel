"""
bench_fusion_width.py — run time vs max_fused_qubits.

Sweeps the gate-fusion width (`max_fused_qubits` ∈ 1..6) across qubit counts and
circuits, timing the real Simulator cost model (fuse + apply) on the available
backends. The point is to show *why macquerel defaults to width 4*:

  - The optimal width drifts upward with qubit count. At small n the one-time
    matrix-composition cost dominates the apply and rewards narrow fusion; as n
    grows the apply (a full pass over the 2**n state) dominates and wider fusion
    wins by making fewer passes.
  - No single width is best everywhere, but width 4 wins on the normalized
    aggregate across the measured 17-30q MLX tier — hence the default.

See the write-up: https://github.com/guenp/macquerel/pull/8#issuecomment-4636543327

Usage:
    uv run python benchmarks/bench_fusion_width.py
    uv run python benchmarks/bench_fusion_width.py --qubits 16 20 22 24 --backend mlx metal
    uv run python benchmarks/bench_fusion_width.py --circuits QFT QAOA --reps 5
    uv run python benchmarks/bench_fusion_width.py --json benchmarks/data/fusion_width.json \
        --plot benchmarks/data/fusion_width.png

Requires the ``viz`` extra (matplotlib) for the plot: ``uv sync --extra viz``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from circuits import build_qaoa, build_qft, build_quantum_volume, build_random

from macquerel.backends.cpu import CPUBackend
from macquerel.circuit import Circuit, Gate
from macquerel.compiler import fuse_gates

WIDTHS = list(range(1, 7))

CIRCUITS = {
    "QFT": lambda n: build_qft(n),
    "random": lambda n: build_random(n, depth=50),
    "QAOA": lambda n: build_qaoa(n),
    "QV": lambda n: build_quantum_volume(n),
}


def _time_fuse_and_apply(backend, circuit: Circuit, width: int, n: int) -> float:
    """Wall-clock seconds to fuse `circuit` at `width` and apply it (the real
    per-call Simulator cost: fusion happens once, then the gates are applied)."""
    t0 = time.perf_counter()
    fused = fuse_gates(circuit, max_fused_qubits=width)
    sv = backend.allocate(n)
    for op in fused.ops:
        if isinstance(op, Gate):
            sv = backend.apply_matrix(sv, op.matrix, op.targets, op.controls or None)
    backend.to_numpy(sv)
    return time.perf_counter() - t0


def _make_backends(requested: list[str] | None) -> dict:
    names = requested or ["cpu", "mlx", "metal"]
    backends: dict = {}
    for name in names:
        try:
            if name == "cpu":
                backends["cpu"] = CPUBackend()
            elif name == "mlx":
                from macquerel.backends.mlx_backend import MLXBackend

                backends["mlx"] = MLXBackend()
            elif name == "metal":
                from macquerel.backends.metal_backend import MetalBackend

                backends["metal"] = MetalBackend()
            else:
                raise ValueError(f"unknown backend {name!r}")
        except Exception as e:  # optional backend not available — skip it
            print(f"  backend {name:6s} unavailable ({str(e)[:60]})")
    return backends


def benchmark(qubit_counts: list[int], circuits: list[str], backends: dict, reps: int) -> dict:
    # results["<backend>"]["<circuit>"]["<n>"] = {width: ms, ...}
    results: dict = {}
    for bname, backend in backends.items():
        results[bname] = {}
        print(f"\n=== backend: {bname} ===")
        for cname in circuits:
            results[bname][cname] = {}
            print(f"  {cname}:")
            header = "    n   " + "  ".join(f"w{w:>6}" for w in WIDTHS) + "   best"
            print(header)
            for n in qubit_counts:
                circuit = CIRCUITS[cname](n)
                row: dict[int, float] = {}
                for w in WIDTHS:
                    _time_fuse_and_apply(backend, circuit, w, n)  # warm-up
                    secs = min(_time_fuse_and_apply(backend, circuit, w, n) for _ in range(reps))
                    row[w] = round(secs * 1e3, 3)
                results[bname][cname][str(n)] = row
                best_w = min(row, key=lambda w: row[w])
                cells = "  ".join(f"{row[w]:7.2f}" for w in WIDTHS)
                print(f"    {n:<3d} {cells}   w{best_w}")
    return results


def _aggregate_optima(results: dict) -> None:
    """Print, per backend, the per-n best width and the normalized aggregate
    winner across all (circuit, n) cells — the figure that justifies the default."""
    print("\n=== normalized aggregate (each cell scaled by its own fastest width) ===")
    for bname, by_circuit in results.items():
        agg = dict.fromkeys(WIDTHS, 0.0)
        cells = 0
        for _cname, by_n in by_circuit.items():
            for _n, row in by_n.items():
                fastest = min(row.values())
                if fastest <= 0:
                    continue
                cells += 1
                for w in WIDTHS:
                    agg[w] += row[w] / fastest
        if not cells:
            continue
        norm = {w: round(agg[w] / cells, 3) for w in WIDTHS}
        winner = min(norm, key=lambda w: norm[w])
        print(f"  {bname:6s}: {norm}  -> aggregate winner = w{winner}")


def make_plot(results: dict, path: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed (uv sync --extra viz) — skipping plot.")
        return

    # One row per backend, columns: per-circuit absolute ms + a normalized-aggregate panel.
    backends = list(results)
    circuits = sorted({c for b in results.values() for c in b})
    ncols = len(circuits) + 1
    nrows = len(backends)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.6 * nrows), squeeze=False)
    cmap = plt.get_cmap("viridis")

    for r, bname in enumerate(backends):
        by_circuit = results[bname]
        for c, cname in enumerate(circuits):
            ax = axes[r][c]
            by_n = by_circuit.get(cname, {})
            ns = sorted(int(n) for n in by_n)
            for i, n in enumerate(ns):
                row = by_n[str(n)]
                ys = [row[w] for w in WIDTHS]
                color = cmap(i / max(1, len(ns) - 1))
                ax.plot(WIDTHS, ys, marker="o", color=color, label=f"{n}q")
                best_w = min(row, key=lambda w: row[w])
                ax.scatter(
                    [best_w],
                    [row[best_w]],
                    s=120,
                    facecolors="none",
                    edgecolors=color,
                    linewidths=2,
                    zorder=5,
                )
            ax.axvline(4, color="crimson", ls="--", lw=1, alpha=0.6)
            ax.set_title(f"{bname} · {cname}")
            ax.set_xlabel("max_fused_qubits")
            ax.set_ylabel("fuse+apply (ms)")
            ax.set_yscale("log")
            ax.set_xticks(WIDTHS)
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=8, title="qubits")

        # Normalized-aggregate panel for this backend.
        ax = axes[r][len(circuits)]
        agg = dict.fromkeys(WIDTHS, 0.0)
        cells = 0
        for by_n in by_circuit.values():
            for row in by_n.values():
                fastest = min(row.values())
                if fastest <= 0:
                    continue
                cells += 1
                for w in WIDTHS:
                    agg[w] += row[w] / fastest
        if cells:
            ys = [agg[w] / cells for w in WIDTHS]
            ax.plot(WIDTHS, ys, marker="s", color="black", lw=2)
            winner = WIDTHS[ys.index(min(ys))]
            ax.scatter(
                [winner],
                [min(ys)],
                s=160,
                facecolors="none",
                edgecolors="crimson",
                linewidths=2.5,
                zorder=5,
                label=f"winner w{winner}",
            )
            ax.axvline(4, color="crimson", ls="--", lw=1, alpha=0.6, label="default w4")
            ax.legend(fontsize=8)
        ax.set_title(f"{bname} · aggregate (normalized)")
        ax.set_xlabel("max_fused_qubits")
        ax.set_ylabel("mean time / per-cell best")
        ax.set_xticks(WIDTHS)
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        "Run time vs max_fused_qubits — optimal width drifts up with qubit count; "
        "4 is the robust default (red dashed)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=130)
    print(f"Chart -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run time vs max_fused_qubits sweep")
    parser.add_argument("--qubits", nargs="+", type=int, default=[16, 18, 20, 22, 24])
    parser.add_argument("--circuits", nargs="+", default=list(CIRCUITS), choices=list(CIRCUITS))
    parser.add_argument("--backend", nargs="+", default=None, help="subset of: cpu mlx metal")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--json", metavar="FILE")
    parser.add_argument("--plot", metavar="FILE", default="benchmarks/data/fusion_width.png")
    args = parser.parse_args()

    backends = _make_backends(args.backend)
    if not backends:
        raise SystemExit("No backends available.")

    results = benchmark(sorted(args.qubits), args.circuits, backends, args.reps)
    _aggregate_optima(results)

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2))
        print(f"\nResults written to {path}")

    if args.plot:
        make_plot(results, args.plot)


if __name__ == "__main__":
    main()
