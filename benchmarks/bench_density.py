#!/usr/bin/env python3
"""bench_density.py — DensityMatrixSimulator runtime across backends and qubit counts.

Times `DensityMatrixSimulator.probabilities` end to end (evolve the vectorized
density matrix + read back the 2**N diagonal) on two noisy circuit families:

- ``ghz_noise``: GHZ + one depolarizing channel per qubit — channel-light,
  dominated by the doubled-state gate applies;
- ``random_noise``: brickwork random rotations + CX layers with a channel per
  layer — fusion-rich, channel-heavy.

Each (circuit, backend, N) cell runs in a fresh subprocess (no GPU memory-pool
or lazy-graph contamination across cells), with one warm-up run and
min-of-reps timing inside the worker — the same protocol as the statevector
benchmarks.

An N-qubit density matrix is a ``4**N * 8``-byte doubled statevector, so cells
are budget-gated with each backend's peak multiplier at the doubled size and a
hard 64 GiB cap: on a 128 GiB machine the largest cell is metal @ N=16
(32 GiB state). MLX is capped at N=15 (int32 ShapeElem at 2N=30) and CPU at
N=13 for patience, not memory.

Usage:
    uv run python benchmarks/bench_density.py
    uv run python benchmarks/bench_density.py --qubits 2 4 6 8 --backends cpu metal \
        --json benchmarks/data/density.json --plot benchmarks/data/density.png
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

# Budget calibration shared with the other harnesses.
from bench_statevector import _PEAK_MULT, _system_ram_gib

BACKENDS = ["cpu", "mlx", "metal"]
CIRCUITS = ["ghz_noise", "random_noise"]
# Per-backend N ceilings at the doubled count (see module docstring).
MAX_QUBITS = {"cpu": 13, "mlx": 15, "metal": 16}
HARD_CAP_GIB = 64.0

BACKEND_COLORS = {"metal": "#d62728", "mlx": "#1f77b4", "cpu": "#2ca02c"}


def build_circuit(kind: str, n: int):
    from macquerel.circuit import Circuit

    qc = Circuit(n)
    if kind == "ghz_noise":
        qc.h(0)
        for i in range(n - 1):
            qc.cx(i, i + 1)
        for q in range(n):
            qc.depolarizing(q, 0.01)
        return qc
    if kind == "random_noise":
        rng = np.random.default_rng(17)
        channels = ["depolarizing", "amplitude_damping", "phase_damping"]
        for d in range(4):
            for q in range(n):
                qc.ry(q, float(rng.uniform(0, 2 * np.pi)))
                qc.rz(q, float(rng.uniform(0, 2 * np.pi)))
            for q in range(d % 2, n - 1, 2):
                qc.cx(q, q + 1)
            ch = channels[d % len(channels)]
            getattr(qc, ch)(int(rng.integers(n)), 0.05)
        return qc
    raise ValueError(f"unknown circuit kind {kind!r}")


def est_peak_gib(backend: str, n: int) -> float:
    mult = _PEAK_MULT.get(f"macquerel-{backend}", 4.0)
    return (4**n) * 8 * mult / 1024**3


# ----------------------------------------------------------------------------
# Worker: one (circuit, backend, n) cell — warm-up + min-of-reps, JSON on stdout.
# ----------------------------------------------------------------------------
def run_worker(args) -> int:
    import time

    from macquerel.density import DensityMatrixSimulator

    qc = build_circuit(args.circuit, args.n)
    dm = DensityMatrixSimulator(backend=args.backend)

    def once() -> float:
        t0 = time.perf_counter()
        dm.probabilities(qc)
        return time.perf_counter() - t0

    once()  # warm-up: pipeline/kernel compilation, buffer pool fill
    secs = min(once() for _ in range(args.reps))
    print(json.dumps({"seconds": secs}))
    return 0


def measure_cell(circuit: str, backend: str, n: int, reps: int, timeout: float) -> float:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--circuit",
        circuit,
        "--backend",
        backend,
        "--n",
        str(n),
        "--reps",
        str(reps),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        msg = (proc.stderr.strip().splitlines() or ["worker exited nonzero"])[-1]
        raise RuntimeError(msg[:120])
    return float(json.loads(proc.stdout.strip().splitlines()[-1])["seconds"])


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--qubits", type=int, nargs="+", default=list(range(2, 17)))
    ap.add_argument("--backends", nargs="+", default=BACKENDS, choices=BACKENDS)
    ap.add_argument("--circuits", nargs="+", default=CIRCUITS, choices=CIRCUITS)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument(
        "--mem-budget-frac",
        type=float,
        default=0.45,
        help="skip cells whose pessimistic peak estimate exceeds this fraction of RAM",
    )
    ap.add_argument("--json", default="benchmarks/data/density.json")
    ap.add_argument("--plot", default="benchmarks/data/density.png")
    ap.add_argument(
        "--replot",
        action="store_true",
        help="skip measuring; redraw --plot from the existing --json",
    )
    # Worker mode (internal).
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--circuit", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--backend", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--n", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        raise SystemExit(run_worker(args))

    if args.replot:
        results = json.loads(Path(args.json).read_text())
        make_plot(results, args.backends, args.plot)
        return

    budget = min(_system_ram_gib() * args.mem_budget_frac, HARD_CAP_GIB)
    print(f"per-cell memory budget: {budget:.0f} GiB (hard cap {HARD_CAP_GIB:.0f} GiB)\n")

    results: dict = {"benchmark": "density_matrix_runtime", "data": {}}
    for circuit in args.circuits:
        results["data"][circuit] = {b: {} for b in args.backends}
        print(f"[{circuit}] {'n':>3}" + "".join(f" {b:>12}" for b in args.backends))
        for n in args.qubits:
            row = f"[{circuit}] {n:>3}"
            for b in args.backends:
                if n > MAX_QUBITS[b] or est_peak_gib(b, n) > budget:
                    row += f" {'-':>12}"
                    continue
                try:
                    secs = measure_cell(circuit, b, n, args.reps, args.timeout)
                    results["data"][circuit][b][str(n)] = secs
                    row += f" {secs:>11.4g}s"
                except Exception as e:
                    row += f" ERR({str(e)[:20]})"
            print(row, flush=True)
            if args.json:
                path = Path(args.json)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(results, indent=2))
        print()

    if args.json:
        print(f"Raw data -> {args.json}")
    make_plot(results, args.backends, args.plot)


def make_plot(results: dict, backends: list[str], path: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping chart (raw data still saved).")
        return

    circuits = list(results["data"])
    fig, axes = plt.subplots(1, len(circuits), figsize=(6.5 * len(circuits), 5), squeeze=False)
    for ax, circuit in zip(axes[0], circuits, strict=True):
        for b in backends:
            cells = {int(k): v for k, v in results["data"][circuit].get(b, {}).items()}
            if not cells:
                continue
            xs = sorted(cells)
            ax.plot(
                xs,
                [cells[x] for x in xs],
                marker="o",
                markersize=4,
                linewidth=1.6,
                color=BACKEND_COLORS.get(b),
                label=b,
            )
        ax.set_xlabel("qubits N (density matrix = 2N-qubit state)")
        ax.set_ylabel("seconds (min of reps)")
        ax.set_yscale("log")
        ax.set_title(circuit)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("DensityMatrixSimulator.probabilities runtime", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"Chart -> {path}")


if __name__ == "__main__":
    main()
