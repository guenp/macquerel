#!/usr/bin/env python3
"""bench_batched.py — BatchedSimulator vs a per-circuit Simulator loop (Step 31).

The batched-simulation pitch is amortization: a parameter sweep of B small
circuits pays the fixed per-run costs once per *gate position* instead of once
per circuit x gate. This times a VQE-style hardware-efficient ansatz sweep —
the workload named in the plan — at several (n, B) points, for the batched
engines (cpu, mlx) against the per-circuit loop on the same backend that
auto-select would pick for a single circuit of that size.

Usage:
    uv run python benchmarks/bench_batched.py
    uv run python benchmarks/bench_batched.py --qubits 8 12 --batches 64 512 \
        --json benchmarks/data/batched.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from macquerel import BatchedSimulator, Circuit, Simulator


def ansatz(n: int, thetas: np.ndarray, layers: int = 3) -> Circuit:
    """Hardware-efficient ansatz: [Ry+Rz layer, CX ladder] x layers."""
    qc = Circuit(n)
    t = iter(thetas)
    for _ in range(layers):
        for q in range(n):
            qc.ry(q, float(next(t)))
            qc.rz(q, float(next(t)))
        for q in range(n - 1):
            qc.cx(q, q + 1)
    return qc


def sweep(n: int, batch: int, layers: int = 3, seed: int = 0) -> list[Circuit]:
    rng = np.random.default_rng(seed)
    return [ansatz(n, rng.uniform(0, 2 * np.pi, size=2 * n * layers), layers) for _ in range(batch)]


def time_call(fn, reps: int) -> float:
    fn()  # warm-up
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qubits", type=int, nargs="+", default=[4, 8, 12, 16])
    ap.add_argument("--batches", type=int, nargs="+", default=[32, 128, 512])
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    results: dict = {}
    print(
        f"{'n':>3} {'B':>5} {'loop(auto)':>12} {'batched-cpu':>12} {'batched-mlx':>12}"
        f" {'speedup':>9}"
    )
    for n in args.qubits:
        for b in args.batches:
            circuits = sweep(n, b, args.layers)
            sim = Simulator()

            def loop_run(sim=sim, circuits=circuits):
                for c in circuits:
                    sim.statevector(c)

            cell: dict = {}
            cell["loop_auto"] = time_call(loop_run, args.reps)
            for engine in ("cpu", "mlx"):
                try:
                    bs = BatchedSimulator(backend=engine)
                    cell[f"batched_{engine}"] = time_call(
                        lambda bs=bs, circuits=circuits: bs.statevectors(circuits), args.reps
                    )
                except ImportError:
                    cell[f"batched_{engine}"] = None
            results[f"{n}q_b{b}"] = cell
            best_batched = min(v for k, v in cell.items() if k != "loop_auto" and v)
            print(
                f"{n:>3} {b:>5} {cell['loop_auto'] * 1e3:>10.1f}ms"
                f" {cell['batched_cpu'] * 1e3:>10.1f}ms"
                + (
                    f" {cell['batched_mlx'] * 1e3:>10.1f}ms"
                    if cell.get("batched_mlx")
                    else f" {'-':>12}"
                )
                + f" {cell['loop_auto'] / best_batched:>8.1f}x",
                flush=True,
            )

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"benchmark": "batched_sweep", "results": results}, indent=2))
        print(f"Raw data -> {args.json}")


if __name__ == "__main__":
    main()
