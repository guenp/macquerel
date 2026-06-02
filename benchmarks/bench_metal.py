"""Benchmark: CPU vs MLX vs Metal, emphasising the qubit range each can reach.

The Metal backend is a *capacity* extension, not an equal-qubit speedup: it is
the only backend that runs 31-32q (MLX's int32 ShapeElem rejects >=2**31
amplitudes; CPU is impractically slow and memory-heavy there). At overlapping
sizes Metal is not expected to beat MLX -- both are memory-bandwidth bound, and
Metal synchronises per gate whereas MLX fuses a lazy graph. This benchmark runs
each backend only where it is viable and records per-backend circuit time.

Usage:
    uv run python benchmarks/bench_metal.py --json benchmarks/data/<commit>-metal.json
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np

import macquerel.gates as g
from macquerel.backends.cpu import CPUBackend

_SINGLE = [g.H, g.X, g.Z, g.S, lambda: g.Rz(0.7), lambda: g.Rx(0.3)]
_TWO = [g.CNOT, g.CZ, g.SWAP]

# Per-backend qubit caps: CPU is impractical above ~24q; MLX hard-caps at 30q.
_CPU_MAX = 24
_MLX_MAX = 30


def _random_ops(n, depth, rng):
    ops = []
    for _ in range(depth):
        if n >= 2 and rng.random() < 0.4:
            pair = rng.choice(n, size=2, replace=False).tolist()
            ops.append((_TWO[rng.integers(len(_TWO))](), pair))
        else:
            ops.append((_SINGLE[rng.integers(len(_SINGLE))](), [int(rng.integers(n))]))
    return ops


def _flush(backend, sv):
    if isinstance(sv, np.ndarray):
        return  # CPU: eager
    if type(sv).__name__ == "MetalState":
        # Metal synchronises per gate (waitUntilCompleted), so all work is done
        # when the loop ends. Touch one amplitude via the zero-copy view to
        # confirm completion -- a full to_numpy would add a second copy (64 GiB
        # at 33q) and blow the memory budget.
        _ = backend._view(sv)[0]
        return
    backend.to_numpy(sv)  # MLX: force the lazy graph to evaluate


def _run(backend, n, ops):
    sv = backend.allocate(n)
    t0 = time.perf_counter()
    for mat, targets in ops:
        sv = backend.apply_matrix(sv, mat, targets)
    _flush(backend, sv)
    dt = time.perf_counter() - t0
    del sv
    gc.collect()
    return dt


def benchmark(qubit_counts, depth, reps, seed=42):
    try:
        from macquerel.backends.mlx_backend import MLXBackend

        mlx = MLXBackend()
    except ImportError:
        mlx = None
    try:
        from macquerel.backends.metal_backend import MetalBackend

        metal = MetalBackend()
    except ImportError:
        metal = None

    cpu = CPUBackend()
    rng = np.random.default_rng(seed)
    results = []

    print(f"\n{'qubits':>6}  {'depth':>5}  {'CPU (ms)':>10}  {'MLX (ms)':>10}  {'Metal (ms)':>11}")
    print("-" * 52)

    for n in qubit_counts:
        ops = _random_ops(n, depth, rng)
        # Large-n Metal runs are expensive (per-gate sync) -> fewer reps.
        n_reps = reps if n <= 24 else 1

        cpu_ms = min(_run(cpu, n, ops) for _ in range(n_reps)) * 1000 if n <= _CPU_MAX else None
        mlx_ms = (
            min(_run(mlx, n, ops) for _ in range(n_reps)) * 1000
            if mlx is not None and n <= _MLX_MAX
            else None
        )

        # Past MLX's range, release its cached arange index (up to ~4 GiB at
        # 30q) + Metal cache pool so the 31-33q in-place buffers (16/32/64 GiB)
        # have maximum headroom.
        if mlx is not None and n >= _MLX_MAX:
            mlx = None
            try:
                import mlx.core as _mx

                _mx.clear_cache()
            except Exception:
                pass
            gc.collect()

        metal_ms = min(_run(metal, n, ops) for _ in range(n_reps)) * 1000 if metal else None

        row = {
            "n_qubits": n,
            "depth": depth,
            "reps": n_reps,
            "cpu_ms": round(cpu_ms, 3) if cpu_ms is not None else None,
            "mlx_ms": round(mlx_ms, 3) if mlx_ms is not None else None,
            "metal_ms": round(metal_ms, 3) if metal_ms is not None else None,
        }
        results.append(row)
        fmt = lambda v: f"{v:>10.1f}" if v is not None else f"{'n/a':>10}"
        print(f"{n:>6}  {depth:>5}  {fmt(cpu_ms)}  {fmt(mlx_ms)}  {fmt(metal_ms):>11}", flush=True)

    print()
    return results


def main():
    p = argparse.ArgumentParser(description="CPU vs MLX vs Metal backend benchmark")
    p.add_argument(
        "--qubits", nargs="+", type=int, default=[16, 18, 20, 22, 24, 26, 28, 30, 31, 32, 33]
    )
    p.add_argument("--depth", type=int, default=30)
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--json", metavar="FILE")
    args = p.parse_args()

    print(f"Benchmarking: {len(args.qubits)} qubit counts, depth={args.depth}, reps={args.reps}")
    print(f"(CPU<= {_CPU_MAX}q, MLX<= {_MLX_MAX}q, Metal unbounded)")
    results = benchmark(sorted(args.qubits), args.depth, args.reps, args.seed)

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"config": vars(args), "results": results}, f, indent=2)
        print(f"Results written to {path}")


if __name__ == "__main__":
    main()
