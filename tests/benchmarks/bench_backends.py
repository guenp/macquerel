"""
Benchmark: CPU backend vs MLX backend.

Measures minimum wall-clock time (over --reps repetitions) to apply a
random circuit of the given depth to an n-qubit state vector, for a range
of qubit counts.  Results are printed as a table and optionally written to
a JSON file for later comparison or plotting.

Usage:
    uv run python tests/benchmarks/bench_backends.py
    uv run python tests/benchmarks/bench_backends.py --qubits 10 14 18 22 --depth 30 --reps 5
    uv run python tests/benchmarks/bench_backends.py --json results/run1.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import macquerel.gates as g
from macquerel.backends.cpu import CPUBackend

_SINGLE_GATES = [g.H, g.X, g.Z, g.S, lambda: g.Rz(0.7), lambda: g.Rx(0.3)]
_TWO_QUBIT_GATES = [g.CNOT, g.CZ, g.SWAP]


def _random_ops(n: int, depth: int, rng: np.random.Generator) -> list[tuple]:
    ops = []
    qubits = list(range(n))
    for _ in range(depth):
        if n >= 2 and rng.random() < 0.4:
            pair = rng.choice(qubits, size=2, replace=False).tolist()
            mat = _TWO_QUBIT_GATES[rng.integers(0, len(_TWO_QUBIT_GATES))]()
            ops.append((mat, pair))
        else:
            q = int(rng.integers(0, n))
            mat = _SINGLE_GATES[rng.integers(0, len(_SINGLE_GATES))]()
            ops.append((mat, [q]))
    return ops


def _run(backend, n: int, ops: list[tuple]) -> float:
    sv = backend.allocate(n)
    t0 = time.perf_counter()
    for mat, targets in ops:
        sv = backend.apply_matrix(sv, mat, targets)
    # flush MLX lazy graph if present
    try:
        import mlx.core as mx
        if isinstance(sv, np.ndarray):
            pass
        else:
            mx.eval(sv)
    except ImportError:
        pass
    return time.perf_counter() - t0


def _warmup(backend, n: int, ops: list[tuple]) -> None:
    sv = backend.allocate(n)
    for mat, targets in ops[:min(10, len(ops))]:
        sv = backend.apply_matrix(sv, mat, targets)
    try:
        import mlx.core as mx
        mx.eval(sv) if not isinstance(sv, np.ndarray) else None
    except ImportError:
        pass


def benchmark(
    qubit_counts: list[int],
    depth: int,
    reps: int,
    seed: int = 42,
) -> list[dict]:
    try:
        from macquerel.backends.mlx_backend import MLXBackend
        mlx_backend = MLXBackend()
        has_mlx = True
    except ImportError:
        mlx_backend = None
        has_mlx = False

    cpu = CPUBackend()
    rng = np.random.default_rng(seed)
    results = []

    for n in qubit_counts:
        ops = _random_ops(n, depth, rng)

        _warmup(cpu, n, ops)
        if has_mlx:
            _warmup(mlx_backend, n, ops)

        cpu_times = [_run(cpu, n, ops) for _ in range(reps)]
        mlx_times = [_run(mlx_backend, n, ops) for _ in range(reps)] if has_mlx else []

        cpu_ms = min(cpu_times) * 1000
        mlx_ms = min(mlx_times) * 1000 if mlx_times else None

        results.append({
            "n_qubits": n,
            "depth": depth,
            "reps": reps,
            "cpu_ms": round(cpu_ms, 3),
            "mlx_ms": round(mlx_ms, 3) if mlx_ms is not None else None,
            "speedup": round(cpu_ms / mlx_ms, 3) if mlx_ms else None,
        })

    return results


def _print_table(results: list[dict]) -> None:
    has_mlx = any(r["mlx_ms"] is not None for r in results)
    if has_mlx:
        print(f"\n{'qubits':>6}  {'depth':>5}  {'CPU (ms)':>10}  {'MLX (ms)':>10}  {'speedup':>9}")
        print("-" * 48)
        for r in results:
            speedup = r["speedup"]
            tag = ""
            if speedup is not None:
                if speedup > 1.5:
                    tag = "  MLX faster"
                elif speedup < 0.67:
                    tag = "  CPU faster"
            mlx_str = f"{r['mlx_ms']:>10.1f}" if r["mlx_ms"] is not None else f"{'n/a':>10}"
            sp_str = f"{speedup:>8.2f}x" if speedup is not None else f"{'n/a':>9}"
            print(f"{r['n_qubits']:>6}  {r['depth']:>5}  {r['cpu_ms']:>10.1f}  {mlx_str}  {sp_str}{tag}")
    else:
        print(f"\n{'qubits':>6}  {'depth':>5}  {'CPU (ms)':>10}")
        print("-" * 26)
        for r in results:
            print(f"{r['n_qubits']:>6}  {r['depth']:>5}  {r['cpu_ms']:>10.1f}")
    print()


def _ascii_chart(results: list[dict]) -> None:
    """Print a simple horizontal bar chart of CPU times."""
    max_ms = max(r["cpu_ms"] for r in results)
    width = 40
    print("CPU time (ms) — relative scale:")
    for r in results:
        bar_len = int(r["cpu_ms"] / max_ms * width)
        bar = "█" * bar_len
        print(f"  {r['n_qubits']:>2}q  {bar:<{width}}  {r['cpu_ms']:.1f} ms")
    if any(r["mlx_ms"] for r in results):
        max_mlx = max(r["mlx_ms"] for r in results if r["mlx_ms"])
        print("\nMLX time (ms) — relative scale:")
        for r in results:
            if r["mlx_ms"] is None:
                continue
            bar_len = int(r["mlx_ms"] / max_mlx * width)
            bar = "█" * bar_len
            print(f"  {r['n_qubits']:>2}q  {bar:<{width}}  {r['mlx_ms']:.1f} ms")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU vs MLX backend benchmark")
    parser.add_argument(
        "--qubits", nargs="+", type=int,
        default=[10, 14, 16, 18, 20, 22, 24],
        help="qubit counts to benchmark (default: 10 14 16 18 20 22 24)",
    )
    parser.add_argument(
        "--depth", type=int, default=50,
        help="circuit depth — number of gates per circuit (default: 50)",
    )
    parser.add_argument(
        "--reps", type=int, default=3,
        help="repetitions per configuration; minimum time is reported (default: 3)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for reproducible circuits (default: 42)",
    )
    parser.add_argument(
        "--json", metavar="FILE",
        help="write results to a JSON file for comparison or plotting",
    )
    parser.add_argument(
        "--no-chart", action="store_true",
        help="suppress the ASCII bar chart",
    )
    args = parser.parse_args()

    print(f"Benchmarking: {len(args.qubits)} qubit counts, depth={args.depth}, reps={args.reps}")
    results = benchmark(
        qubit_counts=sorted(args.qubits),
        depth=args.depth,
        reps=args.reps,
        seed=args.seed,
    )

    _print_table(results)
    if not args.no_chart:
        _ascii_chart(results)

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"config": vars(args), "results": results}, f, indent=2)
        print(f"Results written to {path}")


if __name__ == "__main__":
    main()
