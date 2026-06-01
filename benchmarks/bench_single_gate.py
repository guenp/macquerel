"""
Benchmark: single-gate throughput as a function of qubit count and target index.

Reports GB/s of statevector bandwidth consumed per gate application, allowing
comparison against the theoretical peak (unified-memory bandwidth on Apple Silicon).

Usage:
    uv run python benchmarks/bench_single_gate.py
    uv run python benchmarks/bench_single_gate.py --qubits 16 20 24 --reps 5
    uv run python benchmarks/bench_single_gate.py --json results/single_gate.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import macquerel.gates as g
from macquerel.backends.cpu import CPUBackend

_GATES = {
    "H": g.H,
    "X": g.X,
    "Rz(0.7)": lambda: g.Rz(0.7),
    "CNOT": g.CNOT,
    "SWAP": g.SWAP,
}


def _flush(backend, sv) -> None:
    if not isinstance(sv, np.ndarray):
        backend.to_numpy(sv)


def _time_gate(backend, n: int, gate_mat, targets: list[int], reps: int) -> float:
    sv = backend.allocate(n)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        sv = backend.apply_matrix(sv, gate_mat, targets)
        _flush(backend, sv)
        times.append(time.perf_counter() - t0)
    return min(times)


def _gbps(n: int, k: int, elapsed_s: float) -> float:
    """Estimate GB/s: a k-qubit gate reads and writes 2**n complex64 amplitudes."""
    bytes_rw = 2 * (2**n) * 8  # read + write, 8 bytes per complex64
    return bytes_rw / elapsed_s / 1e9


def benchmark(qubit_counts: list[int], reps: int) -> list[dict]:
    try:
        from macquerel.backends.mlx_backend import MLXBackend
        backends = {"cpu": CPUBackend(), "mlx": MLXBackend()}
    except ImportError:
        backends = {"cpu": CPUBackend()}

    results = []
    for n in qubit_counts:
        for gate_name, gate_fn in _GATES.items():
            mat = gate_fn()
            k = 1 if mat.shape[0] == 2 else 2
            targets = list(range(k))
            row = {"n_qubits": n, "gate": gate_name, "target_qubits": targets}
            for bname, backend in backends.items():
                elapsed = _time_gate(backend, n, mat, targets, reps)
                row[f"{bname}_ms"] = round(elapsed * 1000, 3)
                row[f"{bname}_gbps"] = round(_gbps(n, k, elapsed), 3)
            results.append(row)
    return results


def _print_table(results: list[dict]) -> None:
    bnames = [k.replace("_ms", "") for k in results[0] if k.endswith("_ms")]
    headers = ["qubits", "gate"] + [f"{b} (ms)" for b in bnames] + [f"{b} GB/s" for b in bnames]
    print("  ".join(f"{h:>12}" for h in headers))
    print("-" * (14 * len(headers)))
    for r in results:
        row = [str(r["n_qubits"]), r["gate"]]
        row += [str(r.get(f"{b}_ms", "n/a")) for b in bnames]
        row += [str(r.get(f"{b}_gbps", "n/a")) for b in bnames]
        print("  ".join(f"{v:>12}" for v in row))


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-gate throughput benchmark")
    parser.add_argument("--qubits", nargs="+", type=int, default=[14, 18, 22, 26])
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--json", metavar="FILE")
    args = parser.parse_args()

    results = benchmark(sorted(args.qubits), args.reps)
    _print_table(results)

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {path}")


if __name__ == "__main__":
    main()
