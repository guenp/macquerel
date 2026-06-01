"""
Circuit-level macrobenchmarks: QFT, random circuit sampling, QAOA layers.

Also sweeps max_fused_qubits ∈ {1..6} to validate the fusion-width trade-off.

Usage:
    uv run python benchmarks/bench_circuits.py
    uv run python benchmarks/bench_circuits.py --qubits 16 20 24 --reps 3
    uv run python benchmarks/bench_circuits.py --json results/circuits.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from macquerel.circuit import Circuit
from macquerel.compiler import fuse_gates
from macquerel.backends.cpu import CPUBackend
import macquerel.gates as g


def _flush(backend, sv) -> None:
    if not isinstance(sv, np.ndarray):
        backend.to_numpy(sv)


def _run_circuit(backend, circuit: Circuit, max_fused_qubits: int = 4) -> float:
    fused = fuse_gates(circuit, max_fused_qubits=max_fused_qubits)
    from macquerel.circuit import Gate, MeasureOp
    sv = backend.allocate(circuit.n_qubits)
    t0 = time.perf_counter()
    for op in fused.ops:
        if isinstance(op, Gate):
            sv = backend.apply_matrix(sv, op.matrix, op.targets, op.controls or None)
    _flush(backend, sv)
    return time.perf_counter() - t0


def _build_qft(n: int) -> Circuit:
    """n-qubit Quantum Fourier Transform circuit."""
    qc = Circuit(n)
    for i in range(n):
        qc.h(i)
        for j in range(i + 1, n):
            lam = np.pi / (2 ** (j - i))
            qc.cp(i, j, lam)
    for i in range(n // 2):
        qc.swap(i, n - 1 - i)
    return qc


def _build_random(n: int, depth: int, seed: int = 42) -> Circuit:
    """Random circuit: alternating single-qubit and two-qubit gates."""
    rng = np.random.default_rng(seed)
    qc = Circuit(n)
    gates_1q = [g.H, g.X, g.Z, lambda: g.Rz(rng.uniform(0, 2 * np.pi))]
    gates_2q = [g.CNOT, g.CZ, g.SWAP]
    for _ in range(depth):
        if n >= 2 and rng.random() < 0.4:
            pair = rng.choice(n, size=2, replace=False).tolist()
            mat = gates_2q[rng.integers(len(gates_2q))]()
            qc._add("rand2q", mat, pair)
        else:
            q = int(rng.integers(n))
            mat = gates_1q[rng.integers(len(gates_1q))]()
            qc._add("rand1q", mat, [q])
    return qc


def _build_qaoa(n: int, layers: int = 3) -> Circuit:
    """QAOA circuit ansatz (problem = MaxCut on a ring graph)."""
    qc = Circuit(n)
    gamma, beta = 0.5, 0.3
    for _ in range(layers):
        # Cost layer: CZ along a ring
        for i in range(n):
            qc.cz(i, (i + 1) % n)
        # Mixer layer: Rx on all qubits
        for i in range(n):
            qc.rx(i, 2 * beta)
    return qc


def benchmark(qubit_counts: list[int], reps: int, depth: int) -> dict:
    try:
        from macquerel.backends.mlx_backend import MLXBackend
        backends = {"cpu": CPUBackend(), "mlx": MLXBackend()}
    except ImportError:
        backends = {"cpu": CPUBackend()}

    results = {"circuit_benchmarks": [], "fusion_sweep": []}

    for n in qubit_counts:
        circuits = {
            "QFT": _build_qft(n),
            "random": _build_random(n, depth),
            "QAOA": _build_qaoa(n),
        }
        for circuit_name, circuit in circuits.items():
            row = {"n_qubits": n, "circuit": circuit_name}
            for bname, backend in backends.items():
                times = [_run_circuit(backend, circuit) for _ in range(reps)]
                row[f"{bname}_ms"] = round(min(times) * 1000, 3)
            results["circuit_benchmarks"].append(row)

        # Fusion-width sweep on QFT (CPU only — most representative for tuning)
        qft = _build_qft(n)
        for fw in range(1, 7):
            times = [_run_circuit(CPUBackend(), qft, max_fused_qubits=fw) for _ in range(reps)]
            results["fusion_sweep"].append({
                "n_qubits": n, "circuit": "QFT",
                "max_fused_qubits": fw, "cpu_ms": round(min(times) * 1000, 3),
            })

    return results


def _print_results(results: dict) -> None:
    print("\n=== Circuit benchmarks ===")
    bnames = [k.replace("_ms", "") for k in results["circuit_benchmarks"][0] if k.endswith("_ms")]
    headers = ["n_qubits", "circuit"] + [f"{b} (ms)" for b in bnames]
    print("  ".join(f"{h:>12}" for h in headers))
    for r in results["circuit_benchmarks"]:
        row = [str(r["n_qubits"]), r["circuit"]] + [str(r.get(f"{b}_ms", "n/a")) for b in bnames]
        print("  ".join(f"{v:>12}" for v in row))

    print("\n=== Fusion-width sweep (QFT, CPU) ===")
    print(f"{'n_qubits':>8}  {'fused_w':>7}  {'cpu_ms':>10}")
    for r in results["fusion_sweep"]:
        print(f"{r['n_qubits']:>8}  {r['max_fused_qubits']:>7}  {r['cpu_ms']:>10.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Circuit-level macrobenchmarks")
    parser.add_argument("--qubits", nargs="+", type=int, default=[14, 18, 22])
    parser.add_argument("--depth", type=int, default=50)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--json", metavar="FILE")
    args = parser.parse_args()

    results = benchmark(sorted(args.qubits), args.reps, args.depth)
    _print_results(results)

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {path}")


if __name__ == "__main__":
    main()
