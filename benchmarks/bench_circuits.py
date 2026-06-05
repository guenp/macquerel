"""
Circuit-level macrobenchmarks: QFT, random circuit sampling, QAOA layers,
Quantum Volume.

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

import macquerel.gates as g
from macquerel.backends.cpu import CPUBackend
from macquerel.circuit import Circuit
from macquerel.compiler import fuse_gates


def _flush(backend, sv) -> None:
    if not isinstance(sv, np.ndarray):
        backend.to_numpy(sv)


def _run_circuit(backend, circuit: Circuit, max_fused_qubits: int = 4) -> float:
    fused = fuse_gates(circuit, max_fused_qubits=max_fused_qubits)
    from macquerel.circuit import Gate

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


def _haar_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-random unitary of size dim via QR of a complex Ginibre matrix."""
    z = (rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    # Fix the phase ambiguity so the result is genuinely Haar-distributed.
    ph = np.diagonal(r) / np.abs(np.diagonal(r))
    return q * ph


def _build_quantum_volume(n: int, depth: int | None = None, seed: int = 23) -> Circuit:
    """Quantum Volume model circuit (square: depth = n by default).

    Each layer applies a random permutation of the qubits, pairs them up, and
    applies a Haar-random 2-qubit (SU(4)) gate to each pair. These dense 4x4
    gates exercise the general dense path and fuse poorly, so QV is the natural
    stress test for the worst-case (non-diagonal, non-permutation) workload.
    """
    rng = np.random.default_rng(seed)
    d = depth if depth is not None else n
    qc = Circuit(n)
    for _ in range(d):
        perm = rng.permutation(n)
        for i in range(0, n - 1, 2):
            a, b = int(perm[i]), int(perm[i + 1])
            u = _haar_unitary(4, rng).astype(np.complex64)
            qc._add("su4", u, [a, b])
    return qc


def _build_dense_layers(n: int, layers: int = 40, seed: int = 7) -> Circuit:
    """Dense parallel-friendly circuit: full rotation walls + brick-wall entanglers.

    Every gate touches the whole 2**n state vector, so each op is a large,
    parallelizable workload — ideal for GPU amortization on MLX.
    """
    rng = np.random.default_rng(seed)
    qc = Circuit(n)
    for d in range(layers):
        # Rotation wall: a rotation on every qubit (each op spans all 2**n amps)
        for q in range(n):
            qc.ry(q, rng.uniform(0, 2 * np.pi))
            qc.rz(q, rng.uniform(0, 2 * np.pi))
        # Brick-wall CZ entangler (offset alternates each layer) -> fuses well
        for q in range(d % 2, n - 1, 2):
            qc.cz(q, q + 1)
    return qc


def benchmark(qubit_counts: list[int], reps: int, depth: int) -> dict:
    try:
        from macquerel.backends.mlx_backend import MLXBackend

        backends = {"cpu": CPUBackend(), "mlx": MLXBackend()}
    except ImportError:
        backends = {"cpu": CPUBackend()}

    bnames = list(backends.keys())
    results = {"circuit_benchmarks": [], "fusion_sweep": []}

    print("\n=== Circuit benchmarks ===")
    headers = ["n_qubits", "circuit"] + [f"{b} (ms)" for b in bnames]
    print("  ".join(f"{h:>12}" for h in headers), flush=True)

    for n in qubit_counts:
        circuits = {
            "QFT": _build_qft(n),
            "random": _build_random(n, depth),
            "QAOA": _build_qaoa(n),
            "QV": _build_quantum_volume(n),
            "dense": _build_dense_layers(n),
        }
        for circuit_name, circuit in circuits.items():
            row = {"n_qubits": n, "circuit": circuit_name}
            for bname, backend in backends.items():
                print(f"  running {n}q  {circuit_name}  ({bname})...", end="\r", flush=True)
                times = [_run_circuit(backend, circuit) for _ in range(reps)]
                row[f"{bname}_ms"] = round(min(times) * 1000, 3)
            results["circuit_benchmarks"].append(row)
            cols = [str(n), circuit_name] + [str(row.get(f"{b}_ms", "n/a")) for b in bnames]
            print("  ".join(f"{v:>12}" for v in cols), flush=True)

    print("\n=== Fusion-width sweep (QFT, CPU) ===")
    print(f"{'n_qubits':>8}  {'fused_w':>7}  {'cpu_ms':>10}", flush=True)

    qft_backend = CPUBackend()
    for n in qubit_counts:
        qft = _build_qft(n)
        for fw in range(1, 7):
            print(f"  {n}q  QFT  fused_w={fw}...", end="\r", flush=True)
            times = [_run_circuit(qft_backend, qft, max_fused_qubits=fw) for _ in range(reps)]
            entry = {
                "n_qubits": n,
                "circuit": "QFT",
                "max_fused_qubits": fw,
                "cpu_ms": round(min(times) * 1000, 3),
            }
            results["fusion_sweep"].append(entry)
            print(f"{n:>8}  {fw:>7}  {entry['cpu_ms']:>10.1f}", flush=True)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Circuit-level macrobenchmarks")
    parser.add_argument("--qubits", nargs="+", type=int, default=[14, 18, 22])
    parser.add_argument("--depth", type=int, default=50)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--json", metavar="FILE")
    args = parser.parse_args()

    results = benchmark(sorted(args.qubits), args.reps, args.depth)

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {path}")


if __name__ == "__main__":
    main()
