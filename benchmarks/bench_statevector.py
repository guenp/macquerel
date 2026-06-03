#!/usr/bin/env python3
"""
bench_statevector.py — cross-simulator statevector benchmark harness.

Runs identical circuits (QFT, random, QAOA, GHZ) through:
  - macquerel  (CPU, MLX, and/or Metal backend — Apple Silicon)
  - Qiskit Aer (statevector method)
  - Qulacs

and reports wall-clock statevector-build time vs qubit count, then charts it.

Circuits are defined ONCE as a backend-agnostic op list and translated into each
framework's native API, so every simulator runs the exact same logical circuit
from the same gate set that macquerel supports
(H, X, Y, Z, S, T, Rx, Ry, Rz, P, CX, CZ, SWAP, CP).

Usage:
    python bench_statevector.py                          # full sweep, all backends found
    python bench_statevector.py --qubits 8 12 16 20      # custom qubit counts
    python bench_statevector.py --circuits qft random    # subset of circuits
    python bench_statevector.py --backends macquerel-mlx aer   # subset of backends
    python bench_statevector.py --reps 5 --json out.json # more reps + save raw data

Precision note: macquerel defaults to complex64 (single precision); Qiskit Aer and
Qulacs use complex128 (double). Single precision is ~2x lighter on memory bandwidth,
which matters on a bandwidth-bound machine. Use --double to force macquerel to
complex128 for a like-for-like comparison.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field

import numpy as np

# ----------------------------------------------------------------------------
# Backend-agnostic circuit representation
# ----------------------------------------------------------------------------
# Each op is a tuple: (name, *args). Gate set is restricted to what macquerel
# supports so all three backends can run the identical circuit.
#   ("h", q) ("x", q) ("y", q) ("z", q) ("s", q) ("t", q)
#   ("rx", q, theta) ("ry", q, theta) ("rz", q, theta) ("p", q, lam)
#   ("cx", c, t) ("cz", c, t) ("swap", a, b) ("cp", c, t, lam)
Op = tuple


def gen_ghz(n: int) -> list[Op]:
    ops: list[Op] = [("h", 0)]
    ops += [("cx", i, i + 1) for i in range(n - 1)]
    return ops


def gen_qft(n: int) -> list[Op]:
    ops: list[Op] = []
    for j in range(n):
        ops.append(("h", j))
        for k in range(j + 1, n):
            ops.append(("cp", k, j, math.pi / (2 ** (k - j))))
    for i in range(n // 2):
        ops.append(("swap", i, n - 1 - i))
    return ops


def gen_random(n: int, depth: int = 20, seed: int = 0) -> list[Op]:
    """Hardware-style random circuit: layers of random 1q rotations + an
    alternating brickwork of CX entanglers."""
    rng = np.random.default_rng(seed)
    ops: list[Op] = []
    for d in range(depth):
        for q in range(n):
            gate = rng.choice(["rx", "ry", "rz"])
            ops.append((gate, q, float(rng.uniform(0, 2 * math.pi))))
        start = d % 2
        for q in range(start, n - 1, 2):
            ops.append(("cx", q, q + 1))
    return ops


def gen_qaoa(n: int, layers: int = 3, seed: int = 0) -> list[Op]:
    """LR-QAOA-style Max-Cut on a ring. ZZ rotation is decomposed as
    CX - Rz - CX (all in macquerel's gate set)."""
    rng = np.random.default_rng(seed)
    edges = [(i, (i + 1) % n) for i in range(n)]
    ops: list[Op] = [("h", q) for q in range(n)]
    for p in range(layers):
        gamma = (p + 1) / (layers + 1) * math.pi
        beta = (1 - (p + 1) / (layers + 1)) * math.pi / 2
        for a, b in edges:
            ops.append(("cx", a, b))
            ops.append(("rz", b, 2 * gamma))
            ops.append(("cx", a, b))
        for q in range(n):
            ops.append(("rx", q, 2 * beta))
    return ops


GENERATORS = {
    "ghz": gen_ghz,
    "qft": gen_qft,
    "random": gen_random,
    "qaoa": gen_qaoa,
}


# ----------------------------------------------------------------------------
# Backend adapters — each builds + evaluates a statevector for an op list.
# A backend is available only if its import succeeds.
# ----------------------------------------------------------------------------
@dataclass
class Backend:
    name: str
    build_and_run: callable  # (ops, n) -> statevector (forced/realized)
    available: bool = True
    note: str = ""


def _make_macquerel(backend: str, double: bool):
    from macquerel import Circuit, Simulator

    dtype = "complex128" if double else "complex64"
    sim = Simulator(backend=backend, dtype=dtype)

    def build_and_run(ops, n):
        c = Circuit(n)
        # macquerel uses big-endian qubit order (qubit 0 = MSB) while Qiskit and
        # Qulacs are little-endian. Remap q -> n-1-q so every backend yields the
        # identical statevector. Pure index relabeling — no effect on timing.
        r = lambda q: n - 1 - q
        for op in ops:
            name = str(op[0])
            if name in ("h", "x", "y", "z", "s", "t"):
                getattr(c, name)(r(op[1]))
            elif name in ("rx", "ry", "rz", "p"):
                getattr(c, name)(r(op[1]), op[2])
            elif name in ("cx", "cz", "swap"):
                getattr(c, name)(r(op[1]), r(op[2]))
            elif name == "cp":
                c.cp(r(op[1]), r(op[2]), op[3])
            else:
                raise ValueError(f"unknown op {name}")
        sv = sim.statevector(c)
        # force realization (matters for lazy/GPU backends like MLX)
        np.asarray(sv)[0]
        return sv

    return build_and_run


def _make_aer():
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    sim = AerSimulator(method="statevector")

    def build_and_run(ops, n):
        qc = QuantumCircuit(n)
        for op in ops:
            name = op[0]
            if name == "h":
                qc.h(op[1])
            elif name == "x":
                qc.x(op[1])
            elif name == "y":
                qc.y(op[1])
            elif name == "z":
                qc.z(op[1])
            elif name == "s":
                qc.s(op[1])
            elif name == "t":
                qc.t(op[1])
            elif name == "rx":
                qc.rx(op[2], op[1])
            elif name == "ry":
                qc.ry(op[2], op[1])
            elif name == "rz":
                qc.rz(op[2], op[1])
            elif name == "p":
                qc.p(op[2], op[1])
            elif name == "cx":
                qc.cx(op[1], op[2])
            elif name == "cz":
                qc.cz(op[1], op[2])
            elif name == "swap":
                qc.swap(op[1], op[2])
            elif name == "cp":
                qc.cp(op[3], op[1], op[2])
            else:
                raise ValueError(f"unknown op {name}")
        qc.save_statevector()
        result = sim.run(qc).result()
        return result.get_statevector()

    return build_and_run


def _make_qulacs():
    from qulacs import QuantumCircuit as QCircuit
    from qulacs import QuantumState
    from qulacs.gate import (
        CNOT,
        CZ,
        RX,
        RY,
        RZ,
        SWAP,
        U1,
        DenseMatrix,
        H,
        S,
        T,
        X,
        Y,
        Z,
    )

    def build_and_run(ops, n):
        state = QuantumState(n)
        state.set_zero_state()
        qc = QCircuit(n)
        for op in ops:
            name = op[0]
            if name == "h":
                qc.add_gate(H(op[1]))
            elif name == "x":
                qc.add_gate(X(op[1]))
            elif name == "y":
                qc.add_gate(Y(op[1]))
            elif name == "z":
                qc.add_gate(Z(op[1]))
            elif name == "s":
                qc.add_gate(S(op[1]))
            elif name == "t":
                qc.add_gate(T(op[1]))
            # qulacs RX/RY/RZ use exp(+i theta/2 P); qiskit uses exp(-i theta/2 P).
            # Sign flip keeps the unitary identical across backends.
            elif name == "rx":
                qc.add_gate(RX(op[1], -op[2]))
            elif name == "ry":
                qc.add_gate(RY(op[1], -op[2]))
            elif name == "rz":
                qc.add_gate(RZ(op[1], -op[2]))
            elif name == "p":
                qc.add_gate(U1(op[1], op[2]))
            elif name == "cx":
                qc.add_gate(CNOT(op[1], op[2]))
            elif name == "cz":
                qc.add_gate(CZ(op[1], op[2]))
            elif name == "swap":
                qc.add_gate(SWAP(op[1], op[2]))
            elif name == "cp":
                # controlled-phase via a controlled 1-qubit phase dense matrix
                lam = op[3]
                mat = np.array([[1, 0], [0, np.exp(1j * lam)]], dtype=complex)
                g = DenseMatrix(op[2], mat)
                g.add_control_qubit(op[1], 1)
                qc.add_gate(g)
            else:
                raise ValueError(f"unknown op {name}")
        qc.update_quantum_state(state)
        return state.get_vector()

    return build_and_run


def discover_backends(requested: list[str] | None, double: bool) -> list[Backend]:
    specs = {
        "macquerel-cpu": lambda: _make_macquerel("cpu", double),
        "macquerel-mlx": lambda: _make_macquerel("mlx", double),
        "macquerel-metal": lambda: _make_macquerel("metal", double),
        "aer": _make_aer,
        "qulacs": _make_qulacs,
    }
    if requested:
        specs = {k: v for k, v in specs.items() if k in requested}

    backends: list[Backend] = []
    for name, factory in specs.items():
        try:
            fn = factory()
            backends.append(Backend(name=name, build_and_run=fn))
        except Exception as e:
            backends.append(
                Backend(name=name, build_and_run=None, available=False, note=str(e)[:120])
            )
    return backends


# ----------------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------------
def time_run(fn, ops, n, reps: int) -> float:
    fn(ops, n)  # warm-up (JIT / allocation / lazy-graph realization)
    best = math.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(ops, n)
        best = min(best, time.perf_counter() - t0)
    return best


@dataclass
class Results:
    # data[circuit][backend] = list of (qubits, seconds)
    data: dict = field(default_factory=dict)

    def add(self, circuit, backend, q, secs):
        self.data.setdefault(circuit, {}).setdefault(backend, []).append((q, secs))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--qubits", type=int, nargs="+", default=[6, 8, 10, 12, 14, 16, 18, 20])
    ap.add_argument("--circuits", nargs="+", default=list(GENERATORS), choices=list(GENERATORS))
    ap.add_argument(
        "--backends",
        nargs="+",
        default=None,
        help="subset of: macquerel-cpu macquerel-mlx macquerel-metal aer qulacs",
    )
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument(
        "--double",
        action="store_true",
        help="force macquerel to complex128 for like-for-like precision",
    )
    ap.add_argument("--json", default=None, help="save raw results to this path")
    ap.add_argument("--plot", default="bench_results.png", help="output chart path")
    args = ap.parse_args()

    backends = discover_backends(args.backends, args.double)
    live = [b for b in backends if b.available]
    print("Backends:")
    for b in backends:
        status = "ok" if b.available else f"UNAVAILABLE ({b.note})"
        print(f"  {b.name:16s} {status}")
    if not live:
        print("No backends available — install qiskit-aer / qulacs / macquerel.")
        return
    print()

    results = Results()
    for circuit in args.circuits:
        gen = GENERATORS[circuit]
        print(f"== {circuit} ==")
        for n in args.qubits:
            ops = gen(n)
            row = f"  n={n:<3d}"
            for b in live:
                try:
                    secs = time_run(b.build_and_run, ops, n, args.reps)
                    results.add(circuit, b.name, n, secs)
                    row += f"  {b.name}={secs * 1e3:8.2f}ms"
                except Exception as e:
                    row += f"  {b.name}=ERR({str(e)[:30]})"
            print(row)
        print()

    if args.json:
        import json

        with open(args.json, "w") as f:
            json.dump(results.data, f, indent=2)
        print(f"Raw data -> {args.json}")

    make_plot(results, args.circuits, args.plot)


def make_plot(results: Results, circuits, path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping chart (raw data still saved).")
        return

    circuits = [c for c in circuits if c in results.data]
    if not circuits:
        return
    cols = min(2, len(circuits))
    rows = math.ceil(len(circuits) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4.5 * rows), squeeze=False)

    for idx, circuit in enumerate(circuits):
        ax = axes[idx // cols][idx % cols]
        for backend, series in sorted(results.data[circuit].items()):
            series = sorted(series)
            xs = [q for q, _ in series]
            ys = [s * 1e3 for _, s in series]
            ax.plot(xs, ys, marker="o", label=backend)
        ax.set_title(circuit)
        ax.set_xlabel("qubits")
        ax.set_ylabel("statevector time (ms)")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    for j in range(len(circuits), rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"Chart -> {path}")


if __name__ == "__main__":
    main()
