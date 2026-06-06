#!/usr/bin/env python3
"""
bench_statevector.py - framework comparison statevector benchmark harness.

Runs identical circuits (QFT, random, QAOA, GHZ) through:
  - macquerel  (CPU, MLX, and/or Metal backend - Apple Silicon)
  - Qiskit Aer (statevector method)
  - Qulacs

and reports wall-clock statevector-build time vs qubit count, then charts it.

Circuits are defined ONCE as a backend-agnostic op list and translated into each
framework's native API, so every simulator runs the exact same logical circuit
from the same gate set that macquerel supports
(H, X, Y, Z, S, T, Rx, Ry, Rz, P, CX, CZ, SWAP, CP).

Usage:
    uv run python benchmarks/bench_statevector.py
    uv run python benchmarks/bench_statevector.py --qubits 8 12 16 20
    uv run python benchmarks/bench_statevector.py --circuits qft random
    uv run python benchmarks/bench_statevector.py --backends macquerel-mlx aer
    uv run python benchmarks/bench_statevector.py --reps 5 \
        --json benchmarks/data/framework_comparison.json \
        --plot benchmarks/data/framework_comparison.png

Precision note: macquerel defaults to complex64 (single precision); Qiskit Aer and
Qulacs use complex128 (double). Single precision is ~2x lighter on memory bandwidth,
which matters on a bandwidth-bound machine. Use --double to force macquerel to
complex128 for a like-for-like comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

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


ALL_BACKENDS = ["macquerel-cpu", "macquerel-mlx", "macquerel-metal", "aer", "qulacs"]


def make_backend(name: str, double: bool):
    """Build the build_and_run callable for a backend by name."""
    if name == "macquerel-cpu":
        return _make_macquerel("cpu", double)
    if name == "macquerel-mlx":
        return _make_macquerel("mlx", double)
    if name == "macquerel-metal":
        return _make_macquerel("metal", double)
    if name == "aer":
        return _make_aer()
    if name == "qulacs":
        return _make_qulacs()
    raise ValueError(f"unknown backend {name}")


def discover_backends(requested: list[str] | None, double: bool) -> list[Backend]:
    names = requested or ALL_BACKENDS
    backends: list[Backend] = []
    for name in names:
        try:
            fn = make_backend(name, double)
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


# Sentinel the parent process greps for in a worker's stdout.
_RESULT_TAG = "__BENCH_RESULT__"


def run_worker(args) -> int:
    """Worker mode: time a single (backend, circuit, n) cell and print the result.

    Run as its own process so each measurement starts from a clean slate — no
    GPU memory pool, lazy graph, or resident statevector left over from another
    backend. At 30q a statevector is 8-16 GiB, so timing several backends in one
    process lets an earlier backend's retained memory push a later one into
    swap/compression and produce wildly inflated (and unrepeatable) numbers.
    """
    fn = make_backend(args.backend, args.double)
    ops = GENERATORS[args.circuit](args.n)
    secs = time_run(fn, ops, args.n, args.reps)
    print(f"{_RESULT_TAG} {secs!r}")
    return 0


def measure_isolated(name, circuit, n, reps, double, timeout) -> float:
    """Time one cell in a fresh subprocess; raises on failure/timeout."""
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--worker",
        "--backend",
        name,
        "--circuit",
        circuit,
        "--n",
        str(n),
        "--reps",
        str(reps),
    ]
    if double:
        cmd.append("--double")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        msg = (proc.stderr.strip().splitlines() or ["worker exited nonzero"])[-1]
        raise RuntimeError(msg[:120])
    for line in proc.stdout.splitlines():
        if line.startswith(_RESULT_TAG):
            return float(line.split(maxsplit=1)[1])
    raise RuntimeError("worker produced no result")


# ----------------------------------------------------------------------------
# Memory budget — refuse to launch a cell that could exhaust RAM and hang/crash
# the whole machine. The estimate is deliberately pessimistic: it is a safety
# gate, not an accounting tool. Better to SKIP a cell that might have fit than to
# drive the box into swap.
# ----------------------------------------------------------------------------
#
# A 2**n statevector is `bytes_per_amp * 2**n`. The multipliers below are the
# observed peak-to-base ratio of each backend's *transient* footprint:
#   - metal:  in-place single buffer + tiny scratch                  -> ~1.6x
#   - aer/qulacs: complex128 state + a result/copy                   -> ~2.5x
#   - cpu:    tensordot allocates reshaped + transposed copies       -> ~4x
#   - mlx:    lazy graph holds many full-width intermediates at once,
#             plus a pool cache it never returns to the OS. Calibrated
#             from the fact that mlx @ 30q (8 GiB base) drove a 128 GiB
#             machine into swap, i.e. peak >= ~16x base for deep circuits.
_PEAK_MULT = {
    "macquerel-cpu": 4.0,
    "macquerel-mlx": 16.0,
    "macquerel-metal": 1.6,
    "aer": 2.5,
    "qulacs": 2.5,
}


def _system_ram_gib() -> float:
    """Total physical RAM in GiB (sysctl on macOS, /proc on Linux); safe fallback."""
    try:
        if sys.platform == "darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return int(out.strip()) / 1024**3
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1024**2  # kB -> GiB
    except Exception:
        pass
    return 16.0  # conservative fallback if we cannot tell


def est_peak_gib(name: str, n: int, double: bool) -> float:
    """Pessimistic peak-memory estimate for one (backend, n) cell, in GiB."""
    bytes_per_amp = 16 if (double or not name.startswith("macquerel")) else 8
    base = bytes_per_amp * (2**n)
    return base * _PEAK_MULT.get(name, 4.0) / 1024**3


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
        "--backends", nargs="+", default=None, help="subset of: " + " ".join(ALL_BACKENDS)
    )
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument(
        "--double",
        action="store_true",
        help="force macquerel to complex128 for like-for-like precision",
    )
    ap.add_argument("--json", default=None, help="save raw results to this path")
    ap.add_argument(
        "--plot", default="benchmarks/data/framework_comparison.png", help="output chart path"
    )
    ap.add_argument(
        "--no-isolate",
        action="store_true",
        help="time all backends in this process instead of one subprocess per cell "
        "(faster startup, but large-n numbers are unreliable — see run_worker)",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help="per-cell subprocess timeout in seconds (isolated mode only)",
    )
    ap.add_argument(
        "--mem-budget-frac",
        type=float,
        default=0.45,
        help="skip any cell whose estimated peak memory exceeds this fraction of "
        "system RAM (safety gate against swapping/crashing the machine)",
    )
    ap.add_argument(
        "--max-mem-gib",
        type=float,
        default=None,
        help="absolute peak-memory cap per cell in GiB (overrides --mem-budget-frac if smaller)",
    )
    # Worker mode (internal): time a single cell and print the result.
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--backend", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--circuit", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--n", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        return run_worker(args)

    isolate = not args.no_isolate
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

    mode = "subprocess-isolated" if isolate else "in-process (--no-isolate)"
    ram = _system_ram_gib()
    budget = ram * args.mem_budget_frac
    if args.max_mem_gib is not None:
        budget = min(budget, args.max_mem_gib)
    print(f"Timing mode: {mode}")
    print(f"System RAM: {ram:.0f} GiB | per-cell memory budget: {budget:.0f} GiB\n")

    def payload() -> dict:
        return {
            "benchmark": "framework_comparison",
            "config": {
                "qubits": args.qubits,
                "circuits": args.circuits,
                "backends": args.backends or ALL_BACKENDS,
                "reps": args.reps,
                "double": args.double,
                "isolate": isolate,
                "mem_budget_gib": budget,
            },
            "results": results.data,
        }

    def checkpoint():
        if args.json:
            path = Path(args.json)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload(), indent=2))

    results = Results()
    for circuit in args.circuits:
        gen = GENERATORS[circuit]
        print(f"== {circuit} ==")
        for n in args.qubits:
            ops = gen(n)
            # Each cell is logged + checkpointed as it finishes, so a long sweep
            # that is interrupted (kill / OOM) still leaves every completed cell
            # on disk and on screen.
            for b in live:
                est = est_peak_gib(b.name, n, args.double)
                if est > budget:
                    print(
                        f"  {circuit:7s} n={n:<3d} {b.name:16s} "
                        f"SKIP (est ~{est:.0f} GiB > {budget:.0f} GiB budget)",
                        flush=True,
                    )
                    continue
                try:
                    if isolate:
                        secs = measure_isolated(
                            b.name, circuit, n, args.reps, args.double, args.timeout
                        )
                    else:
                        secs = time_run(b.build_and_run, ops, n, args.reps)
                    results.add(circuit, b.name, n, secs)
                    checkpoint()
                    cell = f"{secs * 1e3:8.2f}ms"
                except subprocess.TimeoutExpired:
                    cell = "TIMEOUT"
                except Exception as e:
                    cell = f"ERR({str(e)[:40]})"
                print(f"  {circuit:7s} n={n:<3d} {b.name:16s} {cell}", flush=True)
        print()

    if args.json:
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
