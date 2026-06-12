#!/usr/bin/env python3
"""bench_vqe.py — variational-circuit (VQE) performance: energy evaluations and gradients.

A VQE iteration is dominated by two primitives, timed separately here on a
1D transverse-field Ising Hamiltonian H = -J sum Z_i Z_{i+1} - h sum X_i
(2n-1 Pauli terms) over a hardware-efficient ansatz ([Ry+Rz layer, CX ladder]
x layers, 2*n*layers parameters):

  - ``energy``:   one <H> evaluation = statevector evolution + Pauli-string
                  expectations, on each backend via
                  `TrajectorySimulator(trajectories=1).expectation_pauli`
                  (exact for a noiseless circuit: one trajectory of a
                  channel-free circuit is plain statevector evolution).
                  Swept vs qubit count and vs ansatz depth (layers).
  - ``gradient``: one full parameter-shift gradient = 2P circuit evaluations
                  for P parameters. The per-circuit `Simulator` loop is
                  compared against `BatchedSimulator` (cpu and mlx), which
                  evolves all 2P shifted circuits as one batch; <H> is then
                  computed host-side, vectorized over the batch, identically
                  for both paths so only the evolution strategy differs.
                  Swept vs P (depth sweep at fixed n).

The energy sweep runs each backend to its ceiling (`MAX_QUBITS` — Metal 31,
MLX 30, CPU 24; see the constant for why Metal stops at 31 here and not its
32-qubit statevector limit); large cells use a light one-rep/no-warm-up
protocol (`LIGHT_QUBITS`).

Usage:
    uv run python benchmarks/bench_vqe.py
    uv run python benchmarks/bench_vqe.py --qubits 4 8 12 --layers 1 2 4 \
        --json benchmarks/data/vqe.json --plot benchmarks/data/vqe.png
    uv run python benchmarks/bench_vqe.py --replot   # redraw plot from JSON
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path

import numpy as np

from macquerel import BatchedSimulator, Circuit, Simulator, TrajectorySimulator
from macquerel.simulator import _make_backend

BACKENDS = ["cpu", "mlx", "metal"]
BACKEND_COLORS = {"metal": "#d62728", "mlx": "#1f77b4", "cpu": "#2ca02c"}
GRAD_COLORS = {"loop": "#7f7f7f", "batched-cpu": "#2ca02c", "batched-mlx": "#1f77b4"}

# Per-backend qubit ceilings for the energy sweep. MLX stops at 30 (int32
# ShapeElem). Metal's statevector ceiling is 32, but `expectation_pauli`'s
# host path holds ~4 state-sized buffers at once (the GPU state, its to_numpy
# copy, the per-term working copy, and a conj() temporary) — at 32q that is
# 4 x 32 GiB, beyond a 128 GiB machine, so the energy sweep stops at 31
# (4 x 16 GiB). The CPU cap is patience, not memory.
MAX_QUBITS = {"cpu": 24, "mlx": 30, "metal": 31}
# At and above this qubit count, energy cells run one rep with no warm-up:
# they are bandwidth-bound multi-second runs where dispatch noise and pipeline
# compilation are negligible.
LIGHT_QUBITS = 24


def ansatz(n: int, thetas: np.ndarray, layers: int) -> Circuit:
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


def tfim_terms(n: int, j: float = 1.0, h: float = 1.0) -> list:
    """H = -J sum Z_i Z_{i+1} - h sum X_i in expectation_pauli's term format."""
    terms = [(-j, [("Z", i), ("Z", i + 1)]) for i in range(n - 1)]
    terms += [(-h, [("X", i)]) for i in range(n)]
    return terms


def tfim_energy_host(states: np.ndarray, n: int, j: float = 1.0, h: float = 1.0) -> np.ndarray:
    """<H> per state, vectorized over a (B, 2**n) batch of statevectors.

    ZZ terms come from the probability tensor with per-axis sign masks; X_i is
    <psi|X_i|psi> = Re sum conj(psi) * psi-flipped-on-axis-i. Axis 1+i of the
    reshaped batch is qubit i (macquerel is big-endian; the chain is symmetric
    anyway).
    """
    b = states.shape[0]
    view = states.reshape((b,) + (2,) * n)
    probs = np.abs(view) ** 2
    sum_axes = tuple(range(1, n + 1))
    z = np.array([1.0, -1.0])
    e = np.zeros(b)
    for i in range(n - 1):
        shape_a = [1] * (n + 1)
        shape_a[1 + i] = 2
        shape_b = [1] * (n + 1)
        shape_b[2 + i] = 2
        e += -j * np.sum(probs * z.reshape(shape_a) * z.reshape(shape_b), axis=sum_axes)
    for i in range(n):
        flipped = np.flip(view, axis=1 + i)
        e += -h * np.real(np.sum(np.conj(view) * flipped, axis=sum_axes))
    return e


def time_call(fn, reps: int, warmup: bool = True) -> float:
    if warmup:
        fn()  # warm-up (kernel/pipeline compilation, buffer pools)
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def random_thetas(n: int, layers: int, seed: int = 7) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0, 2 * np.pi, size=2 * n * layers)


def time_energy(backend_name: str, n: int, layers: int, reps: int) -> float:
    sim = TrajectorySimulator(backend=backend_name, trajectories=1)
    qc = ansatz(n, random_thetas(n, layers), layers)
    ham = tfim_terms(n)
    light = n >= LIGHT_QUBITS
    return time_call(lambda: sim.expectation_pauli(qc, ham), 1 if light else reps, warmup=not light)


# ----------------------------------------------------------------------------
# Parameter-shift gradient: dE/dtheta_k = (E(theta_k + pi/2) - E(theta_k - pi/2)) / 2,
# i.e. 2P circuit evaluations per gradient. <H> is computed host-side from the
# statevectors (tfim_energy_host) for the loop and batched paths alike, so the
# timing difference is purely per-circuit dispatch vs batched evolution.
# ----------------------------------------------------------------------------
def shifted_circuits(n: int, layers: int, thetas: np.ndarray) -> list[Circuit]:
    circuits = []
    for k in range(len(thetas)):
        for sign in (1.0, -1.0):
            shifted = thetas.copy()
            shifted[k] += sign * np.pi / 2
            circuits.append(ansatz(n, shifted, layers))
    return circuits


def gradient_loop(sim: Simulator, circuits: list[Circuit], n: int) -> np.ndarray:
    energies = np.array([tfim_energy_host(sim.statevector(c)[None], n)[0] for c in circuits])
    return (energies[0::2] - energies[1::2]) / 2


def gradient_batched(bs: BatchedSimulator, circuits: list[Circuit], n: int) -> np.ndarray:
    energies = tfim_energy_host(bs.statevectors(circuits), n)
    return (energies[0::2] - energies[1::2]) / 2


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--qubits",
        type=int,
        nargs="+",
        default=[4, 8, 12, 16, 20, 24, 28, 30, 31],
        help="energy-sweep sizes; each backend stops at its MAX_QUBITS ceiling",
    )
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--fixed-layers", type=int, default=4, help="depth for the qubit sweep")
    ap.add_argument("--fixed-qubits", type=int, default=12, help="width for the depth sweep")
    ap.add_argument("--grad-qubits", type=int, default=8, help="width for the gradient sweep")
    ap.add_argument(
        "--grad-layers", type=int, nargs="+", default=[1, 2, 4, 8], help="gradient depth sweep"
    )
    ap.add_argument("--backends", nargs="+", default=BACKENDS, choices=BACKENDS)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--json", default="benchmarks/data/vqe.json")
    ap.add_argument("--plot", default="benchmarks/data/vqe.png")
    ap.add_argument("--replot", action="store_true", help="redraw --plot from existing --json")
    args = ap.parse_args()

    if args.replot:
        make_plot(json.loads(Path(args.json).read_text()), args.plot)
        return

    available = []
    for b in args.backends:
        try:
            _make_backend(b, "complex64", None)
            available.append(b)
        except Exception as e:
            print(f"  {b}: UNAVAILABLE ({str(e)[:80]})")
    print(f"backends: {' '.join(available)}\n")

    results: dict = {
        "benchmark": "vqe",
        "config": vars(args).copy(),
        "data": {"qubits": {}, "depth": {}, "gradient": {}},
    }

    def save() -> None:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2))

    # Sweep 1: single <H> evaluation vs qubit count, fixed depth.
    print(f"== energy vs qubits (layers={args.fixed_layers}) ==")
    print(f"{'n':>3} {'params':>7}" + "".join(f" {b:>12}" for b in available))
    for n in args.qubits:
        row = f"{n:>3} {2 * n * args.fixed_layers:>7}"
        for b in available:
            if n > MAX_QUBITS[b]:
                row += f" {'-':>12}"
                continue
            secs = time_energy(b, n, args.fixed_layers, args.reps)
            results["data"]["qubits"].setdefault(b, []).append([n, secs])
            row += f" {secs * 1e3:>10.2f}ms"
            save()
        print(row, flush=True)

    # Sweep 2: single <H> evaluation vs ansatz depth, fixed width.
    print(f"\n== energy vs depth (n={args.fixed_qubits}) ==")
    print(f"{'layers':>7} {'params':>7}" + "".join(f" {b:>12}" for b in available))
    for layers in args.layers:
        row = f"{layers:>7} {2 * args.fixed_qubits * layers:>7}"
        for b in available:
            secs = time_energy(b, args.fixed_qubits, layers, args.reps)
            results["data"]["depth"].setdefault(b, []).append([layers, secs])
            row += f" {secs * 1e3:>10.2f}ms"
            save()
        print(row, flush=True)

    # Sweep 3: full parameter-shift gradient (2P circuits) vs P, loop vs batched.
    n = args.grad_qubits
    engines: list[tuple[str, object]] = [("loop", Simulator())]
    for eng in ("cpu", "mlx"):
        with contextlib.suppress(ImportError):
            engines.append((f"batched-{eng}", BatchedSimulator(backend=eng)))
    print(f"\n== parameter-shift gradient vs #params (n={n}) ==")
    print(f"{'layers':>7} {'P':>5} {'circuits':>9}" + "".join(f" {e:>13}" for e, _ in engines))
    for layers in args.grad_layers:
        thetas = random_thetas(n, layers)
        circuits = shifted_circuits(n, layers, thetas)
        p = len(thetas)
        row = f"{layers:>7} {p:>5} {len(circuits):>9}"
        for name, engine in engines:
            if name == "loop":
                fn = lambda e=engine, cs=circuits: gradient_loop(e, cs, n)
            else:
                fn = lambda e=engine, cs=circuits: gradient_batched(e, cs, n)
            secs = time_call(fn, args.reps)
            results["data"]["gradient"].setdefault(name, []).append([p, secs])
            row += f" {secs * 1e3:>11.1f}ms"
            save()
        print(row, flush=True)

    print(f"\nRaw data -> {args.json}")
    make_plot(results, args.plot)


def make_plot(results: dict, path: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping chart (raw data still saved).")
        return

    cfg = results["config"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    def draw(ax, sweep: str, colors: dict) -> None:
        for name, series in results["data"][sweep].items():
            series = sorted(series)
            ax.plot(
                [x for x, _ in series],
                [s for _, s in series],
                marker="o",
                markersize=4,
                linewidth=1.6,
                color=colors.get(name),
                label=name,
            )
        ax.set_yscale("log")
        ax.set_ylabel("seconds (min of reps)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9)

    draw(axes[0], "qubits", BACKEND_COLORS)
    axes[0].set_xlabel("qubits")
    axes[0].set_title(f"<H> evaluation vs qubits ({cfg['fixed_layers']} layers)")
    draw(axes[1], "depth", BACKEND_COLORS)
    axes[1].set_xlabel("ansatz layers (depth)")
    axes[1].set_title(f"<H> evaluation vs depth (n={cfg['fixed_qubits']})")
    draw(axes[2], "gradient", GRAD_COLORS)
    axes[2].set_xlabel("parameters P (gradient = 2P circuits)")
    axes[2].set_title(f"parameter-shift gradient (n={cfg['grad_qubits']})")
    fig.suptitle("VQE: TFIM energy evaluations and parameter-shift gradients", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"Chart -> {path}")


if __name__ == "__main__":
    main()
