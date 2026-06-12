#!/usr/bin/env python3
"""bench_mcm.py — mid-circuit measurement (MCM) performance on QEC-style circuits.

QEC experiments are dominated by repeated syndrome extraction: entangle ancillas
with data qubits, measure the ancillas mid-circuit, repeat for R rounds. The
workload here is a distance-d repetition code — n = 2d-1 qubits with the d-1
ancillas interleaved between the d data qubits — which scales cleanly in both
of the metrics that matter:

  - rounds R   (circuit depth / number of MeasureOps; d-1 ancilla MCMs each)
  - distance d (qubit count n = 2d-1)

Two execution paths are timed, because macquerel has two MCM semantics:

  - ``sample``:   `Simulator.run(qc, shots)` — the public path. At each
                  MeasureOp it samples `shots` outcomes from the *uncollapsed*
                  state (one statevector pass total, one `backend.sample` per
                  round). Cheap, but syndrome bits are uncorrelated across
                  rounds — fine for marginals, not for decoding.
  - ``collapse``: a per-shot trajectory using `backend.measure(..., collapse=True)`
                  at every MeasureOp — true projective MCM with correlated
                  round-to-round syndrome histories, the semantics a decoder
                  (or future feed-forward API) needs. One full circuit pass per
                  shot; reported as seconds *per shot*.

Note on backends: the GPU backends currently implement `measure` by copying
the state to the host, collapsing there, and writing it back — so collapse-mode
MCM pays two full PCIe-free-but-not-free unified-memory round trips per round.
This benchmark exists to quantify exactly that.

Usage:
    uv run python benchmarks/bench_mcm.py
    uv run python benchmarks/bench_mcm.py --distances 3 5 7 --rounds 1 2 4 8 \
        --json benchmarks/data/mcm.json --plot benchmarks/data/mcm.png
    uv run python benchmarks/bench_mcm.py --replot   # redraw plot from JSON
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from macquerel import Circuit, Simulator
from macquerel.circuit import Gate, MeasureOp
from macquerel.compiler import fuse_gates
from macquerel.simulator import _make_backend

BACKENDS = ["cpu", "mlx", "metal"]
MODES = ["sample", "collapse"]
BACKEND_COLORS = {"metal": "#d62728", "mlx": "#1f77b4", "cpu": "#2ca02c"}
MODE_STYLES = {"sample": "--", "collapse": "-"}
# Triangles for the dashed sample series so the legend reads without squinting
# at line styles.
MODE_MARKERS = {"sample": "^", "collapse": "o"}


def rep_code_circuit(distance: int, rounds: int) -> Circuit:
    """Distance-d repetition code: d data qubits (even indices) and d-1
    ancillas (odd indices). Each round entangles every ancilla with its two
    data neighbours and measures all ancillas mid-circuit. (No reset op exists;
    a real experiment would reset ancillas, but the collapsed post-measurement
    state makes the next round's gate stream identical, so timing is unaffected.)
    """
    n = 2 * distance - 1
    qc = Circuit(n)
    ancillas = list(range(1, n, 2))
    qc.h(0)  # seed some superposition so measurement probabilities are non-trivial
    for q in range(0, n - 2, 2):
        qc.cx(q, q + 2)
    for _ in range(rounds):
        for a in ancillas:
            qc.cx(a - 1, a)
            qc.cx(a + 1, a)
        qc.measure(ancillas)
    return qc


def time_call(fn, reps: int) -> float:
    fn()  # warm-up (kernel/pipeline compilation, buffer pools)
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


# ----------------------------------------------------------------------------
# collapse mode: per-shot trajectory over the fused op list, projective MCM at
# every MeasureOp. State storage is reused across shots (cf. TrajectorySimulator
# ._fresh_state) so Metal/MLX buffer churn is not charged to the measurement.
# ----------------------------------------------------------------------------
def _fresh_state(backend, prev, n: int):
    if prev is not None:
        host = None
        if isinstance(prev, np.ndarray):
            host = prev
        elif hasattr(backend, "_view"):
            host = backend._view(prev)
        if host is not None:
            host.fill(0)
            host[0] = 1.0
            return prev
    return backend.allocate(n, np.complex64)


def run_collapse(backend, ops, n: int, shots: int) -> None:
    state = None
    for _ in range(shots):
        state = _fresh_state(backend, state, n)
        for op in ops:
            if isinstance(op, Gate):
                state = backend.apply_matrix(state, op.matrix, op.targets, op.controls or None)
            elif isinstance(op, MeasureOp):
                backend.measure(state, op.qubits, collapse=True)


def measure_cell(backend_name: str, mode: str, qc: Circuit, args) -> float:
    """Seconds for one (backend, mode, circuit) cell. collapse mode is
    normalized to seconds per shot; sample mode is one full run() call."""
    # A collapse shot costs ~10 s+ past 24 qubits; one rep of two trajectories
    # keeps the largest cells under a minute without touching the small ones.
    big = qc.n_qubits >= 24
    reps = 1 if big else args.reps
    traj_shots = min(2, args.traj_shots) if big else args.traj_shots
    if mode == "sample":
        sim = Simulator(backend=backend_name)
        return time_call(lambda: sim.run(qc, shots=args.shots), reps)
    backend = _make_backend(backend_name, "complex64", None)
    fused = fuse_gates(qc, backend=backend_name)
    ops = [op for op in fused.ops if isinstance(op, (Gate, MeasureOp))]
    secs = time_call(lambda: run_collapse(backend, ops, qc.n_qubits, traj_shots), reps)
    return secs / traj_shots


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7, 9, 11, 13])
    ap.add_argument("--rounds", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--fixed-distance", type=int, default=7, help="distance for the rounds sweep")
    ap.add_argument("--fixed-rounds", type=int, default=8, help="rounds for the qubit sweep")
    ap.add_argument("--backends", nargs="+", default=BACKENDS, choices=BACKENDS)
    ap.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    ap.add_argument("--shots", type=int, default=1000, help="shots per Simulator.run (sample mode)")
    ap.add_argument("--traj-shots", type=int, default=8, help="trajectories timed in collapse mode")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--json", default="benchmarks/data/mcm.json")
    ap.add_argument("--plot", default="benchmarks/data/mcm.png")
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
        "benchmark": "mcm_qec",
        "config": vars(args).copy(),
        "data": {"rounds": {}, "qubits": {}},
    }

    def save() -> None:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2))

    # Sweep 1: time vs syndrome-extraction rounds at fixed distance.
    d = args.fixed_distance
    print(f"== rounds sweep (distance {d}, n={2 * d - 1}) ==")
    header = f"{'rounds':>7} {'MCMs':>5}"
    for b in available:
        for m in args.modes:
            header += f" {b + '/' + m:>16}"
    print(header)
    for r in args.rounds:
        qc = rep_code_circuit(d, r)
        row = f"{r:>7} {r * (d - 1):>5}"
        for b in available:
            for m in args.modes:
                secs = measure_cell(b, m, qc, args)
                results["data"]["rounds"].setdefault(b, {}).setdefault(m, []).append([r, secs])
                row += f" {secs * 1e3:>14.2f}ms"
                save()
        print(row, flush=True)

    # Sweep 2: time vs qubit count at fixed rounds.
    r = args.fixed_rounds
    print(f"\n== qubit sweep ({r} rounds) ==")
    header = f"{'n':>3} {'dist':>4}"
    for b in available:
        for m in args.modes:
            header += f" {b + '/' + m:>16}"
    print(header)
    for d in args.distances:
        n = 2 * d - 1
        qc = rep_code_circuit(d, r)
        row = f"{n:>3} {d:>4}"
        for b in available:
            for m in args.modes:
                secs = measure_cell(b, m, qc, args)
                results["data"]["qubits"].setdefault(b, {}).setdefault(m, []).append([n, secs])
                row += f" {secs * 1e3:>14.2f}ms"
                save()
        print(row, flush=True)

    # Derived metric: marginal cost per single ancilla MCM, from the rounds sweep
    # (slope between the first and last round counts, divided by MCMs per round).
    d = args.fixed_distance
    print("\nper-MCM marginal cost (rounds sweep slope / (d-1) ancillas):")
    for b in available:
        for m in args.modes:
            series = sorted(results["data"]["rounds"][b][m])
            (r0, t0), (r1, t1) = series[0], series[-1]
            if r1 > r0:
                per_mcm = (t1 - t0) / ((r1 - r0) * (d - 1))
                unit = "s/shot" if m == "collapse" else f"s/{args.shots} shots"
                print(f"  {b:>6}/{m:<9} {per_mcm * 1e6:>10.1f} us per MCM ({unit})")

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
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    def draw(ax, sweep: str) -> None:
        for b, by_mode in results["data"][sweep].items():
            for m, series in by_mode.items():
                series = sorted(series)
                label = f"{b} {m}" + (
                    " (s/shot)" if m == "collapse" else f" ({cfg['shots']} shots)"
                )
                ax.plot(
                    [x for x, _ in series],
                    [s for _, s in series],
                    marker=MODE_MARKERS[m],
                    markersize=5,
                    linewidth=1.6,
                    linestyle=MODE_STYLES[m],
                    color=BACKEND_COLORS.get(b),
                    label=label,
                )
        ax.set_yscale("log")
        ax.set_ylabel("seconds (min of reps)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

    d = cfg["fixed_distance"]
    draw(axes[0], "rounds")
    axes[0].set_xlabel(f"syndrome-extraction rounds ({d - 1} ancilla MCMs each)")
    axes[0].set_title(f"vs MCM rounds — distance {d} ({2 * d - 1} qubits)")
    draw(axes[1], "qubits")
    axes[1].set_xlabel("qubits (n = 2d-1)")
    axes[1].set_title(f"vs qubit count — {cfg['fixed_rounds']} rounds")
    fig.suptitle("Mid-circuit measurement: repetition-code syndrome extraction", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"Chart -> {path}")


if __name__ == "__main__":
    main()
