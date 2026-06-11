#!/usr/bin/env python3
"""bench_memory.py — theoretical vs measured RAM for an N-qubit statevector.

Theoretical: a complex64 statevector is exactly ``2**N * 8`` bytes. Measured:
the **peak memory footprint** of a fresh subprocess that allocates the state on
one backend and runs a GHZ circuit over it (compute-light, so the footprint is
dominated by state storage, not scratch). Each cell runs under
``/usr/bin/time -l`` (macOS), whose *peak memory footprint* comes from the
kernel's per-task ledger — unlike plain RSS it charges unified-memory Metal
allocations to the process, which is what the GPU backends allocate.

The worker forces evaluation without a host readback (``to_numpy`` would add a
full extra state-sized NumPy copy and hide each backend's true overhead), so
the measured/theoretical gap is the backend's real multiplier: Metal updates
in place (expect ~1x + interpreter baseline), MLX double-buffers every gate
and holds lazy-graph temporaries (~2-4x), CPU tensordot makes reshaped copies
(~2-4x). The interpreter baseline (~100-200 MB) dominates below ~22 qubits —
that flat floor on the left of the chart is Python+NumPy, not the state.

Cells whose pessimistic peak estimate exceeds the memory budget are skipped,
exactly like bench_statevector (the MLX 16x multiplier cuts it off at 28q on a
128 GiB machine; its int32 ShapeElem caps it at 30q regardless). The budget is
additionally hard-capped at 64 GiB per cell, regardless of installed RAM.

The ``dm`` series measures the v0.3 `DensityMatrixSimulator` (GHZ + a
depolarizing channel per qubit): an N-qubit density matrix is its row-major
vectorization, a ``4**N * 8``-byte doubled statevector, so it inherits each
backend's multiplier at the doubled count — N=16 on Metal is a 32 GiB state,
the largest cell that fits the 64 GiB cap.

Results merge into the existing --json on every write, so one series can be
re-measured (e.g. ``--series dm``) without re-running the others; the plot
always redraws every series present in the file.

Usage:
    uv run python benchmarks/bench_memory.py
    uv run python benchmarks/bench_memory.py --series dm --backends metal \
        --json benchmarks/data/memory.json --plot benchmarks/data/memory.png
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

# Budget gate + peak-multiplier calibration shared with the timing harness.
from bench_statevector import _PEAK_MULT, _system_ram_gib

BACKENDS = ["cpu", "mlx", "metal"]
# Hard per-backend ceilings: MLX's int32 ShapeElem rejects 2**31 amplitudes;
# CPU past 28q is minutes of tensordot for no extra memory insight.
MAX_QUBITS = {"cpu": 28, "mlx": 30, "metal": 32}
# Density-matrix ceilings at the *doubled* count: an N-qubit density matrix is
# a 2N-qubit vectorized state, so the same limits land at N = ceiling // 2.
MAX_QUBITS_DM = {"cpu": 14, "mlx": 15, "metal": 16}
# Never measure a cell whose pessimistic peak exceeds this, whatever the RAM.
HARD_CAP_GIB = 64.0

BACKEND_COLORS = {"metal": "#d62728", "mlx": "#1f77b4", "cpu": "#2ca02c"}


def theoretical_bytes(n: int) -> int:
    return (2**n) * 8  # complex64


def theoretical_dm_bytes(n: int) -> int:
    return (4**n) * 8  # complex64 density matrix = doubled statevector


# ----------------------------------------------------------------------------
# Worker: allocate + evolve one GHZ state, force evaluation, exit.
# ----------------------------------------------------------------------------
def run_worker(args) -> int:
    from macquerel.circuit import Circuit, Gate
    from macquerel.compiler import fuse_gates
    from macquerel.simulator import _make_backend

    n = args.n
    qc = Circuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)

    if args.dm:
        # Density-matrix cell: the same GHZ plus one depolarizing channel per
        # qubit, evolved over the 4**n vectorized state. Forcing evaluation
        # mirrors the statevector path below: no host readback, so the
        # footprint is the backend's own doubled-state storage + scratch.
        from macquerel.density import DensityMatrixSimulator

        for q in range(n):
            qc.depolarizing(q, 0.01)
        dm = DensityMatrixSimulator(backend=args.backend)
        backend, state = dm._evolve(qc)
        if args.backend == "mlx":
            import mlx.core as mx

            mx.eval(state.data)
        elif args.backend == "metal":
            backend._flush()
        return 0

    backend = _make_backend(args.backend, "complex64")
    fused = fuse_gates(qc, backend=args.backend)
    sv = backend.allocate(n, np.complex64)
    for op in fused.ops:
        if isinstance(op, Gate):
            sv = backend.apply_matrix(sv, op.matrix, op.targets, op.controls or None)

    # Force evaluation *without* to_numpy: a readback would allocate a second
    # full state on the host and obscure the backend's own footprint.
    if args.backend == "mlx":
        import mlx.core as mx

        mx.eval(sv.data)
    elif args.backend == "metal":
        backend._flush()  # commit + wait on the batched command buffer
    return 0


# ----------------------------------------------------------------------------
# Parent: run each cell under /usr/bin/time -l and parse the ledger numbers.
# ----------------------------------------------------------------------------
_FOOTPRINT_RE = re.compile(r"^\s*(\d+)\s+peak memory footprint", re.M)
_MAXRSS_RE = re.compile(r"^\s*(\d+)\s+maximum resident set size", re.M)


def measure_cell(backend: str, n: int, timeout: float, dm: bool = False) -> dict:
    cmd = [
        "/usr/bin/time",
        "-l",
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--backend",
        backend,
        "--n",
        str(n),
    ]
    if dm:
        cmd.append("--dm")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        msg = (proc.stderr.strip().splitlines() or ["worker exited nonzero"])[-1]
        raise RuntimeError(msg[:120])
    fp = _FOOTPRINT_RE.search(proc.stderr)
    rss = _MAXRSS_RE.search(proc.stderr)
    if not fp or not rss:
        raise RuntimeError("could not parse /usr/bin/time -l output")
    return {"footprint": int(fp.group(1)), "maxrss": int(rss.group(1))}


def est_peak_gib(backend: str, n: int, dm: bool = False) -> float:
    mult = _PEAK_MULT.get(f"macquerel-{backend}", 4.0)
    base = theoretical_dm_bytes(n) if dm else theoretical_bytes(n)
    return base * mult / 1024**3


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--qubits", type=int, nargs="+", default=list(range(5, 33)))
    ap.add_argument("--backends", nargs="+", default=BACKENDS, choices=BACKENDS)
    ap.add_argument(
        "--series",
        nargs="+",
        default=["sv", "dm"],
        choices=["sv", "dm"],
        help="sv: statevector cells; dm: density-matrix cells (doubled state)",
    )
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument(
        "--mem-budget-frac",
        type=float,
        default=0.45,
        help="skip cells whose pessimistic peak estimate exceeds this fraction of RAM",
    )
    ap.add_argument("--json", default="benchmarks/data/memory.json")
    ap.add_argument("--plot", default="benchmarks/data/memory.png")
    ap.add_argument(
        "--replot",
        action="store_true",
        help="skip measuring; redraw --plot from the existing --json",
    )
    # Worker mode (internal).
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--backend", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--n", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--dm", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        raise SystemExit(run_worker(args))

    if args.replot:
        results = json.loads(Path(args.json).read_text())
        make_plot(results, args.backends, args.plot)
        return

    if sys.platform != "darwin":
        raise SystemExit("this benchmark relies on macOS /usr/bin/time -l ledger output")

    budget = min(_system_ram_gib() * args.mem_budget_frac, HARD_CAP_GIB)
    print(f"per-cell memory budget: {budget:.0f} GiB (hard cap {HARD_CAP_GIB:.0f} GiB)\n")

    # Merge into whatever the JSON already holds so one series can be
    # (re)measured without discarding the others.
    results: dict = {"benchmark": "memory_footprint"}
    if args.json and Path(args.json).exists():
        results.update(json.loads(Path(args.json).read_text()))
    results.setdefault("theoretical_bytes", {})
    results.setdefault("measured", {})
    if "dm" in args.series:
        results.setdefault("theoretical_dm_bytes", {})
        results.setdefault("measured_dm", {})

    def save() -> None:
        if args.json:
            path = Path(args.json)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(results, indent=2))

    series_cfg = {
        "sv": (theoretical_bytes, MAX_QUBITS, "theoretical_bytes", "measured", False),
        "dm": (theoretical_dm_bytes, MAX_QUBITS_DM, "theoretical_dm_bytes", "measured_dm", True),
    }
    for series in args.series:
        theory_fn, caps, theory_key, measured_key, dm = series_cfg[series]
        print(f"[{series}] {'n':>3} {'theory':>10}" + "".join(f" {b:>12}" for b in args.backends))
        for b in args.backends:
            results[measured_key].setdefault(b, {})
        for n in args.qubits:
            if n > max(caps[b] for b in args.backends):
                continue
            # str keys: merged JSON round-trips through json.loads, which
            # stringifies dict keys; mixing int and str would duplicate them.
            results[theory_key][str(n)] = theory_fn(n)
            row = f"[{series}] {n:>3} {theory_fn(n) / 1024**3:>9.3g}G"
            for b in args.backends:
                if n > caps[b]:
                    row += f" {'-':>12}"
                    continue
                if est_peak_gib(b, n, dm=dm) > budget:
                    row += f" {'SKIP':>12}"
                    continue
                try:
                    cell = measure_cell(b, n, args.timeout, dm=dm)
                    results[measured_key][b][str(n)] = cell
                    row += f" {cell['footprint'] / 1024**3:>11.3g}G"
                except Exception as e:
                    row += f" ERR({str(e)[:20]})"
            print(row, flush=True)
            save()
        print()

    if args.json:
        print(f"Raw data -> {args.json}")
    make_plot(results, args.backends, args.plot)


def _fmt_bytes(x: float, _pos=None) -> str:
    """Human-readable byte tick labels (256 B, 4 KiB, 1 GiB, ...)."""
    for unit, size in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if x >= size:
            return f"{x / size:g} {unit}"
    return f"{x:g} B"


def make_plot(results: dict, backends: list[str], path: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    except ImportError:
        print("matplotlib not installed — skipping chart (raw data still saved).")
        return

    fig, ax = plt.subplots(figsize=(9.5, 6))
    theory = {int(k): v for k, v in results.get("theoretical_bytes", {}).items()}
    ns = sorted(theory)
    if ns:
        ax.plot(
            ns,
            [theory[n] for n in ns],
            "k--",
            linewidth=1.5,
            label="theoretical statevector (2$^N$ × 8 B, complex64)",
        )
    # Floor the y-axis at 16 MiB: below that everything hides under the
    # interpreter baseline anyway, and the empty decades of theoretical line
    # only squash the measured curves.
    lo = 2.0**24
    hi = max(theory.values()) if theory else lo
    dm_theory = {int(k): v for k, v in results.get("theoretical_dm_bytes", {}).items()}
    if dm_theory:
        dns = sorted(dm_theory)
        ax.plot(
            dns,
            [dm_theory[n] for n in dns],
            "k:",
            linewidth=1.5,
            label="theoretical density matrix (4$^N$ × 8 B, complex64)",
        )
        hi = max(hi, max(dm_theory.values()))
    for b in backends:
        cells = {int(k): v for k, v in results["measured"].get(b, {}).items()}
        if not cells:
            continue
        xs = sorted(cells)
        ys = [cells[x]["footprint"] for x in xs]
        hi = max(hi, max(ys))
        ax.plot(
            xs,
            ys,
            marker="o",
            markersize=4,
            linewidth=1.6,
            color=BACKEND_COLORS.get(b),
            label=f"measured peak — {b} statevector",
        )
    for b in backends:
        cells = {int(k): v for k, v in results.get("measured_dm", {}).get(b, {}).items()}
        if not cells:
            continue
        xs = sorted(cells)
        ys = [cells[x]["footprint"] for x in xs]
        hi = max(hi, max(ys))
        ax.plot(
            xs,
            ys,
            marker="^",
            markersize=5,
            linewidth=1.6,
            linestyle="--",
            color=BACKEND_COLORS.get(b),
            label=f"measured peak — {b} density matrix",
        )
    ax.set_xlabel("qubits N")
    ax.set_ylabel("RAM")
    ax.set_yscale("log", base=2)
    ax.set_ylim(bottom=lo)
    # Readable byte ticks every 16x instead of raw 2^k exponents.
    exps = range(int(np.floor(np.log2(lo))) // 4 * 4, int(np.ceil(np.log2(hi))) + 4, 4)
    ax.yaxis.set_major_locator(FixedLocator([2.0**e for e in exps]))
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_bytes))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_title(
        "Statevector vs density-matrix RAM: theoretical vs measured peak footprint\n"
        "(GHZ / GHZ + depolarizing, per-cell subprocess)"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"Chart -> {path}")


if __name__ == "__main__":
    main()
