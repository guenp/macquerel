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
128 GiB machine; its int32 ShapeElem caps it at 30q regardless).

Usage:
    uv run python benchmarks/bench_memory.py
    uv run python benchmarks/bench_memory.py --qubits 5 10 ... --backends metal \
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

BACKEND_COLORS = {"metal": "#d62728", "mlx": "#1f77b4", "cpu": "#2ca02c"}


def theoretical_bytes(n: int) -> int:
    return (2**n) * 8  # complex64


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


def measure_cell(backend: str, n: int, timeout: float) -> dict:
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
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        msg = (proc.stderr.strip().splitlines() or ["worker exited nonzero"])[-1]
        raise RuntimeError(msg[:120])
    fp = _FOOTPRINT_RE.search(proc.stderr)
    rss = _MAXRSS_RE.search(proc.stderr)
    if not fp or not rss:
        raise RuntimeError("could not parse /usr/bin/time -l output")
    return {"footprint": int(fp.group(1)), "maxrss": int(rss.group(1))}


def est_peak_gib(backend: str, n: int) -> float:
    mult = _PEAK_MULT.get(f"macquerel-{backend}", 4.0)
    return theoretical_bytes(n) * mult / 1024**3


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--qubits", type=int, nargs="+", default=list(range(5, 33)))
    ap.add_argument("--backends", nargs="+", default=BACKENDS, choices=BACKENDS)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument(
        "--mem-budget-frac",
        type=float,
        default=0.45,
        help="skip cells whose pessimistic peak estimate exceeds this fraction of RAM",
    )
    ap.add_argument("--json", default="benchmarks/data/memory.json")
    ap.add_argument("--plot", default="benchmarks/data/memory.png")
    # Worker mode (internal).
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--backend", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--n", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        raise SystemExit(run_worker(args))

    if sys.platform != "darwin":
        raise SystemExit("this benchmark relies on macOS /usr/bin/time -l ledger output")

    budget = _system_ram_gib() * args.mem_budget_frac
    print(f"per-cell memory budget: {budget:.0f} GiB\n")
    print(f"{'n':>3} {'theory':>10}" + "".join(f" {b:>12}" for b in args.backends))

    results: dict = {"theoretical_bytes": {}, "measured": {b: {} for b in args.backends}}
    for n in args.qubits:
        results["theoretical_bytes"][n] = theoretical_bytes(n)
        row = f"{n:>3} {theoretical_bytes(n) / 1024**3:>9.3g}G"
        for b in args.backends:
            if n > MAX_QUBITS[b]:
                row += f" {'-':>12}"
                continue
            if est_peak_gib(b, n) > budget:
                row += f" {'SKIP':>12}"
                continue
            try:
                cell = measure_cell(b, n, args.timeout)
                results["measured"][b][n] = cell
                row += f" {cell['footprint'] / 1024**3:>11.3g}G"
            except Exception as e:
                row += f" ERR({str(e)[:20]})"
        print(row, flush=True)
        if args.json:
            path = Path(args.json)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"benchmark": "memory_footprint", **results}, indent=2))

    if args.json:
        print(f"\nRaw data -> {args.json}")
    make_plot(results, args.backends, args.plot)


def make_plot(results: dict, backends: list[str], path: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping chart (raw data still saved).")
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ns = sorted(results["theoretical_bytes"])
    ax.plot(
        ns,
        [results["theoretical_bytes"][n] / 1024**3 for n in ns],
        "k--",
        linewidth=1.5,
        label="theoretical statevector (2$^N$ × 8 B, complex64)",
    )
    for b in backends:
        cells = results["measured"].get(b, {})
        if not cells:
            continue
        xs = sorted(int(k) for k in cells)
        ys = [cells[x]["footprint"] / 1024**3 for x in xs]
        ax.plot(
            xs,
            ys,
            marker="o",
            markersize=4,
            linewidth=1.6,
            color=BACKEND_COLORS.get(b),
            label=f"measured peak footprint — {b}",
        )
    ax.set_xlabel("qubits N")
    ax.set_ylabel("RAM (GiB)")
    ax.set_yscale("log", base=2)
    ax.set_title(
        "Statevector RAM: theoretical vs measured peak footprint (GHZ, per-cell subprocess)"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"Chart -> {path}")


if __name__ == "__main__":
    main()
