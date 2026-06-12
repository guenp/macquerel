#!/usr/bin/env python3
"""Plot per-step speedups for the GPU-perf plan (docs/plan.md Steps 21-30).

Reads the per-step JSONs written by run_step_bench.sh into benchmarks/data/steps/
(named ``<step>-<commit>-<backend>.json``) and produces:

- ``step_speedups.png`` — the headline chart: per-backend speedup of each step
  relative to the **baseline**, aggregated (geometric mean) over circuits and
  qubit counts, plus a per-qubit-count breakdown for the final state.
- ``step_curves_<backend>.png`` — per-circuit runtime curves, one line per step,
  for each backend that has data.
- a text table of step-over-step and cumulative speedups on stdout.

A step only re-measures the backends it touched; untouched backends carry their
previous measurement forward, so cumulative numbers always compare like with like.

Usage:
    uv run python benchmarks/plot_steps.py [--dir benchmarks/data/steps]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

# Execution order of the plan (docs/plan.md): step 24 lands before step 23;
# the v0.2.x+ candidate line (steps 32-34) re-baselined at the 0.2.1 release
# commit ("step32-baseline") and runs 32 -> 33 -> 34. Steps 31/35 are not here:
# 31 (BatchedSimulator) has its own harness (bench_batched.py) and 35 is
# routing-only, which this benchmark pins away.
STEP_ORDER = [
    "step20-baseline",
    "step21",
    "step22",
    "step24",
    "step23",
    "step25",
    "step26",
    "step27",
    "step28",
    "step30",
    "step32-baseline",
    "step32",
    "step33",
    "step34",
]

BACKENDS = ["macquerel-cpu", "macquerel-mlx", "macquerel-metal"]
BACKEND_COLORS = {
    "macquerel-metal": "#d62728",
    "macquerel-mlx": "#1f77b4",
    "macquerel-cpu": "#2ca02c",
}


def load_steps(data_dir: Path):
    """-> {step: {backend: {(circuit, q): secs}}}, ordered per STEP_ORDER."""
    by_step: dict[str, dict[str, dict]] = defaultdict(dict)
    commits: dict[str, str] = {}
    wanted = set(STEP_ORDER)
    for path in sorted(data_dir.glob("step*.json")):
        doc = json.loads(path.read_text())
        if "results" not in doc:
            continue
        step = doc.get("step")
        if step not in wanted:
            continue
        commits[step] = doc.get("commit", "?")
        for circuit, by_backend in doc["results"].items():
            for backend, series in by_backend.items():
                cell = by_step[step].setdefault(backend, {})
                for q, secs in series:
                    cell[(circuit, int(q))] = secs
    steps = [s for s in STEP_ORDER if s in by_step]
    return steps, dict(by_step), commits


def carry_forward(steps, by_step):
    """-> {step: {backend: {(circuit, q): secs}}} with untouched backends carried."""
    state: dict[str, dict] = {}
    timeline: dict[str, dict[str, dict]] = {}
    for step in steps:
        for backend, cells in by_step[step].items():
            state[backend] = cells
        timeline[step] = {b: dict(c) for b, c in state.items()}
    return timeline


def geomean(values):
    vals = [v for v in values if v and v > 0]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def speedup_table(steps, timeline, baseline):
    """Geomean speedups per (step, backend): vs baseline and vs previous step."""
    rows = []
    for i, step in enumerate(steps):
        for backend in BACKENDS:
            cur = timeline[step].get(backend)
            base = timeline[baseline].get(backend)
            if not cur or not base:
                continue
            shared = [k for k in cur if k in base]
            vs_base = geomean([base[k] / cur[k] for k in shared])
            vs_prev = None
            if i > 0:
                prev = timeline[steps[i - 1]].get(backend)
                if prev:
                    shared_p = [k for k in cur if k in prev]
                    vs_prev = geomean([prev[k] / cur[k] for k in shared_p])
            rows.append((step, backend, vs_prev, vs_base))
    return rows


def make_speedup_plot(steps, timeline, baseline, commits, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    # Left: cumulative geomean speedup vs baseline per backend, per step.
    plot_steps_ = steps[1:] if steps and steps[0] == baseline else steps
    width = 0.25
    xs = np.arange(len(plot_steps_))
    for bi, backend in enumerate(BACKENDS):
        ys, labels = [], []
        for step in plot_steps_:
            cur = timeline[step].get(backend)
            base = timeline[baseline].get(backend)
            if cur and base:
                shared = [k for k in cur if k in base]
                ys.append(geomean([base[k] / cur[k] for k in shared]) or np.nan)
            else:
                ys.append(np.nan)
        bars = ax1.bar(
            xs + (bi - 1) * width,
            ys,
            width,
            label=backend,
            color=BACKEND_COLORS.get(backend),
        )
        for x, y in zip(xs, ys, strict=True):
            if y == y:  # not NaN
                ax1.annotate(
                    f"{y:.2f}x",
                    xy=(x + (bi - 1) * width, y),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
    ax1.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax1.set_xticks(xs)
    ax1.set_xticklabels([f"{s}\n({commits.get(s, '?')})" for s in plot_steps_], fontsize=8)
    ax1.set_ylabel("cumulative speedup vs baseline (geomean, higher = better)")
    ax1.set_title("Speedup vs baseline after each step")
    ax1.legend(fontsize=8)
    ax1.grid(True, axis="y", alpha=0.3)

    # Right: final-state speedup vs baseline by qubit count (geomean over circuits).
    final = steps[-1]
    qubit_counts = sorted({q for b in BACKENDS for (c, q) in timeline[final].get(b, {})})
    for backend in BACKENDS:
        cur = timeline[final].get(backend)
        base = timeline[baseline].get(backend)
        if not cur or not base:
            continue
        xs2, ys2 = [], []
        for q in qubit_counts:
            ratios = [base[k] / cur[k] for k in cur if k[1] == q and k in base]
            gm = geomean(ratios)
            if gm:
                xs2.append(q)
                ys2.append(gm)
        ax2.plot(xs2, ys2, marker="o", label=backend, color=BACKEND_COLORS.get(backend))
    ax2.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("qubits")
    ax2.set_ylabel("final speedup vs baseline (geomean over circuits)")
    ax2.set_title(f"Final state ({final}) vs baseline, by qubit count")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"Chart -> {out_path}")


def make_backend_curves(steps, timeline, backend, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    circuits = sorted({c for s in steps for (c, q) in timeline[s].get(backend, {})})
    if not circuits:
        return
    cols = min(2, len(circuits))
    rows = math.ceil(len(circuits) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4.5 * rows), squeeze=False)
    cmap = plt.get_cmap("viridis")

    for idx, circuit in enumerate(circuits):
        ax = axes[idx // cols][idx % cols]
        for si, step in enumerate(steps):
            cells = timeline[step].get(backend, {})
            series = sorted((q, s) for (c, q), s in cells.items() if c == circuit)
            if not series:
                continue
            ax.plot(
                [q for q, _ in series],
                [s * 1e3 for _, s in series],
                marker="o",
                markersize=3,
                label=step,
                color=cmap(si / max(1, len(steps) - 1)),
                linewidth=1.3,
            )
        ax.set_title(f"{backend} — {circuit}")
        ax.set_xlabel("qubits")
        ax.set_ylabel("statevector time (ms)")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)

    for j in range(len(circuits), rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"Chart -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="benchmarks/data/steps")
    args = ap.parse_args()
    data_dir = Path(args.dir)

    steps, by_step, commits = load_steps(data_dir)
    if not steps:
        print(f"No step JSONs found in {data_dir}")
        return
    baseline = steps[0]
    timeline = carry_forward(steps, by_step)

    print(f"{'step':<18} {'backend':<16} {'vs prev':>9} {'vs baseline':>12}")
    for step, backend, vs_prev, vs_base in speedup_table(steps, timeline, baseline):
        prev_s = f"{vs_prev:.2f}x" if vs_prev else "-"
        base_s = f"{vs_base:.2f}x" if vs_base else "-"
        print(f"{step:<18} {backend:<16} {prev_s:>9} {base_s:>12}")

    make_speedup_plot(steps, timeline, baseline, commits, data_dir / "step_speedups.png")
    for backend in BACKENDS:
        short = backend.removeprefix("macquerel-")
        make_backend_curves(steps, timeline, backend, data_dir / f"step_curves_{short}.png")


if __name__ == "__main__":
    main()
