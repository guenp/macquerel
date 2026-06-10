# macquerel Implementation Plan — Remaining Work

macquerel is a quantum state-vector simulator targeting Apple Silicon's unified-memory
architecture. The v0.1 and v0.2 lines (CPU/MLX/Metal backends, gate fusion + qubit
remapping, expectation values, automatic backend selection, Cirq/Qiskit adapters, the full
benchmarking suite, and the shot-batch / fusion-width autotuners) are **complete** — see
[`plan_completed.md`](plan_completed.md) for the shipped record, including the MLX/Metal
performance findings.

This document tracks only work that has **not** been implemented yet. Steps keep their
original numbering for continuity with the completed record.

---

## v0.2.x — GPU backend performance: **SHIPPED (2026-06-10)**

The performance line (Steps 21–28, plus the post-line Step 30) is complete and merged
on `gpu-perf-plan`. Every step was A/B-benchmarked per the measurement protocol; the
full per-step data, charts, and justifications live in
[`benchmarks/data/steps/`](../benchmarks/data/steps/README.md), and the shipped record
with measured results is in
[`plan_completed.md`](plan_completed.md#v02x--gpu-backend-performance-steps-21-30).

Headline (geomean over circuits, vs the pre-line baseline): **Metal 2.5–2.9× across
6–28q**, **MLX 1.9–2.6× at 22–28q** (best cell: random@28 14.7×), **CPU 1.5–2.0×**.
Against Qiskit Aer, macquerel's auto path wins from 20q (random 28 vs 51 ms) and by
5–12× at 24q+. Auto-select is re-tuned to CPU ≤16q / Metal ≥17q (MLX serves 17–30q
only when pyobjc-Metal is absent).

Not shipped: **Step 28** (qubit remapping) is wired but **off by default** — the A/B
lost on every backend (readback inverse transpose outweighs any stride benefit; opt in
with `MACQUEREL_REMAP=1`). **Step 29** (custom MLX dense kernel) was not needed — Step
27 closed the gap it targeted.

### Step 30 — per-backend, qubit-aware fusion-width defaults

A post-line fusion-width re-sweep (`benchmarks/data/fusion_width.json`: widths 1–6 ×
{QFT, random, QAOA, QV} × 16–24q, all three backends) showed the optimal
`max_fused_qubits` is now a **backend** property as well as a qubit-count one: Steps
22/25 removed most of Metal's per-gate overhead — the thing fusion amortizes — so
narrow fusion wins on Metal up to ~22q (1.3–2× vs the old global default of 4), while
at 24q+ every backend is apply-bound and 4 still wins (a *flat* per-backend width 2
regressed metal random@24–28 by 2.7–3.7× in the step A/B — the default must be
qubit-aware). Defaults: **metal 2 ≤22q, cpu 3 ≤18q, otherwise 4; mlx 4 everywhere**.
`fuse_gates` now takes the target backend and resolves the default from
`default_fusion_width(backend, n_qubits)`; `MACQUEREL_FUSION_WIDTH` still pins a
single global width, and `auto` still runs the per-chip autotuner. A/B per the usual
protocol; results in [`benchmarks/data/steps/`](../benchmarks/data/steps/README.md).

---

## v0.2.x+ — performance candidates (not scheduled)

Measured candidates that came out of the backend comparison
([`docs/backends.md`](backends.md)), roughly in order of expected payoff. Each would
follow the same A/B protocol as Steps 21–30 before shipping:

- **Batched small-circuit simulation.** The small-n regime is dispatch-bound, so the
  fix is amortization: pack hundreds of parameter-sweep circuits (VQE/QML workloads)
  into one kernel launch. This attacks the fixed per-run costs from the other side;
  also listed under v0.3 as the `BatchedSimulator` feature.
- **A custom MLX dense kernel** (`mx.fast.metal_kernel`) would bypass `tensordot`'s
  internal permutation — the dominant cost on scattered-target circuits — and close
  most of MLX's random/QFT gap. Deferred during the performance line (Step 29)
  because the native Metal backend already provides those kernels; it matters only
  for the no-PyObjC fallback path.
- **An in-place-style diagonal path for MLX** (compiled elementwise phase multiply
  instead of a gather table) targets the QFT gap specifically — wide diagonal runs
  are where MLX falls furthest behind Metal (6.2× at 24q, 9.5× at 28q).
- **Lowering Metal's small-n floor** — persistent command buffers across `run()`
  calls, a pre-warmed pipeline cache, pooled buffer allocation — could push the
  CPU/Metal crossover below 16q.
- **Per-chip tier boundaries.** The 16q crossover is measured on an M5 Max; base
  M-series chips have different bandwidth/latency ratios. The same measure-and-cache
  approach used for `MACQUEREL_FUSION_WIDTH=auto` could autotune backend selection.

---

## v0.3

- **Noise channels / density matrices** — `DensityMatrixSimulator` with Kraus-operator
  channels.
- **Memory-mapped out-of-core backend** — state vector backed by an NVMe file via
  `np.memmap`, for single large runs past DRAM capacity.
- **Batched small-circuit simulation** — `BatchedSimulator` packing many small circuits
  (QML/VQE parameter sweeps) into one kernel launch.
- **Multi-Mac over Thunderbolt** — distributed state vector using index-bit partitioning
  across machines.

---

## Verification

After each step, run `uv run pytest tests/ -x -q` and confirm the new tests pass before
moving to the next step. Final verification:

```bash
uv run pytest tests/ -v           # full suite green
python -c "
import macquerel as mq
qc = mq.Circuit(n_qubits=3)
qc.h(0); qc.cx(0, 1); qc.cx(0, 2); qc.measure_all()
result = mq.Simulator(backend='cpu').run(qc, shots=1000)
print(result)   # should show ~500 '000' and ~500 '111'
"
```
