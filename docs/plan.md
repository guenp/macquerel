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

The performance line (Steps 21–28) is complete and merged on `gpu-perf-plan`. Every step
was A/B-benchmarked per the measurement protocol; the full per-step data, charts, and
justifications live in [`benchmarks/data/steps/`](../benchmarks/data/steps/README.md),
and the shipped record with measured results is in
[`plan_completed.md`](plan_completed.md#v02x--gpu-backend-performance-steps-21-28).

Headline (geomean over circuits, vs the pre-line baseline): **Metal 2.2–2.8× at
22–28q** (and 1.5–1.7× below that), **MLX 1.9–2.6× at 22–28q** (best cell: random@28
14.6×), **CPU 1.6–2.0× at 20–22q**. Against Qiskit Aer, macquerel's auto path is at
parity at 20q and wins 5–12× at 24q+. Auto-select is re-tuned to CPU ≤16q / Metal ≥17q
(MLX serves 17–30q only when pyobjc-Metal is absent).

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
