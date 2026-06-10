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
