# macquerel Implementation Plan — Remaining Work

macquerel is a quantum state-vector simulator targeting Apple Silicon's unified-memory
architecture. The v0.1 and v0.2 lines (CPU/MLX/Metal backends, gate fusion + qubit
remapping, expectation values, automatic backend selection, Cirq/Qiskit adapters, the full
benchmarking suite, and the shot-batch / fusion-width autotuners) are **complete**, as is
the v0.2.x+ performance-candidate line (MLX broadcast diagonal + custom dense kernel,
Metal small-n floor, per-chip tier autotuning, and the `BatchedSimulator`) — see
[`plan_completed.md`](plan_completed.md) for the shipped record, including the MLX/Metal
performance findings and the per-step A/B results.

This document tracks only work that has **not** been implemented yet. Steps keep their
original numbering for continuity with the completed record.

## A/B Protocol

Each optimization step is measured as a commit-pinned A/B test against the immediately
previous baseline: run the same benchmark cells before and after the change, in isolated
subprocesses to avoid GPU memory-pool and lazy-graph contamination, using min-of-reps
timings to reduce noise. Correctness is gated first with the test suite, then performance
is accepted only if the affected backend/circuit/qubit ranges improve without meaningful
regressions elsewhere. Per-step JSONs and plots are saved under `benchmarks/data/steps/`,
and changes that lose their A/B, like default qubit remapping, are left disabled or
reverted rather than folded into the hot path.

Implemented steps are moved to [`plan_completed.md`](plan_completed.md), where each
shipped optimization records its commit ID alongside the measured A/B result and design
notes.

---

## v0.3

- **Memory-mapped out-of-core backend** — state vector backed by an NVMe file via
  `np.memmap`, for single large runs past DRAM capacity.
- **Multi-Mac over Thunderbolt** — distributed state vector using index-bit partitioning
  across machines.

> `BatchedSimulator` (batched small-circuit simulation) shipped early as Step 31 of the
> v0.2.x+ line, and **Noise channels / density matrices** (`DensityMatrixSimulator`
> with Kraus-operator channels) shipped as the first v0.3 feature — see
> [`plan_completed.md`](plan_completed.md).

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
