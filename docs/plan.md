# macquerel Implementation Plan — Remaining Work

macquerel is a quantum state-vector simulator targeting Apple Silicon's unified-memory
architecture. The v0.1 and v0.2 lines (CPU/MLX/Metal backends, gate fusion + qubit
remapping, expectation values, automatic backend selection, Cirq/Qiskit adapters, the full
benchmarking suite, and the shot-batch / fusion-width autotuners) are **complete** — see
[`plan_completed.md`](plan_completed.md) for the shipped record, including the MLX/Metal
performance findings.

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

## v0.2.x+ — performance candidates

Measured candidates that came out of the backend comparison
([`docs/backends.md`](backends.md)), roughly in order of expected payoff. Each would
follow the same A/B protocol before shipping:

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
