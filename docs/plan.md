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

## v0.3.x — RAM usage candidates

Measured peak-to-theory multipliers (`benchmarks/data/memory.json`, GHZ cells ≥256 MiB,
statevector and density-matrix series agree): **metal 1.0×** (in-place — nothing left to
win), **cpu 3.0×**, **mlx 19-25×**. Candidates in priority order; each follows the A/B
protocol (correctness gate first, then peak footprint *and* runtime measured per cell,
since most of these trade one for the other).

- **Step 36: MLX eval cadence + pool release** — the 19-25× peak is the lazy graph
  holding up to 16 double-buffered full-width intermediates between `async_eval` kicks
  (`_ASYNC_EVAL_INTERVAL = 16`) plus a buffer pool that never returns memory to the OS.
  Shrink the interval as the state grows (e.g. every 2-4 gates above ~26 effective
  qubits), call `mx.clear_cache()` at observation boundaries, and verify buffer donation
  (MLX reuses an input buffer at refcount 1 — check `MLXState` holds no extra refs).
  Target: the ~2-3× double-buffer floor, which un-skips the mlx DM n=15 cell (currently
  estimated 128 GiB) and reaches 29-30q statevectors without swap. Re-run the Step 24
  A/B: cadence trades memory against pipeline fullness.
- **Step 37: quantum-trajectory simulator** — Monte-Carlo wavefunction: K stochastic
  *statevector* trajectories, sampling one Kraus operator per channel per trajectory
  (probability `||K_k psi||^2`, then renormalize). Memory `2**n` per trajectory, run
  sequentially — noisy **33-qubit** simulation on Metal instead of the DM's n=16 cap,
  reusing `Simulator` and the existing `ChannelOp`s. Exact in expectation; sampling
  noise ~1/sqrt(K). Complements `DensityMatrixSimulator` (exact, small n) as a
  `TrajectorySimulator` (stochastic, large n).
- **Step 38: `expectation_pauli` via monomial gather** — a Pauli string is monomial, so
  `tr(rho P) = sum_i phase(i) * rho[i, i XOR mask]` (mask = the X/Y bit pattern): a
  gather of `2**n` elements off the zero-copy `_host_view`, like `probabilities`
  already does for the diagonal. Replaces today's full `4**n x 8 B` readback plus a
  `vec.copy()` per term (~64 GiB of transients at n=16). Same idea gives
  `density_matrix()` an opt-in zero-copy view on Metal.
- **Step 39: CPU dense apply without tensordot copies** — the 3.0× is tensordot's full
  output plus a transposed copy. Stage (a): one `np.einsum(..., out=...)` whose output
  index order lands axes in original positions (removes the transpose pass and one
  temp, → ~2×). Stage (b): chunk the apply over non-target axes with bounded scratch
  (→ ~1× + chunk). Each stage is worth ~1 statevector qubit (≈ half a DM qubit) at
  fixed RAM.
- **Step 40: single-pass ket⊗bra superoperator for narrow gates (DM)** — apply
  `kron(U, conj(U))` on `[t, t+n]` in one pass instead of U then `conj(U)` (two full
  passes over the `4**n` state). Halves memory traffic and, on MLX, the live graph
  intermediates. Only for gates of ≤2-3 original qubits (the doubled width must stay
  within the dense-kernel sweet spots: Metal registers spill and the MLX custom kernel
  caps at k=6); wider fused gates keep the two-pass path. Interacts with the fusion
  width — A/B both knobs together.

Considered and rejected: **Hermitian half-storage** (a clean 2× — rho is determined by
its upper triangle — but it breaks the "reuse the statevector backends unchanged"
design; every kernel would need a triangular-layout variant) and **low-rank ensembles**
(`rho = sum_i p_i |psi_i><psi_i|`, memory `r * 2**n` — dominated by trajectories, which
get the same memory profile without eigenvalue-truncation machinery). The
**memory-mapped out-of-core backend** (v0.3, above) composes with all of these — notably
Metal's `newBufferWithBytesNoCopy:` accepts page-aligned mmap'd memory, so an
NVMe-backed density matrix at n≥17 may be feasible on the GPU path too.

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
