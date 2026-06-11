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
  `np.memmap`, for single large runs past DRAM capacity. Investigated and expanded
  into the explicit spill-to-disk design under [v0.4](#v04-spill-to-disk-statevectors-duckdb-style-out-of-core-execution)
  below — passive OS paging turned out to be the wrong mechanism.
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

## v0.4 — Spill-to-disk statevectors (DuckDB-style out-of-core execution)

[DuckDB](https://duckdb.org/2024/07/09/memory-management.html) runs analytical queries
on larger-than-memory data by giving every operator a fixed memory budget
(`memory_limit`, default 80% of RAM) and a buffer manager: intermediates live in
buffer-managed blocks that are spilled to a temporary directory when the budget is
exceeded and read back in large sequential chunks, so sort/join/aggregate
[degrade gracefully](https://duckdb.org/2021/08/27/external-sorting.html) to NVMe
bandwidth instead of failing or swapping. This section investigates whether that
model transfers to statevector simulation and turns the v0.3 "memory-mapped
out-of-core backend" bullet into a concrete design.

### What transfers and what doesn't

The workloads differ in one fundamental way: a query touches data *selectively* (cold
blocks can sit on disk untouched), while a gate touches **every** amplitude — there is
no cold data during a full-state pass, so a generic LRU buffer pool buys nothing.
Naively backing the state with `np.memmap` and letting the OS page (the v0.3 sketch)
inherits exactly this problem: the kernel evicts 16 KiB pages reactively, with no
knowledge of the access plan, and a 34-qubit gate pass becomes random-fault-ordered
I/O. What *does* transfer from DuckDB is everything around the buffer pool:

1. **An explicit budget instead of passive paging.** A `memory_limit`-style knob
   (e.g. `MACQUEREL_MEMORY_LIMIT`, defaulting to today's min(0.45 × RAM, 64 GiB)
   benchmark gate) and an explicit chunk manager that owns the I/O schedule.
2. **Sequential, partitioned access.** DuckDB radix-partitions a spilling hash join so
   each partition is processed *fully* while resident. The statevector analog is
   index-bit partitioning — the same scheme already planned for multi-Mac: split the
   state into 2^m chunks by the top m index bits. Gates on the low n−m **local**
   qubits act independently inside each chunk; only gates on the top m **global**
   bits pair amplitudes across chunks. Unlike DuckDB, the access pattern is *known at
   compile time* from the gate's target qubits — no adaptivity needed.
3. **Maximize work per residency.** The unit of I/O is a "chunk episode": load a
   chunk (sequential multi-GiB read), apply *every* gate that can run before the next
   global-bit barrier (the existing fusion + commutation-aware grouping machinery,
   re-targeted), write it back, prefetch the next chunk meanwhile. This is Doi–Horii
   cache blocking lifted one level — DRAM as the cache, NVMe as main memory — and it
   is the entire performance story: one episode costs a full-state read+write
   regardless of how many gates it carries.
4. **Graceful degradation.** Budget ≥ state size → one chunk, today's in-RAM Metal
   path, zero overhead. Budget < state size → the same backend streams chunks.
   Metal composes directly: `newBufferWithBytesNoCopy:` accepts the page-aligned
   chunk buffer, so the GPU computes on the resident chunk while the I/O thread
   prefetches the next — DuckDB's pin/unpin, with exactly three pins live
   (compute / prefetch / writeback).
5. **Clean pages skip writeback.** DuckDB drops unmodified cached pages instead of
   spilling them. The analog: read-only passes (sampling, probabilities,
   expectation values) stream chunks without writing them back — half the I/O.

### Feasibility numbers (this machine: 128 GiB RAM, 2 TB NVMe, ~6–8 GB/s sequential)

| n | state (2ⁿ × 8 B) | fits | episode cost (read+write @ 7 GB/s) |
|---|---|---|---|
| 33 | 64 GiB | RAM (today's ceiling) | — (in-memory) |
| 34 | 128 GiB | NVMe | ~37 s |
| 35 | 256 GiB | NVMe | ~75 s |
| 36 | 512 GiB | NVMe | ~150 s |
| 37 | 1 TiB | NVMe (~1.6 TB free) | ~5 min |

So spill-to-disk buys **+1 to +4 qubits** of capacity at a slowdown set by the
local/global structure of the circuit: GHZ at 34q is a handful of episodes (~minutes);
a QFT, whose controlled-phase gates are diagonal (diagonal gates never touch a global
bit's *pairing*, only its phase — they are chunk-local at any width), spills cheaply;
a random dense circuit on scattered qubits is the worst case and needs the remap step
below to stay off the global bits. Note the asymmetry that makes scheduling decisive:
local work runs at unified-memory bandwidth, global work at NVMe bandwidth —
a ~50–100× gap, far steeper than the cache-blocking gap that lost remapping its
single-machine A/B.

### Steps

- **Step 41: chunked state store + budget knob** — a `ChunkedState` owning one
  page-aligned file (`MACQUEREL_TEMP_DIR`, default alongside the platform temp dir;
  checked against free disk up front, deleted on exit — DuckDB's `temp_directory` /
  `max_temp_directory_size` semantics), a `MACQUEREL_MEMORY_LIMIT` budget, and a
  three-buffer streaming pipeline (compute / prefetch / writeback) feeding the Metal
  backend via `newBufferWithBytesNoCopy:`. Correctness gate: identical amplitudes vs
  the in-RAM path at small n with an artificially tiny budget (the budget knob makes
  out-of-core testable at 20q in CI). Sub-step: optional **lz4 chunk compression on
  the writeback path** with a raw fallback when the ratio is poor (DuckDB stores
  spilled blocks the same way) — multithreaded lz4 decompresses at tens of GB/s,
  comfortably above the NVMe, so it is near-free capacity on structured states
  (a GHZ state is two nonzeros) and harmless on Haar-random ones, which are
  incompressible.
- **Step 42: episode scheduler** — compile the fused gate stream into chunk-local
  episodes split at global-bit barriers; route diagonal/phase gates on global bits
  into episodes (they need no cross-chunk pairing); skip writeback for read-only
  passes. A/B metric: episodes per circuit (= full-state I/O passes), then wall time.
- **Step 43: global bits via remap passes** — when a dense gate needs a global bit,
  emit one explicit remap episode (swap a global bit with a cold local bit — a
  paired-chunk streaming pass) instead of pairing chunks ad hoc, then proceed
  chunk-locally. Reuses the existing disabled-by-default `remap_qubits` machinery,
  which finally has the cost asymmetry it was built for. A/B: remap-pass count and
  wall time on the random-circuit benchmark at 34q.
- **Step 44: out-of-core benchmark + ship gate** — extend `bench_memory.py` with a
  spilled series (peak RAM must respect the budget; report episodes and effective
  GB/s against the NVMe roofline) and a 34q GHZ/QFT/random wall-time table. Ship only
  if 34q runs end-to-end under the default budget without swap and the in-RAM path
  shows zero regression. SSD-endurance note in the docs: an episode writes the full
  state (128 GiB at 34q), so deep unfused circuits are a TBW consideration —
  documented, not engineered around, in v0.4.
- **Step 45: block-float `complex32` states** — opt-in half-precision amplitude
  storage (`dtype="complex32"`): two float16s per amplitude plus one float32
  max-magnitude scale per chunk. The per-chunk scale is load-bearing, not a nicety:
  at 34q the typical amplitude magnitude is ~2⁻¹⁷ ≈ 8e-6, *below* float16's normal
  range (6e-5), so naive fp16 lands in subnormals or underflows — block floating
  point restores the range. A deterministic, circuit-independent 2× on every byte
  count, and GPU-native (Metal and MLX both have first-class `half`), so chunks stay
  compressed in unified memory too: it composes with spilling *and* raises the
  in-RAM ceiling 33q → 34q (and the DM ceiling toward n=17) with no disk involved.
  Cost: a 10-bit mantissa gives ~2⁻¹¹ relative rounding per gate pass, accumulating
  roughly with √depth — fine for sampling and expectation values at moderate depth,
  not for amplitude-level verification. Ship opt-in only; gate with the existing
  differential tests against complex64 and publish a fidelity-vs-depth table.

Considered and deferred: **error-bounded lossy compression (SZ/ZFP)** — the
strongest published precedent for compressed full-state simulation (Wu et al.,
*Full-state quantum circuit simulation by using data compression*, SC'19,
[arXiv:1810.14582](https://arxiv.org/abs/1810.14582), which kept the state
compressed in memory, decompressing blocks on the fly, and reached a 61-qubit
Grover), but a poor fit here for now: SZ/ZFP throughput is ~1-5 GB/s per CPU core
(below the NVMe without multithreading or a Metal port — ZFP has CUDA kernels, no
Metal), the error bound is consumed once per compress/decompress cycle and every
chunk episode is a cycle, and the ratio evaporates on Haar-random states — exactly
the circuits that need spilling most. Also deferred: **changed representations**
(matrix product states, decision diagrams, sparse amplitude dictionaries) — these
compress *structure* rather than bytes (MPS memory scales with bond dimension, not
2ⁿ) and are the biggest capacity lever of all, but they are a different simulator,
deserving their own roadmap line rather than a codec bullet under spill-to-disk;
and a **general buffer pool with LRU eviction** (no selectivity in the access
pattern — the streaming three-buffer pipeline is the whole pool). The multi-Mac
item (v0.3) and this section share the index-bit partitioning compiler work;
whichever ships first pays most of the cost of the second.

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
