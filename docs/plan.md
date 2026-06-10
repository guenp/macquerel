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

## v0.2.x — GPU backend performance: competitive with Aer/Qulacs

**Goal:** make the MLX and Metal backends competitive with the other simulation frameworks
on **runtime**, as measured by the existing benchmark suite
(`benchmarks/bench_statevector.py` sweeps, `benchmarks/bench_versions.py` for
regressions). Every step below is decided by A/B benchmark, per the project's standing
practice — implement, measure, keep or revert.

### Baseline (benchmarks/data/large, 2026-06, M5 Max 128 GB)

Where we currently stand against the strongest rival at each size (times in ms,
`random` circuit unless noted):

| n | Aer | Qulacs | macquerel-mlx | macquerel-metal |
|---|---|---|---|---|
| 16 | 7.0 | 19.0 | 40.2 | 67.1 |
| 20 | 51.0 | 383.2 | 97.8 | 107.3 |
| 24 | 913.6 | 7696.6 | 1427.6 | 530.5 |
| 26 | 3886.5 | — | 6081.2 | 1926.3 |
| 28 | — | — | 39788.1 | 7868.0 |

Reading of the data:

1. **Aer beats MLX at every shared qubit count** on random/QFT (1.6–1.9× at 20–26q).
   QAOA/GHZ are closer but still losses at 24q+.
2. **MLX has a cliff at 26–28q** (random@28 = 39.8 s vs Metal 7.9 s). The bench harness
   already documents the cause: the lazy graph holds many full-width intermediates plus a
   pool cache, with observed peak ≥ ~16× the state size — the machine swaps.
3. **Metal wins everything ≥22q** (2.7–5× over MLX at 24–28q, beats Aer from 22q), yet
   `_select_backend` still routes 22–30q to MLX. The default `auto` path leaves most of
   our own performance on the table.
4. **Metal is dispatch-bound below ~20q**: every gate pays `commit` +
   `waitUntilCompleted` (random@6 = 22.5 ms for a 64-amplitude state). Qulacs/Aer own
   the small-n regime on dispatch overhead, not bandwidth.
5. **MLX dense gates move ~2× the bandwidth-bound ideal bytes**: `_dense_apply` does
   `tensordot` (pass 1) then `mx.transpose(...).reshape(-1)`, and the reshape of a
   permuted view materializes a second full-state copy (pass 2). Fused dense gates
   dominate random/QV circuits — exactly where Aer leads.

### Success criteria

- **G1:** MLX ≥ parity with Aer on random/QFT/QAOA at 18–26q (GHZ is close to
  parity by 22q; Aer still leads it at 18–20q).
- **G2:** Metal's "beats every rival" crossover moves from ≥22q down to ≤18q
  (`bench_statevector.py` annotates this crossover on the chart).
- **G3:** the 26–28q MLX cliff is gone (smooth bandwidth-bound scaling to its 30q cap).
- **G4:** `backend="auto"` always selects the measured-fastest backend tier.
- **G5:** no regression on `bench_versions.py` CI (CPU path untouched).

**Execution order (updated after review):** 21 → 22 → **24 → 23** → 25 → 26 → 27/28.
Step 24 lands before Step 23: it is the simpler change, it does not touch basis-order
semantics, and removing the 26–28q memory cliff first means Step 23's large-n A/B
measures the pure bandwidth win rather than a mix of bandwidth and swap relief.

### Step 21: Auto-select — route the 22q+ tier to Metal (quick win)

The data already says Metal wins ≥22q, by up to 5×. Change `_select_backend` tiers to
CPU ≤16 / MLX 17–21 / Metal ≥22, with the crossover as a named constant documented from
`benchmarks/data/large`. Re-measure the MLX/Metal crossover after Steps 23–24 (MLX fixes
may move it back up) and re-tune the constant at the end of this line.

*Expected:* default-path users get the 2.7–5× at 24–30q immediately. ~One-line change.

### Step 22: Metal — batch command-buffer encoding (remove per-gate sync)

`_dispatch` creates a command buffer, encodes one compute pass, commits, and
**waitUntilCompleted** per gate. Instead keep one open command buffer + encoder on the
backend; `apply_matrix` only encodes a dispatch. Metal's default hazard tracking
serializes dispatches that touch the same `MTLBuffer`, so ordering on the state buffer is
preserved without explicit barriers. Flush (commit + wait) only at observation
boundaries — `to_numpy` / `measure` / `sample` / `abs2sum` — and every N encoded gates
(N ≈ 256) to bound encoder growth. Sub-items:

- **Cache `_const` buffers** keyed by content bytes — fused circuits reuse identical
  matrices/index tables, and today every gate allocates fresh `MTLBuffer`s.
- **Hard requirement — constant-buffer lifetime:** today `_const` buffers are safe
  because `_dispatch` waits before returning; with deferred submission, every
  `MTLBuffer` referenced by an encoded-but-uncommitted dispatch must stay alive until
  its command buffer completes. The default `commandBuffer()` retains referenced
  resources (we must *not* switch to `commandBufferWithUnretainedReferences`), and the
  content cache holds Python references on top. A cache reset while dispatches are in
  flight must be covered by a dedicated test (shrink the cache cap so eviction happens
  mid-encoding).
- Reuse a small pool for the 4-byte scalar `setBytes` payloads (already cheap; verify).

*Expected:* the fixed ~ms/gate cost that makes Metal slowest below 20q collapses to
encode-only (~µs); targets G2 directly and also trims large-n times. This is the Metal
analogue of MLX optimization P1 (defer evaluation), which measured 1.1–1.8×there —
here the per-gate sync is far more expensive, so the small-n gain should be larger.

### Step 23: MLX — eliminate the per-dense-gate transpose copy

Track qubit→axis order in `MLXState` instead of restoring canonical order after every
gate. `tensordot` leaves the contracted axes in front; today we pay a full
transpose+copy to put them back (cause (5) in the v0.1 perf analysis — never fixed; the
einsum attempt P5 was a different approach and was rightly reverted). Plan:

- Add an `axis_order` field to `MLXState`; gates look targets up through the map and
  the post-`tensordot` permutation is *recorded*, not materialized.
- Diagonal/permutation paths translate target bit positions through the same map (their
  index math already takes arbitrary bit positions).
- Materialize canonical order once, at readback/measure boundaries.

*Expected:* dense/fused gates drop from 2 full-state passes to 1 — up to ~2× on
random/QV-style circuits at ≥20q, which is precisely the 1.6–1.9× gap to Aer (G1).

*Risk:* this is the **highest correctness-risk step** in the line. The
diagonal/permutation paths build linear indices assuming canonical bit positions
(`_gate_index`), and every readback/sampling path assumes canonical basis order, so the
axis map must be threaded through all of them. Before benchmarking it needs: mixed-kind
fuzz tests (dense + diagonal + permutation + controlled interleaved in one circuit),
partial-qubit `sample`/`abs2sum` against the CPU oracle, and readback-ordering tests
after non-trivial permutations. Scheduled *after* Step 24 (see execution order).

### Step 24: MLX — bound lazy-graph memory with periodic `mx.async_eval`

P1 removed per-gate `mx.eval` and never re-introduced any synchronization between
boundaries, so a depth-d circuit at 26–28q keeps O(d) full-width temporaries in flight
(observed peak ≥ ~16× state size → swap → the 28q cliff). Insert
`mx.async_eval(sv.data)` every N gates — async keeps the pipeline full while letting MLX
retire and free earlier temporaries. Tune N ∈ {8, 16, 32} and consider gating on state
size (only when `2**n * 8` bytes ≥ ~1 GiB so small-n graphs stay fully lazy). Evaluate
`mx.set_memory_limit` / `mx.clear_cache` interaction at 26–30q.

*Expected:* removes the 26–28q cliff (G3); random@28 should land near the
bandwidth-bound trend (~4–6 s, vs 39.8 s today), making MLX usable to its 30q cap.

### Step 25: Metal — kind-specialized kernels

Today permutation gates run through the dense group kernel (2^k×2^k complex MACs per
group) and `k` is a runtime loop bound. Three sub-items, each A/B'd independently:

- **Monomial kernel**: gather + per-row phase, one thread per amplitude — mirrors the
  MLX permutation fast path; pure index math, no MACs.
- **Function-constant specialization**: build pipeline variants for k = 1..4 with the
  loops unrolled at compile time (cached per k), instead of runtime `k`/`nc` loops over
  thread-local arrays.
- **Threadgroup sizing** from `threadExecutionWidth` / `maxTotalThreadsPerThreadgroup`
  instead of the hardcoded 256.

*Expected:* moderate; the dense kernel is bandwidth-bound at large n, so most of the gain
shows at small-to-mid n on permutation-heavy circuits (GHZ, brickwork CX layers).

### Step 26: Compiler — diagonal-run wide fusion (helps both backends)

Diagonal × diagonal = diagonal: composition is O(2^k) elementwise (not an O(8^k) matrix
product) and *apply* cost is one elementwise pass regardless of k. So maximal runs of
diagonal gates (Rz, Z, S, T, P, CZ, CP and diagonal fused blocks) can fuse far wider
than the dense width-4 limit — cap at ~12 qubits (4096-entry table, trivially
constant-memory). Requires kind-aware grouping in `fuse_gates` so a diagonal run is not
absorbed into a dense group (which destroys the fast path for both backends).

*Expected:* QFT is a CP cascade — its diagonal layers collapse to ~one elementwise pass
per H column at every size; QAOA cost layers (CX·Rz·CX = diagonal ZZ block) compose
into wider diagonals too. Biggest algorithmic (vs constant-factor) win available; QFT
is currently our worst circuit vs Aer at 24–26q.

### Step 27: Compiler — commutation-aware fusion grouping

Greedy in-order fusion unions qubit sets across *unrelated* gates and flushes early: on a
brickwork random circuit, a 1q rotation on q5 between two gates on {0,1} forces {0,1,5}
into one group. Gates on disjoint qubits commute — schedule fusion per qubit
neighborhood (the standard Aer/Qulacs approach) so groups are local and reach full
width. Verified, as today, by fused-vs-unfused statevector equivalence tests.

*Expected:* fewer, denser fused gates → fewer full-state passes on random/QV circuits;
multiplies the value of Steps 22–23 since both backends' per-pass costs drop.

### Step 28: Wire `remap_qubits` into the hot path (benchmark-gated)

Step 13 implemented the Doi–Horii remap pass but the `Simulator` never calls it, and
its current signature is insufficient for the hot path: it returns only the remapped
circuit (it rewrites `MeasureOp` qubit labels in place), not the permutation. Spec for
wiring it in:

- `remap_qubits` (or a wrapper) must return **both** the remapped circuit and the
  applied permutation so the simulator can invert it at the boundary.
- `statevector()` must return amplitudes in the **caller's canonical basis order**
  (apply the inverse permutation at readback; with Step 23's axis-order tracking this
  folds into the existing readback materialization for free).
- `run()`/`sample()` must report counts keyed by the **caller's logical qubit order**,
  not the remapped physical order (bit-relabel before counting).

Keep only if A/B wins.

### Step 29 (stretch): MLX — custom group-based dense kernel

If Step 23 doesn't close the Aer gap: revisit Step 10's `mx.fast.metal_kernel` dense
path with the complex64 layout (state viewed as `float2`), one thread per 2^k-amplitude
group, double-buffered (MLX kernel inputs are `const device`). One read + one write per
amplitude with zero transpose, at the cost of maintaining a hand-written kernel inside
the MLX graph.

### Non-goals for this line

- `sample`/`expectation_pauli` GPU round-trips (not on the benchmark's timed path).
- CPU backend tuning (it is the readability/correctness oracle).
- complex128 GPU support; the suite's `--double` flag already covers the comparison.

### Measurement protocol

For each step: `uv run pytest tests/ -x -q` green first (differential tests are the
correctness gate), then A/B with
`benchmarks/bench_statevector.py --circuits qft random qaoa ghz` over the affected qubit
range, ≥5 reps at noisy sizes (the suite's min-of-reps + subprocess isolation handles
thermal/pool noise), and a `bench_versions.py` run against the latest release before
merging. Update `benchmarks/data/large/` and the auto-select tiers once at the end of
the line, then move the shipped steps to `plan_completed.md` with their measured A/B
results.

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
