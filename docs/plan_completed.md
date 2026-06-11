# macquerel Implementation Plan — Completed Work

This document is the historical record of implementation steps that have **shipped**.
It is kept separate from [`plan.md`](plan.md), which tracks only remaining work. Steps
retain their original numbering. For each step the rationale, design notes, and (for the
performance work) the measured A/B results are preserved, because they document *why* the
code is shaped the way it is.

---

## v0.1 — core simulator

macquerel is a quantum state-vector simulator targeting Apple Silicon's unified-memory
architecture. The v0.1 milestone delivered the CPU reference backend + MLX backend,
single/two-qubit gates, gate fusion, measurement/sampling, and a differential test
harness, following the spec's layered architecture: gate library → CPU reference backend →
frontend API → MLX backend → compiler → tests.

### Step 1: Fix packaging and project structure

The original `CMakeLists.txt` wrongly treated `simulator.py` as a C++ library and
`__init__.py` was empty. Fixed: removed the broken `src/` subdirectory add (no C++ in
v0.1), exported `Circuit`, `Simulator`, `Gate` from `__init__.py`, and created the
`src/macquerel/backends/` package.

**Verified:** `python -c "import macquerel"` succeeds; `uv sync` completes cleanly.

### Step 2: Gate matrix library (`src/macquerel/gates.py`)

All standard unitary matrices as `np.ndarray` (complex64), plus helpers.

- Single-qubit: `I`, `H`, `X`, `Y`, `Z`, `S`, `T`, `Rx(θ)`, `Ry(θ)`, `Rz(θ)`, `P(λ)`
- Two-qubit: `CNOT`, `CZ`, `SWAP`, `CP(λ)`
- Helper: `controlled(U)` — lifts a single-qubit gate to a 2-qubit controlled gate
- `classify(matrix) -> GateKind` where `GateKind = Literal["diagonal", "permutation", "dense"]`

**Verified (tests/test_gates.py):** `H @ H ≈ I`, `X @ X ≈ I`, `S @ S ≈ Z`,
`Rz(θ) @ Rz(-θ) ≈ I`, unitarity for all gates, and correct `classify()` kind for each gate.

### Step 3: CPU reference backend (`src/macquerel/backends/cpu.py`)

The oracle against which all other backends are differentially tested. State is an
`np.ndarray` of shape `(2**n,)`, dtype `complex64`, reshaped to `(2,)*n` for gate
application. `apply_matrix` uses `np.tensordot` + transpose back to canonical order;
`measure`, `sample`, and `allocate` implemented per spec.

**Verified (tests/test_cpu_backend.py):** Bell, GHZ, QFT-vs-`np.fft`, norm preservation
after every gate, measurement collapse, and GHZ sampling χ² test.

### Step 4: Frontend API (`src/macquerel/circuit.py`, `src/macquerel/simulator.py`)

`Gate` dataclass (`name`, `matrix`, `targets`, `controls`, `kind`); `Circuit` with gate
builder methods (`h`, `x`, `y`, `z`, `s`, `t`, `rx`, `ry`, `rz`, `cx`, `cz`, `swap`, `cp`),
`measure`/`measure_all`, and index validation; `Simulator` with `run(circuit, shots)` →
`Counter[str]`, `statevector(circuit)`, and backend selection.

**Verified (tests/test_circuit.py):** API round-trip, index validation raises
`ValueError`, `measure_all` on a 3-qubit circuit.

### Step 5: MLX backend (`src/macquerel/backends/mlx_backend.py`)

GPU backend with diagonal/permutation fast paths and a Metal-kernel dense path (see the
perf section for the layout that eventually shipped). `measure`/`sample` use
`mx.random.categorical`. Raises a helpful `ImportError` when `mlx` is absent.

**Verified (tests/test_mlx_backend.py):** differential tests against the CPU oracle to
`1e-5` on all amplitudes — the primary correctness gate.

### Step 6: Gate fusion compiler pass (`src/macquerel/compiler.py`)

`fuse_gates(circuit, max_fused_qubits=4) -> Circuit`: greedily merges adjacent gates whose
combined target set is ≤ `max_fused_qubits`, composing matrices into a single unitary.
`MeasureOp` is a fusion barrier; controlled qubits count toward the set size.

**Verified (tests/test_compiler.py):** fused == unfused statevectors, single-gate circuits
unchanged, measurement barriers respected, fused-matrix unitarity.

### Step 7: Test suite

`tests/test_known_circuits.py` (Bell, GHZ, Grover, QFT on CPU + MLX),
`tests/test_properties.py` (norm preservation + fusion equivalence over random circuits),
`tests/test_edge_cases.py` (1-qubit, all-identity, empty circuit, max-controls).

---

## v0.1 completion — performance-critical gaps

These items were described in the v0.1 spec but initially deferred to a simpler correct
implementation.

### Step 8: Permutation gate fast path (`mlx_backend.py`)

Permutation gates (X, SWAP, CNOT) dispatch to a dedicated `_apply_permutation`
(gather/scatter, no arithmetic) instead of falling through to the general dense path.

### Step 9: SoA state representation (`mlx_backend.py`)

The state stays in struct-of-arrays form across gate calls (later superseded — see P4),
with the complex64 view reconstructed only at the API boundary.

### Step 10: `mx.fast.metal_kernel` gate hot path (`mlx_backend.py`)

Custom Metal kernel where each GPU thread owns one `(i, i ⊕ 2ᵏ)` amplitude pair, with
double-buffering (MLX kernel inputs are `const device`).

### Step 11: Reproducible RNG (`mlx_backend.py`, `simulator.py`)

`Simulator.__init__` takes an optional `seed`, threaded through to the backends'
sampling. (`mx.random.key(seed)` for MLX.)

### Step 12: `Backend` Protocol ABC (`src/macquerel/backends/__init__.py`)

A `typing.Protocol` (`@runtime_checkable`) declaring the full backend interface
(`allocate`, `apply_matrix`, `measure`, `sample`, `to_numpy`, `expectation_pauli`,
`abs2sum`) so type checkers can verify backend conformance.

---

## MLX backend performance optimization

After the permutation fast path (Step 8), SoA state (Step 9), and the custom Metal kernel
(Step 10), the MLX backend was still slower than CPU across 16–24 qubits. Profiling a
depth-50 random circuit at 22 qubits identified these root causes:

1. **Per-gate `mx.eval()`** forced a GPU sync on every gate, defeating MLX's lazy graph
   (50 elementwise ops: 17.4 ms with per-gate eval vs 6.0 ms once — a 3× penalty).
2. **Host-side O(2ⁿ) work per gate** in `_apply_permutation` (NumPy table rebuild ~9 ms/gate
   + ~0.2 ms host→device copy).
3. **Diagonal path rebuilt device scratch** (`arange`, `gate_idx`, bitwise intermediates)
   every call.
4. **SoA real/imag split** turned each complex multiply into 3–4 real kernels.
5. **Full transpose+copy per dense gate** in `_dense_apply`.
6. **The old raw backend benchmark did not fuse** (deliberate - it measured per-gate throughput).
7. **Minor per-gate host overhead** (`classify`, `astype`).

**Optimizations (ranked, as shipped):**

- **P1 — Defer evaluation. ✅ DONE (`3c74da8`).** Dropped per-gate `mx.eval` from all five
  gate paths; eval forced only at segment boundaries (measure / `to_numpy` / sample).
  Targets cause (1). Measured 1.1–1.8×.
- **P2 — Eliminate host-side permutation work. ✅ DONE (`81e96e3`).** Compute the gather
  index on-device with `mx.arange` + bitwise ops; only the tiny `2**k` inverse lookup stays
  on host. Targets cause (2); removed ~9 ms/gate at 22q. **This flipped MLX from slower to
  faster than CPU.**
- **P3 — Cache per-gate device constants. ✅ DONE (`6178f2f`).** Caches the reused
  `arange(2ⁿ)` index, the scalar mask, and `classify()` memoized by matrix bytes. Targets
  causes (3) and (7). Deliberately does **not** cache full `2ⁿ` per-target index tables
  (unbounded memory; on-device rebuild is cheap after P2).
- **P4 — Native `complex64` storage. ✅ DONE (`1eb61db`).** State is now a single complex64
  array: one complex tensordot instead of four real ones, one complex gather for
  diagonal/permutation, and the SoA-only Metal 1q kernel dropped. A/B vs SoA: neutral on
  the raw backend benchmark (1.01-1.04× at 18-22q), faster on dense/fused circuits at 20q:
  dense 1.16×, QFT 1.08×. **This refutes the spec's "SoA up to 6.9× over interleaved"
  claim on MLX 0.31** — the layouts are equivalent here, and complex64 is simpler. Targets
  cause (4).

  > **Note — why complex64 can *look* slower at large n with low reps.** At 20–22q the
  > random circuit is memory-bandwidth-bound, and a complex64 amplitude (8 bytes) equals two
  > float32 (SoA `real`+`imag`), so both layouts move identical bytes per gate — theory
  > predicts parity. The genuine difference is below the benchmark's noise floor:
  > GPU clock/thermal variance swings a single 22q config by ~10–15% (44.8 ms vs 51.4 ms
  > observed). With `--reps 3` the min can land on a lucky-fast SoA run, making complex64
  > look ~0.85×; another pair reverses it. Higher reps (9+) and interleaved measurement
  > collapse the gap to parity. complex64's real win is on dense/fused circuits.

- **P5 — Avoid the full transpose copy via einsum. ❌ TRIED & REVERTED (`b4b0171` →
  `50d3d54`).** A single canonical-order `einsum` was **slower** on every dense/fused
  workload at 18q (QFT 5.3→6.2, QAOA 2.8→4.2, dense 21.3→26.2 ms): MLX's `einsum`
  decomposes to a costlier sequence than `tensordot`+`transpose`. Premise false on MLX 0.31.
- **P6 — Fuse before MLX dispatch. ✅ ALREADY SATISFIED.** The simulator already runs the
  fusion pass on the hot path (`statevector()` / `run()`). The historical raw backend
  benchmark deliberately stayed unfused. No code change needed.
- **P7 — Re-tune the auto-select crossover. ✅ DONE (`351376c`).** `_select_backend` routes
  ≤16 qubits to CPU (was ≤14) and MLX for 17–31, matching the measured crossover.
- **P8 — `mx.compile` the hot gate kernels. ✅ DONE (`b6bef6f`).** Compiled the diagonal
  phase kernel and permutation gather. A/B (reps=5): 20q 1.09×, 22q 1.12×, with a ~0.88×
  compile-overhead regression at 18q. Kept for the large-n gain.

**Net result:** P1 (defer eval) and P2 (kill host-side perm work) together flipped MLX from
slower to faster than CPU. P3, P7, P8, and P4 added incremental gains and simplification,
leaving MLX **2.4–5.6× faster at 18–22 qubits** (see `benchmarks/data/` and
`benchmark-2.png`). P5 was tried and reverted; P6 needed no change. Every perf-sensitive
step was decided by A/B benchmark on the harness rather than implemented blind.

---

## Metal backend, qubit remapping, expectation values

### Step 13: Qubit remapping / cache-blocking compiler pass (`compiler.py`)

`remap_qubits(circuit) -> Circuit`: a second pass after gate fusion (Doi–Horii technique)
that relabels qubits so the most frequently targeted ones land on the lowest indices
(minimizing stride `2ᵏ`). **Verified (tests/test_compiler.py):** remapped circuits produce
measurement distributions identical to the unremapped originals.

### Step 14: `expectation_pauli` and `abs2sum` (`cpu.py`, `mlx_backend.py`)

`abs2sum(sv, qubits)` — marginal probability sum over the given qubits, exposed as a
first-class method. `expectation_pauli(sv, pauli_strings)` — expectation value of a Pauli
operator or sum of Pauli strings. Verified against analytic values in
`tests/test_backend_protocol.py` (⟨Z⟩ = 1 for |0⟩, ⟨X⟩ = 1 for |+⟩).

### Step 15: `MetalBackend` — native backend for the >30-qubit regime

> **Status: IMPLEMENTED (2026-06-02). Gate 0 = GO (outcome 1).** Released MLX caps at
> **≤30 qubits**; 31q+ is reachable only by a native backend. Built as a **PyObjC driver**
> (`src/macquerel/backends/metal_backend.py`), **not** the metal-cpp + nanobind +
> scikit-build-core extension originally specified — that offline `.metal` → `.metallib`
> path is unbuildable on the target machine (Metal Toolchain CLI missing; Xcode downloader
> broken). PyObjC reaches the same goals with no build system: shaders compile at runtime
> via `newLibraryWithSource`, amplitudes live in one unified-memory `MTLBuffer` with 64-bit
> indexing, updates are in-place. **Verified: runs 31q, 32q, and 33q** — the realistic
> 64 GiB ceiling, using 64.06 GiB resident (analytic GHZ spot-check exact), plus
> differential-tested vs CPU to 1e-5, wired into auto-select (Metal ≥31q).
> **Surprise result:** Metal also *beats* MLX from ~22q up (≈13× at 30q) because it avoids
> MLX's double-buffering + gather temporaries under memory pressure — see `benchmark-3.png`.

#### Gate 0 — ceiling re-validation (RESULTS, 2026-06-02, M-series, 128 GB unified memory)

`benchmarks/probe_mlx_ceiling.py` exercised allocate / scalar-multiply / `arange` / gather
/ readback on a `2ⁿ` complex64 state, each `n` in an isolated subprocess:

| n | elements | complex64 size | result |
|---|---|---|---|
| 29 | 5.37e8 | 4 GiB | ✅ all ops OK |
| 30 | 1.07e9 | 8 GiB | ✅ all ops OK |
| 31 | 2.15e9 | 16 GiB | ❌ **rejected before allocation** |

**The break is a hard type ceiling, not a memory limit.** At n=31 (`2³¹` elements) MLX
0.31.2 raises `TypeError` because the shape value `2³¹` does not fit MLX's **`int32`
`ShapeElem`** — the binding refuses the call before any allocation. This is the
`ShapeElem = int32_t` root cause of MLX issue **#3327**, and it would reject 31q even with
infinite RAM. The newest MLX on PyPI is **0.31.2**; building from `main` does not help (PR
**#3524** keeps `ShapeElem = int32_t` and only adds overflow *detection* — verified by
reading the PR). So **no released or development MLX reaches 31q**; the ceiling is a
deliberate upstream design choice.

**Verdict: GO — outcome (1), the only outcome on offer.** A native backend is the only way
to reach 31q+: 31q (16 GiB), 32q (32 GiB), 33q (64 GiB) are all MLX-unreachable. In-place
updates add a **+1-qubit capacity edge** over MLX's double-buffering.

#### The capacity argument (why in-place matters)

MLX custom kernels cannot write back into their inputs (`const device`, issue **#2547**),
so every gate **double-buffers** — costing one qubit of headroom. In **GiB** (`2ⁿ × 8`
bytes; this 128 GB Mac reports ≈ 128 GiB):

| n | 1 state | MLX (double-buffered) | Metal (in-place) |
|---|---|---|---|
| 31 | 16 GiB | 32 GiB ✅ | 16 GiB ✅ |
| 32 | 32 GiB | 64 GiB ✅ | 32 GiB ✅ |
| 33 | 64 GiB | 128 GiB ⚠️ byte-fit only | 64 GiB ✅ **realistic max** |
| 34 | 128 GiB | 256 GiB ❌ impossible | 128 GiB ⚠️ byte-fit only |

**Realistic max: 33 qubits** in-place (64 GiB state leaves 64 GiB for everything else).
34q is byte-fit only. Concrete payoff: **33q vs MLX's 32q** (+1 usable qubit), plus the
analogous gain on a 64 GB machine.

#### As-built architecture

A pure-Python PyObjC driver behind the existing `Backend` Protocol. State held as **one
interleaved complex64 `MTLBuffer`** (bit-identical to NumPy for zero-copy readback, fits
33q at 64 GiB < the 80.6 GiB `maxBufferLength` measured on this M5 Max). Two
runtime-compiled kernels: a per-amplitude `diagonal` fast path and a per-group `dense`
kernel covering dense/permutation/controlled. A **3D grid** reconstructs a `uint64` linear
index from `thread_position_in_grid` (a 1D grid can't address >2³² threads) — the actual
mechanism that lifts the ceiling. In-place is race-free: each kernel writes each output
element exactly once and reads only its own group.

#### Implementation phases (as built)

1. ✅ Gate 0 — ceiling re-validation. GO.
2. PyObjC needs no build step; the `metal` extra (`pyobjc-framework-Metal`) was added to
   `pyproject.toml`/`uv.lock`. (The original CMake/nanobind/metallib skeleton was obsolete.)
3. ✅ Allocate + zero-copy `to_numpy` readback. Differential-tested vs CPU.
4. ✅ Diagonal kernel → differential test vs CPU (exact).
5. ✅ Permutation → handled by the general dense kernel, differential-tested, in-place.
6. ✅ Dense group kernel (1–4q fused) + controlled (control-bit mask) → differential
   tested, including non-adjacent targets and a 12-seed random-circuit fuzz.
7. ✅ Wired into `_select_backend` (Step 16): CPU ≤16q, MLX 17–30q, Metal ≥31q.
8. ✅ Large-n validation: 31q and 32q GHZ spot-check on a subset of amplitudes matches the
   analytic prediction exactly; in-place confirmed (single 1× buffer per state).

#### Testing (`tests/test_metal_backend.py`)

`pytest.importorskip` to keep CI green without the extension. Differential vs CPU to 1e-5
on the fuzzed random-circuit corpus (the most important test), boundary tests (31→32 switch,
all-identity, max-controls, empty), and an env-gated (`MACQUEREL_BIG_TESTS=1`) large-n test.

#### Outcome vs pre-build expectations

The pre-build expectation was that Metal would only *match* MLX's ms/gate (both
bandwidth-bound). In practice Metal **beats** MLX from ~22q up (≈13× at 30q,
`benchmark-3.png`): MLX's per-gate double-buffering and full-width gather temporaries thrash
unified memory at 28–30q, while Metal's in-place single-buffer path scales as the
bandwidth-bound ideal. Metal is slower *below* ~20q (per-gate `waitUntilCompleted` sync vs
MLX's lazy-graph fusion), so auto-select keeps CPU ≤16q and MLX 17–30q. Success criteria —
running 31–33q at all, and half the memory of MLX — both met.

### Step 16: Automatic backend selection (`simulator.py`)

`backend='auto'` (the default): `CPUBackend` for ≤16 qubits, `MLXBackend` for 17–30,
`MetalBackend` for ≥31. Implemented as `_select_backend`. MLX's tier ends at 30q because
its `int32` `ShapeElem` rejects `2**31` amplitudes.

---

## Test additions (`tests/`)

- ✅ `test_compiler.py` — remapping equivalence (remap on/off yields identical
  distributions).
- ✅ `test_metal_backend.py` — differential CPU vs Metal up to 32+ qubits.
- ✅ `test_backend_protocol.py` — Protocol conformance for CPU/MLX/Metal; `expectation_pauli`
  and `abs2sum` against analytic values.
- ✅ `test_simulator.py` — seed reproducibility (two seeded runs return identical counts).

> Remaining test gap: Quantum Volume circuits and random-circuit-sampling spot checks in
> `test_known_circuits.py` are still outstanding — tracked in [`plan.md`](plan.md).

---

## Cirq/Qiskit adapters

### Step 17: Cirq/Qiskit front-end adapters (`src/macquerel/adapters/`)

`macquerel.from_cirq(circuit)` and `macquerel.from_qiskit(circuit)` converters (exported
conditionally from `__init__.py` when the optional extras are installed), so existing
circuits run unmodified on macquerel backends. Tested in `tests/test_adapters.py`.

---

## Benchmarking suite

### Step 18: Benchmarking suite (`benchmarks/`)

The maintained suite was consolidated around the benchmarks that are still used:

- ✅ **Framework comparison** (`bench_statevector.py`): macquerel CPU/MLX/Metal vs
  Qiskit Aer and Qulacs, with endianness/precision/convention handling.
- ✅ **Fusion-width sweep** (`bench_fusion_width.py`): QFT, random, QAOA, and Quantum
  Volume circuits swept over `max_fused_qubits ∈ {1..6}`.
- ✅ **Released-version regression checks** (`bench_versions.py`, `plot_versions.py`):
  compare CPU/MLX/Metal across PyPI releases and the current checkout.

> Quantum Volume benchmark coverage shipped in the **v0.2** line - see below.

---

## v0.2 — autotuning + benchmark completion

### Step 18 (completed): Benchmarking suite gaps

Closed the two outstanding §9 items from the v0.1 benchmarking suite:

- ✅ **Quantum Volume macrobenchmark** (`bench_fusion_width.py`): Haar-random SU(4) model
  circuit (depth = n), added to the swept circuit set alongside QFT/random/QAOA/dense.
  Exercises the worst-case dense (non-diagonal, non-permutation) path.
- ✅ **Framework comparison cleanup** (`bench_statevector.py`): macquerel backends run
  alongside Qiskit Aer and Qulacs, and optional frameworks degrade gracefully when absent.
- ✅ **Companion tests** (`tests/test_known_circuits.py`): QV normalization + exact-inverse
  identity known-answer tests, and a random-circuit-sampling spot check that shot
  frequencies track |ψ|².

### Step 19: Shot batch-size autotuning (`src/macquerel/simulator.py`)

`Simulator` gained a `batch_shots` parameter (default `"auto"`), threaded through to each
backend's `sample()` (and added to the `Backend` Protocol). The MLX backend autotunes the
`mx.random.categorical` batch size by doubling from a 1024-shot base until throughput
plateaus (Tsim heuristic), memoized per category count (`2**len(qubits)`). An explicit int
pins the batch and draws in chunks with deterministic per-chunk subkeys; a seeded `"auto"`
run draws in a single deterministic pass so results stay reproducible. CPU/Metal accept the
kwarg for interface parity (host NumPy sampling has nothing to tune).

### Step 20: Fusion-width default + opt-in per-chip autotuning (`src/macquerel/compiler.py`)

`fuse_gates(max_fused_qubits=None)` resolves to a **fixed width of 4** — the zero-config
default, with no measurement on the hot path. Autotuning is **opt-in** via
`MACQUEREL_FUSION_WIDTH`: an int pins the width; `auto` runs `autotune_fusion_width()`, which
measures the optimal width on the local chip once and caches it
(`~/.cache/macquerel/fusion_width.json` + in-memory). Measurement failures fall back to 4 and
never raise.

**Why 4, and the ideal width per qubit count.** The optimal width *drifts with qubit count*,
because fusion trades a one-time matrix-composition cost against the per-apply cost of a full
pass over the `2ⁿ` state. Benchmarked on an M5 Max (fuse+apply, MLX backend):

| qubits n | ideal `max_fused_qubits` | regime |
|---|---|---|
| ≤ ~16 | 1–2 (immaterial) | apply is sub-ms; composition overhead dominates |
| ~20 | 3 | apply starting to dominate |
| ~22 | 4–5 | |
| ~24 | 5–6 | apply-bound — wider fusion = fewer passes |
| **17–30q aggregate** | **4** | normalized winner across the measured regime |

No single width is optimal everywhere, but **4 wins on normalized aggregate** across the
measured 17–30q MLX tier, so it is the default. This was settled by re-running the benchmarks:
a first cut autotuned on a single QFT at n=18 and picked width 2, which **regressed the
large-n path by up to ~2×** (22q QFT: 617ms→1376ms on CPU; 131→222ms on MLX) while only
shaving negligible ms off sub-16q circuits. The opt-in autotuner now measures fuse+apply
across a *span* straddling the regime (MLX 20q/22q, else CPU 14q/16q), normalizes each qubit
count by its own fastest width, and picks the lowest aggregate (ties within 2% break toward
4); on this chip it confirms 4. Full benchmark write-up:
<https://github.com/guenp/macquerel/pull/8#issuecomment-4636543327>.

---

## v0.2.x — GPU backend performance (Steps 21-30)

> **Status: SHIPPED (2026-06-10), branch `gpu-perf-plan`.** Goal: make the MLX and Metal
> backends competitive with Qiskit Aer / Qulacs on runtime. Every step was A/B-benchmarked
> with `benchmarks/run_step_bench.sh` (subprocess-isolated, min-of-3, per-step JSONs named
> `<step>-<commit>-<backend>.json`); full data, charts, and per-step justifications in
> `benchmarks/data/steps/README.md`. Execution order was 21 → 22 → 24 → 23 → 25 → 26 →
> 27 → 28 (24 moved before 23 after review so the memory-cliff fix would not confound
> 23's large-n A/B).

### Step 21: Auto-select 22q+ → Metal ✅ (`7cad261`, superseded by `0806f3e`)

Routing-only quick win from the baseline data (Metal beat MLX 2.7–5.1× at 24–28q while
auto still routed there to MLX). Re-tuned at the end of the line: after Steps 22+25,
Metal wins at **every** measured count ≥17q, so the tiers are now CPU ≤16q / Metal ≥17q,
with MLX serving 17–30q only when `pyobjc-framework-Metal` is absent.

### Step 22: Metal batched command-buffer encoding ✅ (`58cc612`) — Metal 1.30× geomean

Gates are encoded into one open command buffer and committed at observation boundaries
(`_view`) or every 256 dispatches, instead of `commit` + `waitUntilCompleted` per gate;
`_const` matrix/index buffers are cached by content (in-flight safety rests on the
command buffer's default retention of referenced resources — regression-tested by
shrinking the cache cap). random@6 22.6→12.1 ms, random@20 106.8→64.2 ms; neutral at
26–28q (bandwidth-bound).

### Step 24: MLX periodic `mx.async_eval` ✅ (`bf78f05`) — random@28 1.56×

P1 removed per-gate eval but left a depth-d lazy graph holding O(d) full-width
temporaries (peak ≥ ~16× state size → swap at 28q). `async_eval(sv.data)` every 16
gates (only ≥24 qubits) bounds the working set: random@28 40.0→25.6 s. Interval 8–64
measured flat (±1%); every non-swapping cell unchanged.

### Step 23: MLX axis-order tracking ✅ (`7bbc216`) — 1.05–1.15× on dense circuits

`MLXState.perm` records the post-tensordot axis order instead of paying a full-state
transpose+copy per dense gate; diagonal/permutation/controlled paths translate targets
through the map; the permutation is materialized once at readback. Verified by
mixed-kind fuzz (dense+diagonal+permutation+controlled interleaved), partial
`abs2sum`/sampling, and readback-ordering tests. Gain bounded because `mx.tensordot`
still permutes its *input* internally — the planned Step 29 custom kernel became
unnecessary once Step 27 landed.

### Step 25: Kind-specialized Metal kernels ✅ (`bc346fb`) — Metal 1.36× geomean, random@24–28 2.7–3.1×

(a) Pipelines compiled per gate width with `K_FIXED` baked in as a preprocessor macro:
the per-group `amp[]`/`idx[]` arrays unroll into registers (runtime-`k` indexing was
spilling) — this is where the ~3× on fused 4q dense gates came from. (b) A monomial
kernel applies permutation-class gates with 2^k multiplies per group instead of 4^k
MACs (1.9× on 4q monomials at 20q; neutral ≥24q). (c) Threadgroup sweep 64/256/1024
was flat; 256 kept.

### Step 26: Diagonal-run wide fusion + CPU diagonal path ✅ (`14cdf74`) — QFT-focused

Second fusion pass merges adjacent diagonal gates (including diagonal composites such
as CX·Rz·CX from pass 1) into up-to-**8**-qubit diagonals via an O(2^k) elementwise
product. Width sweep: 7–8 win; 10 *regresses* (materializing + classifying 1024×1024
dense matrices outweighs the saved passes) — `MACQUEREL_DIAG_FUSION_WIDTH` overrides.
The CPU backend gained a broadcast in-place diagonal multiply (with memoized classify;
an unmemoized check cost ~10µs/gate on the raw path, caught by `bench_versions`).
cpu qft@22 680→268 ms; metal qft@28 1393→1025 ms; mlx qaoa@28 1.19×.

### Step 27: Commutation-aware fusion grouping ✅ (`3a742e8`) — largest single step

Replaced the single greedy in-order group with a multi-open-group scheduler: each gate
lands in or after the latest open group it shares a qubit with (the only real ordering
constraint; disjoint-qubit gates commute), joining the first group with capacity, with
at most 8 open groups. Brickwork circuits collapse into rolling neighborhood groups:
mlx random@24 1314→171 ms (7.7×), random@28 23.5→2.7 s (8.5×); metal random@24
180→99 ms; cpu random@22 1528→664 ms. GHZ unchanged (a CX chain has nothing to
reorder). Geomeans step-over-step: mlx 1.46×, cpu 1.17×, metal 1.15×.

### Step 28: Qubit remapping wired, gated OFF ❌ kept disabled (`f55fea2`)

`remap_qubits_with_perm` returns the applied permutation; `statevector()` inverts it at
readback and counts stay in caller order for free (MeasureOp labels are rewritten in
list order). The A/B lost on every backend/circuit at 24q (metal qft 61→79 ms, random
89→126 ms): the readback inverse transpose outweighs any stride benefit and the GPU
kernels are stride-insensitive. Ships disabled; `MACQUEREL_REMAP=1` opts in.

### Step 30: Per-backend, qubit-aware fusion-width defaults ✅ (`3a745fb`) — metal 1.30× / cpu 1.32× geomean

A fusion-width re-sweep (`benchmarks/data/fusion_width.json`: widths 1–6 ×
{QFT, random, QAOA, QV} × 16–24q, all three backends) showed the optimal
`max_fused_qubits` became a backend property after Steps 22/25: with Metal's per-gate
overhead mostly gone, wide fusion at small/mid n only pays host-side matrix
composition and densifies cheap diagonal/monomial gates. Defaults are now resolved per
(backend, qubit count) by `default_fusion_width`: **metal 2 ≤22q, cpu 3 ≤18q,
otherwise 4; mlx 4 everywhere** — `fuse_gates` takes the target backend and the
Simulator passes the one it selected; `MACQUEREL_FUSION_WIDTH` still pins a global
width and `auto` still runs the autotuner. The qubit tiering is load-bearing: a *flat*
metal width 2 won the sweep's normalized aggregate (1.58× vs width 4) but the step A/B
exposed 2.7–3.7× regressions on random@24–28 — at 24q+ every backend is apply-bound
and 4 still wins. Shipped A/B: metal 1.3–2.15× at 6–20q (qft@20 21→10 ms, random@20
41→28 ms), cpu 1.4–1.8× ≤16q, large n flat at 1.00×; mlx re-measured as a no-change
control (1.01×). Known compromise: metal random@22 0.85× (22q nets ~1.17× across
circuits).

### Net result

Cumulative vs the line's baseline (geomean over ghz/qft/random/qaoa): **Metal 2.5–2.9×
across 6–28q**; **MLX 1.9–2.6× at 22–28q** (best cell random@28 **14.7×**: 40.0 s →
2.7 s); **CPU 1.5–2.0×** (best cell qft@22 3.6×). Versus Qiskit Aer
(`benchmarks/data/large`): macquerel-metal wins from 20q (random 28 vs 51 ms), 5–12×
faster at 24q+ (random@24 99 ms vs 914 ms). The release-regression harness
(`bench_versions.py`) shows current faster than v0.2.0 on every backend at 8–12q. G3
(28q cliff) and G4 (auto picks measured-fastest) met; G2 (Metal crossover) moved from
≥22q to ~20q; G1 met at the system level (Metal tier), with MLX itself still behind
Aer only on QFT at 24q+.

---

## v0.2.x+ — performance candidates (Steps 31-35)

> **Status: SHIPPED (2026-06-10), branch `perf-candidates`.** The five measured
> candidates that came out of the backend comparison ([`backends.md`](backends.md)),
> each A/B-benchmarked per the protocol: re-baselined at the 0.2.1 release commit
> (`step32-baseline`, a 0.99-1.00× no-change control vs Step 30), per-step JSONs in
> `benchmarks/data/steps/`, full write-ups in `benchmarks/data/steps/README.md`.
> Execution order was 32 → 33 → 34 → 35 → 31.

### Step 32: MLX broadcast diagonal path ✅ (`fef6417`) — mlx 1.20× geomean, qft@22-28 2.5-4.3×

The diagonal path built a full-width gather table (k shift/or passes to build a
2^n gate-row index, plus a full-width phase gather) before the multiply. Replaced
with a single broadcast elementwise multiply: the state viewed with one length-2
axis per target qubit (gaps collapsed, so the view stays ≤2k+1-dimensional) times
the (2,)*k diagonal. This was the "in-place-style diagonal path" candidate aimed
at the wide-diagonal QFT cells where MLX trailed Metal 6-10×: qft@28 9.08→2.12 s,
qaoa@28 2.4×; everything else flat within GPU clock variance.

### Step 33: custom MLX dense kernel via `mx.fast.metal_kernel` ✅ (`4fff625`) — mlx 1.09× geomean, random@22-28 1.16-1.61×

The deferred Step 29. Dense/controlled gates now use the native Metal backend's
group-per-thread kernel design (Step 25), with gate width and control count baked
into per-(k, nc) generated source so the loops unroll — bypassing `mx.tensordot`'s
internal input permutation, the dominant cost on scattered-target circuits and the
half that Step 23's axis-order tracking could not remove. MLX kernel inputs are
`const device` (issue #2547), so the result double-buffers; tensordot remains the
k>6 fallback (fusion never emits those). Matters for the no-PyObjC fallback tier.
Cumulative with Step 32: mlx 1.30× geomean, 2.16× at 28q vs the 0.2.1 baseline.

### Step 34: lower the Metal small-n floor ✅ (`4e05d01`) — metal 1.07× geomean, ghz@24-26 1.6-1.8×, init 7.5 ms → 30 µs

(a) Device, command queue, and per-k compiled pipelines are process-wide;
`MetalBackend()` construction — paid on every `backend="auto"` call — drops from
~7.5 ms to ~30 µs, and the Simulator now reuses backend instances across
auto-mode calls (seeded simulators keep fresh-per-call backends so reruns stay
bit-identical). (b) State buffers recycle through a size-keyed per-backend pool;
buffers whose state dies while the open command buffer may still reference them
are parked until the next flush. (c) Redundant per-dispatch `setBuffer`/`setBytes`
ObjC calls are skipped. Apparent small-n regressions in the min-of-3 A/B did not
reproduce at reps=9 (ghz@12 1.31→0.86 ms, a win — the baseline cell was a
lucky-fast outlier).

### Step 35: per-chip backend-tier autotuning; CPU tier default 16 → 15 ✅ (`800691d`)

The CPU/GPU crossover is a chip property (the old 16q constant was measured on an
M5 Max). Mirroring the fusion-width pattern: `MACQUEREL_BACKEND_TIERS=<int>` pins
the CPU tier's max qubit count; `auto` measures the crossover once (QFT +
brickwork random on both backends across 10-20q, requiring the GPU to win
*sustainably*, not at one lucky count) and caches to
`~/.cache/macquerel/backend_tiers.json`; failures fall back to the default and
the zero-config path never measures. The default moved to 15: after Step 34,
Metal wins qft/random/qaoa at 16q (qft 5.6 ms vs cpu 9.2 ms) and the autotuner
independently measures cpu_max=15 on this chip.

### Step 31: `BatchedSimulator` ✅ (`9b00708`) — 26-47× at 4q, 20-45× at 8q, 14-23× at 12q, 2.2× at 16q vs a per-circuit loop

The v0.3 batched small-circuit feature, pulled forward. The small-n regime is
dispatch-bound, so a B-circuit parameter sweep run one circuit at a time pays the
fixed per-run costs B times. `BatchedSimulator` groups the batch by structure
signature (gate positions/targets/controls) and evolves each group as one
(B, 2^n) tensor: one batched matmul per gate position — or one broadcast phase
multiply when every circuit's matrix there is diagonal — with controlled gates
lifted to plain unitaries on controls+targets. Engines: NumPy and MLX (lazy
graph; the whole batched circuit evaluates in a few fused kernels); `auto`
routes on total size log2(B)+n against the Step 35 tier boundary, which picks
the faster engine in every measured cell (`benchmarks/data/batched.json`).
Design findings: per-circuit gate fusion was **99% of a fused prototype's
runtime** — batching already amortizes dispatch, so the batched path skips
fusion entirely; and `run()` requires explicit MeasureOps (no implicit
measure-all), keeping batched semantics unambiguous. Differential-tested against
the per-circuit Simulator on parameter sweeps, mixed-structure batches,
controlled/diagonal gates, and sampling distributions (`tests/test_batched.py`).

### Net result

Cumulative over the line (vs 0.2.1 baseline, geomean over ghz/qft/random/qaoa):
**mlx 1.30× overall, growing to 2.16× at 28q** (best cell qft@28 4.5×) — the MLX
fallback tier no longer has a pathological QFT gap; **metal 1.07× overall**
concentrated at 20-26q plus the 250× construction-cost removal on the auto
path; **cpu untouched**. Against the full two-line arc, the cumulative geomean
vs the step20 baseline stands at **metal 2.85×, mlx 2.04×, cpu 1.63×**. The
batched API turns small-circuit sweeps from a per-run-overhead problem into a
single launch per gate position (up to 47× measured).

## v0.3 — noise channels / density matrices

> **Status: SHIPPED (2026-06-11), branch `feat/density-matrix-noise`.** The first
> v0.3 feature: `DensityMatrixSimulator` with Kraus-operator channels.

### `DensityMatrixSimulator` ✅

The density matrix is carried as its **row-major vectorization** — a `4**n`
complex64 array the existing statevector backends treat as an ordinary 2n-qubit
state (ket axes `0..n-1`, bra axes `n..2n-1`). That single representational
choice meant zero backend changes:

- a unitary `rho -> U rho U^dagger` is two `apply_matrix` calls (U on the ket
  axes, `conj(U)` on the bra axes; controls shift with their targets);
- a Kraus channel `rho -> sum_k K_k rho K_k^dagger` is **one** dense
  superoperator `sum_k K_k (x) conj(K_k)` applied to the channel's paired
  ket+bra axes — a width-2q dense gate the backends already dispatch;
- measurement probabilities are the diagonal `rho_ii`, the stride-`2**n + 1`
  slice of the vectorization, read through a zero-copy host view on CPU/Metal
  (no full-matrix readback); `purity` is one BLAS `vdot` over the same view.

Channels live on the circuit (`Circuit.bit_flip/.phase_flip/.depolarizing/`
`.amplitude_damping/.phase_damping/.kraus`, validated for completeness at build
time) as `ChannelOp`s, which the fusion pass treats as barriers — gate runs
between channels still fuse, with the width resolved at the doubled qubit count
where the applies actually happen. The statevector `Simulator` and
`BatchedSimulator` reject noisy circuits with a pointer to the DM simulator.

**Memory**: an n-qubit density matrix costs exactly a 2n-qubit statevector, so
the backend ceilings land at n=15 (MLX, int32 ShapeElem at 2n=30) and n=16
(Metal, 32 GiB). Measured (`bench_memory.py --series dm`): metal stays on the
theoretical `4**N x 8 B` line (32.2 GiB at N=16), cpu peaks ~3x, mlx ~16-25x —
the same multipliers as the statevector series at the doubled count. Runtime
(`bench_density.py`): crossovers mirror the statevector tiers — cpu wins
through N~6-7, metal everywhere above (noisy GHZ@16 in 6.0 s, random@16 8.5 s).

**Correctness**: differential-tested against a direct dense `rho` reference on
random noisy circuits, against `Simulator` on noiseless circuits
(`rho == |psi><psi|`), and against analytic channel formulas; physicality
(trace, hermiticity, positivity, purity decay) holds across all three backends
(`tests/test_density.py`, `tests/test_noise.py`).

## v0.3.x — RAM usage candidates (Steps 36-40)

> In progress on branch `perf/v0.3.x-ram`; remaining steps tracked in
> [`plan.md`](plan.md).

### Step 36: MLX monomial kernel + eval cadence + pool release ✅ (`3decb13`) — peaks 19-25× → 3-5×, ghz@24-28 6.5-7.9× faster

The measured 19-25× MLX memory peaks were mostly **not** the eval cadence: the
monomial gather path built its source-index table on-device out of ~5 full-width
uint32 lazy-graph intermediates per fused permutation gate — a 28q GHZ held
10-12 GiB live for a 2 GiB state, and the lazy graph keeps every index
temporary alive until evaluation. Three changes landed together:

- **Register-resident monomial kernel** (the Step 33 design applied to
  generalized permutations): one thread owns a group, computes its indices in
  registers, reads the group through the gate's row permutation, and applies
  the per-row phase — the only full-width buffers are the input and output
  state. The on-device gather path remains the k>6 fallback.
- **Tight eval cadence with backpressure** above 26 effective qubits:
  `async_eval` every 2 gates (was 16), blocking on the checkpoint from one
  interval ago before kicking the next. The blocking half matters because
  `async_eval` alone never blocks — a shallow circuit encodes its whole graph
  before the GPU retires anything, keeping every intermediate live regardless
  of cadence (measured: cadence alone only halved the peaks).
- **Pool release at observation boundaries**, gated on the freed-buffer pool
  exceeding ⅛ of unified memory; smaller pools stay warm. (An unconditional
  clear cost qaoa@24q 2.4× in re-allocation; a 1 GiB threshold still paid it.)

A/B (GHZ memory cells vs `benchmarks/data/memory.json` baseline): sv peaks drop
3.5-3.8× — 28q 39.2 → 10.2 GiB (19.6× → 5.1× theory), 26q 9.15 → 2.65 GiB —
and dm n=14 49.2 → 8.15 GiB. Newly fitting cells: sv 29q/30q at **3.0×**
(12.1/24.2 GiB; 30q previously drove the machine into swap at ~16×) and the
previously-skipped dm n=15 (32.2 GiB), both un-skipped by recalibrating
`_PEAK_MULT` (mlx 16.0 → 6.0). Runtime improved in every measured cell
(`benchmarks/data/steps/step36-baseline-0c4c9dc-mlx.json` vs
`step36-3decb13-mlx.json`): ghz@24-28 6.5-7.9×, qft 2.1-2.6×, random 1.2-1.8×,
qaoa par-1.09×. Buffer-donation audit: `MLXState` and the simulator hot path
hold no extra state references; the per-n arange cache and the pending-eval
checkpoint (cleared at boundaries) are the only retained arrays.

### Step 37: `TrajectorySimulator` — Monte-Carlo wavefunction noise ✅ (`2b63036`) — noisy circuits at `2**n` memory, 30q GHZ 10.3 s/trajectory

Noisy simulation without the density matrix's `4**n` cost: each trajectory evolves
an ordinary statevector; every Kraus channel applies one operator sampled with the
Born probability `p_k = <psi|K_k^dagger K_k|psi>`, then renormalizes. Exact in
expectation (Mølmer–Castin–Dalibard); error ~`1/sqrt(trajectories)`. Reuses the
backends, `ChannelOp`s, and fusion (channels stay barriers) unchanged, with auto
selection at the *statevector* count — noisy circuits reach the Metal range instead
of the DM's n=16 cap. Jump probabilities need no state copies: every built-in
channel has **diagonal** effect operators `E_k`, so all `p_k` come from one
`abs2sum` marginal over the channel qubits (compatible with Metal's in-place
state); arbitrary non-diagonal `kraus(...)` channels fall back to the channel-qubit
reduced density matrix off a host view.

Two measured-at-30q memory findings shaped the implementation: (a) a fresh seeded
backend per trajectory pinned a new state buffer each time — now one derived-seed
backend per call keeps reproducibility (the whole stream derives from `seed`) and
shares the pool; (b) re-allocating the state per trajectory accumulated one
state-sized footprint *per trajectory* — released multi-GiB MTLBuffers are
reclaimed lazily by the driver and the Step 34 pool caps at 1 GiB — so trajectories
reset the state in place through the zero-copy view (4 trajectories at 30q:
47.3 → 21.5 GB peak, now constant in trajectory count). Demo: noisy 30q GHZ
(depolarizing ×3), 4 trajectories, 4000 shots on Metal — 10.3 s/trajectory.
Tests: noiseless exactness vs `Simulator`, stochastic agreement with
`DensityMatrixSimulator` (probabilities, Pauli expectations, a non-diagonal-effect
channel forcing the fallback), sampling semantics, seeding, backend parity
(`tests/test_trajectory.py`).

### Step 38: `expectation_pauli` via monomial gather ✅ (`fe6e0a7`) — 6.3× faster, 2.9× less peak (n=14, 4 terms, Metal)

A Pauli string is monomial — `P|i> = phase(i)|i ^ mask>` with `mask` the X/Y bit
pattern — so `tr(rho P) = sum_i phase(i) * rho[i, i ^ mask]` with
`phase(i) = i**(#Y) * (-1)**popcount(i & zy_mask)`: one `2**n`-element gather off
the zero-copy host view (CPU array / Metal unified-memory buffer; MLX pays one
readback), like `probabilities` already does for the diagonal. Replaces the full
`4**n` host readback plus a state-sized copy and two CPU gate applies per term —
at the n=16 Metal ceiling the old path's transients alone were ~3 state sizes
(~96 GB), past the machine's safe budget. `density_matrix()` gains an opt-in
`copy=False` zero-copy view with the same mechanism. A/B at n=14 (GHZ +
depolarizing, 4 mixed terms incl. Y, post-evolve call): 1.63 → 0.26 s and
13.0 → 4.45 GB peak, identical values; correctness gated against `tr(rho P)` on
the dense reference across backends (`tests/test_density.py`).

### Step 39: CPU in-place chunked dense apply ✅ (`bb230f0`) — peaks 3.0× → 1.03×, runtime up to 1.70× faster

The tensordot dense path made ~3 full state copies per gate (its internal input
permutation, its output, and the non-contiguous reshape on write-back). Replaced
with an in-place chunked apply: transpose the `(2,)*n` view so the target axes
lead (a stride trick, no data movement), then loop bounded chunks over the
non-target axes — gather a chunk into contiguous scratch, one GEMM into a second
scratch, scatter back to the same positions (safe in place: each group depends
only on its own amplitudes). Chunk size 2¹⁶ columns measured fastest (16 MiB of
scratch at k=4); the controlled path reuses the same routine on its control-slice
view. Stage (a) of the plan — `np.einsum(..., out=...)` — was prototyped and
**rejected**: without `optimize` einsum loses BLAS (14× slower at 24q/k=4), with
it the intermediates return. A/B: cpu peaks drop from 3.0× to ~1.03× of theory
(28q GHZ 6.04 → 2.05 GiB, dm n=14 6.04 → 2.06 GiB) and runtime improves in every
measured cell, growing with size — random@24q 1.70×, qaoa@22q 1.49×, ghz@22-24q
1.33×, qft@24q 1.31× (`benchmarks/data/steps/step39-*.json`). `_PEAK_MULT` cpu
4.0 → 1.5.

### Step 40: single-pass ket⊗bra superoperator for narrow DM gates ✅ (`8361126`) — metal/cpu/mlx 1.0-1.09×, two variants rejected on measurement

Control-free DM unitaries now apply as one `kron(U, conj(U))` gate over their
paired ket+bra axes — one pass over the `4**n` state instead of two — when the
doubled gate stays inside the backend fast envelope. Eligibility is per gate
*kind*, because kron preserves it: diagonal superoperators stay diagonal
(broadcast multiply; eligible to k=4, where the materialized `4**k × 4**k`
matrix is still small), monomial stays monomial (k≤3 — doubled width 6, the MLX
kernel cap), dense caps at k=2. Superoperators are memoized by matrix bytes;
controlled gates keep two passes (a controlled-U's ket⊗bra product
cross-multiplies its control projectors and is not a single controlled gate).
The decisive findings were the **rejections**: dense superops at doubled width 6
spill GPU registers (random_noise@12-14q on Metal **5× slower**), and capping
the DM fusion width at the superop width (so every fused gate qualifies) loses
more to extra passes than single-pass saves (ghz_noise on cpu −21%). The
shipped kind-aware rule is a small, consistent win with no regressions: metal
1.00-1.09×, cpu 1.02-1.07× (reps=7 where reps=3 looked noisy), mlx 0.99-1.06×
(`benchmarks/data/steps/step40-*.json`).
