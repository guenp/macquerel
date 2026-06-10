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

## v0.2.x — GPU backend performance (Steps 21-28)

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

### Net result

Cumulative vs the line's baseline (geomean over ghz/qft/random/qaoa): **Metal 2.2–2.8×
at 22–28q** and 1.5–1.7× below; **MLX 1.9–2.6× at 22–28q** (best cell random@28
**14.6×**: 40.0 s → 2.7 s); **CPU 1.6–2.0× at 20–22q**. Versus Qiskit Aer
(`benchmarks/data/large`): parity at 20q, 5–12× faster at 24q+ (random@24 99 ms vs
914 ms). The release-regression harness (`bench_versions.py`) shows current faster than
v0.2.0 on every backend at 8–12q. G3 (28q cliff) and G4 (auto picks measured-fastest)
met; G2 (Metal crossover) moved from ≥22q to ~20q; G1 met at the system level (Metal
tier), with MLX itself still behind Aer only on QFT at 24q+.
