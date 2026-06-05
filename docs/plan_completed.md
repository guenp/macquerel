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
6. **`bench_backends.py` does not fuse** (deliberate — measures raw per-gate throughput).
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
  `bench_backends` (1.01–1.04× at 18–22q), faster on dense/fused (`bench_circuits` @20q:
  dense 1.16×, QFT 1.08×). **This refutes the spec's "SoA up to 6.9× over interleaved"
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
  fusion pass on the hot path (`statevector()` / `run()`). `bench_backends.py` deliberately
  stays unfused. No code change needed.
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

Implemented the bulk of the §9 benchmark plan:

- ✅ **API-level microbenchmarks** (`bench_single_gate.py`): single-gate throughput vs
  target index and qubit count, reported as GB/s vs theoretical peak.
- ✅ **Circuit-level macrobenchmarks** (`bench_circuits.py`): QFT, random circuit sampling,
  QAOA layers, swept across qubit counts, plus the `max_fused_qubits ∈ {1..6}` sweep.
- ✅ **Comparison harness** (`bench_statevector.py`): vs Qiskit Aer (statevector) and qulacs,
  with endianness/precision/convention handling.
- ✅ **Metal-specific benchmarks + plots** (`bench_metal.py`, `plot_metal.py`,
  `plot_results.py`).

> Remaining benchmark gaps (Quantum Volume macrobenchmark, qsim CPU comparison) are tracked
> in [`plan.md`](plan.md).
