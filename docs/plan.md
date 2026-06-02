# macquerel v0.1 Implementation Plan

## Context

macquerel is a quantum state-vector simulator targeting Apple Silicon's unified-memory architecture. The design spec is complete and a scaffold exists (protocol stubs, empty gate methods, a placeholder test). Everything functional needs to be built. This plan covers the **v0.1 milestone**: CPU reference backend + MLX backend, single/two-qubit gates, gate fusion, measurement/sampling, and a differential test harness.

The implementation follows the spec's layered architecture: gate library → CPU reference backend → frontend API → MLX backend → compiler → tests.

---

## Step 1: Fix packaging and project structure

**Why first:** the CMakeLists.txt `src/CMakeLists.txt` wrongly treats `simulator.py` as a C++ library, and `__init__.py` is empty. These need to be correct before anything else.

Files to touch:
- `CMakeLists.txt` — remove the `src/` subdirectory add for now (no C++ in v0.1)
- `src/macquerel/__init__.py` — export `Circuit`, `Simulator`, `Gate`
- Create `src/macquerel/backends/` package (`__init__.py`)

**Test:** `python -c "import macquerel"` succeeds; `uv sync` completes cleanly.

---

## Step 2: Gate matrix library (`src/macquerel/gates.py`)

Define all standard unitary matrices as `np.ndarray` (complex64), plus helpers.

**Gates to implement:**
- Single-qubit: `I`, `H`, `X`, `Y`, `Z`, `S`, `T`, `Rx(θ)`, `Ry(θ)`, `Rz(θ)`, `P(λ)` (phase)
- Two-qubit: `CNOT`, `CZ`, `SWAP`, `CP(λ)` (controlled-phase)
- Helper: `controlled(U)` — lifts a single-qubit gate to a 2-qubit controlled gate

**Gate classification** (used by compiler and kernel dispatch):
```python
GateKind = Literal["diagonal", "permutation", "dense"]
def classify(matrix: np.ndarray) -> GateKind: ...
```
Diagonal: all off-diagonal entries zero. Permutation: each row/col has exactly one nonzero entry of magnitude 1.

**Test (tests/test_gates.py):** `H @ H ≈ I`, `X @ X ≈ I`, `S @ S ≈ Z`, `Rz(θ) @ Rz(-θ) ≈ I`, unitarity (`U @ U†  ≈ I`) for all gates, `classify()` returns correct kind for each gate.

---

## Step 3: CPU reference backend (`src/macquerel/backends/cpu.py`)

The oracle against which all other backends are differentially tested. Correctness over speed.

**State representation:** `np.ndarray` of shape `(2**n,)`, dtype `complex64`. Stored flat; reshaped to `(2,)*n` for gate application.

**`apply_matrix`:** reshape state to `(2,)*n`, use `np.tensordot(gate, state, axes=([1,...], targets))`, then transpose axes back into canonical order. This is the textbook trick — one extra copy per gate is fine for a reference.

**`measure(sv, qubits, collapse=True)`:**
1. Reshape state to `(2,)*n`
2. Marginal probability of each qubit via `np.sum(|ψ|², axis=complementary_axes)`
3. Sample outcome via `np.random.choice([0,1], p=[p0, p1])`
4. If `collapse`: zero amplitudes inconsistent with outcome, renormalize

**`sample(sv, qubits, shots)`:** compute full `2^len(qubits)` probability vector, draw `shots` samples via `np.random.choice`.

**`allocate(n_qubits, dtype)`:** returns `|0...0⟩` state (all zeros except index 0 = 1).

**Test (tests/test_cpu_backend.py):**
- Bell state: `H(0)` then `CNOT(0,1)` on 2 qubits → statevector `[1/√2, 0, 0, 1/√2]`
- GHZ: `H(0)`, `CNOT(0,1)`, `CNOT(0,2)` → `[1/√2, 0, ..., 0, 1/√2]`
- QFT on 4 qubits: compare against `np.fft.fft` (up to normalization and bit reversal)
- Norm preservation after every gate (‖ψ‖ = 1, within 1e-5)
- Measurement collapse: post-collapse state is normalized and consistent with outcome
- Sampling: GHZ 3-qubit gives ~50/50 between `000` and `111` (χ² test, 10k shots)

---

## Step 4: Refactor frontend API

Replace the current stub with proper objects. Files: `src/macquerel/circuit.py`, update `src/macquerel/simulator.py`.

**`Gate` dataclass:**
```python
@dataclass
class Gate:
    name: str
    matrix: np.ndarray        # unitary, shape (2^k, 2^k)
    targets: list[int]
    controls: list[int] = field(default_factory=list)
    kind: GateKind = "dense"  # set by gates.py classify()
```

**`Circuit`:**
- `__init__(n_qubits)` — stores `n_qubits`, `gates: list[Gate | MeasureOp]`
- Gate builder methods: `h(q)`, `x(q)`, `y(q)`, `z(q)`, `s(q)`, `t(q)`, `rx(q, θ)`, `ry(q, θ)`, `rz(q, θ)`, `cx(ctrl, tgt)`, `cz(ctrl, tgt)`, `swap(q0, q1)`, `cp(ctrl, tgt, λ)`
- `measure(qubits)`, `measure_all()` — append `MeasureOp`
- Validation: qubit indices in `[0, n_qubits)`, no duplicate targets/controls

**`Simulator`:**
- `__init__(backend='cpu', dtype='complex64')` — instantiates the right backend
- `run(circuit, shots=1000)` → `Counter[str]` (bitstrings as keys)
- `statevector(circuit)` → `np.ndarray` or `mx.array`
- Backend selection: `'cpu'` → `CPUBackend`, `'mlx'` → `MLXBackend` (v0.1)
- For `run()`: compile circuit (fusion pass), then apply gates in order via backend

**Test (tests/test_circuit.py):** API round-trip (build circuit, run, check return type), qubit-index validation raises `ValueError`, measure_all on 3-qubit circuit.

---

## Step 5: MLX backend (`src/macquerel/backends/mlx_backend.py`)

**State representation (SoA):** two `mx.array` of shape `(2**n,)` dtype `float32`: `real` and `imag`.

**`allocate`:** `real = mx.zeros(2**n); imag = mx.zeros(2**n); real[0] = 1.0`

**`apply_matrix` — double-buffering with `mx.fast.metal_kernel`:**

The Metal kernel body for a single-qubit gate (indices as per §5.4 of spec):
```metal
uint p = thread_position_in_grid.x;
uint k = targets[0];
uint low = (1u << k) - 1u;
uint i0 = ((p & ~low) << 1) | (p & low);
uint i1 = i0 | (1u << k);
// 8-float matrix layout: [re(m00), im(m00), re(m01), im(m01), ...]
float a0r = real_in[i0], a0i = imag_in[i0];
float a1r = real_in[i1], a1i = imag_in[i1];
real_out[i0] = mat[0]*a0r - mat[1]*a0i + mat[2]*a1r - mat[3]*a1i;
imag_out[i0] = mat[0]*a0i + mat[1]*a0r + mat[2]*a1i + mat[3]*a1r;
real_out[i1] = mat[4]*a0r - mat[5]*a0i + mat[6]*a1r - mat[7]*a1i;
imag_out[i1] = mat[4]*a0i + mat[5]*a0r + mat[6]*a1i + mat[7]*a1r;
```

Kernel is registered once at module import via `mx.fast.metal_kernel(...)`. Grid: `(2**(n-1),)` threads. For a k-qubit fused gate, `2**(n-k)` threads; each gathers `2**k` amplitudes.

**Diagonal gate fast path:** kernel reduces to `out[i] = phase[i & mask] * in[i]` — no pairing, no stride problem.

**`measure` and `sample`:**
- Reconstruct complex view: `probs = real**2 + imag**2` (elementwise, then sum over irrelevant axes)
- For sampling: `mx.random.categorical(mx.log(probs), num_samples=shots)`
- Collapse: zero inconsistent amplitudes, renormalize

**Fallback:** if `mlx` is not installed, `MLXBackend` raises `ImportError` with a helpful message.

**Test (tests/test_mlx_backend.py):** Differential tests — every circuit in `test_cpu_backend.py` run on both backends must agree to `1e-5` on all amplitudes. This is the primary correctness gate.

---

## Step 6: Gate fusion compiler pass (`src/macquerel/compiler.py`)

**`fuse_gates(circuit, max_fused_qubits=4) → Circuit`:**
- Walk gates left-to-right; greedily merge adjacent gates whose combined target set has size ≤ `max_fused_qubits`
- Merge: compute the combined unitary via tensor contraction (or direct Kronecker + matrix multiply for simple cases)
- `MeasureOp` is a fusion barrier — never merge across it
- Controlled qubits count toward the qubit-set size

**`GateFuser` implementation:**
1. Maintain a "pending group": a list of commuting/adjacent gates and their joint qubit set
2. When adding a new gate would exceed `max_fused_qubits` or hits a barrier: emit the fused gate, start a new group
3. For a group of 1: emit as-is. For a group of k: compose matrices into a single `(2**k × 2**k)` unitary

**Test (tests/test_compiler.py):**
- Fused and unfused circuits produce identical statevectors on CPU backend (≤1e-6 difference)
- Single-gate circuits are unchanged by fusion
- Measurement barriers are respected (no fusion across measure)
- Unitarity of fused matrices

---

## Step 7: Complete test suite

Final pass to ensure full coverage matching §8 of the spec.

**tests/test_known_circuits.py:**
- Bell state, GHZ, Grover (2-qubit, marked state `|11⟩`), QFT 4-qubit
- Each checked on CPU and MLX backends

**tests/test_properties.py:**
- Norm preservation: run 50 random circuits (random gates, depths 1–20, 2–8 qubits), assert `‖ψ‖ = 1` within 1e-5 after each gate
- Fusion equivalence: same 50 random circuits with fusion on/off must agree

**tests/test_edge_cases.py:**
- 1-qubit circuit
- All-identity circuit (statevector unchanged)
- Empty circuit (returns `|0⟩`)
- max-controls gate (CNOT with 3 controls on 5 qubits)

---

## Files created / modified

| File | Action |
|---|---|
| `src/macquerel/__init__.py` | update exports |
| `src/macquerel/gates.py` | **new** — gate matrices + classify() |
| `src/macquerel/circuit.py` | **new** — Gate, MeasureOp, Circuit |
| `src/macquerel/simulator.py` | **rewrite** — Simulator orchestration |
| `src/macquerel/compiler.py` | **new** — gate fusion pass |
| `src/macquerel/backends/__init__.py` | **new** — package |
| `src/macquerel/backends/cpu.py` | **new** — CPUBackend |
| `src/macquerel/backends/mlx_backend.py` | **new** — MLXBackend |
| `CMakeLists.txt` | fix — remove broken src/ subdirectory |
| `tests/test_gates.py` | **new** |
| `tests/test_cpu_backend.py` | **new** |
| `tests/test_mlx_backend.py` | **new** |
| `tests/test_circuit.py` | **new** |
| `tests/test_compiler.py` | **new** |
| `tests/test_known_circuits.py` | **new** |
| `tests/test_properties.py` | **new** |
| `tests/test_edge_cases.py` | **new** |
| `tests/test_simulator.py` | update — replace placeholder |

---

## v0.2+ Implementation Plan

What follows is everything in the design spec that is not yet implemented, grouped by milestone and priority.

---

### v0.1 completion — performance-critical gaps

These items are described in the v0.1 spec but were deferred to a simpler correct implementation. They are the highest-value changes before moving to v0.2.

**Step 8: Permutation gate fast path (`src/macquerel/backends/mlx_backend.py`)**

Gate classification already identifies `"permutation"` gates (X, SWAP, CNOT) but `apply_matrix` only special-cases `"diagonal"` — permutation gates fall through to `_apply_general`. Permutation gates need no arithmetic, just a gather/scatter (or index-bit swap for SWAP/CNOT), so they should dispatch to a dedicated `_apply_permutation` method.

**Step 9: SoA state representation (`src/macquerel/backends/mlx_backend.py`)**

The spec mandates two `float32` arrays (`real`, `imag`) as the persistent state, not an interleaved `complex64` array. SoA is measured at up to 6.9× over interleaved storage and is described as "the single most important data-layout decision." Currently the MLX backend converts to SoA inside gate methods and immediately converts back. The state should stay in SoA form (two `mx.array`s) across gate calls, with the complex64 view reconstructed only at the API boundary (`statevector()`).

**Step 10: `mx.fast.metal_kernel` gate hot path (`src/macquerel/backends/mlx_backend.py`)**

The current `_apply_general` uses NumPy `tensordot` + MLX elementwise ops. The spec's optimized path uses a custom Metal kernel registered via `mx.fast.metal_kernel`: each GPU thread owns one `(i, i ⊕ 2ᵏ)` amplitude pair, loads both SoA `float32` values, applies the matrix, and scatters back. Double-buffering (ping-pong between two SoA buffer pairs) is required because MLX kernel inputs are `const device`. The kernel pseudocode is in §5.4 of the design spec and needs to be validated and debugged against a real build. A fused k-qubit kernel generalizes this to `2ⁿ⁻ᵏ` threads each gathering `2ᵏ` amplitudes.

**Step 11: Reproducible RNG (`src/macquerel/backends/mlx_backend.py`, `src/macquerel/simulator.py`)**

The spec requires `mx.random.key(seed)` for reproducible sampling. Expose an optional `seed` parameter on `Simulator.__init__` and thread it through to `MLXBackend.sample`.

**Step 12: `Backend` Protocol ABC (`src/macquerel/backends/__init__.py`)**

Define a formal `typing.Protocol` class with the full backend interface so that type checkers can verify backend conformance.

```python
class Backend(Protocol):
    def allocate(self, n_qubits: int, dtype) -> np.ndarray: ...
    def apply_matrix(self, sv, matrix, targets, controls) -> np.ndarray: ...
    def measure(self, sv, qubits, *, collapse: bool) -> list[int]: ...
    def sample(self, sv, qubits, shots) -> Counter: ...
    def expectation_pauli(self, sv, pauli_strings) -> np.ndarray: ...
    def abs2sum(self, sv, qubits) -> np.ndarray: ...
```

---

### MLX backend performance optimization

The permutation fast path (Step 8) and SoA state (Step 9) are implemented, and the
custom Metal kernel (Step 10) handles single-qubit dense gates. Despite this, the MLX
backend is still slower than the CPU backend across the benchmarked range (16–24
qubits). Profiling identified the causes below; the optimizations are ranked by
impact/effort.

**Context — is this expected?** Partly. At 16–22 qubits the statevector is only a few
MB, so the GPU runs in the *dispatch-bound* regime: per-kernel launch latency dominates
the actual compute, and NumPy is simply faster for small arrays. GPUs only pull ahead
once arrays are large enough that compute ≫ launch overhead (typically 24–28+ qubits).
On top of that inherent disadvantage, the current backend has fixable inefficiencies
that widen the gap well beyond the floor.

**Measured root causes** (depth-50 random circuit, 22 qubits unless noted):

1. **Per-gate `mx.eval()` forces a GPU sync on every gate.** Each of `_apply_diagonal`,
   `_apply_permutation`, `_apply_general`, `_apply_controlled`, and
   `_apply_metal_kernel_1q` ends with `mx.eval(...)`. MLX is lazy by design — it should
   build a large graph and evaluate once, fusing kernels and overlapping dispatch.
   Evaluating per gate defeats this. *Measured: 50 elementwise ops cost 17.4 ms with
   per-gate eval vs 6.0 ms evaluating once — a 3× penalty before any kernel-fusion
   gains.*
2. **Host-side O(2ⁿ) work per gate in the permutation path.** `_apply_permutation`
   rebuilds the full `2ⁿ` permutation table in NumPy on the host every call, then copies
   it to the device. *Measured: the NumPy table build alone is ~9 ms/gate — 3× the cost
   of the entire CPU gate (~3 ms) — plus a ~0.2 ms/gate 16 MB host→device copy.* This is
   why permutation gates remain ~7–9× slower than CPU even after the Python per-element
   loop was vectorized.
3. **Diagonal path rebuilds device scratch every call.** `_apply_diagonal` allocates
   `mx.arange(2ⁿ)`, a `gate_idx` of size `2ⁿ`, and several `2ⁿ` bitwise intermediates on
   every invocation to apply a phase vector that depends only on `(n, targets, matrix)`.
4. **SoA real/imag split multiplies kernel count.** Two float32 arrays turn every complex
   multiply into 3–4 real kernels; the dense path issues four `mx.tensordot` calls (rr,
   ri, ir, ii) where native `complex64` would need one.
5. **Full transpose+copy per dense gate.** `_dense_apply` ends with
   `mx.transpose(...).reshape(-1)`, a full `2ⁿ` memory reorder + copy on every dense gate.
6. **The `bench_backends.py` path does not fuse** — its `_run` applies raw ops one at a
   time, so MLX eats every tiny gate individually with its own eval (worst case for
   dispatch overhead). `bench_circuits.py` does fuse.
7. **Minor per-gate host overhead:** `classify(mat)` and `matrix.astype(...)` run on every
   `apply_matrix` call.

**Optimizations** (ranked):

- **Step P1 — Defer evaluation (high impact, low effort). ✅ DONE (`3c74da8`).** Dropped
  per-gate `mx.eval` from all five gate paths; evaluation is now forced only at segment
  boundaries (measure / `to_numpy` / sample). Targets cause (1). Measured 1.1–1.8× on its
  own.
- **Step P2 — Eliminate host-side permutation work (high impact, medium effort). ✅ DONE
  (`81e96e3`).** Compute the permutation gather index on-device with `mx.arange` + bitwise
  ops (mirroring `_apply_diagonal`); only the tiny `2**k` inverse lookup stays on the host.
  Targets cause (2); removed ~9 ms/gate at 22q. This is the change that flipped MLX from
  slower to faster than CPU.
- **Step P3 — Cache per-gate device constants (medium–high impact, medium effort). ✅ DONE
  (`6178f2f`).** Cache the constants reused on every gate: the `arange(2ⁿ)` index (bounded
  by distinct qubit counts), the scalar mask, and `classify()` results memoized by matrix
  bytes. Targets causes (3) and (7). Deliberately does **not** cache full `2ⁿ` per-target
  index tables — those would grow memory unboundedly and the on-device rebuild is cheap
  after P2.
- **Step P4 — Native `complex64` storage (medium–high impact, higher effort). ✅ DONE
  (`1eb61db`).** State is now a single complex64 array: one complex tensordot instead of
  four real ones, one complex gather for diagonal/permutation, and the SoA-only Metal 1q
  kernel is dropped (1q dense gates go through the complex tensordot). A/B vs the SoA backend:
  neutral on `bench_backends` (1.01–1.04× at 18–22q — complex tensordot matches the dropped
  Metal kernel), and faster on dense/fused circuits (`bench_circuits` @20q: dense 1.16×, QFT
  1.08×). **This refutes the spec's "SoA up to 6.9× over interleaved" claim on MLX 0.31** —
  the layouts are equivalent here, and complex64 is simpler (one array, one tensordot, no
  custom kernel). Targets cause (4).

  > **Note — why complex64 can *look* slower at large n with low reps.** Recorded
  > runs use `--reps 3` and report the min of 3. At 20–22 qubits the random circuit
  > is **memory-bandwidth-bound**, and a complex64 amplitude is 8 bytes = exactly two
  > float32 (the SoA `real`+`imag`), so **both layouts move identical bytes per gate** —
  > theory predicts parity, and the workload has essentially no dense multi-qubit gates,
  > so complex64's one-tensordot-vs-four advantage never triggers. The genuine
  > SoA↔complex64 difference at large n is therefore **below the benchmark's noise
  > floor**: GPU clock/thermal variance across process launches swings a *single*
  > configuration's 22q time by ~10–15% (one SoA run measured 44.8 ms, another 51.4 ms).
  > With only 3 samples the min can land on a lucky-fast SoA run and an ordinary
  > complex64 run, making complex64 look ~0.85×; a different pair of runs reverses it.
  > Higher reps (9+) and interleaved measurement (both backends in one process, sharing
  > thermal state) collapse the gap to parity. The large-n dip in the committed plots is
  > measurement noise, not a real regression; complex64's real win is on dense/fused
  > circuits (`bench_circuits`), which `bench_backends` does not exercise.
- **Step P5 — Avoid the full transpose copy via einsum (medium impact). ❌ TRIED & REVERTED
  (`b4b0171` → `50d3d54`).** Replaced `tensordot`+`transpose` in `_dense_apply` with a single
  canonical-order `einsum`. A/B on `bench_circuits` @18q showed einsum **slower** on every
  dense/fused workload (QFT 5.3→6.2, QAOA 2.8→4.2, dense 21.3→26.2 ms): MLX's `einsum`
  decomposes to a costlier sequence than `tensordot`+`transpose`. Premise false on MLX 0.31;
  reverted. Targets cause (5).
- **Step P6 — Fuse before MLX dispatch. ✅ ALREADY SATISFIED.** The simulator already runs
  the fusion pass on the hot path (`statevector()` and `run()` call `fuse_gates`). The
  `bench_backends.py` microbenchmark deliberately stays unfused to measure raw per-gate
  throughput, so it should not be changed. No code change needed. Targets cause (6).
- **Step P7 — Re-tune the auto-select crossover (low effort). ✅ DONE (`351376c`).**
  `_select_backend` now routes ≤16 qubits to CPU (was ≤14) and MLX for 17–31, matching the
  measured crossover (CPU wins through 16q, MLX from 18q+).
- **Step P8 — `mx.compile` the hot gate kernels (low–medium impact, low effort). ✅ DONE
  (`b6bef6f`).** Compiled the diagonal phase kernel and permutation gather. A/B (single
  qubit count, reps=5): modest net win where MLX is used — 20q 1.09×, 22q 1.12× — with a
  small ~0.88× compile-overhead regression at 18q. Kept for the large-n gain; benefit is
  incremental because the diagonal chain already fuses under deferred eval (P1).

The two highest-leverage changes were **P1** (defer eval) and **P2** (kill host-side perm
work); together they flipped MLX from slower to faster than CPU. P3, P7, P8, and P4 then
added incremental gains and simplification, leaving MLX **2.4–5.6× faster at 18–22 qubits**
(see `benchmarks/data/` and `benchmark-2.png`). **P5 was tried and reverted** (einsum
regressed). **P6** needed no change (the simulator already fuses). Every perf-sensitive
step was decided by A/B benchmark on the harness rather than implemented blind — which is
how P5 was caught and P4's expected SoA advantage was found not to hold on MLX 0.31.

---

### v0.2 — Metal backend, qubit remapping, expectation values

**Step 13: Qubit remapping / cache-blocking compiler pass (`src/macquerel/compiler.py`)**

Add a second compiler pass after gate fusion. The Doi–Horii technique: relabel qubits so the most frequently targeted qubits in each window land on the lowest indices (minimizing stride `2ᵏ`), inserting logical SWAP gates at chunk boundaries. Expose as `remap_qubits(circuit) → Circuit`. The existing fusion equivalence tests should also cover remapping: fused+remapped circuits must produce statevectors identical to unfused+unremapped.

**Step 14: `expectation_pauli` and `abs2sum` (`src/macquerel/backends/cpu.py`, `src/macquerel/backends/mlx_backend.py`)**

`abs2sum(sv, qubits)` — marginal probability sum over the given qubits (already computed internally in `sample`; expose as a first-class method).

`expectation_pauli(sv, pauli_strings)` — expectation value of a Pauli operator or sum of Pauli strings. For a single Pauli term `P` on qubit `q`: reshape state to `(2,)*n`, apply the Pauli matrix, take the inner product with the original state. For a sum, loop over terms.

**Step 15: `MetalBackend` — native compiled backend for the >31-qubit regime**

> **Status: IMPLEMENTED (2026-06-02).** Gate 0 = GO (outcome 1). Released MLX caps at
> **≤30 qubits**; 31q+ is reachable only by a native backend. Built as a **PyObjC driver**
> (`src/macquerel/backends/metal_backend.py`), not the metal-cpp + nanobind + scikit-build-core
> extension this plan originally specified — that offline `.metal` → `.metallib` path is
> unbuildable on the target machine (the Metal Toolchain CLI is missing and Xcode's downloader
> is broken: `DVTDownloads.framework` absent, `-runFirstLaunch` fails). PyObjC reaches the same
> goals with no build system: shaders compile at runtime via `newLibraryWithSource` (Metal
> *framework* only), amplitudes live in one unified-memory `MTLBuffer` with 64-bit indexing,
> updates are in-place. **Verified: runs 31q, 32q, and 33q** — the realistic 64 GiB ceiling,
> using 64.06 GiB resident (16/32/64 GiB states, analytic GHZ spot-check exact), plus
> differential-tested vs CPU to 1e-5, wired into auto-select (Metal ≥31q).
> **Surprise result:** Metal also *beats* MLX from ~22q up (≈13× at 30q) because it avoids
> MLX's double-buffering + gather temporaries under memory pressure — see `benchmark-3.png`.

#### Rationale check (do this *before* writing any C++)

The original premise was "MLX breaks at 32 qubits (uint32 element count), so we need a
64-bit-indexed Metal kernel." Research in June 2026 sharpens and partly undercuts this:

- The real ceiling is **2³¹ elements = 31 qubits**, not 2³². Root cause is
  `ShapeElem = int32_t` in `mlx/array.h`: at ≥2³¹ amplitudes, shape/size math overflows to
  negative and allocation wraps (MLX issue **#3327**, reported 2026-03-26). Our installed
  **MLX 0.31.2 still has this bug** — so today MLX is effectively capped at **30 qubits**
  for safety (31q is exactly on the cliff).
- That issue was **closed by PR #3524 (merged 2026-05-21)** — but the PR **does NOT lift the
  ceiling.** Its description states *"ShapeElem stays int32_t. No public API change."* It only
  adds **overflow detection** (a clean `std::overflow_error` instead of silent corruption at
  the 2³¹ boundary) and promotes some Metal kernel *offsets* to `size_t`. The 2³¹-element
  limit remains, by design — so neither released MLX nor `main` reaches 31q. *(Verified by
  reading the PR, 2026-06-02; corrects an earlier draft of this plan that assumed #3524
  widened the type.)*

**Action — gate 0 (½ day, blocking). ✅ DONE (2026-06-02).** Empirically probe the ceiling on
this machine (`benchmarks/probe_mlx_ceiling.py`):

```python
# does MLX allocate, fill, gate, and read back a 2**n complex64 state without overflow?
for n in (30, 31, 32, 33):
    sv = mx.zeros(2**n, dtype=mx.complex64); sv = sv * 1.0; mx.eval(sv)
```

Decision tree (which outcome the backend's reason-to-exist hinges on):

1. **If MLX still caps at ≤31q:** full justification — Metal is the only path to 32q+.
2. **If MLX now reaches 32q+ but doubles memory** (its lazy graph + any gather temporaries):
   Metal's edge is **capacity**, not speed — see the in-place argument below.
3. **If MLX reaches 32q+ cleanly:** **recommend NOT building this backend.** Document the
   finding, mark Step 15 obsolete, and redirect the effort to Step 13/16 polish or v0.3.
   A hand-maintained Metal/Objective-C++ extension is a large permanent maintenance tax
   that only pays off if it does something MLX cannot.

**Outcome: (1).** See "Gate 0 — RESULTS" below for the data. Two follow-up checks were also
resolved while running the gate:

- *Is there a newer release that lifts it?* No — **0.31.2 is the latest on PyPI**.
- *Would building MLX `main` help?* **No, and we did not need to build it to know.** PR #3524
  (the only relevant change since) keeps `ShapeElem = int32_t` and merely *detects* the
  overflow; it does not widen the type. `main` rejects 31q exactly like the release. Building
  from source would be purely confirmatory (same 30q ceiling, cleaner error), so it was
  **evaluated and skipped** rather than run.

The spec itself already concedes (`§3`): *"MLX gets you to 90% of peak with 20% of the
effort … a hand-written Metal kernel can't beat physics, it can only match it."* So this
backend is **never about throughput** at equal qubit count — both saturate memory bandwidth.
Its only possible wins are **(a) reach beyond MLX's index ceiling** and **(b) +1 qubit of
capacity** from genuine in-place updates.

#### Gate 0 — RESULTS (2026-06-02, this machine: M-series, 128 GB unified memory)

Probe: `benchmarks/probe_mlx_ceiling.py` exercised allocate / scalar-multiply / `arange` /
gather / readback on a `2ⁿ` complex64 state, each `n` in an isolated subprocess.

| n | elements | complex64 size | result |
|---|---|---|---|
| 29 | 5.37e8 | 4 GiB | ✅ all ops OK |
| 30 | 1.07e9 | 8 GiB | ✅ all ops OK |
| 31 | 2.15e9 | 16 GiB | ❌ **rejected before allocation** |

**The break is a hard type ceiling, not a memory limit.** At n=31 (`2³¹` elements) MLX
0.31.2 raises `TypeError: zeros(): incompatible function arguments … Invoked with types: int`
— the shape value `2³¹` does not fit MLX's **`int32` `ShapeElem`**, so the binding refuses
the call *before any allocation is attempted*. Confirmed it is not an overload quirk:
`mx.zeros((2**31,))` (tuple shape) and `mx.arange(2**31)` fail identically. This is exactly
the `ShapeElem = int32_t` root cause of issue **#3327**, and it would reject 31q even on a
machine with infinite RAM (the 16 GiB state fits this 128 GB box easily).

**Released-MLX status:** the newest MLX on PyPI is **0.31.2** (release list ends
`… 0.31.0, 0.31.1, 0.31.2`). Building from `main` does **not** help: PR #3524 keeps
`ShapeElem = int32_t` and only adds overflow *detection* (see rationale check) — `main`
rejects 31q just like 0.31.2, only with a cleaner error. So **no released or development MLX
reaches 31q**; the ceiling is a deliberate upstream design choice, not a bug awaiting a fix.

**Verdict: GO — outcome (1), and it is the only outcome on offer.** A native backend is the
**only** way to reach 31q+ at all, now and for the foreseeable future — there is no "wait for
the MLX fix" shortcut, because MLX has chosen to stay int32. This *strengthens* Step 15's
justification versus the earlier draft:

- **The full 31q+ regime is Metal-exclusive.** 31q (16 GiB), 32q (32 GiB), and 33q (64 GiB)
  are all unreachable by MLX on this 128 GB machine and would each require the native backend.
- On top of that, in-place updates add the **capacity / +1-qubit** edge: a realistic **33q on
  this 128 GiB machine** (64 GiB state, in-place) vs MLX double-buffering's 32q, and 32q vs
  31q on a 64 GB machine. (34q is byte-fit only — it would consume all RAM; see the capacity
  table.)
- Lifting the ceiling *inside* MLX (widening `ShapeElem` to int64 and submitting upstream) is
  the design spec's open question `§11` and remains a theoretical alternative — but it is an
  MLX-core change Apple has so far declined, not something we can pin today.
- **Recommendation for review:** the only routes to 31q+ are (a) **build the Metal backend**
  (this Step 15), or (b) **upstream an int64-`ShapeElem` change to MLX** (large, external,
  uncertain acceptance). If 31q+ is a near-term goal, (a) is the realistic path. If it is not,
  Step 15 can be **deferred** with no impact on ≤30q work — that is the genuine decision point,
  not "wait for an MLX release."

#### The capacity argument (why in-place matters)

State is complex64 = 8 bytes/amplitude (stored SoA as two `float32` buffers that sum to the
same 8 bytes/amp). MLX custom kernels cannot write back into their inputs — inputs are
`const device` (issue **#2547**, still effectively unresolved as of 2026; the only workaround
is a dummy/extra output), so every gate **double-buffers**: it reads one `2ⁿ×8` buffer and
writes another. That doubling costs exactly one qubit of headroom.

The table below is in **GiB** (`2ⁿ × 8` bytes; this 128 GB Mac reports `hw.memsize` ≈ 137 GB
= **128 GiB**). Distinguish two ceilings: the *byte-fit* ceiling (does the buffer fit in
nominal RAM at all?) and the *realistic* ceiling (does it fit with enough free memory for
macOS, the Python process, and the Metal resident working set — a kernel can't run against
swapped-out buffers, so the whole state must stay resident):

| n | 1 state | MLX (double-buffered) | Metal (in-place) |
|---|---|---|---|
| 31 | 16 GiB | 32 GiB ✅ | 16 GiB ✅ |
| 32 | 32 GiB | 64 GiB ✅ | 32 GiB ✅ |
| 33 | 64 GiB | 128 GiB ⚠️ byte-fit only (= all RAM) | 64 GiB ✅ **realistic max** |
| 34 | 128 GiB | 256 GiB ❌ impossible | 128 GiB ⚠️ byte-fit only (= all RAM) |

**Realistic max on this 128 GiB machine: 33 qubits** with the in-place Metal backend (64 GiB
state leaves 64 GiB for everything else). **34q is byte-fit only** — the state alone consumes
100% of RAM, leaving nothing resident for the OS/runtime/Metal, so it would thrash and is not
achievable in practice. The in-place backend's concrete payoff is therefore **33q vs MLX's
32q** (MLX double-buffering needs 128 GiB for 33q = the same all-RAM wall), i.e. **+1 usable
qubit**, plus the analogous 32q-vs-31q gain on a 64 GB machine.

Two more constraints at the top end:
- **Don't materialize the full statevector on the host.** `to_numpy()` at 33q allocates
  *another* 64 GiB ndarray (also unified memory) → 128 GiB total, hitting the wall. So at
  32–33q, **sample / measure / compute expectations on-device** and read back only small
  results (the `§6.5` prepare-once/sample-many guarantee); full host readback for differential
  testing is itself limited to ~**32q**, which is why the large-n test uses a subset-amplitude
  check.
- **Per-buffer limit.** SoA helps here: splitting into two `float32` buffers (each `2ⁿ × 4`)
  keeps any single `MTLBuffer` at half the state size (e.g. 32 GiB each at 33q), comfortably
  under `MTLDevice.maxBufferLength`. Verify `maxBufferLength` and
  `recommendedMaxWorkingSetSize` on the target machine during the build skeleton (phase 2).

#### Architecture (only if gate 0 says "build it")

> **As-built note:** the design below (metal-cpp + nanobind + scikit-build-core + offline
> `.metallib`) was the original target but was *not* built — the offline Metal Toolchain is
> unavailable on the machine (see Status). The shipped backend is a **pure-Python PyObjC
> driver** with the *same* kernel logic (SoA dropped in favour of one interleaved complex64
> `MTLBuffer`, which is bit-identical to NumPy for zero-copy readback and still fits 33q at
> 64 GiB < the 80.6 GiB `maxBufferLength` measured on this M5 Max). Two runtime-compiled
> kernels: a per-amplitude `diagonal` fast path and a per-group `dense` kernel covering
> dense/permutation/controlled. The 3D-grid 64-bit index reconstruction is unchanged from the
> design. Keep the rest of this section for historical/design context.

A native extension behind the existing `Backend` Protocol, so the simulator dispatches to it
unchanged. State held as **two `MTLBuffer`s** (`real`, `imag` — SoA per `§5.2`; Metal has no
native complex type, and separate streams coalesce on the M-series memory controller) aliased
onto unified-memory allocations via `newBufferWithBytesNoCopy`, so there is genuinely no host
copy. All index math is `size_t`/`uint64_t`.

```
macquerel/
  src/macquerel/backends/
    metal_backend.py        # thin Python wrapper: holds buffer handles, satisfies Backend
  src_native/metalq/        # the extension
    bindings.cpp            # nanobind module: allocate / apply_gate / gather / readback
    engine.mm               # metal-cpp (Obj-C++): device, queue, pipeline cache, dispatch
    kernels.metal           # compute shaders (compiled to .metallib at build time)
    Metal.hpp               # vendored metal-cpp single header (header-only)
  CMakeLists.txt            # scikit-build-core target + metallib compile step
```

**Build system.** Move packaging to **scikit-build-core + CMake** (replacing the
pure-`hatchling` wheel) with **nanobind** for the bindings (the combo nanobind's own docs
recommend; CMake ≥3.15, `nanobind_add_module`). metal-cpp is header-only — vendor the
single-header `Metal.hpp` and define `NS_PRIVATE_IMPLEMENTATION` / `CA_PRIVATE_IMPLEMENTATION`
/ `MTL_PRIVATE_IMPLEMENTATION` in exactly one `.mm`. CMake invokes `xcrun -sdk macosx metal`
then `metallib` to compile `kernels.metal` → `default.metallib`, packaged alongside the
extension and loaded with `newLibraryWithURL`. Keep the pure-Python wheel path working when
the extension is absent: `metal_backend` import failure must degrade gracefully (MLX/CPU stay
the default), exactly like the current optional `mlx` import.

**Kernels** (mirror the MLX paths already validated, now in-place with 64-bit indices):

- *Diagonal* (`§6.3`, highest value): one elementwise complex multiply, read-once/write-once,
  no pairing, no stride problem — `out[i] = diag[gate_row(i)] * in[i]`, written in place.
- *Permutation*: pure gather/index-bit rewrite, no arithmetic.
- *Dense 1–2q (and fused k-q)*: the qHiPSTER `(i, i ⊕ 2ᵏ)` pairing loop — each thread owns
  one amplitude group, loads `2ᵏ` SoA float pairs, applies the matrix, scatters back. For the
  high-stride regime, stage a contiguous tile through `threadgroup` memory and permute on-chip
  (`§4.3`/`§6.4`) to convert strided global access into contiguous.
- *Controlled*: mask threads whose index doesn't satisfy the control bits (`§5.6`).

Grid: a 1D grid can't address >2³² threads, so dispatch a **3D grid** (or one thread per
`2ᵏ`-group with each thread striding) and reconstruct a `uint64` linear index from
`thread_position_in_grid` — this is the actual mechanism that lifts the ceiling.

In-place correctness: the pairing/diagonal kernels write each output element exactly once and
read only their own group, so an in-place update over one buffer is race-free without
double-buffering (the spec's own argument, `§5.4`).

#### Implementation phases (as-built status)

1. ✅ **Gate 0 — re-validate the ceiling** (above). GO.
2. **~~Build skeleton~~ (obsolete):** no CMake/nanobind/metallib — PyObjC needs no build step.
   The `metal` extra (`pyobjc-framework-Metal`) was added to `pyproject.toml`/`uv.lock`.
3. ✅ **Allocate + readback:** `allocate(n)` → one unified-memory `MTLBuffer`; `to_numpy` is a
   zero-copy `complex64` view. Differential-tested vs CPU.
4. ✅ **Diagonal kernel** → differential test vs CPU (exact).
5. ✅ **Permutation** → handled by the general dense kernel (a permutation matrix is dense),
   differential-tested; in-place via the per-group structure (no gather race).
6. ✅ **Dense group kernel** (1–4q fused) + **controlled** (control-bit mask) → differential
   tested, including non-adjacent targets and a 12-seed random-circuit fuzz.
7. ✅ **Wire into `_select_backend`** (Step 16): CPU ≤16q, MLX 17–30q, Metal ≥31q.
8. ✅ **Large-n validation:** 31q and 32q GHZ spot-check on a subset of amplitudes matches the
   analytic prediction exactly; in-place confirmed (single 1× buffer per state).

#### Testing (`tests/test_metal_backend.py`)

- Skip the whole module unless the extension imported (`pytest.importorskip`), like the MLX
  tests — keeps CI green on machines without it / non-Apple.
- **Differential vs CPU** to 1e-5 on the fuzzed random-circuit corpus (`§8`) — the single most
  important test; catches indexing/stride/control-mask bugs. Reuse the existing CPU↔MLX
  differential harness, adding Metal as a third backend.
- Boundary tests: the 31→32 qubit switch, all-identity, max-controls, empty circuit.
- A capped large-n test (env-gated, e.g. `MACQUEREL_BIG_TESTS=1`) so 32q+ runs are opt-in.

#### Risks / decision triggers

- **MLX widens `ShapeElem` to int64 upstream:** would obsolete the index-ceiling rationale,
  leaving only the in-place +1-qubit edge. Gate 0 showed this has *not* happened (Apple
  explicitly kept int32 in #3524), so it is a low-probability future risk, not current.
- **Build/maintenance tax:** scikit-build-core + a `.metallib` step + metal-cpp churn is a
  permanent cost on every contributor's machine and in CI. Only justified by a real capacity
  win.
- **~~At equal qubits it won't be faster~~ — refuted at large n.** The pre-build expectation
  was that Metal would only match MLX's ms/gate (both bandwidth-bound). In practice Metal
  *beats* MLX from ~22q up (≈13× at 30q, `benchmark-3.png`): MLX's per-gate double-buffering
  and full-width gather temporaries thrash unified memory at 28–30q, while Metal's in-place
  single-buffer path scales as the bandwidth-bound ideal (~2× per +1 qubit). Metal is slower
  *below* ~20q (per-gate `waitUntilCompleted` sync vs MLX's lazy-graph fusion), so auto-select
  keeps CPU ≤16q and MLX 17–30q. Success criteria — *running 31–33q at all* and *half the
  memory of MLX* — both met (33q = 64 GiB in-place vs MLX's notional 128 GiB).
- **Memory to test 33q** (64 GiB state) is fine on this 128 GiB machine, but a full host
  readback (another 64 GiB) is not — plan for a subset-amplitude differential check rather
  than full statevector equality at the top end. 34q is byte-fit only and not a test target.

**Step 16: Automatic backend selection (`src/macquerel/simulator.py`)**

When `backend='auto'` (the default): select `CPUBackend` for ≤16 qubits (GPU launch overhead
dominates), `MLXBackend` for 17–30 qubits, `MetalBackend` for ≥31 qubits. **Done** — see
`_select_backend` in `simulator.py`. MLX's tier ends at 30q (not 31q): its `int32` `ShapeElem`
rejects `2**31` amplitudes, so 31q is the first Metal-only point.

---

### v0.2 test additions (`tests/`)

- `test_compiler.py` — add remapping equivalence: circuits run with remap on/off must yield identical statevectors.
- `test_metal_backend.py` — differential tests (CPU vs Metal) for circuits up to 32+ qubits, once MetalBackend exists.
- `test_backend_protocol.py` — verify CPUBackend, MLXBackend, and MetalBackend all satisfy the Backend Protocol; test `expectation_pauli` and `abs2sum` against analytic values (e.g. ⟨Z⟩ = 1 for |0⟩, ⟨X⟩ = 1 for |+⟩).
- `test_simulator.py` — add seed reproducibility test: two runs with the same seed must return identical shot counts.
- `tests/test_known_circuits.py` — add Quantum Volume circuits and random-circuit-sampling spot checks.

---

### v0.3

**Step 17: Cirq/Qiskit front-end adapters**

`macquerel.from_cirq(circuit)` and `macquerel.from_qiskit(circuit)` converters so existing circuits can run unmodified on macquerel backends. Depends on optional `cirq-core` / `qiskit` extras in `pyproject.toml`.

---

### v1.0

**Step 18: Benchmarking suite (`benchmarks/`)**

Implement the full plan from §9 of the design spec:
- API-level microbenchmarks: single-gate throughput as a function of target qubit index and qubit count, reported as GB/s vs theoretical peak.
- Circuit-level macrobenchmarks: QFT, random circuit sampling, QAOA layers, Quantum Volume, swept 20–32 qubits.
- Comparison harness vs qsim CPU and Qiskit Aer statevector.
- `max_fused_qubits` sweep ∈ {1,2,3,4,5,6} to validate the "4 is optimal" finding on Apple hardware.

**Step 19: Shot batch-size autotuning (`src/macquerel/simulator.py`)**

Autotune the shot batch passed to `mx.random.categorical` by doubling until throughput plateaus (the Tsim approach). Expose `batch_shots` parameter on `Simulator` with `'auto'` default.

**Step 20: Per-chip fusion-width autotuning (`src/macquerel/compiler.py`)**

At install time (or first run), measure bandwidth/FLOP ratio on the local chip and pick the `max_fused_qubits` value that maximises throughput rather than hardcoding 4.

---

### v2

- **Noise channels / density matrices** — `DensityMatrixSimulator` with Kraus-operator channels.
- **Memory-mapped out-of-core backend** — state vector backed by an NVMe file via `np.memmap`, for single large runs past DRAM capacity.
- **Batched small-circuit simulation** — `BatchedSimulator` packing many small circuits (QML/VQE parameter sweeps) into one kernel launch.
- **Multi-Mac over Thunderbolt** — distributed state vector using index-bit partitioning across machines.

---

## Verification

After each step, run `uv run pytest tests/ -x -q` and confirm the new tests pass before moving to the next step. Final verification:

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
