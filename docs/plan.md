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
- **Step P4 — Evaluate native `complex64` storage (medium–high impact, higher effort).
  ⏸ DEFERRED.** One tensordot instead of four, half the kernels for elementwise ops. The
  spec cites up to 6.9× for SoA over interleaved, so this needs an A/B benchmark on current
  MLX before committing — deferred until that measurement can be run. Targets cause (4).
- **Step P5 — Avoid the full transpose copy (medium impact, medium effort). ⏸ DEFERRED.**
  Only affects `_dense_apply`, which is rarely hit: single-qubit dense gates use the Metal
  kernel and the common 2-qubit gates (CNOT/CZ/SWAP) are permutation/diagonal. CPU pays the
  same transpose, so it is not a CPU-vs-MLX gap. Low value; needs a benchmark to justify the
  rewrite risk. Targets cause (5).
- **Step P6 — Fuse before MLX dispatch. ✅ ALREADY SATISFIED.** The simulator already runs
  the fusion pass on the hot path (`statevector()` and `run()` call `fuse_gates`). The
  `bench_backends.py` microbenchmark deliberately stays unfused to measure raw per-gate
  throughput, so it should not be changed. No code change needed. Targets cause (6).
- **Step P7 — Re-tune the auto-select crossover (low effort). ✅ DONE (`351376c`).**
  `_select_backend` now routes ≤16 qubits to CPU (was ≤14) and MLX for 17–31, matching the
  measured crossover (CPU wins through 16q, MLX from 18q+).
- **Step P8 — `mx.compile` the hot gate kernels (low–medium impact, low effort).
  ⏸ DEFERRED.** Interacts with the deferred-evaluation graph from P1 and the variable
  per-gate shapes; benefit is uncertain without a benchmark. Deferred.

The two highest-leverage changes were **P1** (defer eval) and **P2** (kill host-side perm
work). Combined with P3 and P7, MLX went from slower than CPU everywhere to **2.4–5.5×
faster at 18–22 qubits** (see `benchmarks/data/`). The remaining steps (P4, P5, P8) are
deferred because they are perf-sensitive and need A/B benchmarking to justify — they should
be revisited with the benchmark harness rather than implemented blind.

---

### v0.2 — Metal backend, qubit remapping, expectation values

**Step 13: Qubit remapping / cache-blocking compiler pass (`src/macquerel/compiler.py`)**

Add a second compiler pass after gate fusion. The Doi–Horii technique: relabel qubits so the most frequently targeted qubits in each window land on the lowest indices (minimizing stride `2ᵏ`), inserting logical SWAP gates at chunk boundaries. Expose as `remap_qubits(circuit) → Circuit`. The existing fusion equivalence tests should also cover remapping: fused+remapped circuits must produce statevectors identical to unfused+unremapped.

**Step 14: `expectation_pauli` and `abs2sum` (`src/macquerel/backends/cpu.py`, `src/macquerel/backends/mlx_backend.py`)**

`abs2sum(sv, qubits)` — marginal probability sum over the given qubits (already computed internally in `sample`; expose as a first-class method).

`expectation_pauli(sv, pauli_strings)` — expectation value of a Pauli operator or sum of Pauli strings. For a single Pauli term `P` on qubit `q`: reshape state to `(2,)*n`, apply the Pauli matrix, take the inner product with the original state. For a sum, loop over terms.

**Step 15: `MetalBackend` (`src/macquerel/backends/metal_backend.py` or a C extension)**

A compiled extension using metal-cpp and nanobind. Hand-written `.metal` shaders with 64-bit (`size_t`) indexing to exceed the MLX `uint32` ceiling at 32 qubits. Genuine in-place updates (no double-buffering needed). Build system: scikit-build-core + CMake. For ≤31 qubits it offers no throughput advantage over the MLX kernel, so its only purpose is the >31-qubit regime.

**Step 16: Automatic backend selection (`src/macquerel/simulator.py`)**

When `backend='auto'` (make this the new default): select `CPUBackend` for ≤14 qubits (GPU launch overhead dominates), `MLXBackend` for 15–31 qubits, `MetalBackend` for ≥32 qubits. Requires MetalBackend (Step 15).

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
