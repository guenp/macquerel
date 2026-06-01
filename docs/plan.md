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
