# macquerel — Design Document

*A quantum state-vector simulator for Apple Silicon*

**Status:** Draft v0.1 · **Target:** Python 3.11+ on macOS (M1–M5, arm64)

> **Verification legend.** Throughout this document:
> - ✅ **Verified** — API signature or behavior confirmed against current MLX / cuQuantum / qsim documentation (see §11 References).
> - 🧪 **Pseudocode / untested** — illustrative; the logic has *not* been compiled or run and should be treated as a specification to validate, not finished code.
> - 💭 **Design judgment** — a reasoned proposal, not yet benchmarked on real hardware.

---

## 1. Motivation

State-vector simulation is the workhorse of quantum-circuit development: it holds the full `2ⁿ`-amplitude wavefunction in memory and applies gates as in-place linear-algebra updates. The dominant accelerated implementations — NVIDIA's **cuStateVec** (part of the cuQuantum SDK) and Google's **qsim** — are built for CUDA GPUs and AVX/FMA x86 CPUs respectively. Neither targets Apple Silicon's GPU or its unified-memory architecture, so Mac users fall back to CPU NumPy paths or the unoptimized PyTorch MPS backend.

Apple Silicon is, however, unusually well-suited to this workload for one specific reason: **state-vector simulation is memory-bandwidth-bound, not compute-bound.** A recent controlled study on the M4 Pro confirms that every standard gate (H, CNOT, CP, SWAP) sits at ≤0.38 FLOP/byte and the general single-qubit unitary tops out at 0.875 FLOP/byte — deep in the memory-bound regime at every qubit count. Unified memory gives the GPU access to the entire RAM pool (up to 192 GB on Mac Studio) at high bandwidth with zero host↔device copies, which is exactly the resource this problem is starved for.

**macquerel** ("Mac" + "querel", and a nod to the fish) is a from-scratch state-vector simulator that treats unified memory and the Apple GPU as first-class citizens.

### Goals
- A clean, Qiskit/Cirq-adjacent Python API for building and running circuits.
- Single-machine simulation of ~30 qubits (`complex64`) on a 64 GB Mac.
- Competitive or better wall-clock time than qsim's CPU path on the same hardware.
- A pluggable backend abstraction so the compute engine can evolve.

### Non-goals (v1)
- Multi-node / distributed simulation (cuStateVec's MPI mode).
- Tensor-network or stabilizer methods (cuTensorNet / cuStabilizer analogues).
- Noise channels and density matrices (deferred to v2).

---

## 2. What the state of the art does

A quick survey to anchor the design.

**cuStateVec** exposes a handle-based C API with a small, orthogonal set of primitives: gate application (`applyMatrix`), measurement, expectation values (including on a Pauli basis), and sampling. Its newer "Ex" API adds a *State Vector Updater* that queues operators for a circuit and applies them in a batch, plus optimized paths for dense / diagonal / anti-diagonal matrices. It scales past one GPU by distributing the state vector across devices using index-bit schemes, and past one node via MPI.

**qsim** is a C++ state-vector simulator using three techniques to hit state-of-the-art CPU performance: **gate fusion** (greedily combining adjacent 1- and 2-qubit gates into composite matrices of up to ~6 qubits, with `max_fused_gate_size=4` usually optimal), **AVX/FMA vectorization**, and **OpenMP multithreading**. It reaches ~40 qubits on a 90-core Xeon and ~30 qubits in 16 GB of RAM. It is integrated as a Cirq backend.

**The shared inner loop.** Both descend from the qHiPSTER pattern: applying a single-qubit gate on qubit `k` means walking the `2ⁿ` amplitudes in pairs `(i, i ⊕ 2ᵏ)`, multiplying each pair by the 2×2 gate matrix, and writing back. The outer loop iterates over groups of length `2ᵏ⁺¹`; the inner loop applies the gate to pairs separated by stride `2ᵏ`. The key portability and performance lessons from prior work:

- **Memory-bound everywhere.** Compute throughput barely matters; bandwidth and access pattern dominate.
- **Stride is the enemy.** As `k` grows the stride `2ᵏ` blows past cache lines and defeats the hardware prefetcher, so high-index target qubits run slower than low-index ones. Cache-blocking (Qiskit Aer's approach: remap gates onto low-index qubits within fixed chunks, insert SWAPs at boundaries) recovers locality.
- **Fusion adapts arithmetic intensity.** Fusing gates raises FLOP/byte toward the roofline balance point, turning several bandwidth-limited passes into one.

These three findings drive macquerel's design directly.

---

## 3. The central decision: Metal vs MLX

This is the most consequential engineering choice, so it gets its own section.

**MLX** is Apple's array framework (NumPy-like API, lazy computation graph, unified-memory arrays, op fusion via `mx.compile`, and a path to custom Metal kernels via `mx.fast.metal_kernel`). **Metal** is the bare GPU API — you write `.metal` compute shaders and manage command buffers, threadgroups, and buffers yourself.

| Dimension | MLX | Raw Metal |
|---|---|---|
| Dev velocity | High — array ops, autograd, lazy fusion for free | Low — manual shaders, buffers, command encoding |
| Unified memory | Zero-copy, automatic | Zero-copy but manual `MTLBuffer` lifetime |
| Inner-loop control | Limited by what ops/graph express | Total — own the threadgroup tiling, register use |
| Custom kernels | Supported (`mx.fast.metal_kernel`, source string) | Native |
| Multi-device CPU/GPU | Transparent | Manual |
| Maintenance burden | Low — Apple maintains the substrate | High — track Metal API churn yourself |
| Ceiling on perf | Slightly below hand-tuned Metal | Highest achievable |
| Known limits | uint32 element-count ceiling hits at **32 qubits** | None imposed by framework |

The honest summary: **MLX gets you to 90% of peak with 20% of the effort, because the workload is bandwidth-bound and there isn't much compute to hand-optimize.** When you're already saturating memory bandwidth, a hand-written Metal kernel can't beat physics — it can only match it. Raw Metal's advantage is real only for the access-pattern problems (stride, tiling), and even those can be expressed as custom Metal kernels *launched from MLX*, which keeps the unified-memory plumbing and graph machinery for free.

**Decision: MLX as the primary backend, with custom Metal kernels injected through `mx.fast.metal_kernel` for the gate-application hot path.** This is a hybrid that captures most of the upside of both. Pure Metal is kept as an optional, advanced backend behind the same interface for the cases where MLX's abstractions get in the way (notably the 32-qubit uint32 ceiling, which a custom 64-bit-indexed Metal kernel can bypass).

---

## 4. Architecture

macquerel is layered so the front end is stable while the compute engine evolves.

```
┌─────────────────────────────────────────────┐
│  Frontend API   Circuit, Gate, register DSL  │   user-facing, backend-agnostic
├─────────────────────────────────────────────┤
│  Compiler       fusion + qubit remapping      │   circuit → fused op schedule
├─────────────────────────────────────────────┤
│  Backend API    Backend protocol (ABC)        │   apply_matrix / measure / sample / expval
├──────────────┬──────────────┬─────────────────┤
│  MLXBackend  │ MetalBackend │  CPUBackend     │   interchangeable engines
│  (default)   │ (advanced)   │  (reference)    │
└──────────────┴──────────────┴─────────────────┘
```

The **Backend protocol** is deliberately the same minimal, orthogonal set cuStateVec settled on, because it's been proven sufficient:

```python
class Backend(Protocol):
    def allocate(self, n_qubits: int, dtype) -> StateVector: ...
    def apply_matrix(self, sv, matrix, targets, controls) -> None: ...
    def measure(self, sv, qubits, *, collapse=True) -> list[int]: ...
    def sample(self, sv, qubits, shots) -> Counter: ...
    def expectation_pauli(self, sv, pauli_strings) -> np.ndarray: ...
    def abs2sum(self, sv, qubits) -> np.ndarray: ...
```

### 4.1 Frontend

A Cirq/Qiskit-flavored builder. Gates are objects carrying a unitary and target/control qubits; the circuit is an ordered list with measurement boundaries. *(💭 Proposed API — illustrative of the intended surface, not yet implemented.)*

```python
import macquerel as mq

qc = mq.Circuit(n_qubits=24)
qc.h(0)
qc.cx(0, 1)
qc.rz(2, theta=0.3)
qc.measure_all()

sim = mq.Simulator(backend="mlx", dtype="complex64")
result = sim.run(qc, shots=1000)
print(result.counts)
state = sim.statevector(qc)      # returns an mx.array view, zero-copy
```

### 4.2 Compiler (the part that actually buys performance)

Two passes, both lifted directly from the prior-art lessons:

**Gate fusion.** A greedy fuser combines adjacent gates acting on overlapping small qubit sets into a single dense matrix, capped at a configurable `max_fused_qubits` (default 4, matching qsim's empirical sweet spot). This raises arithmetic intensity so each memory pass does more work — the right lever for a bandwidth-bound machine. Measurement gates at equal times act as fusion barriers.

**Qubit remapping / cache-blocking.** Because stride `2ᵏ` destroys locality for high-index target qubits, the compiler relabels qubits so that the most frequently hit targets land on low indices within a chunk that fits the GPU threadgroup memory / CPU cache, inserting logical SWAPs at chunk boundaries. This is the Doi–Horii technique adapted to Apple's memory hierarchy.

### 4.3 The MLX backend's gate kernel

Single- and two-qubit gates are applied with a custom Metal kernel registered through `mx.fast.metal_kernel`, so it joins MLX's lazy graph and runs in unified memory with no copies. The kernel implements the pairing loop: each GPU thread owns one `(i, i ⊕ 2ᵏ)` amplitude pair, loads both `complex64` values, applies the (possibly fused) matrix, and scatters back. Control qubits are handled by masking threads whose index doesn't satisfy the control bits.

State is stored as a structure-of-arrays (`real[]`, `imag[]`) rather than interleaved complex, which vectorizes cleanly and matches how the M-series GPU coalesces loads.

For the high-stride regime, the kernel reads a contiguous tile into threadgroup memory, permutes in-place there (the on-chip analogue of CUDA's `__shfl_xor`), and writes back contiguously — converting a strided global-memory pattern into a contiguous one.

### 4.4 Backend selection and the 32-qubit boundary

MLX indexes elements with uint32, which caps it at `2³²` amplitudes — i.e. it breaks at **32 qubits**. macquerel handles this with automatic backend selection: ≤31 qubits use `MLXBackend`; ≥32 qubits (memory permitting) fall through to `MetalBackend`, whose hand-written kernel uses 64-bit indexing. The CPU backend (NumPy, the reference implementation that defines correctness) is always available and is the default for ≤14 qubits where GPU launch overhead dominates.

---

## 5. Implementation specifics

This section pins down *how* each backend is actually built, including the exact libraries and API surface, and flags the constraints the MLX API imposes.

### 5.1 Libraries used

| Concern | Library / API |
|---|---|
| Arrays, dtypes, reductions, RNG | `mlx.core` (`mx.array`, `mx.sum`, `mx.cumsum`, `mx.random.*`) |
| Gate hot path | `mlx.core.fast.metal_kernel` (JIT-compiled custom Metal) |
| 64-bit / >31-qubit backend | hand-written `.metal` shaders compiled to a `.metallib`, driven via Apple's **metal-cpp** (`Metal.hpp`), bound to Python with **nanobind** |
| CPU reference backend | NumPy, which links **Accelerate/vDSP** for BLAS on macOS |
| Build | scikit-build-core + CMake for the metal-cpp extension; pure-Python wheel otherwise |

### 5.2 State representation

State is a structure-of-arrays: two `float32` arrays `real` and `imag` of length `2ⁿ`, not one interleaved `complex64` array. Metal has no native complex type, so SoA lets the kernel do plain float math with no struct unpacking, and separate contiguous real/imag streams coalesce better on the M-series memory controller. The `mx.complex64` view is reconstructed only at the API boundary when returning a statevector to the user. In MLX these are two `mx.array`s; in the Metal backend they are two `MTLBuffer`s aliased onto unified-memory allocations.

### 5.3 Reference gate path (pure MLX, v0.1)

The simplest correct backend reshapes the `2ⁿ` vector to shape `(2,)*n`, contracts the gate's matrix against the target axes with `mx.tensordot`, and transposes back. This is the textbook NumPy-simulator trick: trivially correct, and the oracle the differential tests check everything else against. It is *not* the fast path — the transpose materializes a full extra copy of the state per gate, doubling memory traffic on a bandwidth-bound workload.

### 5.4 Optimized gate path via `mx.fast.metal_kernel`

`mx.fast.metal_kernel(name, input_names, output_names, source, header="", ensure_row_contiguous=True, atomic_outputs=False)` takes the **body** of a Metal function (the signature is auto-generated from `input_names`/`output_names`) and returns a callable. The call site supplies `inputs=`, `template=[("T", dtype)]`, `grid=`, `threadgroup=`, `output_shapes=`, and `output_dtypes=`; the result joins MLX's lazy graph and runs in unified memory. MLX exposes `thread_position_in_grid` and, if referenced in the source, auto-passes `<name>_shape`, `<name>_strides`, and `<name>_ndim` for each input. *(✅ Verified — this signature and call convention match the current MLX docs.)*

For a single-qubit gate, the grid is `2ⁿ⁻¹` threads; each thread owns one amplitude pair that differs only in target bit `k`, where pair index `p` maps to indices by inserting a zero bit at position `k`. *(🧪 The kernel body below is pseudocode — the bit-insertion index math and the matrix-multiply have not been compiled or run, and are the parts most likely to need debugging against a real build.)*

```metal
// 🧪 PSEUDOCODE — not compiled or tested. Specification to validate.
uint p = thread_position_in_grid.x;
uint k = targets[0];
uint low = (1u << k) - 1u;
uint i0 = ((p & ~low) << 1) | (p & low);   // insert 0 at bit k
uint i1 = i0 | (1u << k);

float a0 = real_in[i0], b0 = imag_in[i0];
float a1 = real_in[i1], b1 = imag_in[i1];

// mat = [m00, m01, m10, m11], each complex -> 8 floats
real_out[i0] = mat[0]*a0 - mat[1]*b0 + mat[2]*a1 - mat[3]*b1;
imag_out[i0] = mat[0]*b0 + mat[1]*a0 + mat[2]*b1 + mat[3]*a1;
real_out[i1] = mat[4]*a0 - mat[5]*b0 + mat[6]*a1 - mat[7]*b1;
imag_out[i1] = mat[4]*b0 + mat[5]*a0 + mat[6]*b1 + mat[7]*a1;
```

A fused k-qubit gate (k≤4) generalizes this: `2ⁿ⁻ᵏ` threads, each gathering `2ᵏ` amplitudes by inserting zeros at the k target-bit positions, doing a small mat-vec, and scattering back. **Controls** are a thread mask, not a separate kernel: test `(i0 >> c) & 1` for each control qubit and copy the pair through unchanged if any control bit is unset.

**Important constraint — no true in-place update.** MLX declares all kernel inputs `const device`, so a kernel cannot write back into its input arrays through the stable API (an open MLX issue, #2547, proposes an `input_rw_status` flag, but it has not landed). *(✅ Verified against MLX issue #2547.)* macquerel therefore uses **double-buffering**: each gate reads `real_in/imag_in` and writes fresh `real_out/imag_out`, and the simulator ping-pongs the two buffers between gates. Because every amplitude belongs to exactly one pair, every output element is written, so this is correct without zero-initialization. The cost is holding two state vectors — one extra `2ⁿ × 8` bytes — which lowers the maximum simulable qubit count by one. (The raw Metal backend, which controls its own buffers, *can* update in place and avoids this.)

**High-stride targets.** For large `k` the global-memory stride `2ᵏ` defeats the prefetcher. The kernel reads a contiguous tile into `threadgroup` memory, permutes on-chip, and writes back contiguously — the Apple analogue of CUDA's `__shfl_xor`. *(💭 Design judgment — gated behind a benchmark since MLX's default coalescing may already suffice; see open questions.)*

### 5.5 Measurement and sampling

These use MLX array ops, not custom kernels. The marginal probability of a qubit is `mx.sum(real**2 + imag**2, axis=...)` over the complementary axes after reshaping to `(2,)*n`. For `shots` samples, macquerel computes the `2ⁿ` probability vector once and draws via `mx.random.categorical(logits, num_samples=shots)` over the log-probabilities. *(✅ `mx.random.categorical(logits, ..., num_samples=...)`, `mx.cumsum`, and `mx.random.key` verified against MLX docs; MLX has **no** `searchsorted`, also verified.)* (A `mx.cumsum` + uniform-draw + comparison path is the fallback when the full logits array is too large; lacking `searchsorted`, this uses an `argmax` over the comparison mask rather than a binary search — 💭 design judgment, not yet benchmarked.) **Collapse** is a masked multiply zeroing amplitudes inconsistent with the outcome, followed by a renormalization divide — both elementwise MLX ops. RNG uses MLX's explicit `key` mechanism (`mx.random.key(seed)`) so runs are reproducible.

### 5.6 Raw Metal backend

This is a compiled extension, not Python-driven Metal. It uses the same metal-cpp substrate MLX itself is built on: hand-written `.metal` shaders compiled to a `.metallib`, command-buffer and `MTLBuffer` management in Objective-C++/metal-cpp, exposed to Python via nanobind. Its sole advantage over the MLX kernel is **64-bit indexing** (`size_t` thread/index math) to exceed `2³²` amplitudes, plus genuine in-place updates. For ≤31 qubits it is no faster than the MLX kernel, since both saturate memory bandwidth.

---

## 6. Optimization techniques (lessons from cuQuantum and qsim)

A survey of how cuStateVec and qsim reach state-of-the-art performance. Each lesson is tagged with whether it *validates* an existing macquerel choice or is a *new* technique to adopt. The recurring theme: this workload is memory-bandwidth-bound, so the wins come from moving less data, not computing faster.

### 6.1 Gate fusion as an arithmetic-intensity knob *(validates §4.2; refines it)*
Both simulators fuse adjacent 1- and 2-qubit gates into composite matrices — qsim greedily up to ~6 qubits (`max_fused_gate_size=4` usually optimal ✅), cuStateVec via its operator-queue updater. The ARM/qsim work frames fusion precisely as *adapting arithmetic intensity to the roofline balance point*: on a bandwidth-bound machine, fusion's value is turning N memory passes into one, not doing math faster. **Refinement for macquerel:** auto-tune fusion width per chip against measured bandwidth/FLOP ratio rather than hardcoding 4 (💭).

### 6.2 Structure-of-arrays layout *(strongly validates §5.2)*
A batched-simulator layout study measured separate real/imag arrays (SoA) at **up to 6.9× over interleaved-contiguous and 1.9× over interleaved** complex storage. This is a larger effect than I had weighted; it confirms SoA is the single most important data-layout decision, and the ARM "VLEN-adaptive memory layout" work reinforces that layout dominates compute.

### 6.3 Specialize kernels by gate structure *(NEW — highest-value addition)*
cuStateVec has distinct optimized paths for **dense, diagonal, and anti-diagonal** matrices, plus a dedicated Pauli-rotation primitive and Pauli-basis expectation. The key insight for a bandwidth-bound simulator:

- **Diagonal gates** (Z, S, T, RZ, CZ, phase, and any controlled-phase) require **no amplitude pairing at all** — they are a single elementwise complex multiply. Each amplitude is read and written once instead of being gathered into pairs, which is roughly a **2× bandwidth win** and has *no stride problem* regardless of target qubit. This is the most valuable single change to the kernel design in §5.4.
- **Permutation gates** (X, SWAP, CNOT) are pure index relabeling — no arithmetic, just a gather/scatter, and for SWAP/CNOT often expressible as an index-bit swap (see §6.4).

**For macquerel:** the compiler should classify each (possibly fused) gate as `diagonal | permutation | dense` and dispatch to a specialized kernel. A diagonal kernel is `out[i] = phase[i & mask] * in[i]` over the full grid — trivial to write and verify, and a big win for phase-heavy circuits like QFT and QAOA.

### 6.4 Index-bit permutation as a first-class primitive *(refines §4.2)*
cuStateVec exposes APIs to swap state-vector index bits, used both for distributed simulation and to keep frequently-hit qubits on low (cache-friendly) indices. macquerel's cache-blocking (§4.2) does the same thing internally; the lesson is to **expose index-bit permutation as an explicit backend operation** rather than burying it in the compiler, so it composes with measurement, with permutation-gate handling (§6.3), and with the high-stride tile trick.

### 6.5 Prepare-once, sample-many *(makes §5.5 an explicit guarantee)*
Both simulators separate the expensive `O(2ⁿ)` state preparation from sampling, drawing many bitstrings from the *final* state without re-running the circuit (qsim's `run` vs `compute_amplitudes` ✅; cuStateVec's batched sampling). macquerel should make "prepare once, sample many" an explicit API guarantee, and **autotune the shot batch size** by doubling until throughput plateaus (the approach the Tsim simulator uses).

### 6.6 Where NOT to spend effort *(saves wasted work)*
- **SIMD/NEON micro-optimization of the gate kernel.** A SIMD encoding study found vectorization helps only the compute-bound portions; full-state-vector gate application is memory-bound and gains essentially nothing. The CPU backend should rely on NumPy/Accelerate and not hand-write NEON intrinsics for gates.
- **The M5 Neural Accelerators.** They accelerate matmul; our bottleneck is bandwidth, so don't expect them to help gate application (the benchmarks in §9 should confirm this).
- **FP16 storage.** The Tensor-Core emulation literature is explicit: FP16's 5-bit exponent underflows at the `2^(-n/2)` amplitude scale and its mantissa is insufficient — FP32 is the floor for general circuits. Our `complex64` default is correct; mixed FP16-storage/FP32-accumulate is at best a v2 research item, not a default.

### 6.7 Batched small-circuit simulation *(NEW — v2 track)*
cuStateVec's `applyMatrixBatched` and the VE study show that for many *small* circuits (QML/VQE parameter sweeps), packing them into one contiguous block and one kernel launch beats per-circuit calls. Unified memory makes this natural on Apple Silicon. Proposed as a v2 `BatchedSimulator` (see §10 Roadmap).

### Summary of changes pulled into the design
| Lesson | Status | Where it lands |
|---|---|---|
| Diagonal/permutation gate fast path | NEW | new kernel class in §5.4 |
| Per-chip fusion-width autotuning | NEW | §4.2 / install-time tuning |
| Index-bit permutation as explicit primitive | refine | backend protocol §4 |
| Prepare-once/sample-many + batch autotune | refine | §5.5 |
| SoA layout | validated | §5.2 (keep) |
| Avoid SIMD/FP16/Neural-Accelerator chasing | validated | §5, §9 (don't pursue) |
| Batched small circuits | NEW | v2 roadmap §10 |

---

## 7. Memory and capacity

`complex64` is the default (8 bytes/amplitude); `complex128` is available for verification. State-vector size is `2ⁿ × 8` bytes:

| Qubits | complex64 size | Fits on |
|---|---|---|
| 24 | 256 MB | any M-series |
| 28 | 4 GB | 16 GB Mac |
| 30 | 16 GB | 32 GB Mac |
| 32 | 64 GB | 64 GB Mac (Metal backend) |
| 34 | 256 GB | 192 GB Mac Studio |

Unified memory is the headline advantage: a 64 GB MacBook can simulate state vectors that won't fit a 24 GB discrete GPU, with no copy tax. An optional memory-mapped backend (state vector backed by an NVMe file) can extend past DRAM at a large bandwidth penalty — useful for one-off large runs, off by default.

---

## 8. Test plan

Correctness first, performance second. Nothing ships until the reference and accelerated backends agree.

**Unit / correctness**
- **Gate identities:** `H·H = I`, `X·X = I`, `S·S = Z`, controlled-gate truth tables, parametric rotations against closed forms.
- **Differential testing:** every backend (CPU, MLX, Metal) must produce statevectors agreeing to `1e-5` (complex64) / `1e-12` (complex128) on a fuzzed corpus of random circuits (random gates, targets, controls, depths). This is the single most important test — it catches indexing, stride, and control-mask bugs.
- **Cross-simulator validation:** statevectors and sampled distributions checked against Qiskit Aer and qsim on circuits up to ~20 qubits.
- **Fusion equivalence:** circuits run with fusion on/off must yield identical statevectors — proves the compiler is semantics-preserving.
- **Remapping equivalence:** same property for the qubit-remapping pass including inserted SWAPs.
- **Measurement statistics:** sampled distributions pass a χ² goodness-of-fit against analytic probabilities (e.g. GHZ gives ~50/50 on all-zeros / all-ones); collapse leaves a correctly normalized post-measurement state.

**Known-answer circuits**
- GHZ, QFT (compare to analytic phases), Grover (amplitude peaks on the marked state), Quantum Volume circuits, and random circuit sampling (cross-entropy benchmarking, the same yardstick qsim uses).

**Property-based**
- Norm preservation (`‖ψ‖ = 1`) after every gate, within tolerance.
- Unitarity of fused matrices.

**Edge cases**
- 1-qubit circuits, all-identity circuits, the 31→32 qubit backend-switch boundary, max-controls gates, empty circuits.

CI runs the full suite on each supported chip generation available in the runner pool; the large-qubit tests are gated behind a nightly job.

---

## 9. Benchmarking plan

Borrowing cuStateVec's split between **API-level** and **circuit-level** benchmarks.

**API-level (microbenchmarks).** Single-gate apply throughput as a function of (a) target qubit index — to expose the stride/locality cliff — and (b) qubit count. Reported as effective GB/s of memory bandwidth, so we can compare against the chip's theoretical peak and see how close to the roofline we are. This is the metric that matters for a bandwidth-bound workload.

**Circuit-level (macrobenchmarks).** Wall-clock for full circuits: QFT, random circuit sampling (the supremacy-style benchmark), QAOA layers, and Quantum Volume, swept from 20–32 qubits.

**Comparisons.**
- macquerel MLX vs macquerel Metal vs macquerel CPU — internal backend bake-off.
- vs **qsim** CPU path on the same Mac (the most honest baseline; qsim is the incumbent CPU simulator and runs on arm64).
- vs **Qiskit Aer** statevector and **PyTorch MPS** as reference points.
- Cross-generation scaling: M1 → M2 → M3 → M4 → M5 (the M5's GPU Neural Accelerators won't help much here since the workload is bandwidth- not matmul-bound, which the benchmarks should confirm).

**Sweeps.** `max_fused_qubits` ∈ {1,2,3,4,5,6} to reproduce/validate the "4 is optimal" finding on Apple hardware; complex64 vs complex128; fusion and remapping on/off to quantify each pass's contribution.

**Methodology.** Pin to a thermally steady state (warm-up runs, watch for the thermal cliff that prior M4 work documents at 27–30 qubits), report median and IQR over ≥10 runs, fix clock/power mode, record chip + memory config with every result. Output a roofline plot (achieved vs peak bandwidth) per gate type and a speedup table vs qsim.

---

## 10. Roadmap

**v0.1** — CPU reference + MLX backend, single/two-qubit gates, fusion, measurement, sampling. Differential test harness green.
**v0.2** — Metal backend with 64-bit indexing (>31 qubits), cache-blocking remap pass, expectation values on a Pauli basis.
**v0.3** — Cirq/Qiskit front-end adapters so existing circuits run unmodified.
**v1.0** — Stabilized backend API, full benchmark suite published, docs.
**v2 (exploratory)** — Noise channels / density matrices, mmap out-of-core backend, multi-Mac over Thunderbolt.

---

## 11. Open questions

- Does MLX's `mx.compile` graph fusion meaningfully help once the gate kernel is already custom, or is it redundant with our circuit-level fusion?
- Where exactly is the GPU-vs-CPU crossover qubit count on each chip — and is it stable enough to hardcode, or should it be auto-tuned at install time?
- Is the threadgroup-memory tile-permute worth its complexity, or does MLX's default coalescing already get close enough on the M-series memory subsystem?
- Can we lift the 32-qubit boundary inside MLX itself (upstream contribution) rather than maintaining a parallel Metal kernel?

---

## 12. References

**cuStateVec / cuQuantum**
- NVIDIA, *cuStateVec: A High-Performance Library for State Vector Quantum Simulators* (documentation, Overview, and API reference). https://docs.nvidia.com/cuda/cuquantum/latest/custatevec/index.html — primitive set (apply_matrix, measure, expectation on Pauli basis, sample), the "Ex" State Vector Updater, dense/diagonal/anti-diagonal matrix paths, multi-GPU index-bit distribution.
- NVIDIA Technical Blog, *Accelerating Quantum Circuit Simulation with NVIDIA cuStateVec* (2023). https://developer.nvidia.com/blog/accelerating-quantum-circuit-simulation-with-nvidia-custatevec/ — integration with Cirq/qsim, measurement/expectation API design.
- NVIDIA/cuQuantum, *cuStateVec Library* (DeepWiki). https://deepwiki.com/NVIDIA/cuQuantum/4-custatevec-library — API-level vs circuit-level benchmark split (nv-quantum-benchmarks), MPI multi-node scaling.

**qsim**
- Google Quantum AI, *qsim* and *qsim/qsimh overview*. https://quantumai.google/qsim — gate fusion + AVX/FMA + OpenMP design, ~40 qubits / ~30 qubits in 16 GB, Cirq integration.
- Google Quantum AI, *Choosing hardware for your qsim simulation*. https://quantumai.google/qsim/choose_hw — `max_fused_gate_size = 4` empirical optimum.
- Google Quantum AI, *qsim::BasicGateFuser / Fuser class references*. https://quantumai.google/reference/cc/qsim/class/qsim/basic-gate-fuser — greedy fusion of 1- and 2-qubit gates, measurement gates as fusion barriers.

**Inner loop, memory access, and cache-blocking**
- M. Smelyanskiy et al., *qHiPSTER: The Quantum High Performance Software Testing Environment*, arXiv:1601.07195. https://arxiv.org/pdf/1601.07195 — the `(i, i ⊕ 2ᵏ)` amplitude-pairing loop, outer/inner stride structure, controlled-gate kernels.
- *Accelerating State-Vector Quantum Simulation on Integrated GPUs via Cache Locality Optimization*, arXiv:2605.15098. https://arxiv.org/html/2605.15098 — stride/locality cliff, Doi–Horii cache-blocking (gate remapping onto low-index qubits with boundary SWAPs).
- *QVecOpt: An Efficient Storage and Computing Optimization...*, arXiv:2508.15545. https://www.arxiv.org/pdf/2508.15545 — amplitude-pairing single-pass mechanism, sliding-window/block storage.
- *Prototype of a Batched Quantum Circuit Simulator for the Vector Engine*, ACM (10.1145/3624062.3624226). https://dl.acm.org/doi/fullHtml/10.1145/3624062.3624226 — contiguous vs strided vs gather-scatter kernel tradeoffs, in-register shuffle analogy.
- *High-performance Vector-length Agnostic Quantum Circuit Simulations on ARM Processors*, arXiv:2602.09604. https://arxiv.org/pdf/2602.09604 — gate fusion as arithmetic-intensity adaptation under the roofline model; qsim on ARM (Grace, Graviton3, A64FX).

**Optimization techniques (§6)**
- *cuQuantum SDK: A High-Performance Library for Accelerating Quantum Science*, arXiv:2308.01999. https://arxiv.org/pdf/2308.01999 — index-bit swap APIs for qubit reordering, gate-fusion design, distributed SV layout.
- *Prototype of a Batched Quantum Circuit Simulator for the Vector Engine*, ACM 10.1145/3624062.3624226. https://dl.acm.org/doi/fullHtml/10.1145/3624062.3624226 — measured SoA layout advantage (up to 6.9× / 1.9×), `applyMatrixBatched` / `MAP_TYPE_MATRIX_INDEXED` batched simulation.
- *Low-Level and NUMA-Aware Optimization for High-Performance Quantum Simulation*, arXiv:2506.09198. https://arxiv.org/pdf/2506.09198 — comparison of SIMD/FMA/threading optimizations across qsim, QuEST, PennyLane Lightning, etc.
- *Accelerating Quantum State Encoding with SIMD*, arXiv:2604.06270. https://arxiv.org/pdf/2604.06270 — SIMD helps compute-bound encoding but not memory-bound full-state gate application (on Apple Silicon).
- *Quantum Circuit Simulation by SGEMM Emulation on Tensor Cores*, arXiv:2303.08989. https://arxiv.org/pdf/2303.08989 — FP16 exponent underflow / mantissa insufficiency; FP32 needed for general circuits.
- NVIDIA cuQuantum, *cuStateVec Ex examples* (Pauli rotation, index-bit permutation, diagonal/anti-diagonal paths). https://docs.nvidia.com/cuda/cuquantum/latest/custatevec/examples.html
- Google Quantum AI, *The Cirq Interface* (`run` vs `compute_amplitudes` — prepare-once/sample-many). https://quantumai.google/qsim/cirq_interface
- *Tsim: Fast Universal Simulator for Quantum Error Correction*, arXiv:2604.01059. https://arxiv.org/html/2604.01059v1 — shot-batch-size autotuning by doubling until throughput plateaus.

**Apple Silicon: MLX, Metal, unified memory**
- Apple, *Get started with MLX for Apple silicon*, WWDC25. https://developer.apple.com/videos/play/wwdc2025/315/ — custom Metal kernels as lazy graph nodes (`mx.fast.metal_kernel`), unified memory, lazy evaluation.
- ml-explore, *MLX: An array framework for Apple silicon* (GitHub). https://github.com/ml-explore/mlx — lazy computation, dynamic graphs, unified shared-memory arrays, CPU/GPU without transfers.
- Apple Machine Learning Research, *Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU* (2025). https://machinelearning.apple.com/research/exploring-llms-mlx-m5 — M5 TensorOps/Neural Accelerators (relevant to why they *don't* help a bandwidth-bound workload).
- *Profiling Apple Silicon Performance for ML Training*, arXiv:2501.14925. https://arxiv.org/pdf/2501.14925 — unified memory capacity (up to 128/192 GB) vs lower raw compute throughput tradeoff.

**MLX API (implementation specifics in §5)**
- MLX, *Custom Metal Kernels* and *mlx.core.fast.metal_kernel*. https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html and https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.metal_kernel.html — kernel signature, `inputs/template/grid/threadgroup/output_shapes/output_dtypes` call convention, auto-passed `_shape/_strides/_ndim`, `ensure_row_contiguous`, `atomic_outputs`.
- ml-explore/mlx, issue #2547, *read+write input arguments for custom kernels*. https://github.com/ml-explore/mlx/issues/2547 — confirms inputs are `const device` by default (the constraint that forces double-buffering).
- MLX, *mlx.core.random.categorical* and *mlx.core.cumsum*. https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.random.categorical.html — sampling API used for shots.

**The key Apple-Silicon-specific result**
- *A Controlled Study of Memory Hierarchy Transitions in Quantum Circuit Simulation on Apple M4 Pro Unified Memory Architecture*, arXiv:2605.08792. https://arxiv.org/html/2605.08792 — gates are memory-bound at ≤0.38 FLOP/byte (0.875 upper bound), the **MLX uint32 ceiling at 32 qubits**, the 27–30 qubit thermal cliff, and direct-index scatter-write backends including Metal compute shaders.