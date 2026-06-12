# Changelog

All notable changes to this project are documented here, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-06-12

### Changed

- Apply narrow control-free density-matrix unitaries as a single
  `kron(U, conj(U))` superoperator pass over the paired ket+bra axes (Step 40),
  one full-state pass instead of two, with kind-aware eligibility (diagonal k<=4,
  monomial k<=3, dense k<=2): a small consistent win (1.0-1.09x) across backends.
- Replace the CPU backend's tensordot dense apply with an in-place chunked
  gather/GEMM/scatter (Step 39): peak memory falls from ~3x to ~1.03x of the state
  size (a 28-qubit GHZ now peaks at 2.05 GiB) and runtime improves in every
  measured cell, up to 1.70x on random@24q. The controlled path reuses the same
  in-place routine on its control-slice view.
- Compute `DensityMatrixSimulator.expectation_pauli` as a monomial gather (Step 38):
  `tr(rho P) = sum_i phase(i) * rho[i, i ^ mask]` off the zero-copy host view,
  replacing a full `4**n` readback plus a state-sized copy per term — 6.3x faster
  at 2.9x less peak (n=14, 4 terms, Metal). `density_matrix()` gains an opt-in
  `copy=False` zero-copy view.
- Cut MLX peak memory from 19-25x to 3-5x of the state size (Step 36): monomial
  (permutation) gates now use a register-resident group-per-thread Metal kernel
  instead of materializing full-width gather-index intermediates in the lazy graph;
  the async-eval cadence tightens to every 2 gates with two-deep backpressure above
  26 effective qubits; and observation boundaries return the buffer pool to the OS
  when it exceeds an eighth of unified memory. MLX now reaches 29-30q statevectors
  and 15-qubit density matrices without swap, and every measured runtime cell
  improved (GHZ@24-28q 6.5-7.9x, QFT 2.1-2.6x, random 1.2-1.8x).
- Refresh the large statevector and memory benchmark artifacts after the v0.3.x RAM
  line: the MLX statevector series now extends to 30 qubits (previously skipped for
  memory) and the density-matrix series to n=15; measured peak multipliers drop to
  ~1.03x (CPU) and ~3-5x (MLX) of the state size. MLX remains capped at 30 qubits by
  the upstream int32 `ShapeElem` limit (arrays of 2**31+ elements are not
  representable); Metal covers 31-33q.

### Added

- Add `TrajectorySimulator` (Step 37): Monte-Carlo wavefunction simulation of noisy
  circuits at statevector (`2**n`) memory — each trajectory samples one Kraus
  operator per channel with its Born probability and renormalizes, exact in
  expectation with error ~ `1/sqrt(trajectories)`. Built-in channels jump-sample
  from a single `abs2sum` marginal (no state copies, Metal-compatible), so noisy
  simulation reaches the full statevector range instead of the density matrix's
  n=16 cap: a noisy 30-qubit GHZ runs at 10.3 s/trajectory on Metal with a
  footprint constant in trajectory count.
- Add `DensityMatrixSimulator`: noisy circuit simulation over the vectorized density
  matrix (a `4**n` doubled statevector), reusing the CPU/MLX/Metal backends unchanged —
  unitaries apply to the ket and bra axes, Kraus channels as one dense superoperator per
  channel. API: `density_matrix`, `probabilities` (diagonal-only readback), `run`,
  `expectation_pauli`, `purity`. Auto backend selection at the doubled qubit count;
  ceilings at 15 qubits (MLX) and 16 qubits (Metal, a 32 GiB state).
- Add Kraus-operator noise channels on `Circuit`: `bit_flip`, `phase_flip`,
  `depolarizing`, `amplitude_damping`, `phase_damping`, and arbitrary (multi-qubit)
  channels via `kraus(qubits, operators)`, validated for trace preservation at build
  time. Channels act as gate-fusion barriers; the statevector `Simulator` and
  `BatchedSimulator` reject noisy circuits with a pointer to `DensityMatrixSimulator`.
- Add per-cell GPU-utilization sampling to `benchmarks/bench_memory.py` (device-wide
  busy-percent from the IORegistry, no sudo) with a utilization panel in the memory
  chart, and document GPU usage: every simulation runs on Apple Silicon's single
  GPU — MLX and Metal are two paths to the same device — and the measured curves
  show cpu cells at the idle baseline and the GPU saturating from 24q (MLX) / 29q
  (Metal) / n=13 density-matrix cells.
- Add a "How it works" documentation section: a gentle but thorough introduction to
  statevector simulation and the library's internals — gate application and gate-kind
  fast paths, the three backend designs, the optimization playbook, and the vectorized
  density-matrix noise simulation — with diagrams and literature references.
- Add `benchmarks/bench_density.py` (runtime across backends/qubit counts on noisy GHZ
  and random-brickwork circuits) and a density-matrix series in
  `benchmarks/bench_memory.py`; both budget-gate cells at min(0.45 × RAM, 64 GiB).
  Measured: Metal stays on the theoretical `4**N × 8 B` line (32.2 GiB at N=16) and
  runs a noisy 16-qubit GHZ in 6.0 s.

### Fixed

- Fix sampled bitstring order for `measure(qubits)` lists that permute 3+ qubits out of
  ascending order (e.g. `measure([1, 2, 0])`): every sampler transposed the marginal by
  `argsort(qubits)` where the rank permutation (its inverse) is required, producing bits
  in the wrong positions and disagreeing with the `measure()` collapse path. Sorted
  lists, `measure_all()`, and two-qubit measure lists were unaffected.

## [0.2.2] - 2026-06-11

### Added

- Add `BatchedSimulator` (Step 31): parameter sweeps of same-width circuits run as one
  batched evolution — one batched matmul per gate position instead of one run per
  circuit — with NumPy and MLX engines and automatic engine routing. Measured 2-47x
  over a per-circuit loop on VQE-style sweeps (`benchmarks/bench_batched.py`).
- Add a custom MLX dense-gate kernel via `mx.fast.metal_kernel` (Step 33), replacing
  `mx.tensordot`'s internally-permuting dense path with the Metal backend's
  group-per-thread design: random@22-28q 1.16-1.61x.
- Add opt-in per-chip backend-tier autotuning (Step 35): `MACQUEREL_BACKEND_TIERS=<int>`
  pins the CPU tier boundary, `=auto` measures the CPU/GPU crossover once and caches it.

### Changed

- Replace the MLX diagonal gather-table path with a broadcast elementwise phase multiply
  (Step 32): qft@22-28q 2.5-4.3x, closing most of the MLX/Metal QFT gap.
- Lower the Metal small-n floor (Step 34): process-wide shared device/queue/pipelines
  (backend construction 7.5 ms -> 30 us), pooled state-buffer allocation, fewer
  per-dispatch ObjC calls, and backend-instance reuse on the `auto` path.
- Retune automatic backend selection to CPU through 15 qubits (was 16): after Step 34,
  Metal wins three of the four benchmark circuits at 16q.

## [0.2.1] - 2026-06-11

### Added

- Add native Metal command-buffer batching and kind-specialized kernels for lower dispatch
  overhead and faster dense/monomial gate application.
- Add MLX axis-order tracking and periodic `mx.async_eval` cadence to reduce dense-gate
  transpose work and eliminate the 28-qubit lazy-graph memory cliff.
- Add diagonal-run wide fusion, commutation-aware fusion grouping, and backend/qubit-aware
  default fusion widths.
- Add opt-in qubit remapping via `MACQUEREL_REMAP=1`; it remains disabled by default after
  A/B benchmarking showed losses on the measured backends.
- Add benchmark artifacts and documentation for the shipped Steps 21-30 performance line,
  including per-step A/B data under `benchmarks/data/steps/`.

### Changed

- Retune automatic backend selection to use CPU through 16 qubits, Metal at 17+ qubits
  when available, and MLX as the 17-30 qubit fallback when Metal is absent.
- Refresh large statevector and fusion-width benchmark JSON/plot artifacts after the
  backend and compiler optimization line.
- Reorganize `docs/plan.md` so it tracks remaining work while `docs/plan_completed.md`
  records shipped steps, commit IDs, and measured A/B results.

### Fixed

- Fix quality-check issues in benchmark scripts, backend typing, and GPU differential tests
  so Ruff, formatting, `ty`, and the full test suite pass together.

## [0.2.0] - 2026-06-06

### Added

- Add Quantum Volume macrobenchmarks using Haar-random SU(4) model circuits.
- Add `qsimcirq` support to the cross-simulator statevector benchmark harness, with graceful
  degradation when qsim is not installed.
- Add `Simulator(batch_shots=...)` and backend `sample(..., batch_shots=...)` support. MLX
  sampling can autotune `mx.random.categorical` batch size, while explicit integer values pin
  chunk size.
- Add opt-in fusion-width autotuning via `MACQUEREL_FUSION_WIDTH=auto`, with in-memory and
  disk caching. The zero-config fusion default remains width 4.
- Add a fusion-width sweep benchmark and committed plot/data showing why width 4 is the
  normalized aggregate default across the measured MLX tier.
- Add Quantum Volume, random-circuit-sampling, MLX batched-sampling, and fusion-width
  resolver/autotuner tests.
- Add `docs/plan_completed.md` and reorganize the implementation plan so `docs/plan.md`
  tracks only remaining v0.3 work.

### Changed

- Improve `bench_statevector.py` with subprocess-isolated timing, memory-budget skips,
  per-cell logging, and JSON checkpointing so large-qubit benchmark runs do not contaminate
  later measurements or drive the host into swap.
- Regenerate cross-simulator and fusion-width benchmark artifacts with corrected isolated
  measurements.
- Extend the backend protocol and CPU/Metal samplers to accept `batch_shots` for interface
  parity.

### Fixed

- Fix MLX permutation handling for composed non-involutive permutation gates produced by
  fusion.
- Fix MLX monomial/permutation handling so phased permutation matrices preserve their row
  phases instead of doing a phase-dropping gather.
- Validate non-positive fusion widths: explicit `fuse_gates(..., max_fused_qubits=...)`
  values now raise, while invalid env/cache values fall back safely.

## [0.1.1] - 2026-06-02

### Fixed

- Use an absolute logo URL in the README so it renders on the PyPI project page
  (relative image paths don't resolve there).

## [0.1.0] - 2026-06-02

### Added

- Initial release: quantum state-vector simulator for Apple Silicon.
- `Circuit` builder with a chainable gate API and `Simulator` with `run()` / `statevector()`.
- CPU backend (NumPy reference, tensordot reshape).
- MLX backend for Apple Silicon GPU acceleration (17–30 qubits), with graceful fallback.
- Metal backend (PyObjC driver, 64-bit indexing, in-place updates) reaching 31–33 qubits past
  MLX's int32 ceiling.
- Automatic backend selection (CPU ≤16q, MLX 17–30q, Metal 31q+).
- Gate-fusion compiler and diagonal/permutation/dense gate classification.
- Cirq and Qiskit import adapters (optional extras).

[Unreleased]: https://github.com/guenp/macquerel/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/guenp/macquerel/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/guenp/macquerel/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/guenp/macquerel/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/guenp/macquerel/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/guenp/macquerel/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/guenp/macquerel/releases/tag/v0.1.0
