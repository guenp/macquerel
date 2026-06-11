# The backends

All three backends implement the same small protocol:

```text
 allocate(n)            → state            (|0…0⟩, 2ⁿ complex64 amplitudes)
 apply_matrix(state, U, targets, controls) → state
 measure(state, qubits, collapse=…)        → outcomes      (projective)
 sample(state, qubits, shots)              → Counter       (Born-rule sampling)
 abs2sum / expectation_pauli / to_numpy    → host-side observations
```

They are differential-tested against each other to ~1e-5, so the choice never affects
*what* you compute — only how fast, and how many qubits fit. What actually differs is
**where the amplitudes live and who sweeps them**:

```text
              CPU                    MLX                      Metal
        ┌──────────────┐     ┌────────────────┐      ┌──────────────────┐
 state  │ NumPy array  │     │ mx.array in    │      │ one MTLBuffer in │
 lives  │ in RAM       │     │ unified memory │      │ unified memory   │
        └──────┬───────┘     └───────┬────────┘      └────────┬─────────┘
               │                     │                        │
 gates  CPU cores sweep it    GPU kernels from a       GPU kernels, encoded
 run    (tensordot/BLAS)      lazily-built graph       eagerly, in place
               │                     │                        │
 per    out-of-place:         out-of-place: every      in place: zero extra
 gate   ~3× state in temps    gate makes a new         copies (1× state,
        at peak               buffer (≥2× state)       measured)
               │                     │                        │
 ceiling  RAM & patience      30 qubits (int32         33 qubits (64 GiB,
          (~24q practical)    shape limit)             64-bit indexing)
```

This page explains each backend's design; the measured performance comparison and
tuning guidance live in [Backends: CPU vs MLX vs Metal](../backends.md).

## Backend selection

`Simulator()` (i.e. `backend="auto"`) routes by qubit count using measured crossover
points, not guesses:

```text
   n ≤ 15      → cpu     (state ≤ 0.25 MB: GPU dispatch latency would dominate)
   n ≥ 16      → metal   (if installed)
                 mlx     (fallback, up to its 30-qubit ceiling)
                 cpu     (last resort)
```

The boundary is a property of the chip; `MACQUEREL_BACKEND_TIERS=auto` re-measures
the CPU/GPU crossover once on *your* machine and caches it, and an integer value pins
it. The reasoning is a fixed-vs-scaling cost argument: GPU work per gate doubles with
every qubit while launch overhead stays constant, so there is always a crossover —
see [the performance page](../backends.md#why-metal-trails-at-low-qubit-counts) for
the measurement.

## CPU: the NumPy reference

`backends/cpu.py` is deliberately the simplest implementation — "correctness over
speed" — and the semantic reference the GPU backends are tested against. A dense gate
is one `np.tensordot` contracting the gate tensor against the target axes of the
`(2,)*n` view, followed by a transpose to put the axes back; diagonal gates take a
broadcast in-place multiply instead (one read + one write per amplitude, no temporary
— required for the wide fused diagonals the compiler emits, where a 10-qubit diagonal
as a dense 1024×1024 tensordot would be slower than the gates it replaced).

Its one structural weakness is that `tensordot` is out-of-place: the contraction
materializes a full-size output and the transpose another, so peak memory runs ~3×
the state (the measured multiplier in `benchmarks/data/memory.json`). Practical to
~24 qubits; beyond that it is GPU territory.

## MLX: a lazy compute graph on the GPU

[MLX](https://github.com/ml-explore/mlx) is Apple's machine-learning array framework.
Arrays live in unified memory and operations are **lazy**: `mx.multiply`, `mx.matmul`
etc. don't execute — they append nodes to a compute graph, which runs only when
something forces evaluation (`mx.eval`, or reading values back). For a circuit this
is a natural fit: the whole gate sequence becomes one graph that MLX's scheduler
fuses and executes in a handful of kernels, and nothing synchronizes with the host
until you ask for results.

```text
   h(0)      cx(0,1)     rz(1,θ)         statevector()
    │           │           │                  │
    ▼           ▼           ▼                  ▼
  [node] ──► [node] ──► [node] ──► … ──►  mx.eval(graph)
                (graph building: microseconds,        (GPU executes the
                 nothing runs yet)                     whole fused graph)
```

How the gate kinds map onto MLX (`backends/mlx_backend.py`):

- **Diagonal** — reshape the state so each target qubit gets its own length-2 axis
  (gaps between targets collapsed, so the view stays ≤ 2k+1 dimensional) and
  broadcast-multiply by the `(2,)*k` diagonal. One elementwise kernel, no index
  table.
- **Permutation** — build the per-amplitude source index *on the GPU* with
  `mx.arange` + bitwise ops (no host-side 2ⁿ table), gather, then apply the per-row
  phases. The index math + gather is wrapped in `mx.compile` so it fuses into one
  kernel.
- **Dense** — a custom kernel written through `mx.fast.metal_kernel`, with the same
  one-thread-per-group design as the native Metal backend (the gate width is baked
  into the kernel source so the per-row loops unroll into registers). This replaced
  `mx.tensordot`, whose internal permutation of the state was the dominant cost on
  scattered-target circuits. Gates wider than 6 qubits (which fusion never emits)
  fall back to tensordot.

Two structural costs are inherent to the design. First, MLX arrays are immutable, so
**every gate writes a fresh buffer** — the state is effectively double-buffered, and
a long lazy graph can keep many full-width intermediates alive. The backend manages
this with a cadence: every 16 gates (on states ≥ 24 qubits) it calls
`mx.async_eval`, which starts executing the graph *without blocking* gate encoding,
letting MLX retire and free earlier intermediates. Without it, deep 28-qubit
circuits kept O(depth) temporaries and drove the machine into swap. Second, MLX's
shape elements are `int32`, so an array of 2³¹ elements is rejected outright: **30
qubits is a hard ceiling**, and the reason the native Metal backend exists.

One more trick worth knowing about because it shows up in the code as `MLXState.perm`:
after a dense tensordot the contracted axes land in *front* of the output rather than
back in their original positions. Restoring them costs a full transpose (a copy), so
the backend doesn't — it records the axis permutation on the state object and folds
it in once, at the next readback. Bookkeeping instead of bandwidth.

## Metal: in-place kernels, 64-bit indexing

`backends/metal_backend.py` drives the GPU directly through
[Metal](https://developer.apple.com/documentation/metal) (via PyObjC), with shader
source compiled at runtime — no offline toolchain, the package stays pure Python.
It exists for what MLX structurally cannot do:

- **64-bit indexing.** All index math in the kernels is `ulong`; states of 2³¹+
  amplitudes are fine. With in-place updates, a 33-qubit state is one 64 GiB buffer —
  it fits a 128 GiB machine.
- **Genuine in-place updates.** Each thread owns one group (the 2ᵏ amplitudes sharing
  its non-target bits — disjoint by construction, so there are no races): read the
  group, multiply, write back to the same locations. Measured peak memory sits *on*
  the theoretical 2ⁿ×8-byte line. Halving the bytes moved per gate relative to a
  double-buffered design is also a speed win in a bandwidth-bound regime.

The state is a single `MTLBuffer` of `float2` (bit-identical to NumPy `complex64`),
in unified memory with shared storage mode — so host readback is a **zero-copy
`np.frombuffer` view**, and `measure`/`sample` can reuse the CPU implementations on
that view directly. Three kernels mirror the gate kinds: `diagonal` (one thread per
amplitude), `monomial` (gather + phase per group), `dense` (matrix–vector per group,
controls checked with an early-out).

The driver layer is where Metal's overheads get engineered away:

```text
  apply_matrix calls          one open command buffer            GPU
  ───────────────────         ───────────────────────────       ─────
  h(0)      ─ encode ──►  ┌─────────────────────────────┐
  cx(0,1)   ─ encode ──►  │ dispatch │ dispatch │ … (≤256)│ ──► executes in
  rz(1,θ)   ─ encode ──►  └─────────────────────────────┘      encoding order
  …                                   │
  statevector()  ── flush: commit + waitUntilCompleted ─┘
                    (the only CPU↔GPU sync point)
```

- **Batched command encoding** — gate dispatches are encoded into one open command
  buffer and only committed at *observation boundaries* (readback, measure, sample)
  or every 256 dispatches. A run of gates pays one commit and one sync instead of one
  each; a serial encoder guarantees execution in encoding order, so no barriers are
  needed.
- **Per-width specialized pipelines** — the gate width `k` is baked into the shader
  as a preprocessor macro, so the compiler unrolls every per-row loop and sizes the
  per-thread arrays exactly; pipelines are cached per (kernel, k).
- **Shared process-wide objects** — device, queue, compiled libraries, and pipelines
  are module-level singletons: creating them cost ~7.5 ms per `Simulator` call in
  auto mode, now paid once per process.
- **Buffer pooling and constant caching** — state buffers are recycled on a free
  list (re-touching warm pages beats faulting fresh ones), small constant buffers
  (gate matrices, index tables) are cached by content, and redundant
  per-dispatch buffer re-binds are skipped.

The result is a backend that is fastest *everywhere* ≥ 16 qubits, not just past
MLX's ceiling — see [the measurements](../backends.md#who-wins-where).

## Memory at a glance

Peak resident memory, measured by `benchmarks/bench_memory.py` against the
theoretical 2ⁿ×8-byte state (multipliers from `benchmarks/data/memory.json`):

```text
            theory      cpu        mlx           metal
 statevector 2ⁿ×8 B     ~3×        up to ~20×    ~1.0×  (in place)
 density mx  4ⁿ×8 B     ~3×        ~19–25×       ~1.0×
```

The MLX multiplier is the lazy graph's double-buffered intermediates; reducing it is
Step 36 on the [roadmap](../plan.md). When a working set approaches physical RAM,
macOS starts swapping and runtimes fall off a cliff — the memory benchmark
budget-gates its cells for exactly this reason, and it is usually what a "slow" 28q+
run turns out to be.

---

Next: [Optimizations](optimizations.md) — the compiler passes and tuning machinery
that sit on top of these three engines.
