# Optimizations

Every optimization in macquerel follows from one observation: **gate application is
memory-bandwidth-bound**. A gate does a few multiply-adds per amplitude but must
stream the whole 2ⁿ-amplitude state through the memory system, so at scale the
simulator's speed is bytes moved, not FLOPs. That leaves exactly three levers:

1. **Make fewer passes over the state** — gate fusion, batching many circuits into
   one pass.
2. **Move fewer bytes per pass** — gate-kind fast paths, in-place updates.
3. **Stop paying fixed costs per gate** — kernel-launch batching, pipeline caching,
   lazy evaluation.

Below ~16 qubits the state fits in cache and a fourth lever appears (dispatch
overhead dominates everything), which is its own section. Everything here was
validated by a measure-first A/B discipline, described at the end; the per-step
results live in [the completed plan](../plan_completed.md).

## Gate fusion: fewer passes

Three one-qubit gates on the same qubit are three passes over the state. But gates
compose: multiply their matrices together (cheap — the matrices are tiny) and apply
the product in *one* pass. Fusion generalizes this to neighborhoods of qubits
(`compiler.fuse_gates`):

```text
 before (5 passes over 2ⁿ amplitudes):       after (1 pass):

 q0 ──H──●─────────●──                       q0 ──┐
         │         │                              │
 q1 ─────X────●────X──            ──►        q1 ──┼─[ U 8×8 ]──
              │                                   │
 q2 ──T───────X───────                       q2 ──┘

         H, CX, T, CX, CX                    U = CX·CX·T·CX·H  (embedded into
                                                 the 3-qubit space, composed
                                                 as 8×8 matrix products)
```

The compiler greedily accumulates adjacent gates while their combined qubit set stays
within `max_fused_qubits`, embeds each into the group's joint space, and multiplies.
Gate fusion is standard practice in fast simulators — qHiPSTER introduced it for
exactly this bandwidth argument, and Qulacs and qsim both rely on it [\[2, 3\]](#references).

**The width is a real trade-off**, which is why it is tuned rather than hardcoded:
each extra qubit of width doubles the matrix dimension (composition cost, and 4ᵏ MACs
per group at apply time) but removes passes over the state. The optimum drifts with
qubit count — at small n the apply is sub-millisecond and composition overhead
dominates (narrow wins); at large n the apply dominates (wide wins) — *and* with
backend, because fusion amortizes per-gate overhead and each backend has a different
amount left. The measured defaults: Metal fuses to width 2 up to 22q then 4 (its
per-gate overhead mostly vanished with batched encoding, so wide fusion just costs
composition), CPU 3 up to 18q then 4, MLX 4 everywhere (largest per-gate lazy-graph
overhead, so it rewards the widest fusion). `MACQUEREL_FUSION_WIDTH=<int>` pins the
width; `=auto` runs a small autotuner once and caches the result for your chip.

Two refinements make fusion bite harder:

- **Commutation-aware grouping.** Gates on disjoint qubits commute, so a gate on an
  unrelated qubit shouldn't break a group. The compiler keeps several groups open at
  once and routes each gate to the earliest open group it fits in — constrained only
  by the *latest* group it shares a qubit with, the one real ordering constraint.
  QAOA's nearest-neighbor layers collapse from 144 gates into 24 contiguous 4-qubit
  blocks this way.
- **Diagonal-run wide fusion.** Diagonal × diagonal is diagonal, composition is an
  O(2ᵏ) elementwise product (not an O(8ᵏ) matmul), and *applying* a diagonal is one
  cheap pass regardless of width — so adjacent diagonal gates fuse far past the dense
  limit, to width 8 by default. This runs as a second pass, deliberately: the first
  pass is what composes e.g. QAOA's CX·RZ·CX into a diagonal ZZ block; the second
  merges those blocks and QFT's controlled-phase cascades into wide diagonals. The
  cap is itself measured: at width 10 the 1024×1024 matrix the `Gate` still carries
  costs more to materialize and classify than the saved passes return.

Fusion stops at any non-gate op: measurements, and noise channels (which do not
commute with the gates around them — see [Noise](noise.md)).

## Gate-kind fast paths: fewer bytes

The diagonal / permutation / dense classification from
[Applying gates](applying-gates.md#not-all-gates-cost-the-same-diagonal-permutation-dense)
is itself the second lever. Two examples of how it played out in this codebase:

- The **MLX diagonal path** originally built a full 2ⁿ gather table (k shift/or
  passes to compute a per-amplitude row index, then a full-width phase gather).
  Replacing it with a broadcast elementwise multiply — reshape so each target qubit
  gets its own axis, multiply by the `(2,)*k` diagonal — cut QFT runtimes by 2.5–4.3×
  at 22–28q. Same math, fewer bytes.
- The **MLX dense path** used `mx.tensordot`, which internally permutes a *copy* of
  the state to bring target axes together — the dominant cost on scattered-target
  circuits. A custom `mx.fast.metal_kernel` with the Metal backend's
  one-thread-per-group design (no permutation in either direction) bought 1.16–1.61×
  on random circuits.

## Killing fixed costs: the small-n and dispatch story

A GPU kernel launch costs the same whether the state is 512 bytes or 32 GiB. Three
mechanisms keep those fixed costs off the per-gate path:

- **Metal batched command encoding** — one commit + one CPU↔GPU sync per observation
  boundary instead of per gate (encode up to 256 dispatches into one command buffer).
- **MLX lazy evaluation** — graph building costs microseconds per gate; the
  `async_eval` cadence (every 16 gates at ≥24q) keeps the GPU busy *and* bounds how
  many intermediates the graph holds live.
- **Construction caching** — Metal's device/queue/pipelines are process-wide
  singletons (7.5 ms → ~30 µs per backend construction); state buffers are pooled;
  constant buffers are cached by content; classification is memoized by matrix bytes.

The residue that can't be engineered away — command-buffer scheduling latency, a few
µs of encode per gate — is why CPU wins below 16 qubits and why `auto` mode exists.
Two autotuners handle the boundaries that are chip properties rather than constants:
`MACQUEREL_BACKEND_TIERS=auto` measures the CPU/GPU crossover, and
`Simulator(batch_shots="auto")` (the default) tunes the GPU sampling batch size by
doubling until throughput plateaus.

## Batching circuits: `BatchedSimulator`

For parameter sweeps the small-n regime has one more trick. Running N variants of a
12-qubit ansatz as N `Simulator` calls pays the per-gate dispatch cost N times for
trivially small states. `BatchedSimulator` packs the sweep into one `(B, 2ⁿ)` tensor
and applies each gate position across the whole batch in a single batched matmul
(or one broadcast multiply when every matrix at that position is diagonal):

```text
 per-circuit loop:                      batched:
 B × (per-gate dispatch × depth)        depth × (one batched op over (B, 2ⁿ))

 ┌c₀: h cx rz(θ₀) …┐
 ├c₁: h cx rz(θ₁) …┤      ──►          [ B×2ⁿ tensor ] ─h─ cx ─rz(θ⃗)─ …
 ├c₂: h cx rz(θ₂) …┤
 └…                ┘                    (one launch per gate position)
```

Circuits are grouped by structure signature (same gate positions/targets/controls —
exactly what a parameter sweep produces), and `auto` routes on the *total* size
log₂(B) + n, since a batch moves as many bytes as one state that big. Measured
2–47× over the per-circuit loop on VQE-style sweeps.

## The A/B discipline

Optimizing a bandwidth-bound program is full of plausible ideas that lose. Every
candidate here ran as a commit-pinned A/B against the previous baseline: same
benchmark cells before and after, each cell in an **isolated subprocess** (GPU memory
pools and lazy graphs contaminate later measurements in-process), min-of-reps
timings, correctness gated by the test suite first. Accepted only if the targeted
regime improved without meaningful regressions elsewhere — and ideas that lost their
A/B stayed out:

- **Qubit remapping** (relabel hot qubits to low indices, the Doi–Horii cache-blocking
  idea [\[6\]](#references)) is implemented but **disabled by default**: it regressed the measured
  backends, whose kernels are largely insensitive to target position. It remains
  opt-in via `MACQUEREL_REMAP=1`.
- A naive **single-point fusion autotuner** (measure at one small n) picked widths
  that regressed the large-n path by up to ~2×, which is why the autotuner measures
  across a span of qubit counts and scores the aggregate.

Per-step data and plots live under `benchmarks/data/steps/`; the narrative record
with commit IDs is [plan_completed.md](../plan_completed.md).

---

Next: [Noise and density matrices](noise.md) — simulating what real hardware
actually does to your circuit.

## References

1. S. Williams, A. Waterman, D. Patterson, "Roofline: an insightful visual
   performance model for multicore architectures", *CACM* **52**(4), 2009 — the
   bandwidth-vs-compute framing.
2. M. Smelyanskiy, N. P. D. Sawaya, A. Aspuru-Guzik, "qHiPSTER" (2016).
   [arXiv:1601.07195](https://arxiv.org/abs/1601.07195) — gate fusion for
   bandwidth-bound simulation.
3. Y. Suzuki et al., "Qulacs" (2021).
   [arXiv:2011.13524](https://arxiv.org/abs/2011.13524) — fusion and specialized
   per-kind kernels.
4. NVIDIA, *cuQuantum SDK documentation* (cuStateVec gate fusion) —
   [https://docs.nvidia.com/cuda/cuquantum/](https://docs.nvidia.com/cuda/cuquantum/)
5. A. Hannun et al., *MLX: Efficient and flexible machine learning on Apple
   silicon* — [https://github.com/ml-explore/mlx](https://github.com/ml-explore/mlx)
6. J. Doi, H. Horii, "Cache Blocking Technique to Large Scale Quantum Computing
   Simulation on Supercomputers" (2020).
   [arXiv:2102.02957](https://arxiv.org/abs/2102.02957)
