# macquerel Implementation Plan — Remaining Work

macquerel is a quantum state-vector simulator targeting Apple Silicon's unified-memory
architecture. Completed implementation and optimization work lives in
[`plan_completed.md`](plan_completed.md), including the per-step A/B results and design
notes for completed lines.

This document tracks only work that has **not** been implemented yet. Steps keep their
original numbering for continuity with the completed record.

## A/B Protocol

Each optimization step is measured as a commit-pinned A/B test against the immediately
previous baseline: run the same benchmark cells before and after the change, in isolated
subprocesses to avoid GPU memory-pool and lazy-graph contamination, using min-of-reps
timings to reduce noise. Correctness is gated first with the test suite, then performance
is accepted only if the affected backend/circuit/qubit ranges improve without meaningful
regressions elsewhere. Per-step JSONs and plots are saved under `benchmarks/data/steps/`,
and changes that lose their A/B, like default qubit remapping, are left disabled or
reverted rather than folded into the hot path.

Implemented steps are moved to [`plan_completed.md`](plan_completed.md), where each
completed optimization records its commit ID alongside the measured A/B result and design
notes.

---

## v0.3

- **Memory-mapped out-of-core backend** — state vector backed by an NVMe file via
  `np.memmap`, for single large runs past DRAM capacity. Investigated and expanded
  into the explicit out-of-core design under [v0.3.x](#v03x-out-of-core-statevectors-disk-streamed-past-ram-capacity)
  below — passive OS paging turned out to be the wrong mechanism.
- **Multi-Mac over Thunderbolt** — distributed state vector using index-bit partitioning
  across machines.

---

## v0.3.x — Out-of-core statevectors (disk-streamed past RAM capacity)

[DuckDB](https://duckdb.org/2024/07/09/memory-management.html) runs analytical queries
on larger-than-memory data by giving every operator a fixed memory budget
(`memory_limit`, default 80% of RAM) and a buffer manager: intermediates live in
buffer-managed blocks that are spilled to a temporary directory when the budget is
exceeded and read back in large sequential chunks, so sort/join/aggregate
[degrade gracefully](https://duckdb.org/2021/08/27/external-sorting.html) to NVMe
bandwidth instead of failing or swapping. This section investigates whether that
model transfers to statevector simulation and turns the v0.3 "memory-mapped
out-of-core backend" bullet into a concrete design.

### What transfers and what doesn't

DuckDB's buffer manager does two separable jobs. It is a **cache**: keep hot blocks
in RAM, evict cold ones by LRU, fault blocks in on demand — necessary because a
query's access pattern is unknown until it runs. And it is a **bounded-memory
streaming mechanism**: a fixed budget, explicitly pinned working blocks, large
sequential spill/reload I/O instead of random page faults.

The caching job does *not* transfer. A cache earns its keep on skew — a query touches
data selectively, so cold blocks can sit on disk untouched. A gate pass touches
**every** amplitude exactly once: there is no cold data to leave behind and no reuse
for a replacement policy to exploit, so a demand-paged cache degenerates to pure
overhead (LRU is the textbook-worst policy for a cyclic scan — each block is evicted
just before it is needed again). Naively backing the state with `np.memmap` and
letting the OS page (the v0.3 sketch) fails for exactly this reason: the kernel is a
generic reactive cache, evicting 16 KiB pages with no knowledge of the access plan,
and a 34-qubit gate pass becomes random-fault-ordered I/O.

The streaming job is what this design keeps — and because a gate's access pattern is
known at *compile time* from its target qubits (unlike a query's), it can be kept in
a stronger, policy-free form: no faulting, no eviction decisions, just a fixed
schedule. Concretely, what transfers:

1. **An explicit budget instead of passive paging.** A `memory_limit`-style knob
   (e.g. `MACQUEREL_MEMORY_LIMIT`, defaulting to today's min(0.45 × RAM, 64 GiB)
   benchmark gate) and an explicit chunk manager that owns the I/O schedule.
2. **Sequential, partitioned access.** DuckDB radix-partitions a spilling hash join so
   each partition is processed *fully* while resident. The statevector analog is
   index-bit partitioning — the same scheme already planned for multi-Mac: split the
   state into 2^m chunks by the top m index bits. Gates on the low n−m **local**
   qubits act independently inside each chunk; only gates on the top m **global**
   bits pair amplitudes across chunks. The chunk visit order falls straight out of
   the gate's target qubits — static, not adaptive.
3. **Maximize work per residency.** The unit of I/O is a "chunk episode": load a
   chunk (sequential multi-GiB read), apply *every* gate that can run before the next
   global-bit barrier (the existing fusion + commutation-aware grouping machinery,
   re-targeted), write it back, prefetch the next chunk meanwhile. This is Doi–Horii
   cache blocking lifted one level — DRAM as the cache, NVMe as main memory — and it
   is the entire performance story: one episode costs a full-state read+write
   regardless of how many gates it carries.
4. **Graceful degradation.** Budget ≥ state size → explicitly bypass the chunk manager
   and route to today's in-RAM backend, preserving the existing hot path. Budget <
   state size → stream chunks. Metal should compose directly via
   `newBufferWithBytesNoCopy:` over page-aligned chunk buffers, but this is the
   riskiest systems assumption and must be validated before scheduler work: confirm
   no hidden copies, no page-fault collapse, and stable throughput with a tiny-budget
   prototype.
5. **Clean pages skip writeback.** DuckDB drops unmodified cached pages instead of
   spilling them. The analog: read-only passes (sampling, probabilities,
   expectation values) stream chunks without writing them back — half the I/O.

### What replaces the buffer pool

The pool's replacement is the Step 42 **three-buffer ring**: exactly three pinned
chunk-sized RAM buffers — one being computed on, one being prefetched (the next chunk
in the schedule, read sequentially while compute runs), one being written back (the
previous chunk) — rotating roles as the schedule advances. It is a buffer pool
collapsed to its degenerate case. What survives: *pinning* (a chunk never moves while
the GPU is reading it) and the *budget* (3 × chunk size ≤ `MACQUEREL_MEMORY_LIMIT`
fixes the partition count 2^m). What disappears: the cache. There is no hit/miss, no
replacement policy, no demand faulting — "eviction" is the scheduled writeback of a
chunk the plan is finished with, "admission" is the scheduled prefetch of the chunk
the plan needs next.

That also pins down *how* it helps, because it is not by avoiding I/O — an episode
reads and writes the full state no matter what. The ring's entire contribution is
**hiding** the I/O: with compute and both I/O directions overlapped, an episode costs
max(compute, I/O) rather than their sum, and since local gate work runs at
unified-memory bandwidth (~93 GiB/s measured full-state pass) while the NVMe sustains
~7 GB/s, I/O is the roofline — the pipeline's one job is to keep the NVMe queue
never idle.

Relative to the shipped in-RAM backends, the genuinely new concepts are four —
everything else is existing machinery re-targeted:

1. **A memory budget** (`MACQUEREL_MEMORY_LIMIT`). Today allocation is
   all-or-nothing: the state fits in RAM or the run fails.
2. **A chunked state** (`ChunkedState`, Step 41): one logical state stored as 2^m
   file-backed partitions split by top index bits, vs today's single contiguous
   buffer.
3. **The episode as the unit of cost** (Step 43). Today's cost unit is the per-gate
   full-state pass and fusion minimizes passes; out of core, the unit is the chunk
   *residency* — load once, run every gate up to the next global-bit barrier, write
   once — and the same fusion + commutation-aware grouping machinery is re-aimed at
   minimizing episodes (full-state I/O round-trips) instead.
4. **The three-buffer ring** above.

The remap pass (Step 44) is not new machinery either: it is the existing
disabled-by-default `remap_qubits` pass, finally facing the ~13× bandwidth asymmetry
it was built for (it lost its in-RAM A/B because no such asymmetry exists there).

### Feasibility numbers (assuming a 128 GiB Mac with NVMe at ~6–8 GB/s sequential)

| n | state (2ⁿ × 8 B) | fits | episode cost (read+write @ 7 GB/s) |
|---|---|---|---|
| 33 | 64 GiB | RAM (today's ceiling) | — (in-memory) |
| 34 | 128 GiB | NVMe | ~37 s |
| 35 | 256 GiB | NVMe | ~75 s |
| 36 | 512 GiB | NVMe | ~150 s |
| 37 | 1 TiB | NVMe | ~5 min |

To be explicit about what kind of feature this is: **capacity, never speed**. There
is no input where the out-of-core path beats the in-RAM path — when the state fits
the budget, Step 41's bypass routes to today's backend untouched (Step 46 gates the
ship on zero in-RAM regression). Per axis: **qubits** +1 to +4 (33q → 37q, bounded by
free disk); **RAM** pinned to the budget regardless of state size — the state is not
smaller, it lives on disk; **speed** strictly a cost, with a hard floor of ~13× per
full-state pass (~93 GiB/s in RAM vs ~7 GB/s NVMe), ~2× more again when the pass
writes back. The alternative the spilled path competes against is not "slower" — it
is "fails to allocate". (The one piece of the line that *can* speed things up is
Step 47's complex32: halving bytes per amplitude in a bandwidth-bound simulator.)

So out-of-core execution buys **+1 to +4 qubits** of capacity at a slowdown set by the
local/global structure of the circuit: GHZ at 34q is a handful of episodes (~minutes);
a QFT, whose controlled-phase gates are diagonal (diagonal gates never touch a global
bit's *pairing*, only its phase — they are chunk-local at any width), spills cheaply;
a random dense circuit on scattered qubits is the worst case and needs the remap step
below to stay off the global bits. Note the asymmetry that makes scheduling decisive:
local work runs at unified-memory bandwidth, global work at NVMe bandwidth —
a ~13× gap on measured numbers (~93 GiB/s effective full-state pass vs ~7 GB/s
sequential NVMe), far steeper than the cache-blocking gap that lost remapping its
single-machine A/B.

### Steps

- **Step 41: chunked state store + no-copy Metal proof** — first build the smallest
  useful `ChunkedState`: one page-aligned file (`MACQUEREL_TEMP_DIR`, default
  alongside the platform temp dir; checked against free disk up front, deleted on
  exit — DuckDB's `temp_directory` / `max_temp_directory_size` semantics), a
  `MACQUEREL_MEMORY_LIMIT` budget, and an explicit in-RAM bypass when the budget
  covers the state. Prototype one local gate over artificially chunked small states,
  feeding Metal with `newBufferWithBytesNoCopy:`. Ship this step only if the prototype
  produces identical amplitudes vs the in-RAM path, peak RSS respects the tiny budget,
  Instruments/benchmarks show no hidden full-state copy, and sequential read/write +
  compute throughput stays near the measured NVMe/Metal roofline.
- **Step 42: three-buffer streaming pipeline** — generalize Step 41 into the compute /
  prefetch / writeback pipeline with exactly three live pins, plus read-only streaming
  that skips writeback for sampling, probabilities, and expectation values. Benchmark
  this before the full scheduler: report effective GB/s for raw sequential
  read/write, read-only streaming, and one local-gate episode at forced-small budgets.
- **Step 43: episode scheduler + gate classification** — compile the fused gate stream
  into chunk-local episodes split at global-bit barriers. Make locality rules explicit:
  gates whose non-diagonal target action is wholly local stay inside a chunk;
  diagonal/phase gates on global bits stay local because they never pair chunks;
  controls on global bits are local when the target action is local because they only
  mask chunks; non-diagonal global targets are barriers. Skip writeback for read-only
  passes. A/B metric: episodes per circuit (= full-state I/O passes), then wall time.
- **Step 44: global bits via remap passes** — when a non-diagonal gate needs a global
  target bit, emit one explicit remap episode (swap a global bit with a cold local bit
  — a paired-chunk streaming pass) instead of pairing chunks ad hoc, then proceed
  chunk-locally. Reuses the existing disabled-by-default `remap_qubits` machinery,
  which finally has the cost asymmetry it was built for. A/B: remap-pass count and
  wall time on the random-circuit benchmark at 34q.
- **Step 45: optional lz4 writeback compression** — only after the raw pipeline works,
  add opt-in lz4 chunk compression on the writeback path with a raw fallback when the
  ratio is poor. Being lossless, its ceiling is information-theoretic, not a tooling
  question: a structured state like GHZ is mostly zero bytes and compresses nearly
  for free, but a Haar-random chunk is incompressible past ~1.1–1.2× — the mantissa
  bits are effectively uniform random, and the only slack is in the exponents, which
  Porter-Thomas concentration leaves a few bits of entropy. So gate it on wall time,
  not just ratio, and make the raw fallback short-circuit quickly enough that failed
  compression does not disturb GPU/I/O overlap.
- **Step 46: out-of-core benchmark + ship gate** — extend `bench_memory.py` with a
  spilled series (peak RAM must respect the budget; report episodes and effective
  GB/s against the NVMe roofline) and a 34q GHZ/QFT/random wall-time table. Ship only
  if 34q runs end-to-end under the default budget without swap and the in-RAM path
  shows zero regression. SSD-endurance note in the docs: an episode writes the full
  state (128 GiB at 34q), so deep unfused circuits are a TBW consideration —
  documented, not engineered around, in v0.3.x.
- **Step 47: block-float `complex32` states** — opt-in half-precision amplitude
  storage (`dtype="complex32"`): two float16s per amplitude plus one float32
  max-magnitude scale per chunk. The per-chunk scale is load-bearing, not a nicety:
  at 34q the typical amplitude magnitude is ~2⁻¹⁷ ≈ 8e-6, *below* float16's normal
  range (6e-5), so naive fp16 lands in subnormals or underflows — block floating
  point restores the range. A deterministic, circuit-independent 2× on every byte
  count, and GPU-native (Metal and MLX both have first-class `half`), so chunks stay
  compressed in unified memory too: it composes with spilling *and* raises the
  in-RAM ceiling 33q → 34q (and the DM ceiling toward n=17) with no disk involved.
  Cost: a 10-bit mantissa gives ~2⁻¹¹ relative rounding per gate pass, accumulating
  roughly with √depth — fine for sampling and expectation values at moderate depth,
  not for amplitude-level verification. Ship opt-in only; gate with the existing
  differential tests against complex64 and publish a fidelity-vs-depth table.
  **This is the cheapest +1 qubit of the v0.3.x line** and depends only on
  chunk-granular scaling, not on the disk pipeline — it can ship before Steps 41–44.

Considered and deferred: **error-bounded lossy compression (SZ/ZFP)** — the
strongest published precedent for compressed full-state simulation (Wu et al.,
*Full-state quantum circuit simulation by using data compression*, SC'19,
[arXiv:1810.14582](https://arxiv.org/abs/1810.14582), which kept the state
compressed in memory, decompressing blocks on the fly, and reached a 61-qubit
Grover), but a poor fit here for now. Unlike lossless coding, lossy compression
*does* keep working on dense Haar-random states — truncating mantissas is exactly
how high-entropy floats compress, regardless of structure — so the objection is not
the ratio in principle but the practical terms: SZ/ZFP throughput is ~1-5 GB/s per
CPU core (below the NVMe without multithreading or a Metal port — ZFP has CUDA
kernels, no Metal), the error bound is consumed once per compress/decompress cycle
and every chunk episode is a cycle, and on random states the ratio settles toward
the precision-reduction floor — at which point it is buying the same bytes as Step
47's complex32, at lower speed and with an adaptive error model instead of a fixed,
characterizable one. The credible way back in is a **zfp fixed-rate Metal port**
(guaranteed ratio, bounded error, GPU speed), which would slot into Step 45's
wall-time gate unchanged. Also deferred: **an in-RAM "blocked backend"
as its own feature** — partitioning never reduces the bytes a state occupies, and
Metal already stores the state once at a measured ~1.0x of theory, so in-RAM block
partitioning adds zero qubits; it falls out of Step 41 anyway as
the trivial budget ≥ state bypass. (A gate on a global bit is a 2×2 *block matrix
multiply* over chunk pairs at identical traffic to a local pass, so the in-RAM
partitioning tax is ~zero — but so is the benefit.) One conditional exception: an
**`mlx-chunked` mode** (chunks ≤ 2³⁰ elements) would lift MLX's upstream int32
`ShapeElem` ceiling from 30q to Metal parity at 33q while collapsing its ~3-5×
peak multiplier toward ~1× (lazy-graph temporaries become chunk-sized) — same
scheduler, RAM tier; only worth building if MLX-only machines (no PyObjC) turn out
to matter. Also deferred: **changed representations** — matrix product states now
have their own roadmap line ([v0.4](#v04-matrix-product-state-simulator) below);
decision diagrams and sparse amplitude dictionaries remain deferred without one —
and a **general buffer pool with LRU eviction** (no selectivity in the access
pattern, so there is nothing to cache — see "What replaces the buffer pool" above;
the three-buffer ring is the whole pool). The multi-Mac
item (v0.3) and this section share the index-bit partitioning compiler work;
whichever ships first pays most of the cost of the second.

---

## v0.4 — Matrix product state simulator

Everything in v0.3.x buys single qubits at the 2ⁿ byte wall (complex32 +1, spilling
+1 to +4). A **changed representation** is the only lever that escapes it: an
`MPSSimulator` stores the state as a chain of n tensors of shape `(χ, 2, χ)`, where
the bond dimension χ measures entanglement across each cut — memory is
`n · 2χ² · 8 B`, scaling with *entanglement* instead of 2ⁿ. 100 qubits at χ = 1024
is 1.6 GiB; a GHZ state is χ = 2 at any width; MPS storage beats the full vector at
33q for any χ below ~10⁴. The honest trade: each entangling layer can double χ, so a
deep random circuit forces χ → 2^(n/2) and the representation collapses back to 2ⁿ —
this is a new regime (50–1000+ low-entanglement qubits: GHZ/Clifford-lite states,
QFT, shallow VQE/QAOA ansätze, 1D Trotterized time evolution), not a replacement for
the statevector simulator.

Design constraints, from a back-of-envelope study:

- **Sibling-simulator pattern.** Same `Circuit` API, alongside
  `DensityMatrixSimulator` and `TrajectorySimulator`. χ capped with **exact tracked
  truncation fidelity** (computed from the discarded singular values, reported per
  run), nearest-neighbor routing via SWAP chains, exact `O(n χ²)`-per-shot sampling
  by sweeping the chain.
- **Gate costs.** 1-qubit gates are free contractions. A 2-qubit neighbor gate
  contracts two site tensors (~χ³ GEMM), applies the gate, then SVDs the `2χ × 2χ`
  matrix to split back and truncate — `O(8χ³)`, the dominant cost and the one new
  primitive.
- **CPU/GPU split.** The χ³ contractions are GEMMs — exactly what the existing
  Metal/MLX backends are best at, on far friendlier tensor shapes than 2ⁿ vectors.
  But **SVD has no GPU path on Apple Silicon today**: MLX's `mx.linalg.svd` is
  CPU-stream-only (verified on 0.31.2) and Metal Performance Shaders has no SVD, so
  truncation runs on Accelerate LAPACK (~0.1–0.5 s at χ = 1024). Unified memory
  makes the GPU-GEMM/CPU-SVD hybrid copy-free — the spot where discrete-GPU
  simulators pay PCIe tax. QR-based or randomized-SVD truncation is the later GPU
  offload path.
- **A/B references.** Qiskit Aer `method="matrix_product_state"`, quimb, ITensor,
  TeNPy. Document the `random` benchmark as the adversarial case MPS is *not* for.

Considered and deferred: **contraction-based simulation** (qsimh / cuTensorNet /
cotengra style — the whole circuit as one tensor network, contracted per amplitude
batch with slicing). It is a sampling/amplitude tool with no state, no cheap
mid-circuit measurement, and a contraction-order search problem that mature dedicated
tools already own — a different product than this simulator's contract.

---

## Verification

After each step, run `uv run pytest tests/ -x -q` and confirm the new tests pass before
moving to the next step. Final verification:

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
