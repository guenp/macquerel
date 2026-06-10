# GPU-perf plan: per-step benchmark data (docs/plan.md Steps 21–28)

One JSON per `(step, backend)` written by `benchmarks/run_step_bench.sh`, named
`<step>-<commit>-<backend>.json`. Each step was benchmarked in an isolated git
worktree pinned to the commit it names; a step only re-measures the backends it
touched (untouched backends carry forward). `benchmarks/plot_steps.py` renders:

- `step_speedups.png` — cumulative geomean speedup vs baseline per step (left)
  and the final state's speedup by qubit count (right);
- `step_curves_{cpu,mlx,metal}.png` — per-circuit runtime curves, one line per step.

Steps appear in **execution order**: 21 → 22 → 24 → 23 → 25 → 26 → 27 → 28
(24 was moved before 23 after review, so the memory-cliff fix wouldn't confound
23's large-n A/B). All numbers: M5 Max, 128 GB, macOS, MLX 0.31.2, min of 3
reps, subprocess-isolated cells.

## Per-step results and why

**Step 21 (`7cad261`) — auto-select 22q+ → Metal.** Routing only, so no
per-backend bars: the explicit-backend benchmarks are unchanged, but `auto`
users went from MLX to Metal at 22–30q, worth 2.7×(24q)–5.1×(28q) on the
baseline data. Superseded at the end of the line by the final re-tune
(`0806f3e`): after Steps 22+25 Metal wins **everywhere ≥17q**, so the MLX tier
exists only as the no-pyobjc fallback.

**Step 22 (`58cc612`) — Metal batched command buffers + const cache.
Metal 1.30× geomean.** Gates are *encoded* into one open command buffer and
committed at observation boundaries (or every 256 gates) instead of paying
`commit` + `waitUntilCompleted` per gate. Big where the fixed sync cost
dominated — random@6 22.6→12.1 ms, @20 106.8→64.2 ms (1.7×) — and neutral at
26–28q, which is memory-bandwidth-bound: batching changes when work is
submitted, not how many bytes a gate moves.

**Step 24 (`bf78f05`) — MLX periodic `async_eval`. random@28 1.56×; geomean
~1.00×.** With no synchronization between observation boundaries, a depth-d
lazy graph keeps O(d) full-width temporaries alive (peak ≥ ~16× state size);
at 28q that working set swaps and random@28 took 40.0 s. An `mx.async_eval`
every 16 gates (≥24q only) bounds the working set: 40.0→25.6 s. Interval
8/16/32/64 measured within 1% of each other; every other cell unchanged
(nothing else was swapping), which is why the aggregate looks flat — this fix
targeted exactly one cliff and removed it.

**Step 23 (`7bbc216`) — MLX axis-order tracking. MLX 1.03× geomean, 1.05–1.15×
on dense circuits.** Dense gates no longer pay a full-state transpose+copy to
restore canonical axis order after `tensordot`; the permutation is recorded and
folded in once at readback. The win is real (random@24 1419→1308 ms, qaoa@24
192→167 ms) but bounded: `mx.tensordot` still permutes its *input* internally
to position the contracted axes for matmul, so only one of the per-gate passes
was eliminated. Closing the rest needs the Step 29 custom-kernel route — which
became unnecessary (see Step 27).

**Step 25 (`bc346fb`) — Metal kind-specialized kernels. Metal 1.36× geomean;
random@24–28 2.7–3.1× vs Step 22.** Two changes: (a) pipelines are compiled
per gate width with `K_FIXED` baked in as a preprocessor macro, so the
per-group `amp[]`/`idx[]` arrays are unrolled into registers — with runtime
`k` they were runtime-indexed and spilled, which is where the 3× on fused
4-qubit dense gates came from; (b) a monomial kernel handles
permutation-class gates with 2^k multiplies per group instead of 4^k MACs
(1.9× on 4q monomials at 20q, neutral ≥24q where bandwidth dominates). A
threadgroup-size sweep (64/256/1024) was flat; 256 kept.

**Step 26 (`14cdf74`) — diagonal-run wide fusion + CPU diagonal path.
cpu 1.07× / mlx 1.04× / metal 1.02× geomean — concentrated in QFT/QAOA.**
A second fusion pass merges adjacent diagonal gates (including diagonal
composites like CX·Rz·CX from pass 1) into up-to-8-qubit diagonals: one
elementwise pass each, O(2^k) to compose. cpu qft@20 122→60 ms, qft@22
680→268 ms (the CPU backend also gained a broadcast in-place diagonal multiply
instead of tensordot); metal qft@28 1393→1025 ms; mlx qaoa@28 1.19×. The cap
matters: a width-10 sweep *regressed* because materializing and classifying
1024×1024 dense matrices costs more than the saved passes — 8 (256×256) won
the sweep.

**Step 27 (`3a742e8`) — commutation-aware fusion. mlx 1.46×, cpu 1.17×,
metal 1.15× geomean step-over-step — the largest single step.** The single
greedy in-order fusion group flushed whenever a gate on an unrelated qubit
inflated the union past the cap; brickwork circuits therefore emitted
~depth×(n/width) fused gates. The scheduler keeps several open groups and
routes each gate to (or after) the latest group it shares a qubit with —
disjoint-qubit gates commute, so this is the only real ordering constraint.
Random brickwork now collapses into a handful of rolling neighborhood groups
(random@22: 79 fused ops): mlx random@24 1314→171 ms (7.7×), random@28
23.5→2.7 s (8.5×), metal random@24 180→99 ms, cpu random@22 1528→664 ms.
MLX gains most because its per-gate graph overhead was the largest. GHZ is
unchanged — a CX chain shares qubits at every link, so there is genuinely
nothing to reorder.

**Step 28 (`f55fea2`) — qubit remapping: wired, measured, OFF.** A/B at 24q
lost on every backend/circuit (metal qft 61→79 ms, metal random 89→126 ms):
the `statevector()` readback must invert the relabeling with a full-state
transpose, and the GPU kernels are stride-insensitive so there is no win to
pay for it. Per the plan's benchmark gate it ships disabled
(`MACQUEREL_REMAP=1` opts in).

## Cumulative result (Step 27 state vs baseline)

Geomean over circuits, by qubit count — see `step_speedups.png` right panel:

| backend | 16q | 20q | 22q | 24q | 26q | 28q | best single cell |
|---|---|---|---|---|---|---|---|
| metal | 1.53× | 1.68× | 2.21× | 2.52× | 2.82× | 2.73× | random@28 6.2× |
| mlx | 0.99× | 1.24× | 1.87× | 2.46× | 2.34× | 2.58× | random@28 **14.6×** |
| cpu | 1.06× | 1.63× | 1.95× | — | — | — | qft@22 3.5× |

Against the strongest external rival in `benchmarks/data/large` (Qiskit Aer):
at 20q macquerel-metal is at parity (random 41 vs 51 ms, qft 21 vs 19 ms), and
from 22q up it wins decisively — random@24 99 ms vs Aer's 914 ms (9.2×),
qft@24 63 vs 342 ms, qaoa@24 46 vs 271 ms. Below ~16q Aer/Qulacs still win on
dispatch overhead (auto-select keeps CPU there). The plan's success criteria:
G2 (Metal crossover, was ≥22q) landed at ~20q; G3 (28q cliff) gone; G4
(auto picks measured-fastest) re-tuned to CPU ≤16q / Metal ≥17q; G1 holds for
the system (Metal tier) everywhere ≥20q, with MLX itself still behind Aer only
on QFT at 24q+. Step 29 (custom MLX dense kernel) was not needed.
