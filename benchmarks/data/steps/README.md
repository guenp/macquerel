# GPU-perf plan: per-step benchmark data (docs/plan.md Steps 21–30 and 31–35)

One JSON per `(step, backend)` written by `benchmarks/run_step_bench.sh`, named
`<step>-<commit>-<backend>.json`. Each step is benchmarked through the ASV
harness (`benchmarks/asv_benchmarks/`): `asv run` builds the step's commit in
an isolated clone, times each cell in its own process, and the results are
exported into these JSONs via `plot_steps.py --export-asv`. A step only
re-measures the backends it touched (untouched backends carry forward).
`benchmarks/plot_steps.py` renders:

- `step_speedups.png` — cumulative geomean speedup vs baseline per step (left)
  and the final state's speedup by qubit count (right);
- `step_curves_{cpu,mlx,metal}.png` — per-circuit runtime curves, one line per step.

Steps appear in **execution order**: 21 → 22 → 24 → 23 → 25 → 26 → 27 → 28 → 30
(24 was moved before 23 after review, so the memory-cliff fix wouldn't confound
23's large-n A/B; 30 landed after the line shipped), then the v0.2.x+ candidate
line re-baselined at the 0.2.1 release commit (`step32-baseline`, a no-change
control vs step30: 0.99–1.00×) and ran 32 → 33 → 34. All numbers: M5 Max,
128 GB, macOS, MLX 0.31.2, min of 3 reps, process-isolated cells.

The JSONs and charts were re-measured in June 2026 by replaying every plotted
step through the ASV harness; the per-step prose below quotes the original
runs. Step-over-step ratios reproduced within a few percent; the compounded
final cumulative landed at metal 2.44× / mlx 2.06× / cpu 1.52× (originally
2.85× / 2.04× / 1.63×).

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

**Step 30 (`3a745fb`) — per-backend, qubit-aware fusion-width defaults.
metal 1.30× / cpu 1.32× geomean step-over-step; mlx 1.01× (no-change
control).** The width sweep (`benchmarks/data/fusion_width.json`, widths 1–6
× 4 circuits × 16–24q) showed the optimal `max_fused_qubits` became a backend
property after Steps 22/25: with Metal's per-gate overhead mostly gone, wide
fusion at small/mid n only pays host-side matrix composition and densifies
cheap diagonal/monomial gates. Defaults are now tiered per (backend, n):
**metal 2 ≤22q, cpu 3 ≤18q, otherwise 4; mlx 4 everywhere**. Wins are
1.3–2.15× at 6–20q (metal qft@20 21→10 ms, random@20 41→28 ms; cpu 1.4–1.8×
≤16q) with large n flat at 1.00×. The tiering is load-bearing: a *flat*
metal width of 2 won the sweep's normalized aggregate (1.58× vs width 4) but
regressed random@24–28 by 2.7–3.7× in the step A/B — at 24q+ every backend
is apply-bound and width 4 still wins. One compromise cell remains: metal
random@22 is 0.85× (the other 22q circuits win 1.07–1.57×, so the 22q
boundary nets ~1.17×).

## Cumulative result (Step 30 state vs baseline)

Geomean over circuits, by qubit count — see `step_speedups.png` right panel:

| backend | 16q | 20q | 22q | 24q | 26q | 28q | best single cell |
|---|---|---|---|---|---|---|---|
| metal | 2.63× | 2.61× | 2.58× | 2.54× | 2.85× | 2.72× | random@28 6.1× |
| mlx | 0.96× | 1.24× | 1.86× | 2.45× | 2.34× | 2.59× | random@28 **14.7×** |
| cpu | 1.49× | 1.65× | 1.97× | — | — | — | qft@22 3.6× |

Against the strongest external rival in `benchmarks/data/large` (Qiskit Aer):
at 20q macquerel-metal now wins outright (random 28 vs 51 ms, qft 10 vs
19 ms — it was at parity before Step 30), and from 22q up it wins decisively
— random@24 99 ms vs Aer's 914 ms (9.2×), qft@24 63 vs 342 ms, qaoa@24 46 vs
271 ms. Below ~16q Aer/Qulacs still win on dispatch overhead (auto-select
keeps CPU there). The plan's success criteria: G2 (Metal crossover, was ≥22q)
landed at ~20q; G3 (28q cliff) gone; G4 (auto picks measured-fastest)
re-tuned to CPU ≤16q / Metal ≥17q; G1 holds for the system (Metal tier)
everywhere ≥20q, with MLX itself still behind Aer only on QFT at 24q+.
Step 29 (custom MLX dense kernel) was not needed.

## v0.2.x+ candidate line (Steps 31–35)

**Step 32 (`fef6417`) — MLX broadcast diagonal path. mlx 1.20× geomean;
qft@22–28 2.5–4.3×.** The diagonal path built a full-width gather table —
k shift/or passes over a 2^n uint32 index plus a full-width phase gather —
before the actual multiply. It is now a single broadcast elementwise multiply:
the state is viewed with one length-2 axis per target qubit (gaps between
targets collapsed, so the view stays ≤2k+1-dimensional at any n) and the
(2,)*k diagonal broadcasts across the rest. This was the plan's "in-place-style
diagonal path" candidate, aimed at exactly the wide-diagonal QFT cells where
MLX trailed Metal 6–10×: qft@28 9.08→2.12 s, qaoa@28 2.4×, everything else
flat within the documented GPU clock variance.

**Step 33 (`4fff625`) — custom MLX dense kernel (`mx.fast.metal_kernel`).
mlx 1.09× geomean; random@22–28 1.16–1.61×, qaoa@24–26 1.3–1.6×.** Dense and
controlled gates now run the same group-per-thread kernel design as the native
Metal backend (Step 25), generated per (gate width, control count) so the
per-group loops unroll, instead of `mx.tensordot` — whose internal input
permutation was the dominant cost on scattered-target circuits (the half Step
23 couldn't remove). MLX kernel inputs are `const device` (issue #2547), so the
result is double-buffered like every other MLX path; tensordot remains the
fallback for k>6 gates that fusion never emits. This was the deferred Step 29:
it matters for the no-PyObjC fallback tier, and it closes most of MLX's
remaining random-circuit gap — cumulative with Step 32, mlx is 1.3× geomean
and 2.2× at 28q vs the 0.2.1 baseline (qft@28 4.5×).

**Step 34 (`4e05d01`) — Metal small-n floor. metal 1.07× geomean; 1.6–1.8× on
ghz@24–26; init 7.5 ms → 30 µs.** Three fixed costs went away: (a) the device,
command queue, and per-k compiled pipelines are now process-wide, so
constructing a `MetalBackend` — which `backend="auto"` used to do on *every*
call — drops from ~7.5 ms (runtime shader compile) to ~30 µs, and the
Simulator now also reuses backend instances across auto-mode calls (seeded
simulators keep fresh-per-call backends for bit-identical reruns); (b) state
buffers are recycled through a size-keyed pool (in-flight buffers are parked
until the next flush), which is where the 24–26q wins come from — re-touching
warm pages instead of faulting fresh ones; (c) redundant per-dispatch
`setBuffer`/`setBytes` ObjC calls are skipped. Apparent small-n regressions in
the min-of-3 A/B did not reproduce at reps=9 (ghz@12 is 1.31→0.86 ms, a win;
the original baseline cell was a lucky-fast outlier).

**Step 35 (`800691d`) — per-chip backend-tier autotuning; default CPU tier
15q.** Routing-only (the pinned-backend benchmarks here cannot see it).
`MACQUEREL_BACKEND_TIERS=<int>` pins the CPU tier's max qubit count; `auto`
measures the CPU/GPU crossover once (QFT + brickwork random on both backends
across 10–20q, requiring the GPU to win *sustainably*) and caches it to
`~/.cache/macquerel/backend_tiers.json`, mirroring `MACQUEREL_FUSION_WIDTH`.
The default boundary moved 16 → 15: after Step 34, Metal wins qft/random/qaoa
at 16q (qft 5.6 ms vs cpu 9.2 ms) and the autotuner independently measures
cpu_max=15 on this chip (0.7 s, one-time).

**Step 31 (`9b00708`) — BatchedSimulator.** Not in these JSONs — the
statevector harness times one circuit at a time, and the batched API's whole
point is amortizing across a sweep. Its own harness (`bench_batched.py`,
`benchmarks/data/batched.json`; VQE-style ansatz sweeps) measures the batched
evolution vs a per-circuit `Simulator` loop: **26–47× at 4q, 20–45× at 8q,
14–23× at 12q, 2.2× at 16q**. Two findings shaped it: per-circuit gate fusion
was 99% of a fused prototype's runtime (batching already amortizes dispatch,
so the batched path skips fusion entirely), and the engine crossover tracks
total problem size, so `auto` routes on log2(B)+n against the Step 35 tier
boundary — which picks the faster engine in every measured cell.

### Cumulative result (Step 34 state vs the 0.2.1 baseline, this line only)

| backend | 6q | 12q | 16q | 20q | 22q | 24q | 26q | 28q | best single cell |
|---|---|---|---|---|---|---|---|---|---|
| mlx | 0.99× | 0.97× | 0.97× | 1.06× | 1.32× | 1.61× | 1.83× | 2.16× | qft@28 **4.5×** |
| metal | 0.96× | 0.90× | 1.01× | 1.06× | 1.11× | 1.25× | 1.32× | 1.04× | ghz@26 1.8× |

(cpu untouched this line: 1.00×. The sub-1.0 small-n metal/mlx cells are
single noisy min-of-3 cells — the reps=9 recheck above puts them at parity or
better.) Against the full arc, the cumulative geomean vs the *step20* baseline
now stands at **metal 2.85×, mlx 2.04×, cpu 1.63×** (`plot_steps.py` table).

The refreshed framework comparison (`benchmarks/data/large`, macquerel
backends re-measured at this line's head; Aer/Qulacs/cpu data unchanged)
closes the previous line's one remaining external loss: MLX was behind Aer
on QFT at 24q+ (438 vs 364 ms), and now wins **every circuit from 20q up**
(qft@24 121 vs 364 ms, qft@26 481 vs 1719 ms). Metal's lead is unchanged or
better (qft@24 62 ms, random@24 104 ms vs Aer's 364/1014 ms), and MLX even
takes a cell from Metal at qaoa@28 (478 vs 601 ms).
