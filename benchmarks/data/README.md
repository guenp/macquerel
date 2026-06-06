# Benchmark data

Per-commit results from `tests/benchmarks/bench_backends.py` (CPU vs MLX, depth-50
random circuit, qubits 16/18/20/22, reps 9), plus the comparison plot rendered by
`benchmarks/plot_results.py`. Each file was (re)measured against its own commit's
backend code using the current harness.

Files are named `<commit>-<label>.json` so each run is traceable to the code that
produced it. The optimizations correspond to the **MLX backend performance
optimization** section of [`docs/plan.md`](../../docs/plan.md).

## Runs

| File | Commit | Step | What changed |
|---|---|---|---|
| `a17db04-baseline.json` | `a17db04` | — | Baseline before the performance work (after the earlier permutation-loop vectorization and native controlled gates). MLX slower than CPU everywhere. |
| `3c74da8-p1-defer-eval.json` | `3c74da8` | **P1** | Defer MLX evaluation across gates — removed per-gate `mx.eval()` so the lazy graph fuses across gates; evaluation forced only at segment boundaries. |
| `81e96e3-p2-ondevice-perm.json` | `81e96e3` | **P2** | Build the permutation gather index on-device with `mx.arange` + bitwise ops, eliminating the per-gate host-side NumPy table build and host→device copy. |
| `b6bef6f-p8-mxcompile.json` | `b6bef6f` | **P8** | Compile the diagonal phase + permutation gather kernels with `mx.compile`. Modest net win at large n (1.09–1.12× at 20–22q). |
| `1eb61db-p4-complex64.json` | `1eb61db` | **P4** | Native complex64 storage (single array, one complex tensordot vs four real ones, Metal 1q kernel dropped). Neutral on this random workload; faster on dense/fused circuits. Refutes the spec's "SoA 6.9×" claim on MLX 0.31. |

Two steps are not represented by a data file: **P3** (`6178f2f`, cache device
constants) and **P7** (`351376c`, retune auto-select crossover) — both verified
by the test suite, with no measurable effect on this microbenchmark. **P5**
(einsum dense path) was tried and reverted after an A/B regression — see the
plan.

`benchmark.png` covers baseline→P1→P2; `benchmark-2.png` is the full comparison
across all five recorded runs (log-scale runtime + speedup over CPU).

## Metal backend (Step 15)

`<commit>-metal.json` is produced by `benchmarks/bench_metal.py` (CPU vs MLX vs
Metal, depth-30 random circuit) and plotted by `benchmarks/plot_metal.py` into
`benchmark-3.png`. The Metal backend (PyObjC driver, 64-bit indexing, in-place)
is the only backend that runs **31-32q** — MLX's `int32` `ShapeElem` rejects
`>=2**31` amplitudes (Gate 0, `docs/plan.md`).

Circuit time (ms), depth-30 random circuit on Apple M5 Max (128 GiB),
`bedccb3-metal.json`:

| qubits | state size | CPU | MLX | Metal |
|---|---|---|---|---|
| 16 | 0.5 MiB | 2.2 | 2.6 | 6.8 |
| 20 | 8 MiB | 29.0 | 9.1 | 7.4 |
| 24 | 128 MiB | 718 | 126 | 24.9 |
| 26 | 0.5 GiB | n/a | 630 | 85 |
| 28 | 2 GiB | n/a | 2,081 | 292 |
| 30 | 8 GiB | n/a | 15,637 | 1,188 |
| 31 | 16 GiB | n/a | **—** | 2,289 |
| 32 | 32 GiB | n/a | **—** | 4,599 |
| 33 | 64 GiB | n/a | **—** | 10,973 |

Two findings. (1) **Capacity:** Metal reaches **31-33q** (16/32/64 GiB states)
that no other backend can allocate — 33q (64 GiB) is the realistic ceiling on a
128 GiB machine and uses 64.06 GiB resident (in-place, single buffer). (2)
**Unexpected large-n speedup:** Metal also beats MLX from ~22q up, by ~13x at
30q. Metal scales as the bandwidth-bound ideal (~2x per +1 qubit: 1,188 → 2,289
→ 4,599 → 10,973); MLX degrades super-linearly because every gate double-buffers
and the permutation path builds full-width gather temporaries, so at 28-30q the
working set outgrows comfortable residency and the unified memory thrashes.
Below ~20q Metal is *slower* — it synchronises per gate (`waitUntilCompleted`)
while MLX fuses a lazy graph and CPU avoids dispatch latency — so auto-select
still uses CPU ≤16q and MLX 17-30q.

Large-n (>=28q) points are single-rep. MLX times there include real
memory-pressure effects and vary run-to-run (an earlier single-rep 30q hit
~65s), so treat them as indicative of the trend, not tight measurements; the
Metal points are stable. The 33q row needs a 64 GiB in-place buffer — the
benchmark touches only one amplitude to flush (a full `to_numpy` would add a
second 64 GiB copy and exhaust the 128 GiB), matching the plan's on-device
large-n guidance.

## Fusion width (Step 20)

`fusion_width.json` / `fusion_width.png` are produced by
`benchmarks/bench_fusion_width.py`, which sweeps `max_fused_qubits` ∈ 1..6 across
qubit counts and circuits, timing the real Simulator cost model (fuse + apply).
It explains **why macquerel defaults to `max_fused_qubits=4`**.

The optimal width **drifts upward with qubit count**: small n is composition-bound
(narrow fusion wins), large n is apply-bound (wide fusion wins, fewer passes over
the `2**n` state). Best width per cell, MLX on Apple M5 Max (`fusion_width.json`):

| circuit | 18q | 20q | 22q | 24q |
|---|---|---|---|---|
| QFT | 2 | 3 | 5 | 6 |
| random | 1 | 3 | 6 | 6 |
| QAOA | 2 | 3 | 4 | 5 |
| QV | 3 | 4 | 5 | 6 |

No single width is best everywhere, but the **normalized aggregate** across the
17-30q MLX tier (each cell scaled by its own fastest width) bottoms out at **4**:

| width | 1 | 2 | 3 | **4** | 5 | 6 |
|---|---|---|---|---|---|---|
| mean (time / per-cell best) | 2.01 | 1.70 | 1.45 | **1.23** | 1.39 | 2.27 |

So 4 is the aggregate winner across these cells — the robust zero-config default.
A naive autotuner that measured a single small-n point instead picked 2 and
regressed the large-n path by up to ~2×; see the write-up:
<https://github.com/guenp/macquerel/pull/8#issuecomment-4636543327>.

## Results summary

MLX runtime (ms) on the depth-50 random circuit, baseline → latest (P4), reps 9:

| qubits | CPU | MLX baseline | MLX P2 | MLX P4 | P4 vs CPU |
|---|---|---|---|---|---|
| 16 | 3.7 | 15.1 | 2.6 | 1.4 | **2.6× faster** |
| 18 | 14.1 | 32.3 | 3.3 | 2.5 | **5.6× faster** |
| 20 | 50.9 | 112.3 | 11.9 | 10.9 | **4.7× faster** |
| 22 | 297.0 | 353.6 | 52.4 | 51.3 | **5.8× faster** |

P1 + P2 flipped MLX from slower-than-CPU everywhere to faster; P3/P7/P8/P4 added
incremental gains and simplification. MLX is now **up to ~5.8× faster** at 22
qubits and wins across this range.

At large n (20–22q) the P8 (SoA) and P4 (complex64) MLX lines sit within the
benchmark's run-to-run noise floor (~10–15% from GPU clock/thermal variance):
the workload is memory-bandwidth-bound and a complex64 amplitude is exactly two
float32, so both layouts move identical bytes per gate. See the P4 note in
[`docs/plan.md`](../../docs/plan.md) for why low-reps runs can make complex64
*look* slower there. Use higher reps (9+) for stable large-n numbers.

## Reproducing

```bash
uv sync --extra viz   # matplotlib, for the plot

CID=$(git rev-parse --short HEAD)
uv run python tests/benchmarks/bench_backends.py \
    --qubits 16 18 20 22 --depth 50 --reps 3 \
    --json benchmarks/data/${CID}-mychange.json --no-chart

uv run python benchmarks/plot_results.py   # regenerates benchmark.png
```
