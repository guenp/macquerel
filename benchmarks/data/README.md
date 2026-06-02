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
