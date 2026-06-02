# Benchmark data

Per-commit results from `tests/benchmarks/bench_backends.py` (CPU vs MLX, depth-50
random circuit, qubits 16/18/20/22, reps 3), plus the comparison plot rendered by
`benchmarks/plot_results.py`.

Files are named `<commit>-<label>.json` so each run is traceable to the code that
produced it. The optimizations correspond to the **MLX backend performance
optimization** section of [`docs/plan.md`](../../docs/plan.md).

## Runs

| File | Commit | Step | What changed |
|---|---|---|---|
| `a17db04-baseline.json` | `a17db04` | — | Baseline before the performance work (after the earlier permutation-loop vectorization and native controlled gates). MLX slower than CPU everywhere. |
| `3c74da8-p1-defer-eval.json` | `3c74da8` | **P1** | Defer MLX evaluation across gates — removed per-gate `mx.eval()` so the lazy graph fuses across gates; evaluation forced only at segment boundaries. |
| `81e96e3-p2-ondevice-perm.json` | `81e96e3` | **P2** | Build the permutation gather index on-device with `mx.arange` + bitwise ops, eliminating the per-gate host-side NumPy table build and host→device copy. |

`benchmark.png` is the two-panel chart (log-scale runtime + speedup over CPU)
generated from all of the above.

## Results summary

MLX runtime (ms) on the depth-50 random circuit:

| qubits | CPU | MLX baseline | MLX P1 | MLX P2 | P2 vs CPU |
|---|---|---|---|---|---|
| 16 | 3.7 | 15.7 | 8.6 | 4.2 | 0.89× |
| 18 | 14.0 | 32.5 | 26.2 | 5.7 | **2.4× faster** |
| 20 | 50.4 | 114.1 | 94.3 | 11.8 | **4.3× faster** |
| 22 | 290.1 | 354.2 | 314.8 | 53.0 | **5.5× faster** |

P1 + P2 took MLX from slower-than-CPU everywhere to **2.4–5.5× faster** at 18–22
qubits. CPU still wins at 16 qubits (dispatch-bound regime). Remaining steps
(P3–P8) are tracked in the plan.

## Reproducing

```bash
uv sync --extra viz   # matplotlib, for the plot

CID=$(git rev-parse --short HEAD)
uv run python tests/benchmarks/bench_backends.py \
    --qubits 16 18 20 22 --depth 50 --reps 3 \
    --json benchmarks/data/${CID}-mychange.json --no-chart

uv run python benchmarks/plot_results.py   # regenerates benchmark.png
```
