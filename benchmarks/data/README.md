# Benchmark data

The maintained benchmark suite focuses on three outputs:

- `framework_comparison.json/.png`: macquerel CPU, MLX, and Metal compared with Qiskit Aer and Qulacs.
- `fusion_width.json/.png`: runtime vs `max_fused_qubits`.
- `version_regression.json/.png`: released macquerel versions compared with the current checkout for CPU, MLX, and Metal.

Regenerate the plots with:

```bash
uv sync --extra viz

uv run python benchmarks/bench_statevector.py \
  --json benchmarks/data/framework_comparison.json \
  --plot benchmarks/data/framework_comparison.png

uv run python benchmarks/bench_fusion_width.py \
  --json benchmarks/data/fusion_width.json \
  --plot benchmarks/data/fusion_width.png

uv run python benchmarks/bench_versions.py \
  --versions latest \
  --json benchmarks/data/version_regression.json \
  --plot benchmarks/data/version_regression.png
```

`bench_versions.py` is also used by CI on pull requests. The CI job runs a small CPU-only
comparison against the latest PyPI release and prints the benchmark table to stdout. On Apple
machines, pass `--backends cpu mlx metal --extras mlx metal` to include the optional GPU
backends for released-version comparisons.
