# Benchmark data

The maintained benchmark suite focuses on three outputs:

- `framework_comparison.json/.png`: macquerel CPU, MLX, and Metal compared with Qiskit Aer and Qulacs.
- `large/`: the same comparison pushed to 32 qubits, with one JSON per backend (each run in
  isolation so a memory-hungry backend doesn't crowd the others out). `large/aggregate.png`
  merges them into one chart and marks where Metal overtakes the best alternative.
- `fusion_width.json/.png`: runtime vs `max_fused_qubits`.
- `version_regression.json/.png`: released macquerel versions compared with the current checkout for CPU, MLX, and Metal.
- `mcm.json/.png`: mid-circuit measurement on QEC-style repetition-code syndrome extraction —
  runtime vs number of MCM rounds and vs qubit count, for both MCM semantics (`Simulator.run`
  aggregate sampling and per-shot projective collapse via `backend.measure`), per backend.
- `vqe.json/.png`: variational workloads — TFIM `<H>` evaluation time vs qubit count and vs
  ansatz depth per backend, plus full parameter-shift gradients vs parameter count comparing a
  per-circuit `Simulator` loop against `BatchedSimulator` (cpu/mlx).

Regenerate the plots with:

```bash
uv sync --extra bench

# On Apple Silicon, include macquerel's optional GPU backends too:
uv sync --extra bench --extra mlx --extra metal

uv run python benchmarks/bench_statevector.py \
  --json benchmarks/data/framework_comparison.json \
  --plot benchmarks/data/framework_comparison.png

# Merge the per-backend runs under large/ into one annotated chart:
uv run python benchmarks/bench_statevector.py \
  --aggregate benchmarks/data/large \
  --plot benchmarks/data/large/aggregate.png

uv run python benchmarks/bench_fusion_width.py \
  --json benchmarks/data/fusion_width.json \
  --plot benchmarks/data/fusion_width.png

uv run python benchmarks/bench_versions.py \
  --versions latest \
  --json benchmarks/data/version_regression.json \
  --plot benchmarks/data/version_regression.png

# Mid-circuit measurement (QEC syndrome extraction) and VQE workloads:
uv run python benchmarks/bench_mcm.py
uv run python benchmarks/bench_vqe.py
```

`bench_versions.py` is also used by CI on pull requests. The CI job runs a small CPU-only
comparison against the latest PyPI release and prints the benchmark table to stdout. On Apple
machines, pass `--backends cpu mlx metal --extras mlx metal` to include the optional GPU
backends for released-version comparisons.

`bench_statevector.py` prints backend availability before timing starts. If Qulacs is skipped
on Python 3.14, install it in an environment with a prebuilt wheel or provide its local
C++/Boost build prerequisites.
