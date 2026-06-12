# ASV Benchmarks

This directory contains the [Airspeed Velocity](https://asv.readthedocs.io/)
harness for the existing `benchmarks/` suite. The benchmark cases reuse the
current circuit generators and timing paths where practical, but keep ASV
discovery isolated from the CLI scripts.

ASV runs each parameter combination in its own process, so cells are isolated
the same way `bench_statevector.py --worker` isolates them.

Run a quick local smoke test:

```bash
uv run --extra bench asv check
```

Run benchmarks for the current checkout:

```bash
uv run --extra bench asv run --quick
```

Build the report:

```bash
uv run --extra bench asv publish
uv run --extra bench asv preview
```

## Per-step GPU-perf benchmarks

`benchmarks/run_step_bench.sh <step-label> <commit> <backend> [...]` runs the
`statevector.StatevectorRuntime` benchmark for a single historical commit
through `asv run`, then exports the results with
`plot_steps.py --export-asv` into `benchmarks/data/steps/` in the same JSON
format the old `bench_statevector.py`-based runner wrote. `plot_steps.py`
renders the step charts from those JSONs. The step qubit grid (cpu 6–22,
mlx/metal 6–28) and min-of-3 timing live in `statevector.py`.
