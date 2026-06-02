# macquerel

A quantum state-vector simulator for Apple Silicon.

## Documentation

- [`design_spec.md`](design_spec.md) — architecture, backend design decisions, optimization strategy, and benchmarking plan
- [`plan.md`](plan.md) — v0.1 implementation plan with step-by-step breakdown

## Quick start

```bash
uv sync            # CPU backend only
uv sync --extra mlx  # add MLX backend for Apple Silicon GPU
uv run pytest      # run test suite
```

See the root [`README.md`](../README.md) for full usage documentation.
