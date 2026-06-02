# Contributing

## Development setup

This project uses [`uv`](https://docs.astral.sh/uv/); `uv.lock` is canonical and CI runs `UV_FROZEN=true`.

```sh
git clone https://github.com/guenp/macquerel
cd macquerel
uv sync
pre-commit install
```

The MLX and Metal backends only run on Apple Silicon; their tests are skipped automatically
elsewhere. The `cirq` and `qiskit` adapter tests are skipped unless those extras are installed.

```sh
uv sync --extra mlx --extra metal   # Apple Silicon GPU backends
uv sync --all-extras                # also cirq, qiskit, matplotlib
```

## Running checks locally

```sh
uv run pytest
uv run ruff check
uv run ruff format --check
uv run ty check src/
pre-commit run --all-files
```

## Pull request workflow

1. Branch off `main`, make changes, add/update tests.
2. Run the checks above.
3. Open a PR. PR titles become release notes via `gh release create --generate-notes`,
   so write them imperative and user-facing.
4. CI must pass: lint, the test matrix, and `pip-audit`.

Reflect user-visible changes in [CHANGELOG.md](CHANGELOG.md) under `[Unreleased]`.

## Releasing

Bump `version` in `pyproject.toml`, then push a matching tag:

```sh
git tag v0.1.0 && git push origin v0.1.0
```

The release workflow verifies the tag matches the version, builds, publishes to PyPI
via Trusted Publishing (OIDC), and creates a GitHub Release.
