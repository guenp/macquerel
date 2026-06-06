# macquerel

<img src="https://raw.githubusercontent.com/guenp/macquerel/main/logo.png" width="200" />

A quantum state-vector simulator targeting Apple Silicon's unified-memory architecture.

## Features

- **CPU backend** — pure NumPy reference implementation using the tensordot reshape trick
- **MLX backend** — Apple Silicon GPU acceleration via [MLX](https://ml-explore.github.io/mlx/), with graceful fallback on other platforms
- **Gate fusion compiler** — greedy left-to-right fusion of adjacent gates (up to 4 qubits) into composite unitaries
- **Gate classification** — automatically classifies gates as `diagonal`, `permutation`, or `dense` for optimized dispatch
- **Full gate library** — I, H, X, Y, Z, S, T, Rx, Ry, Rz, P, CNOT, CZ, SWAP, CP

## Installation

Requires Python 3.11+.

```bash
uv sync
```

To enable the MLX backend on Apple Silicon:

```bash
uv sync --extra mlx
```

## Usage

```python
from macquerel import Circuit, Simulator

# Bell state
circuit = Circuit(2)
circuit.h(0).cx(0, 1).measure_all()

sim = Simulator()               # defaults to CPU backend
counts = sim.run(circuit, shots=1000)
print(counts)  # Counter({'00': ~500, '11': ~500})
```

```python
# Statevector (no measurement)
circuit = Circuit(2)
circuit.h(0).cx(0, 1)

sv = sim.statevector(circuit)
print(sv)  # [0.707+0j, 0+0j, 0+0j, 0.707+0j]
```

```python
# Apple Silicon GPU backend
sim = Simulator(backend="mlx")
```

### Available gates

| Method | Gate |
|---|---|
| `.h(q)` | Hadamard |
| `.x(q)` / `.y(q)` / `.z(q)` | Pauli X/Y/Z |
| `.s(q)` / `.t(q)` | S / T phase |
| `.rx(q, θ)` / `.ry(q, θ)` / `.rz(q, θ)` | Rotation gates |
| `.p(q, λ)` | Phase gate |
| `.cx(ctrl, tgt)` | CNOT |
| `.cz(ctrl, tgt)` | Controlled-Z |
| `.swap(q0, q1)` | SWAP |
| `.cp(ctrl, tgt, λ)` | Controlled-Phase |
| `.measure(qubits)` / `.measure_all()` | Measurement |

Builders return `self` for chaining.

## Architecture

```
macquerel/
├── gates.py         # Gate matrices + classify()
├── circuit.py       # Circuit / Gate / MeasureOp dataclasses
├── compiler.py      # Gate fusion (fuse_gates)
├── simulator.py     # Simulator — statevector() and run()
└── backends/
    ├── cpu.py       # NumPy backend
    └── mlx_backend.py  # MLX backend (Apple Silicon)
```

## Running tests

```bash
uv run pytest
```

MLX backend tests are automatically skipped when MLX is not installed.

## Benchmarks

Quick smoke run (completes in a few seconds):

```bash
uv run python benchmarks/bench_versions.py --versions latest --backends cpu --qubits 6 8 --depth 8 --reps 1
```

Full benchmark commands:

```bash
# macquerel CPU/MLX/Metal vs Qiskit Aer and Qulacs
uv run python benchmarks/bench_statevector.py \
  --json benchmarks/data/framework_comparison.json \
  --plot benchmarks/data/framework_comparison.png

# Fusion-width sweep
uv run python benchmarks/bench_fusion_width.py \
  --json benchmarks/data/fusion_width.json \
  --plot benchmarks/data/fusion_width.png

# Released-version regression comparison for CPU/MLX/Metal
uv run python benchmarks/bench_versions.py \
  --versions latest --json benchmarks/data/version_regression.json \
  --plot benchmarks/data/version_regression.png
```

Install `uv sync --extra viz` before regenerating plots. Optional frameworks and Apple-only
backends are reported as skipped when they are unavailable.

## Requirements

- Python ≥ 3.11
- numpy ≥ 1.24
- mlx ≥ 0.3.0 *(optional, Apple Silicon only)*
