# macquerel

<img src="logo.png" width="200" />

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
uv run python tests/benchmarks/bench_backends.py --qubits 10 14 --reps 1
```

Full benchmarks (may take several minutes at 24+ qubits):

```bash
# CPU vs MLX backend — random circuits, various qubit counts
uv run python tests/benchmarks/bench_backends.py

# Single-gate throughput (GB/s)
uv run python benchmarks/bench_single_gate.py

# QFT, random, QAOA circuits + fusion-width sweep
uv run python benchmarks/bench_circuits.py
```

All scripts print results live as each configuration completes. Pass `--json FILE` to save results for later comparison.

## Requirements

- Python ≥ 3.11
- numpy ≥ 1.24
- mlx ≥ 0.3.0 *(optional, Apple Silicon only)*
