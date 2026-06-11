# Getting Started

## Requirements

- Python 3.11+
- macOS on Apple Silicon for the MLX and Metal backends (the CPU backend is cross-platform)

## Install

```sh
pip install macquerel              # CPU backend only
pip install "macquerel[mlx]"       # + MLX (Apple Silicon GPU)
pip install "macquerel[mlx,metal]" # + Metal (31q+ via PyObjC)
```

## Building a circuit

Gate builders return `self`, so calls chain:

```python
from macquerel import Circuit, Simulator

circuit = Circuit(3)
circuit.h(0).cx(0, 1).cx(1, 2)   # GHZ state
circuit.measure_all()

sim = Simulator()
counts = sim.run(circuit, shots=1000)
```

For the raw amplitudes, skip measurement and call `statevector`:

```python
circuit = Circuit(2)
circuit.h(0).cx(0, 1)
sv = sim.statevector(circuit)      # [0.707+0j, 0, 0, 0.707+0j]
```

## Simulating noise

Channel builders add Kraus-operator noise to a circuit; run it with the
`DensityMatrixSimulator`:

```python
from macquerel import Circuit, DensityMatrixSimulator

circuit = Circuit(2)
circuit.h(0).cx(0, 1)              # Bell state...
circuit.depolarizing(0, 0.05)      # ...through a depolarizing channel
circuit.amplitude_damping(1, 0.1)  # ...and T1 decay
circuit.measure_all()

dm = DensityMatrixSimulator()
counts = dm.run(circuit, shots=1000)   # noisy counts: some 01/10 leakage
rho = dm.density_matrix(circuit)       # full (2**n, 2**n) density matrix
purity = dm.purity(circuit)            # tr(rho^2) < 1 under noise
```

Built-in channels: `bit_flip`, `phase_flip`, `depolarizing`, `amplitude_damping`,
`phase_damping`, plus arbitrary (multi-qubit) channels via
`circuit.kraus(qubits, operators)`. An n-qubit density matrix costs what a
2n-qubit statevector costs, so noisy simulation tops out around 16 qubits —
see the [API reference](reference/api.md) for details.

## Choosing a backend

```python
Simulator()                 # auto: CPU ≤15q, Metal 16q+ (MLX fallback 16–30q)
Simulator(backend="cpu")
Simulator(backend="mlx")
Simulator(backend="metal")
```

## Local development

```sh
uv sync
uv run pytest
uv run zensical serve   # live-preview the docs at http://localhost:8000
```
