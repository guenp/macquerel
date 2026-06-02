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

## Choosing a backend

```python
Simulator()                 # auto: CPU ≤16q, MLX 17–30q, Metal 31q+
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
