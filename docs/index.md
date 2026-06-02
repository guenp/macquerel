# macquerel

A quantum state-vector simulator targeting Apple Silicon's unified-memory architecture.

!!! note
    These docs are built with [Zensical](https://zensical.org) and deployed to
    GitHub Pages on every push to `main`.

## Install

```sh
pip install macquerel          # CPU backend
pip install "macquerel[mlx]"   # + Apple Silicon GPU backend
```

## Quickstart

```python
from macquerel import Circuit, Simulator

circuit = Circuit(2)
circuit.h(0).cx(0, 1).measure_all()

sim = Simulator()                  # auto-selects a backend
counts = sim.run(circuit, shots=1000)
print(counts)  # Counter({'00': ~500, '11': ~500})
```

## Backends

| Backend | Range | Notes |
|---|---|---|
| CPU (NumPy) | ≤16q | Reference implementation |
| MLX | 17–30q | Apple Silicon GPU |
| Metal | 31–33q | PyObjC driver, 64-bit indexing, in-place |

`Simulator()` selects automatically by qubit count; pass `backend="cpu" / "mlx" / "metal"`
to force one.
