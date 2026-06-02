# API

## `Circuit(n_qubits)`

A quantum circuit over `n_qubits`. Gate builders mutate the circuit and return `self`,
so calls chain.

```python
from macquerel import Circuit

circuit = Circuit(2)
circuit.h(0).cx(0, 1).measure_all()
```

### Gate builders

| Method | Gate |
|---|---|
| `.h(q)` | Hadamard |
| `.x(q)` / `.y(q)` / `.z(q)` | Pauli X / Y / Z |
| `.s(q)` / `.t(q)` | S / T phase |
| `.rx(q, theta)` / `.ry(q, theta)` / `.rz(q, theta)` | Rotation gates |
| `.p(q, lam)` | Phase gate |
| `.cx(control, target)` | CNOT |
| `.cz(control, target)` | Controlled-Z |
| `.swap(q0, q1)` | SWAP |
| `.cp(control, target, lam)` | Controlled-phase |
| `.measure(qubits)` | Measure a subset |
| `.measure_all()` | Measure every qubit |

## `Simulator(backend="auto", dtype="complex64", seed=None)`

Runs circuits. `backend` is one of `"auto"`, `"cpu"`, `"mlx"`, `"metal"`; `"auto"`
selects by qubit count (CPU ≤16q, MLX 17–30q, Metal 31q+). `seed` makes sampling
reproducible.

### `statevector(circuit) -> numpy.ndarray`

Return the final state vector as a complex NumPy array (length `2 ** n_qubits`).

```python
sim = Simulator()
sv = sim.statevector(circuit)
```

### `run(circuit, shots=1000) -> collections.Counter`

Sample measurement outcomes, returning a `Counter` mapping bitstrings to counts.

```python
counts = sim.run(circuit, shots=1000)
```

## Adapters

Available when the matching extra is installed:

```python
from macquerel import from_cirq      # pip install "macquerel[cirq]"
from macquerel import from_qiskit    # pip install "macquerel[qiskit]"
```

Both convert a foreign circuit into a macquerel `Circuit`.
