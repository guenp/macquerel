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

### Noise channels

Channel builders append Kraus-operator channels; circuits containing them run only
on the `DensityMatrixSimulator` (the statevector simulators reject them).

| Method | Channel |
|---|---|
| `.bit_flip(q, p)` | X with probability `p` |
| `.phase_flip(q, p)` | Z with probability `p` |
| `.depolarizing(q, p)` | X, Y, Z each with probability `p/3` |
| `.amplitude_damping(q, gamma)` | \|1⟩ decays to \|0⟩ with probability `gamma` |
| `.phase_damping(q, gamma)` | Coherence decays by `sqrt(1-gamma)` |
| `.kraus(qubits, operators, name=...)` | Arbitrary channel from explicit Kraus matrices |

`.kraus` accepts any number of qubits; operators must be `2**len(qubits)`-dimensional
square matrices satisfying the completeness relation `Σₖ Kₖ†Kₖ = I` (validated at
build time). The factories behind the named builders are importable from
`macquerel.noise` (`bit_flip_kraus(p)`, …, plus `validate_kraus`).

## `Simulator(backend="auto", dtype="complex64", seed=None)`

Runs circuits. `backend` is one of `"auto"`, `"cpu"`, `"mlx"`, `"metal"`; `"auto"`
selects by qubit count (CPU ≤15q, Metal 16q+, with MLX serving 16–30q when Metal is
unavailable; `MACQUEREL_BACKEND_TIERS` pins or autotunes the boundary). `seed` makes
sampling reproducible.

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

## `BatchedSimulator(backend="auto", dtype="complex64", seed=None)`

Simulates many same-width circuits as one batched evolution — built for parameter
sweeps (VQE/QML), where running B small circuits one at a time pays the fixed
per-run costs B times. Circuits sharing a structure (same gate positions, targets,
controls) are evolved together as one `(B, 2**n)` tensor, one batched matmul per
gate position; mixed-structure batches are grouped automatically. `backend` is one
of `"auto"`, `"cpu"`, `"mlx"` (`"auto"` routes on the total size `log2(B) + n`).

### `statevectors(circuits) -> numpy.ndarray`

Final statevectors, shape `(len(circuits), 2 ** n_qubits)`. Measurements are ignored.

### `run(circuits, shots=1000) -> list[collections.Counter]`

Per-circuit counts. Circuits must contain explicit `measure(...)`/`measure_all()`
ops; one without any returns an empty `Counter`.

```python
from macquerel import BatchedSimulator
circuits = [ansatz(theta) for theta in thetas]
svs = BatchedSimulator().statevectors(circuits)   # one sweep, few kernel launches
```

## `DensityMatrixSimulator(backend="auto", dtype="complex64", seed=None)`

Simulates noisy circuits (any circuit, with or without channels) as a density
matrix. The matrix is carried as its vectorization — a `4 ** n_qubits` array the
statevector backends treat as a `2n`-qubit state — so backends, gate fusion, and
the memory model are shared with `Simulator`. `"auto"` selects the backend at the
*doubled* qubit count (an n-qubit density matrix moves as many bytes as a 2n-qubit
statevector); MLX caps at **15 qubits**, Metal at **16** (a 32 GiB state).

```python
from macquerel import Circuit, DensityMatrixSimulator

circuit = Circuit(2)
circuit.h(0).cx(0, 1).depolarizing(0, 0.05).measure_all()
dm = DensityMatrixSimulator()
counts = dm.run(circuit, shots=1000)
```

### `density_matrix(circuit) -> numpy.ndarray`

The final density matrix, shape `(2**n, 2**n)`. Materializes the full matrix on
the host — prefer the methods below for large `n`.

### `probabilities(circuit) -> numpy.ndarray`

Measurement probabilities `p(i) = ρᵢᵢ`, shape `(2**n,)`. Reads only the diagonal;
no full-matrix host copy on the CPU and Metal backends.

### `run(circuit, shots=1000) -> collections.Counter`

Sample measurement outcomes; mirrors `Simulator.run` semantics (per-`MeasureOp`
sampling, counters summed; a circuit without measurements samples all qubits).

### `expectation_pauli(circuit, pauli_strings) -> numpy.ndarray`

`tr(ρP)` for each `(coeff, [(pauli_char, qubit), ...])` term.

### `purity(circuit) -> float`

`tr(ρ²)` — 1.0 for pure states, `1/2**n` when fully mixed.

## Adapters

Available when the matching extra is installed:

```python
from macquerel import from_cirq      # pip install "macquerel[cirq]"
from macquerel import from_qiskit    # pip install "macquerel[qiskit]"
```

Both convert a foreign circuit into a macquerel `Circuit`.
