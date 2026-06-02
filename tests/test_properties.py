"""Property-based tests: norm preservation and fusion equivalence."""

import numpy as np

import macquerel.gates as g
from macquerel.backends.cpu import CPUBackend
from macquerel.circuit import Circuit
from macquerel.compiler import fuse_gates

GATE_FACTORIES = [
    lambda: (g.H(), [0]),
    lambda: (g.X(), [0]),
    lambda: (g.Y(), [0]),
    lambda: (g.Z(), [0]),
    lambda: (g.S(), [0]),
    lambda: (g.Rz(np.random.uniform(0, 2 * np.pi)), [0]),
]

TWO_QUBIT_FACTORIES = [
    lambda qubits: (g.CNOT(), list(qubits[:2])),
    lambda qubits: (g.CZ(), list(qubits[:2])),
    lambda qubits: (g.SWAP(), list(qubits[:2])),
]


def _random_circuit(n_qubits: int, depth: int, rng: np.random.Generator) -> Circuit:
    qc = Circuit(n_qubits)
    qubits = list(range(n_qubits))
    for _ in range(depth):
        if n_qubits >= 2 and rng.random() < 0.4:
            pair = rng.choice(qubits, size=2, replace=False).tolist()
            factory = TWO_QUBIT_FACTORIES[rng.integers(0, len(TWO_QUBIT_FACTORIES))]
            matrix, targets = factory(pair)
            from macquerel.circuit import Gate

            qc.ops.append(
                Gate(name="rand", matrix=matrix.astype(np.complex64), targets=targets, controls=[])
            )
        else:
            q = int(rng.integers(0, n_qubits))
            theta = float(rng.uniform(0, 2 * np.pi))
            matrix = g.Rz(theta)
            from macquerel.circuit import Gate

            qc.ops.append(Gate(name="Rz", matrix=matrix, targets=[q], controls=[]))
    return qc


def _run_circuit(qc: Circuit) -> np.ndarray:
    from macquerel.circuit import Gate

    cpu = CPUBackend()
    sv = cpu.allocate(qc.n_qubits)
    for op in qc.ops:
        if isinstance(op, Gate):
            sv = cpu.apply_matrix(sv, op.matrix, op.targets, op.controls or None)
    return sv


def test_norm_preservation_random():
    rng = np.random.default_rng(42)
    for _ in range(50):
        n = int(rng.integers(2, 9))
        depth = int(rng.integers(1, 21))
        qc = _random_circuit(n, depth, rng)

        from macquerel.circuit import Gate

        cpu = CPUBackend()
        sv = cpu.allocate(n)
        for op in qc.ops:
            if isinstance(op, Gate):
                sv = cpu.apply_matrix(sv, op.matrix, op.targets, op.controls or None)
            norm = np.linalg.norm(sv)
            assert abs(norm - 1.0) < 1e-4, f"norm={norm} after gate"


def test_fusion_equivalence_random():
    rng = np.random.default_rng(123)
    for _ in range(50):
        n = int(rng.integers(2, 6))
        depth = int(rng.integers(1, 15))
        qc = _random_circuit(n, depth, rng)

        sv_unfused = _run_circuit(qc)
        fused = fuse_gates(qc)
        sv_fused = _run_circuit(fused)

        assert np.allclose(sv_unfused, sv_fused, atol=1e-5), (
            f"fusion broke circuit: max diff {np.max(np.abs(sv_unfused - sv_fused))}"
        )
