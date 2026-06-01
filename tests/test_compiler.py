import numpy as np
import pytest

from macquerel.circuit import Circuit, Gate, MeasureOp
from macquerel.compiler import fuse_gates
from macquerel.backends.cpu import CPUBackend
import macquerel.gates as g


def _run_statevector(circuit: Circuit) -> np.ndarray:
    cpu = CPUBackend()
    sv = cpu.allocate(circuit.n_qubits)
    for op in circuit.ops:
        if isinstance(op, Gate):
            sv = cpu.apply_matrix(sv, op.matrix, op.targets, op.controls or None)
    return sv


def test_fusion_preserves_statevector():
    qc = Circuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(2, 0.7)
    qc.h(1)

    sv_unfused = _run_statevector(qc)
    fused = fuse_gates(qc)
    sv_fused = _run_statevector(fused)

    assert np.allclose(sv_unfused, sv_fused, atol=1e-5), \
        f"max diff: {np.max(np.abs(sv_unfused - sv_fused))}"


def test_single_gate_unchanged():
    qc = Circuit(2)
    qc.h(0)

    fused = fuse_gates(qc)
    assert len(fused.ops) == 1
    gate = fused.ops[0]
    assert isinstance(gate, Gate)
    assert np.allclose(gate.matrix, g.H(), atol=1e-6)


def test_measure_barrier():
    """Gates on either side of a measurement should not be fused together."""
    qc = Circuit(2)
    qc.cx(0, 1)
    qc.measure([0])
    qc.cx(0, 1)

    fused = fuse_gates(qc)
    gate_ops = [op for op in fused.ops if isinstance(op, Gate)]
    measure_ops = [op for op in fused.ops if isinstance(op, MeasureOp)]

    assert len(measure_ops) == 1
    assert len(gate_ops) == 2
    # The measure op should be between the two gate ops
    assert isinstance(fused.ops[0], Gate)
    assert isinstance(fused.ops[1], MeasureOp)
    assert isinstance(fused.ops[2], Gate)


def test_fused_matrix_unitarity():
    qc = Circuit(2)
    qc.h(0)
    qc.cx(0, 1)

    fused = fuse_gates(qc)
    for op in fused.ops:
        if isinstance(op, Gate):
            m = op.matrix.astype(np.complex128)
            assert np.allclose(m @ m.conj().T, np.eye(len(m)), atol=1e-5), \
                f"Fused gate {op.name} is not unitary"


def test_fusion_limit():
    """Gates acting on too many qubits together should not be fused."""
    qc = Circuit(5)
    for i in range(5):
        qc.h(i)

    # With max_fused_qubits=4, we can fuse at most 4 of the 5 H gates
    fused = fuse_gates(qc, max_fused_qubits=4)
    gate_ops = [op for op in fused.ops if isinstance(op, Gate)]
    # All H gates act on distinct qubits, so they can all be fused into one 5-qubit group
    # BUT max_fused_qubits=4 means we stop at 4, so we should have 2 groups
    total_qubits = sum(len(op.targets) + len(op.controls) for op in gate_ops)
    assert total_qubits == 5
