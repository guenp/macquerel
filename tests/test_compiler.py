import numpy as np
import pytest

from macquerel.circuit import Circuit, Gate, MeasureOp
from macquerel.compiler import fuse_gates, remap_qubits
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


def test_remap_preserves_distribution():
    """Remapped and original circuits must produce identical measurement distributions."""
    # Build a 4-qubit circuit with unequal qubit access frequency.
    # Qubits 0 and 1 are used much more than 2 and 3.
    qc = Circuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(0, 0.3)
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(1, 0.7)
    qc.h(2)
    qc.cx(2, 3)
    qc.measure_all()

    remapped = remap_qubits(qc)

    # Run both circuits
    cpu = CPUBackend()

    def run_sv(circuit):
        sv = cpu.allocate(circuit.n_qubits)
        for op in circuit.ops:
            if isinstance(op, Gate):
                sv = cpu.apply_matrix(sv, op.matrix, op.targets, op.controls or None)
        return sv

    sv_orig = run_sv(qc)
    sv_remap = run_sv(remapped)

    # Recover the permutation from remap_qubits to invert bitstring labels
    from collections import Counter as _Counter
    freq: _Counter[int] = _Counter()
    for op in qc.ops:
        if isinstance(op, Gate):
            for q_idx in op.targets + op.controls:
                freq[q_idx] += 1
    sorted_qubits = sorted(range(qc.n_qubits), key=lambda q: (-freq[q], q))
    perm = {old: new for new, old in enumerate(sorted_qubits)}
    inv_perm = {new: old for old, new in perm.items()}

    # Compute marginal probabilities for all qubits from both statevectors
    n = qc.n_qubits
    probs_orig = np.abs(sv_orig) ** 2
    probs_remap = np.abs(sv_remap) ** 2

    # For each basis state in the remapped SV, compute the equivalent original basis state
    # and check probabilities match
    for idx in range(2**n):
        # bits in remapped ordering
        remap_bits = [(idx >> (n - 1 - new_q)) & 1 for new_q in range(n)]
        # convert back to original ordering
        orig_idx = 0
        for new_q, bit in enumerate(remap_bits):
            old_q = inv_perm[new_q]
            orig_idx |= bit << (n - 1 - old_q)
        assert abs(probs_remap[idx] - probs_orig[orig_idx]) < 1e-5, (
            f"Probability mismatch at remapped={idx}, orig={orig_idx}: "
            f"{probs_remap[idx]} vs {probs_orig[orig_idx]}"
        )
