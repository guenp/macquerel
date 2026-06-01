"""Edge case tests."""
import numpy as np
import pytest

from macquerel.circuit import Circuit, Gate
from macquerel.simulator import Simulator
from macquerel.backends.cpu import CPUBackend
import macquerel.gates as g


def test_single_qubit_circuit():
    sim = Simulator(backend="cpu")
    qc = Circuit(1)
    qc.h(0)
    qc.measure_all()
    result = sim.run(qc, shots=100)
    assert sum(result.values()) == 100
    assert set(result.keys()).issubset({"0", "1"})


def test_all_identity_circuit():
    """Applying identity gates should leave statevector as |0⟩."""
    n = 3
    cpu = CPUBackend()
    sv = cpu.allocate(n)
    for q in range(n):
        sv = cpu.apply_matrix(sv, g.I(), [q])
    expected = np.zeros(2**n, dtype=np.complex64)
    expected[0] = 1.0
    assert np.allclose(sv, expected, atol=1e-6)


def test_empty_circuit():
    """Empty circuit statevector should be |0⟩."""
    sim = Simulator(backend="cpu")
    qc = Circuit(3)
    sv = sim.statevector(qc)
    expected = np.zeros(8, dtype=np.complex64)
    expected[0] = 1.0
    assert np.allclose(sv, expected, atol=1e-6)


def test_multi_control_gate():
    """CNOT with an extra control (Toffoli-like) on 5 qubits."""
    cpu = CPUBackend()
    n = 5
    sv = cpu.allocate(n)

    # Set all control qubits to |1⟩: apply X to qubits 0, 1, 2
    sv = cpu.apply_matrix(sv, g.X(), [0])
    sv = cpu.apply_matrix(sv, g.X(), [1])
    sv = cpu.apply_matrix(sv, g.X(), [2])

    # Now qubits 0,1,2 are all |1⟩. Apply X to qubit 3 controlled on qubit 2.
    sv = cpu.apply_matrix(sv, g.X(), [3], [2])

    # qubit 3 should now be |1⟩ since control (qubit 2) was |1⟩
    sv_shaped = sv.reshape((2,) * n)
    # marginal prob of qubit 3 being 1
    probs = np.abs(sv_shaped) ** 2
    sum_axes = tuple(i for i in range(n) if i != 3)
    p_qubit3 = np.sum(probs, axis=sum_axes)
    assert p_qubit3[1] > 0.99, f"qubit 3 not in |1⟩: prob={p_qubit3[1]}"


def test_circuit_n_qubits_one():
    qc = Circuit(1)
    qc.x(0)
    sim = Simulator(backend="cpu")
    sv = sim.statevector(qc)
    assert np.allclose(sv, [0, 1], atol=1e-6)


def test_invalid_n_qubits():
    with pytest.raises(ValueError):
        Circuit(0)
