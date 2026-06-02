"""Round-trip tests for Cirq and Qiskit adapters. Skipped when extras are absent."""

import numpy as np
import pytest

# ── Cirq ──────────────────────────────────────────────────────────────────────

cirq = pytest.importorskip("cirq")


def test_cirq_bell():
    from macquerel.adapters.cirq import from_cirq
    from macquerel.simulator import Simulator

    q0, q1 = cirq.LineQubit.range(2)
    cirq_circuit = cirq.Circuit(
        [
            cirq.H(q0),
            cirq.CNOT(q0, q1),
            cirq.measure(q0, q1, key="result"),
        ]
    )

    qc = from_cirq(cirq_circuit)
    assert qc.n_qubits == 2

    sv = Simulator(backend="cpu").statevector(qc)
    expected = np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], dtype=np.complex64)
    assert np.allclose(np.abs(sv), np.abs(expected), atol=1e-5)


def test_cirq_unsupported_gate():
    from macquerel.adapters.cirq import from_cirq

    # CCX (Toffoli) is not in the supported set
    cirq_circuit = cirq.Circuit([cirq.CCX(*cirq.LineQubit.range(3))])
    with pytest.raises(NotImplementedError):
        from_cirq(cirq_circuit)


# ── Qiskit ────────────────────────────────────────────────────────────────────

qiskit = pytest.importorskip("qiskit")


def test_qiskit_bell():
    from qiskit import QuantumCircuit

    from macquerel.adapters.qiskit import from_qiskit
    from macquerel.simulator import Simulator

    qk_circuit = QuantumCircuit(2)
    qk_circuit.h(0)
    qk_circuit.cx(0, 1)
    qk_circuit.measure_all()

    qc = from_qiskit(qk_circuit)
    assert qc.n_qubits == 2

    result = Simulator(backend="cpu").run(qc, shots=1000)
    assert set(result.keys()) <= {"00", "11"}
    assert sum(result.values()) == 1000


def test_qiskit_unsupported_gate():
    from qiskit import QuantumCircuit

    from macquerel.adapters.qiskit import from_qiskit

    qk_circuit = QuantumCircuit(3)
    qk_circuit.ccx(0, 1, 2)  # Toffoli — not in supported set
    with pytest.raises(NotImplementedError):
        from_qiskit(qk_circuit)
