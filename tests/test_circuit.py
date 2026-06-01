from collections import Counter

import pytest

from macquerel.circuit import Circuit, Gate, MeasureOp
from macquerel.simulator import Simulator


def test_circuit_build():
    qc = Circuit(3)
    qc.h(0).cx(0, 1).cx(0, 2)
    gates = [op for op in qc.ops if isinstance(op, Gate)]
    assert len(gates) == 3


def test_measure_all_appends_measure_op():
    qc = Circuit(3)
    qc.h(0)
    qc.measure_all()
    assert isinstance(qc.ops[-1], MeasureOp)
    assert qc.ops[-1].qubits == [0, 1, 2]


def test_measure_specific_qubits():
    qc = Circuit(4)
    qc.measure([0, 2])
    assert isinstance(qc.ops[-1], MeasureOp)
    assert qc.ops[-1].qubits == [0, 2]


def test_qubit_out_of_range_raises():
    qc = Circuit(2)
    with pytest.raises(ValueError):
        qc.h(2)
    with pytest.raises(ValueError):
        qc.h(-1)


def test_duplicate_qubit_raises():
    qc = Circuit(2)
    with pytest.raises(ValueError):
        qc.cx(0, 0)


def test_run_returns_counter():
    qc = Circuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    sim = Simulator(backend="cpu")
    result = sim.run(qc, shots=200)

    assert isinstance(result, Counter)
    assert sum(result.values()) == 200
    assert all(len(k) == 2 for k in result.keys())


def test_all_gate_methods():
    """Smoke test that all gate builder methods work without error."""
    qc = Circuit(3)
    qc.i(0)
    qc.h(0)
    qc.x(1)
    qc.y(2)
    qc.z(0)
    qc.s(1)
    qc.t(2)
    qc.rx(0, 0.5)
    qc.ry(1, 0.3)
    qc.rz(2, 1.0)
    qc.p(0, 0.7)
    qc.cx(0, 1)
    qc.cz(1, 2)
    qc.swap(0, 2)
    qc.cp(0, 1, 0.4)
    assert len([op for op in qc.ops if isinstance(op, Gate)]) == 15
