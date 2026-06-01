from collections import Counter

from macquerel.circuit import Circuit
from macquerel.simulator import Simulator


def test_basic_simulation():
    circuit = Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.rz(1, 0.3)
    circuit.measure_all()

    sim = Simulator(backend="cpu")
    result = sim.run(circuit, shots=1000)

    assert isinstance(result, Counter)
    assert all(isinstance(k, str) for k in result.keys())
    assert all(isinstance(v, int) for v in result.values())
    assert sum(result.values()) == 1000


def test_statevector_bell():
    import numpy as np

    circuit = Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)

    sim = Simulator(backend="cpu")
    sv = sim.statevector(circuit)

    expected = np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], dtype=np.complex64)
    assert np.allclose(np.abs(sv), np.abs(expected), atol=1e-5)


def test_no_measurement_run():
    circuit = Circuit(1)
    circuit.h(0)

    sim = Simulator(backend="cpu")
    result = sim.run(circuit, shots=500)

    assert isinstance(result, Counter)
    assert sum(result.values()) == 500
