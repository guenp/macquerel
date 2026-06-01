"""Known-answer circuit tests."""
import numpy as np
import pytest

from macquerel.circuit import Circuit
from macquerel.simulator import Simulator
from macquerel.backends.cpu import CPUBackend
import macquerel.gates as g


@pytest.fixture
def cpu():
    return CPUBackend()


def test_bell_state():
    sim = Simulator(backend="cpu")
    qc = Circuit(2)
    qc.h(0)
    qc.cx(0, 1)
    sv = sim.statevector(qc)
    inv_sqrt2 = 1 / np.sqrt(2)
    expected = np.array([inv_sqrt2, 0, 0, inv_sqrt2], dtype=np.complex64)
    assert np.allclose(np.abs(sv), np.abs(expected), atol=1e-5)


def test_ghz():
    sim = Simulator(backend="cpu")
    qc = Circuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    sv = sim.statevector(qc)
    inv_sqrt2 = 1 / np.sqrt(2)
    assert abs(sv[0] - inv_sqrt2) < 1e-5
    assert abs(sv[7] - inv_sqrt2) < 1e-5
    assert np.allclose(sv[1:7], 0, atol=1e-5)


def test_grover_2qubit_marked_11():
    """Grover search on 2 qubits with marked state |11⟩."""
    cpu = CPUBackend()
    sv = cpu.allocate(2)

    # Initialize uniform superposition
    sv = cpu.apply_matrix(sv, g.H(), [0])
    sv = cpu.apply_matrix(sv, g.H(), [1])

    # Oracle: flip sign of |11⟩ — that's CZ
    sv = cpu.apply_matrix(sv, g.CZ(), [0, 1])

    # Diffusion operator: 2|+><+| - I
    # = H X H (for each qubit), then CZ, then H X H
    sv = cpu.apply_matrix(sv, g.H(), [0])
    sv = cpu.apply_matrix(sv, g.H(), [1])
    sv = cpu.apply_matrix(sv, g.X(), [0])
    sv = cpu.apply_matrix(sv, g.X(), [1])
    sv = cpu.apply_matrix(sv, g.CZ(), [0, 1])
    sv = cpu.apply_matrix(sv, g.X(), [0])
    sv = cpu.apply_matrix(sv, g.X(), [1])
    sv = cpu.apply_matrix(sv, g.H(), [0])
    sv = cpu.apply_matrix(sv, g.H(), [1])

    # After 1 Grover iteration on 2 qubits, |11⟩ should have highest amplitude
    probs = np.abs(sv) ** 2
    assert np.argmax(probs) == 3  # |11⟩ = index 3


def test_qft_4qubit():
    cpu = CPUBackend()
    n = 4
    sv = cpu.allocate(n)
    sv[0] = 0
    sv[1] = 1  # |0001⟩

    # Apply QFT
    for i in range(n):
        sv = cpu.apply_matrix(sv, g.H(), [i])
        for j in range(i + 1, n):
            angle = np.pi / (2 ** (j - i))
            sv = cpu.apply_matrix(sv, g.P(angle), [j], [i])

    # Bit-reversal
    sv_shaped = sv.reshape((2,) * n)
    sv_shaped = np.transpose(sv_shaped, list(range(n - 1, -1, -1)))
    sv_qft = sv_shaped.reshape(-1)

    # Quantum QFT uses exp(+2πi); numpy fft uses exp(-2πi), so use ifft as reference
    input_vec = np.zeros(2**n, dtype=np.complex128)
    input_vec[1] = 1.0
    qft_ref = np.fft.ifft(input_vec) * np.sqrt(2**n)

    assert np.allclose(sv_qft, qft_ref, atol=1e-4)


def test_auto_backend_bell():
    """Simulator with backend='auto' must run correctly on a Bell circuit."""
    sim = Simulator()  # auto
    qc = Circuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    result = sim.run(qc, shots=1000)
    assert sum(result.values()) == 1000
    # GHZ/Bell: only '00' and '11' should appear
    assert set(result.keys()) <= {"00", "11"}
    # Both outcomes should have roughly 50% probability
    for bits in ("00", "11"):
        assert 350 < result.get(bits, 0) < 650, f"Unexpected count for {bits}: {result}"
