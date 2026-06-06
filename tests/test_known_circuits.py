"""Known-answer circuit tests."""

import numpy as np
import pytest

import macquerel.gates as g
from macquerel.backends.cpu import CPUBackend
from macquerel.circuit import Circuit
from macquerel.simulator import Simulator


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


def _haar_special_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-random special unitary via QR of a complex Ginibre matrix (complex64)."""
    z = (rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    ph = np.diagonal(r) / np.abs(np.diagonal(r))
    u = q * ph
    return (u * np.linalg.det(u) ** (-1 / dim)).astype(np.complex64)


def _qv_layers(n: int, rng: np.random.Generator) -> list[tuple[np.ndarray, list[int]]]:
    """Quantum Volume model circuit as (SU(4) matrix, [a, b]) layers (depth = n)."""
    layers: list[tuple[np.ndarray, list[int]]] = []
    for _ in range(n):
        perm = rng.permutation(n)
        for i in range(0, n - 1, 2):
            u = _haar_special_unitary(4, rng)
            layers.append((u, [int(perm[i]), int(perm[i + 1])]))
    return layers


def test_quantum_volume_gates_are_special_unitary():
    """QV dense 2q gates are SU(4): unitary with determinant 1."""
    rng = np.random.default_rng(5)
    u = _haar_special_unitary(4, rng)
    assert np.allclose(u.conj().T @ u, np.eye(4), atol=1e-6)
    assert np.allclose(np.linalg.det(u), 1.0, atol=1e-5)


def test_quantum_volume_normalized():
    """A Quantum Volume circuit of Haar-random SU(4) gates preserves the norm."""
    n = 4
    rng = np.random.default_rng(7)
    qc = Circuit(n)
    for u, qubits in _qv_layers(n, rng):
        qc._add("su4", u, qubits)
    sv = Simulator(backend="cpu").statevector(qc)
    assert abs(float(np.linalg.norm(sv)) - 1.0) < 1e-5


def test_quantum_volume_inverse_is_identity():
    """QV circuit followed by its exact inverse returns to |0...0⟩ (known answer)."""
    n = 4
    rng = np.random.default_rng(11)
    layers = _qv_layers(n, rng)

    qc = Circuit(n)
    for u, qubits in layers:
        qc._add("su4", u, qubits)
    for u, qubits in reversed(layers):
        qc._add("su4_inv", u.conj().T.copy(), qubits)  # U† on the same target order

    sv = Simulator(backend="cpu").statevector(qc)
    expected = np.zeros(2**n, dtype=np.complex64)
    expected[0] = 1.0
    assert np.allclose(sv, expected, atol=1e-4)


def test_random_circuit_sampling_matches_statevector():
    """Random-circuit-sampling spot check: empirical shot frequencies track |ψ|²."""
    n = 4
    rng = np.random.default_rng(3)
    qc = Circuit(n)
    for q in range(n):
        qc.h(q)  # spread amplitude across the whole basis
    for _ in range(20):
        if rng.random() < 0.5:
            a, b = rng.choice(n, size=2, replace=False)
            qc.cx(int(a), int(b))
        else:
            q = int(rng.integers(n))
            qc.rx(q, float(rng.uniform(0, 2 * np.pi)))

    sim = Simulator(backend="cpu", seed=0)
    probs = np.abs(sim.statevector(qc)) ** 2

    shots = 20000
    qc.measure_all()
    counts = sim.run(qc, shots=shots)
    assert sum(counts.values()) == shots

    empirical = np.zeros(2**n)
    for bits, c in counts.items():
        empirical[int(bits, 2)] = c / shots
    # Sampling error at 20k shots is ~3e-3; 0.05 is a safe, flake-free bound.
    assert np.max(np.abs(empirical - probs)) < 0.05


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
