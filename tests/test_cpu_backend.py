import numpy as np
import pytest

from macquerel.backends.cpu import CPUBackend
import macquerel.gates as g


@pytest.fixture
def cpu():
    return CPUBackend()


def test_bell_state(cpu):
    sv = cpu.allocate(2)
    sv = cpu.apply_matrix(sv, g.H(), [0])
    sv = cpu.apply_matrix(sv, g.CNOT(), [0, 1])
    expected = np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], dtype=np.complex64)
    assert np.allclose(sv, expected, atol=1e-5)


def test_ghz_state(cpu):
    sv = cpu.allocate(3)
    sv = cpu.apply_matrix(sv, g.H(), [0])
    sv = cpu.apply_matrix(sv, g.CNOT(), [0, 1])
    sv = cpu.apply_matrix(sv, g.CNOT(), [0, 2])
    inv_sqrt2 = 1 / np.sqrt(2)
    assert abs(sv[0] - inv_sqrt2) < 1e-5
    assert abs(sv[7] - inv_sqrt2) < 1e-5
    assert np.allclose(sv[1:7], 0, atol=1e-5)


def test_qft_4qubit(cpu):
    """QFT on |0001⟩ should match numpy FFT (up to normalization and bit-reversal)."""
    n = 4
    sv = cpu.allocate(n)
    # prepare |0001⟩ = index 1
    sv[0] = 0
    sv[1] = 1

    # Apply QFT: H, controlled-P gates sequence
    for i in range(n):
        sv = cpu.apply_matrix(sv, g.H(), [i])
        for j in range(i + 1, n):
            angle = np.pi / (2 ** (j - i))
            sv = cpu.apply_matrix(sv, g.P(angle), [j], [i])

    # Reverse qubit order (bit reversal)
    sv_shaped = sv.reshape((2,) * n)
    sv_shaped = np.transpose(sv_shaped, list(range(n - 1, -1, -1)))
    sv_qft = sv_shaped.reshape(-1)

    # Quantum QFT uses exp(+2πi); numpy fft uses exp(-2πi), so use ifft as reference
    input_vec = np.zeros(2**n, dtype=np.complex128)
    input_vec[1] = 1.0
    qft_ref = np.fft.ifft(input_vec) * np.sqrt(2**n)

    assert np.allclose(sv_qft, qft_ref, atol=1e-4), f"max diff: {np.max(np.abs(sv_qft - qft_ref))}"


def test_norm_preservation(cpu):
    n = 4
    sv = cpu.allocate(n)
    gates_to_apply = [
        (g.H(), [0]),
        (g.CNOT(), [0, 1]),
        (g.Rz(0.5), [2]),
        (g.SWAP(), [0, 3]),
        (g.CZ(), [1, 2]),
    ]
    for gate_args in gates_to_apply:
        matrix = gate_args[0]
        targets = gate_args[1]
        controls = gate_args[2] if len(gate_args) > 2 else None
        sv = cpu.apply_matrix(sv, matrix, targets, controls)
        norm = np.linalg.norm(sv)
        assert abs(norm - 1.0) < 1e-5, f"norm {norm} after gate"


def test_measure_collapse(cpu):
    # Bell state: measure qubit 0
    sv = cpu.allocate(2)
    sv = cpu.apply_matrix(sv, g.H(), [0])
    sv = cpu.apply_matrix(sv, g.CNOT(), [0, 1])

    outcomes = cpu.measure(sv, [0], collapse=True)
    assert outcomes[0] in (0, 1)
    assert abs(np.linalg.norm(sv) - 1.0) < 1e-5

    # Post-collapse: qubit 1 should match qubit 0's outcome
    sv_shaped = sv.reshape(2, 2)
    if outcomes[0] == 0:
        assert np.allclose(np.abs(sv_shaped[1, :]), 0, atol=1e-5)
    else:
        assert np.allclose(np.abs(sv_shaped[0, :]), 0, atol=1e-5)


def test_sampling_ghz(cpu):
    """GHZ 3-qubit should give ~50/50 between 000 and 111."""
    sv = cpu.allocate(3)
    sv = cpu.apply_matrix(sv, g.H(), [0])
    sv = cpu.apply_matrix(sv, g.CNOT(), [0, 1])
    sv = cpu.apply_matrix(sv, g.CNOT(), [0, 2])

    shots = 10000
    counts = cpu.sample(sv, [0, 1, 2], shots)

    assert set(counts.keys()).issubset({"000", "111"})
    n0 = counts.get("000", 0)
    n1 = counts.get("111", 0)
    assert n0 + n1 == shots

    # Roughly 50/50: allow 5% margin
    assert 0.45 * shots <= n0 <= 0.55 * shots, f"n0={n0} outside expected ~50%"
    assert 0.45 * shots <= n1 <= 0.55 * shots, f"n1={n1} outside expected ~50%"
