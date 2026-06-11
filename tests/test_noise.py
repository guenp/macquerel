"""Tests for the Kraus channel factories, validation, and circuit builders."""

import numpy as np
import pytest

from macquerel.circuit import ChannelOp, Circuit
from macquerel.noise import (
    amplitude_damping_kraus,
    bit_flip_kraus,
    channel_superoperator,
    depolarizing_kraus,
    phase_damping_kraus,
    phase_flip_kraus,
    validate_kraus,
)

FACTORIES = [
    bit_flip_kraus,
    phase_flip_kraus,
    depolarizing_kraus,
    amplitude_damping_kraus,
    phase_damping_kraus,
]


# --- factories ---


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize("p", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_factory_completeness(factory, p):
    """Every built-in channel satisfies sum_k K_k^dagger K_k = I."""
    ops = factory(p)
    acc = sum(op.conj().T @ op for op in ops)
    np.testing.assert_allclose(acc, np.eye(2), atol=1e-6)


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize("p", [-0.01, 1.01, 2.0])
def test_factory_rejects_bad_probability(factory, p):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        factory(p)


def test_bit_flip_kraus_matrices():
    k0, k1 = bit_flip_kraus(0.25)
    np.testing.assert_allclose(k0, np.sqrt(0.75) * np.eye(2), atol=1e-7)
    np.testing.assert_allclose(k1, np.sqrt(0.25) * np.array([[0, 1], [1, 0]]), atol=1e-7)


def test_amplitude_damping_kraus_matrices():
    k0, k1 = amplitude_damping_kraus(0.36)
    np.testing.assert_allclose(k0, [[1, 0], [0, 0.8]], atol=1e-7)
    np.testing.assert_allclose(k1, [[0, 0.6], [0, 0]], atol=1e-7)


def test_depolarizing_has_four_operators():
    assert len(depolarizing_kraus(0.3)) == 4


# --- validate_kraus ---


def test_validate_accepts_identity_channel():
    validate_kraus([np.eye(2)], 1)


def test_validate_accepts_two_qubit_channel():
    # Correlated bit flip: X(x)X with probability p.
    p = 0.2
    xx = np.kron([[0, 1], [1, 0]], [[0, 1], [1, 0]])
    validate_kraus([np.sqrt(1 - p) * np.eye(4), np.sqrt(p) * xx], 2)


def test_validate_rejects_empty():
    with pytest.raises(ValueError, match="at least one"):
        validate_kraus([], 1)


def test_validate_rejects_wrong_shape():
    with pytest.raises(ValueError, match="shape"):
        validate_kraus([np.eye(4)], 1)


def test_validate_rejects_non_trace_preserving():
    with pytest.raises(ValueError, match="completeness"):
        validate_kraus([0.5 * np.eye(2)], 1)


# --- channel_superoperator ---


def test_superoperator_identity_channel_is_identity():
    s = channel_superoperator([np.eye(2)])
    np.testing.assert_allclose(s, np.eye(4), atol=1e-7)


def test_superoperator_matches_direct_application():
    """S @ vec(rho) == sum_k K rho K^dagger for a random rho and channel."""
    rng = np.random.default_rng(7)
    a = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    rho = a @ a.conj().T
    rho /= np.trace(rho)
    ops = amplitude_damping_kraus(0.3)
    expected = sum(k @ rho @ k.conj().T for k in ops)
    s = channel_superoperator(ops).astype(np.complex128)
    out = (s @ rho.reshape(-1)).reshape(2, 2)
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_superoperator_preserves_trace():
    """Row-major vec: tr(rho') = sum over diagonal positions of S @ vec(rho)."""
    rng = np.random.default_rng(3)
    a = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    rho = a @ a.conj().T
    rho /= np.trace(rho)
    p = 0.4
    xx = np.kron([[0, 1], [1, 0]], [[0, 1], [1, 0]])
    s = channel_superoperator([np.sqrt(1 - p) * np.eye(4), np.sqrt(p) * xx])
    out = (s.astype(np.complex128) @ rho.reshape(-1)).reshape(4, 4)
    assert abs(np.trace(out) - 1.0) < 1e-6


# --- Circuit builders ---


def test_builders_append_channel_ops_and_chain():
    qc = (
        Circuit(3)
        .h(0)
        .bit_flip(0, 0.1)
        .phase_flip(1, 0.2)
        .depolarizing(2, 0.3)
        .amplitude_damping(0, 0.4)
        .phase_damping(1, 0.5)
    )
    channels = [op for op in qc.ops if isinstance(op, ChannelOp)]
    assert len(channels) == 5
    assert channels[0].qubits == [0]
    assert channels[0].name.startswith("BitFlip")
    assert all(k.dtype == np.complex64 for ch in channels for k in ch.kraus)


def test_generic_kraus_builder():
    p = 0.15
    xx = np.kron([[0, 1], [1, 0]], [[0, 1], [1, 0]])
    qc = Circuit(2).kraus([0, 1], [np.sqrt(1 - p) * np.eye(4), np.sqrt(p) * xx], name="CorrelatedX")
    (op,) = qc.ops
    assert isinstance(op, ChannelOp)
    assert op.qubits == [0, 1]
    assert op.name == "CorrelatedX"


def test_kraus_builder_validates_operators():
    with pytest.raises(ValueError, match="completeness"):
        Circuit(1).kraus([0], [0.7 * np.eye(2)])


def test_channel_qubit_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        Circuit(2).bit_flip(2, 0.1)


def test_channel_duplicate_qubits_rejected():
    p = 0.15
    xx = np.kron([[0, 1], [1, 0]], [[0, 1], [1, 0]])
    with pytest.raises(ValueError, match="Duplicate"):
        Circuit(2).kraus([0, 0], [np.sqrt(1 - p) * np.eye(4), np.sqrt(p) * xx])


def test_builder_probability_validation_propagates():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        Circuit(1).depolarizing(0, 1.5)
