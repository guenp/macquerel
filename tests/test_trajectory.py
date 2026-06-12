"""Tests for TrajectorySimulator: noiseless equivalence with the statevector
simulator, stochastic agreement with DensityMatrixSimulator on noisy circuits,
the non-diagonal-effect fallback path, sampling semantics, seeding, and
backend parity."""

import numpy as np
import pytest

from macquerel import Circuit, DensityMatrixSimulator, Simulator, TrajectorySimulator

try:
    from macquerel.backends.metal_backend import _METAL_AVAILABLE
except ImportError:  # pragma: no cover
    _METAL_AVAILABLE = False
try:
    import mlx.core  # noqa: F401

    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False

BACKENDS = ["cpu"]
if _METAL_AVAILABLE:
    BACKENDS.append("metal")
if _MLX_AVAILABLE:
    BACKENDS.append("mlx")

backend = pytest.fixture(params=BACKENDS)(lambda request: request.param)


def noisy_circuit() -> Circuit:
    """A small circuit exercising every built-in (diagonal-effect) channel."""
    qc = Circuit(3)
    qc.h(0).cx(0, 1).cx(1, 2)
    qc.depolarizing(0, 0.10)
    qc.rx(1, 0.7)
    qc.amplitude_damping(1, 0.20)
    qc.bit_flip(2, 0.15)
    qc.rz(0, 0.3).phase_flip(0, 0.05).phase_damping(2, 0.10)
    return qc


# --- noiseless: trajectories are exact, not stochastic ----------------------


def test_noiseless_matches_statevector():
    qc = Circuit(3)
    qc.h(0).cx(0, 1).t(1).cx(1, 2).ry(2, 0.4)
    sv = Simulator(backend="cpu").statevector(qc)
    probs = TrajectorySimulator(backend="cpu", trajectories=1).probabilities(qc)
    np.testing.assert_allclose(probs, np.abs(sv) ** 2, atol=1e-6)


# --- stochastic agreement with the exact density matrix ---------------------


def test_probabilities_match_density_matrix():
    qc = noisy_circuit()
    exact = DensityMatrixSimulator(backend="cpu").probabilities(qc)
    est = TrajectorySimulator(backend="cpu", seed=7).probabilities(qc, trajectories=3000)
    # Monte-Carlo error ~ sqrt(p/T) ~ 0.01; 3.5 sigma headroom.
    np.testing.assert_allclose(est, exact, atol=0.035)


def test_expectation_pauli_matches_density_matrix():
    qc = noisy_circuit()
    terms = [(1.0, [("Z", 0)]), (0.5, [("Z", 1), ("Z", 2)]), (2.0, [("X", 0)])]
    exact = DensityMatrixSimulator(backend="cpu").expectation_pauli(qc, terms)
    est = TrajectorySimulator(backend="cpu", seed=11).expectation_pauli(
        qc, terms, trajectories=3000
    )
    np.testing.assert_allclose(est, exact, atol=0.08)


def test_non_diagonal_effect_channel_fallback():
    # K_1 = sqrt(p) |0><+| has effect E_1 = p |+><+| — genuinely non-diagonal,
    # forcing the reduced-density-matrix path.
    p = 0.3
    plus = np.array([1, 1], dtype=np.complex128) / np.sqrt(2)
    k1 = np.sqrt(p) * np.outer([1, 0], plus.conj())
    e1 = k1.conj().T @ k1
    w, v = np.linalg.eigh(np.eye(2) - e1)
    k0 = (v * np.sqrt(np.clip(w, 0, None))) @ v.conj().T
    qc = Circuit(2)
    qc.h(0).cx(0, 1).kraus([0], [k0, k1])
    exact = DensityMatrixSimulator(backend="cpu").probabilities(qc)
    est = TrajectorySimulator(backend="cpu", seed=3).probabilities(qc, trajectories=4000)
    np.testing.assert_allclose(est, exact, atol=0.035)


# --- sampling semantics ------------------------------------------------------


def test_run_splits_shots_across_trajectories():
    qc = Circuit(2)
    qc.h(0).cx(0, 1).depolarizing(0, 0.05).measure_all()
    counts = TrajectorySimulator(backend="cpu", seed=5, trajectories=64).run(qc, shots=1000)
    assert sum(counts.values()) == 1000
    # GHZ plus weak depolarizing: 00/11 dominate.
    assert (counts["00"] + counts["11"]) / 1000 > 0.9


def test_run_without_measure_samples_all_qubits():
    qc = Circuit(2)
    qc.h(0).cx(0, 1).bit_flip(1, 0.1)
    counts = TrajectorySimulator(backend="cpu", seed=5, trajectories=16).run(qc, shots=200)
    assert sum(counts.values()) == 200
    assert all(len(k) == 2 for k in counts)


def test_run_sums_counters_per_measure_op():
    qc = Circuit(2)
    qc.h(0).measure([0]).cx(0, 1).measure([1])
    counts = TrajectorySimulator(backend="cpu", seed=5, trajectories=8).run(qc, shots=100)
    assert sum(counts.values()) == 200  # one share per MeasureOp, like Simulator


def test_more_trajectories_than_shots():
    qc = Circuit(1)
    qc.h(0).phase_damping(0, 0.2).measure_all()
    counts = TrajectorySimulator(backend="cpu", seed=5, trajectories=50).run(qc, shots=10)
    assert sum(counts.values()) == 10


# --- seeding / validation ----------------------------------------------------


def test_seeded_runs_are_reproducible():
    qc = noisy_circuit()
    qc.measure_all()
    a = TrajectorySimulator(backend="cpu", seed=42, trajectories=32).run(qc, shots=500)
    b = TrajectorySimulator(backend="cpu", seed=42, trajectories=32).run(qc, shots=500)
    assert a == b


def test_seeded_instance_is_reproducible_across_calls():
    qc = noisy_circuit()
    qc.measure_all()
    sim = TrajectorySimulator(backend="cpu", seed=42, trajectories=32)
    assert sim.run(qc, shots=500) == sim.run(qc, shots=500)
    np.testing.assert_array_equal(
        sim.probabilities(qc, trajectories=16), sim.probabilities(qc, trajectories=16)
    )


def test_invalid_args_raise():
    with pytest.raises(ValueError, match="backend"):
        TrajectorySimulator(backend="gpu")
    with pytest.raises(ValueError, match="trajectories"):
        TrajectorySimulator(trajectories=0)
    with pytest.raises(ValueError, match="trajectories"):
        TrajectorySimulator().probabilities(Circuit(1), trajectories=0)


# --- backend parity ----------------------------------------------------------


def test_backend_parity_probabilities(backend):
    qc = noisy_circuit()
    exact = DensityMatrixSimulator(backend="cpu").probabilities(qc)
    est = TrajectorySimulator(backend=backend, seed=9).probabilities(qc, trajectories=400)
    # Same seed -> same Kraus draws on every backend; only float error differs
    # between backends, but each is independently within MC error of exact.
    np.testing.assert_allclose(est, exact, atol=0.08)


def test_backend_parity_same_draws(backend):
    # The Kraus sampling lives in the host RNG, so with one trajectory the
    # result is a deterministic pure state — identical across backends.
    qc = Circuit(2)
    qc.h(0).cx(0, 1).amplitude_damping(0, 0.3).rx(1, 0.5)
    ref = TrajectorySimulator(backend="cpu", seed=1, trajectories=1).probabilities(qc)
    est = TrajectorySimulator(backend=backend, seed=1, trajectories=1).probabilities(qc)
    np.testing.assert_allclose(est, ref, atol=1e-5)
