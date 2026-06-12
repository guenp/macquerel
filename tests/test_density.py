"""Tests for DensityMatrixSimulator: noiseless equivalence with the
statevector simulator, analytic channel checks, differential tests against a
direct dense reference implementation, sampling, and backend parity."""

import numpy as np
import pytest

from macquerel import BatchedSimulator, Circuit, DensityMatrixSimulator, Simulator
from macquerel.circuit import ChannelOp, Gate
from macquerel.compiler import _embed, fuse_gates

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


# --- reference implementation: direct dense rho evolution -------------------


def reference_density_matrix(circuit: Circuit) -> np.ndarray:
    """Evolve the full 2**n x 2**n density matrix directly (no vectorization)."""
    n = circuit.n_qubits
    dim = 2**n
    rho = np.zeros((dim, dim), dtype=np.complex128)
    rho[0, 0] = 1.0
    full = list(range(n))
    for op in circuit.ops:
        if isinstance(op, Gate):
            assert not op.controls, "builders never emit separate controls"
            u = _embed(op.matrix, op.targets, full)
            rho = u @ rho @ u.conj().T
        elif isinstance(op, ChannelOp):
            ks = [_embed(k, op.qubits, full) for k in op.kraus]
            rho = sum(k @ rho @ k.conj().T for k in ks)
    return rho


def random_noisy_circuit(n: int, depth: int, seed: int) -> Circuit:
    """Random 1q rotations + entanglers, interleaved with random channels."""
    rng = np.random.default_rng(seed)
    qc = Circuit(n)
    channels = ["bit_flip", "phase_flip", "depolarizing", "amplitude_damping", "phase_damping"]
    for d in range(depth):
        for q in range(n):
            qc.ry(q, float(rng.uniform(0, 2 * np.pi)))
            qc.rz(q, float(rng.uniform(0, 2 * np.pi)))
        for q in range(d % 2, n - 1, 2):
            qc.cx(q, q + 1)
        ch = channels[int(rng.integers(len(channels)))]
        getattr(qc, ch)(int(rng.integers(n)), float(rng.uniform(0, 0.3)))
    return qc


# --- noiseless circuits: rho == |psi><psi| -----------------------------------


@pytest.mark.parametrize("n", [1, 2, 4])
def test_noiseless_matches_statevector(backend, n):
    rng = np.random.default_rng(n)
    qc = Circuit(n)
    qc.h(0)
    for q in range(n - 1):
        qc.cx(q, q + 1)
    for q in range(n):
        qc.ry(q, float(rng.uniform(0, np.pi)))
        qc.t(q)
    sv = Simulator(backend="cpu").statevector(qc)
    rho = DensityMatrixSimulator(backend=backend).density_matrix(qc)
    np.testing.assert_allclose(rho, np.outer(sv, sv.conj()), atol=1e-5)


def test_noiseless_probabilities_match_amplitudes(backend):
    qc = Circuit(3)
    qc.h(0).cx(0, 1).cx(1, 2).rx(1, 0.9)
    sv = Simulator(backend="cpu").statevector(qc)
    probs = DensityMatrixSimulator(backend=backend).probabilities(qc)
    np.testing.assert_allclose(probs, np.abs(sv) ** 2, atol=1e-5)


def test_noiseless_purity_is_one(backend):
    qc = Circuit(3)
    qc.h(0).cx(0, 1).t(2).cz(1, 2)
    assert abs(DensityMatrixSimulator(backend=backend).purity(qc) - 1.0) < 1e-4


# --- analytic single-channel checks ------------------------------------------


def test_bit_flip_populations(backend):
    qc = Circuit(1).bit_flip(0, 0.3)
    probs = DensityMatrixSimulator(backend=backend).probabilities(qc)
    np.testing.assert_allclose(probs, [0.7, 0.3], atol=1e-6)


def test_phase_flip_dampens_coherence(backend):
    # On |+>: off-diagonals scale by (1 - 2p), populations untouched.
    p = 0.1
    qc = Circuit(1).h(0).phase_flip(0, p)
    rho = DensityMatrixSimulator(backend=backend).density_matrix(qc)
    np.testing.assert_allclose(rho[0, 0], 0.5, atol=1e-6)
    np.testing.assert_allclose(rho[0, 1], 0.5 * (1 - 2 * p), atol=1e-6)


def test_depolarizing_analytic(backend):
    # rho' = (1-p) rho + (p/3)(X rho X + Y rho Y + Z rho Z); on |0><0| the
    # excited population becomes 2p/3.
    p = 0.3
    qc = Circuit(1).depolarizing(0, p)
    rho = DensityMatrixSimulator(backend=backend).density_matrix(qc)
    np.testing.assert_allclose(np.diag(rho).real, [1 - 2 * p / 3, 2 * p / 3], atol=1e-6)


def test_full_depolarizing_gives_maximally_mixed(backend):
    qc = Circuit(1).h(0)
    for _ in range(60):
        qc.depolarizing(0, 0.5)
    rho = DensityMatrixSimulator(backend=backend).density_matrix(qc)
    np.testing.assert_allclose(rho, np.eye(2) / 2, atol=1e-4)


def test_amplitude_damping_decays_excited_state(backend):
    gamma = 0.25
    qc = Circuit(1).x(0).amplitude_damping(0, gamma)
    probs = DensityMatrixSimulator(backend=backend).probabilities(qc)
    np.testing.assert_allclose(probs, [gamma, 1 - gamma], atol=1e-6)


def test_phase_damping_preserves_populations(backend):
    gamma = 0.4
    qc = Circuit(1).h(0).phase_damping(0, gamma)
    rho = DensityMatrixSimulator(backend=backend).density_matrix(qc)
    np.testing.assert_allclose(np.diag(rho).real, [0.5, 0.5], atol=1e-6)
    np.testing.assert_allclose(rho[0, 1], 0.5 * np.sqrt(1 - gamma), atol=1e-6)


def test_two_qubit_correlated_kraus_channel(backend):
    # Correlated XX flip on |00>: ends in |11> with probability p.
    p = 0.2
    xx = np.kron([[0, 1], [1, 0]], [[0, 1], [1, 0]])
    qc = Circuit(2).kraus([0, 1], [np.sqrt(1 - p) * np.eye(4), np.sqrt(p) * xx])
    probs = DensityMatrixSimulator(backend=backend).probabilities(qc)
    np.testing.assert_allclose(probs, [1 - p, 0, 0, p], atol=1e-6)


# --- differential tests vs the dense reference -------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("n", [2, 3, 4])
def test_random_noisy_circuit_matches_reference(backend, n, seed):
    qc = random_noisy_circuit(n, depth=4, seed=seed)
    ref = reference_density_matrix(qc)
    rho = DensityMatrixSimulator(backend=backend).density_matrix(qc)
    np.testing.assert_allclose(rho, ref, atol=1e-4)


def test_fusion_width_does_not_change_result(monkeypatch):
    qc = random_noisy_circuit(4, depth=5, seed=9)
    base = DensityMatrixSimulator(backend="cpu").density_matrix(qc)
    monkeypatch.setenv("MACQUEREL_FUSION_WIDTH", "1")
    unfused = DensityMatrixSimulator(backend="cpu").density_matrix(qc)
    np.testing.assert_allclose(base, unfused, atol=1e-5)


# --- physicality of the evolved state ----------------------------------------


@pytest.mark.parametrize("seed", [3, 4])
def test_trace_hermiticity_positivity(backend, seed):
    qc = random_noisy_circuit(3, depth=6, seed=seed)
    rho = DensityMatrixSimulator(backend=backend).density_matrix(qc)
    assert abs(np.trace(rho) - 1.0) < 1e-4
    np.testing.assert_allclose(rho, rho.conj().T, atol=1e-5)
    assert np.linalg.eigvalsh(rho).min() > -1e-5


def test_purity_decreases_under_noise(backend):
    dm = DensityMatrixSimulator(backend=backend)
    pure = Circuit(2)
    pure.h(0).cx(0, 1)
    noisy = Circuit(2)
    noisy.h(0).cx(0, 1).depolarizing(0, 0.2).depolarizing(1, 0.2)
    assert dm.purity(noisy) < dm.purity(pure) - 0.05


# --- sampling -----------------------------------------------------------------


def test_run_noiseless_ghz_counts(backend):
    qc = Circuit(3)
    qc.h(0).cx(0, 1).cx(1, 2).measure_all()
    counts = DensityMatrixSimulator(backend=backend, seed=11).run(qc, shots=2000)
    assert set(counts) == {"000", "111"}
    assert abs(counts["000"] - 1000) < 150


def test_run_bit_flip_statistics():
    qc = Circuit(1).bit_flip(0, 0.3).measure_all()
    counts = DensityMatrixSimulator(backend="cpu", seed=5).run(qc, shots=20000)
    assert abs(counts["1"] / 20000 - 0.3) < 0.02


def test_run_without_measure_samples_all_qubits():
    qc = Circuit(2)
    qc.x(0)
    counts = DensityMatrixSimulator(backend="cpu", seed=0).run(qc, shots=100)
    assert counts == {"10": 100}


def test_run_measure_subset_orders_bits_by_qubit_list():
    qc = Circuit(3)
    qc.x(2).measure([2, 0])
    counts = DensityMatrixSimulator(backend="cpu", seed=0).run(qc, shots=50)
    assert counts == {"10": 50}


def test_run_measure_unsorted_qubit_list_bit_order():
    """Regression: bit i must be qubits[i] for 3-cycle measure lists (the
    diagonal marginal used argsort(qubits) where the rank permutation is
    needed, mirroring the statevector samplers)."""
    qc = Circuit(3)
    qc.x(1).x(2).measure([1, 2, 0])  # |q0 q1 q2> = |011> -> bits (q1,q2,q0)
    counts = DensityMatrixSimulator(backend="cpu", seed=0).run(qc, shots=20)
    assert counts == {"110": 20}


def test_run_mid_circuit_measurements_sum_counters():
    qc = Circuit(1)
    qc.measure_all()  # all |0>
    qc.x(0)
    qc.measure_all()  # all |1>
    counts = DensityMatrixSimulator(backend="cpu", seed=0).run(qc, shots=100)
    assert counts == {"0": 100, "1": 100}


def test_seeded_runs_are_reproducible(backend):
    qc = Circuit(2)
    qc.h(0).cx(0, 1).depolarizing(0, 0.1).measure_all()
    a = DensityMatrixSimulator(backend=backend, seed=42).run(qc, shots=500)
    b = DensityMatrixSimulator(backend=backend, seed=42).run(qc, shots=500)
    assert a == b


def test_run_matches_simulator_distribution_noiseless():
    qc = Circuit(2)
    qc.h(0).cx(0, 1).measure_all()
    dm_counts = DensityMatrixSimulator(backend="cpu", seed=1).run(qc, shots=4000)
    sv_counts = Simulator(backend="cpu", seed=1).run(qc, shots=4000)
    for key in set(dm_counts) | set(sv_counts):
        assert abs(dm_counts[key] - sv_counts[key]) < 250


# --- expectation values ---------------------------------------------------------


def test_expectation_pauli_noiseless_matches_statevector(backend):
    qc = Circuit(2)
    qc.h(0).cx(0, 1).rz(1, 0.4)
    strings = [(1.0, [("Z", 0)]), (0.5, [("X", 0), ("X", 1)]), (2.0, [("I", 0)])]
    sv = Simulator(backend="cpu").statevector(qc)
    from macquerel.backends.cpu import CPUBackend

    expected = CPUBackend().expectation_pauli(sv, strings)
    got = DensityMatrixSimulator(backend=backend).expectation_pauli(qc, strings)
    np.testing.assert_allclose(got, expected, atol=1e-5)


def test_expectation_pauli_noisy_analytic():
    # <Z> on a bit-flipped |0>: (1-p) - p = 1 - 2p.
    p = 0.3
    qc = Circuit(1).bit_flip(0, p)
    (got,) = DensityMatrixSimulator(backend="cpu").expectation_pauli(qc, [(1.0, [("Z", 0)])])
    assert abs(got - (1 - 2 * p)) < 1e-6
    # <X> on a phase-damped |+>: sqrt(1-gamma).
    gamma = 0.36
    qc = Circuit(1).h(0).phase_damping(0, gamma)
    (got,) = DensityMatrixSimulator(backend="cpu").expectation_pauli(qc, [(1.0, [("X", 0)])])
    assert abs(got - 0.8) < 1e-5


def test_expectation_pauli_gather_matches_dense_reference(backend):
    # Step 38 path: Y phases, mixed multi-qubit strings, identity, and a
    # complex-weighted sum, checked against tr(rho P) on the dense reference.
    qc = Circuit(3)
    qc.h(0).cx(0, 1).ry(2, 0.7).depolarizing(1, 0.1).rz(0, 0.3).amplitude_damping(2, 0.2)
    strings = [
        (1.0, [("Y", 1)]),
        (0.5, [("Y", 0), ("Y", 2)]),
        (-2.0, [("X", 0), ("Y", 1), ("Z", 2)]),
        (1.5, [("I", 1)]),
        (0.25, [("Z", 1), ("X", 2)]),
    ]
    rho = reference_density_matrix(qc)
    from macquerel.gates import I as I_gate
    from macquerel.gates import X, Y, Z

    mats = {"X": X(), "Y": Y(), "Z": Z(), "I": I_gate()}
    expected = []
    for coeff, terms in strings:
        per_qubit = [np.eye(2, dtype=complex) for _ in range(3)]
        for ch, q in terms:
            per_qubit[q] = mats[ch]
        full = per_qubit[0]
        for m in per_qubit[1:]:
            full = np.kron(full, m)
        expected.append(coeff * np.real(np.trace(rho @ full)))
    got = DensityMatrixSimulator(backend=backend).expectation_pauli(qc, strings)
    np.testing.assert_allclose(got, expected, atol=1e-5)


def test_density_matrix_view_matches_copy(backend):
    qc = Circuit(2)
    qc.h(0).cx(0, 1).depolarizing(0, 0.1)
    sim = DensityMatrixSimulator(backend=backend)
    copied = sim.density_matrix(qc)
    view = sim.density_matrix(qc, copy=False)
    np.testing.assert_allclose(view, copied, atol=1e-6)


# --- compiler interaction --------------------------------------------------------


def test_channels_are_fusion_barriers():
    qc = Circuit(2)
    qc.h(0).h(1).depolarizing(0, 0.1).h(0).h(1)
    fused = fuse_gates(qc, max_fused_qubits=4)
    kinds = [type(op).__name__ for op in fused.ops]
    assert kinds == ["Gate", "ChannelOp", "Gate"]


def test_remap_relabels_channel_qubits():
    from macquerel.compiler import remap_qubits_with_perm

    qc = Circuit(3)
    qc.h(2).h(2).h(2).amplitude_damping(2, 0.2)
    remapped, perm = remap_qubits_with_perm(qc)
    (channel,) = [op for op in remapped.ops if isinstance(op, ChannelOp)]
    assert channel.qubits == [perm[2]]


# --- guards & selection -----------------------------------------------------------


def test_statevector_simulator_rejects_channels():
    qc = Circuit(1).bit_flip(0, 0.1)
    with pytest.raises(ValueError, match="DensityMatrixSimulator"):
        Simulator(backend="cpu").statevector(qc)
    with pytest.raises(ValueError, match="DensityMatrixSimulator"):
        Simulator(backend="cpu").run(qc, shots=10)


def test_batched_simulator_rejects_channels():
    qc = Circuit(1).bit_flip(0, 0.1)
    with pytest.raises(ValueError, match="DensityMatrixSimulator"):
        BatchedSimulator(backend="cpu").statevectors([qc])


def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match="Unknown backend"):
        DensityMatrixSimulator(backend="cuda")


@pytest.mark.skipif(not _MLX_AVAILABLE, reason="mlx not installed")
def test_mlx_density_qubit_cap():
    qc = Circuit(16)
    qc.h(0)
    with pytest.raises(ValueError, match="caps density-matrix simulation at 15"):
        DensityMatrixSimulator(backend="mlx").probabilities(qc)


def test_auto_selection_uses_doubled_count(monkeypatch):
    import macquerel.simulator as sim

    monkeypatch.delenv("MACQUEREL_BACKEND_TIERS", raising=False)
    dm = DensityMatrixSimulator()
    # 2n <= cpu tier boundary -> cpu; above it -> a GPU backend when present.
    assert dm._backend_name_for(7) == "cpu"
    if sim._METAL_AVAILABLE or sim._MLX_AVAILABLE:
        assert dm._backend_name_for(10) in ("metal", "mlx")
