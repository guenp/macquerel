"""BatchedSimulator (Step 31): batched results must match per-circuit Simulator runs."""

from collections import Counter

import numpy as np
import pytest

from macquerel import BatchedSimulator, Circuit, Simulator
from macquerel.batched import _MLX_AVAILABLE


def ansatz(thetas: list[float], n: int = 4) -> Circuit:
    """A VQE-style ansatz: Ry layer, CX ladder, Rz layer — one structure."""
    qc = Circuit(n)
    for q in range(n):
        qc.ry(q, thetas[q % len(thetas)])
    for q in range(n - 1):
        qc.cx(q, q + 1)
    for q in range(n):
        qc.rz(q, thetas[(q + 1) % len(thetas)])
    return qc


def _sweep(b: int, n: int = 4, seed: int = 3) -> list[Circuit]:
    rng = np.random.default_rng(seed)
    return [ansatz(list(rng.uniform(0, 2 * np.pi, size=n)), n) for _ in range(b)]


def _engines():
    yield "cpu"
    if _MLX_AVAILABLE:
        yield "mlx"


@pytest.mark.parametrize("backend", list(_engines()))
def test_statevectors_match_simulator(backend):
    circuits = _sweep(7)
    batched = BatchedSimulator(backend=backend).statevectors(circuits)
    single = Simulator(backend="cpu")
    for i, c in enumerate(circuits):
        np.testing.assert_allclose(batched[i], single.statevector(c), atol=1e-5)


@pytest.mark.parametrize("backend", list(_engines()))
def test_mixed_structures_grouped_correctly(backend):
    """Different structures in one batch: grouped per signature, order kept."""
    rng = np.random.default_rng(5)
    circuits = []
    for i in range(6):
        if i % 2 == 0:
            circuits.append(ansatz(list(rng.uniform(0, 6.28, size=4))))
        else:
            qc = Circuit(4)
            qc.h(0).cx(0, 1).rz(2, float(rng.uniform(0, 6.28))).swap(2, 3)
            circuits.append(qc)
    batched = BatchedSimulator(backend=backend).statevectors(circuits)
    single = Simulator(backend="cpu")
    for i, c in enumerate(circuits):
        np.testing.assert_allclose(batched[i], single.statevector(c), atol=1e-5)


def test_controlled_and_diagonal_gates():
    rng = np.random.default_rng(9)
    circuits = []
    for _ in range(5):
        qc = Circuit(3)
        qc.h(0)
        qc.cp(0, 2, float(rng.uniform(0, 6.28)))
        qc.rz(1, float(rng.uniform(0, 6.28)))
        qc.cz(1, 2)
        circuits.append(qc)
    batched = BatchedSimulator(backend="cpu").statevectors(circuits)
    single = Simulator(backend="cpu")
    for i, c in enumerate(circuits):
        np.testing.assert_allclose(batched[i], single.statevector(c), atol=1e-5)


def test_run_counts_match_distribution():
    """Batched sampling tracks |psi|^2: a swept Bell pair stays correlated."""
    circuits = []
    for _ in range(4):
        qc = Circuit(2)
        qc.h(0).cx(0, 1).measure_all()
        circuits.append(qc)
    counts = BatchedSimulator(backend="cpu", seed=7).run(circuits, shots=600)
    assert len(counts) == 4
    for c in counts:
        assert sum(c.values()) == 600
        assert set(c) <= {"00", "11"}
        assert c["00"] > 200 and c["11"] > 200


def test_run_no_measure_returns_empty_counter():
    qc = Circuit(2)
    qc.h(0)
    (counts,) = BatchedSimulator(backend="cpu").run([qc], shots=10)
    assert counts == {}


def test_run_seed_reproducible():
    circuits = _sweep(3)
    for c in circuits:
        c.measure_all()
    a = BatchedSimulator(backend="cpu", seed=11).run(circuits, shots=200)
    b = BatchedSimulator(backend="cpu", seed=11).run(circuits, shots=200)
    assert a == b


def test_width_mismatch_raises():
    with pytest.raises(ValueError, match="n_qubits"):
        BatchedSimulator().statevectors([Circuit(2), Circuit(3)])


def test_empty_batch_raises():
    with pytest.raises(ValueError, match="at least one"):
        BatchedSimulator().statevectors([])


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown batched backend"):
        BatchedSimulator(backend="metal")


def test_single_circuit_batch():
    qc = Circuit(2)
    qc.h(0).cx(0, 1)
    sv = BatchedSimulator(backend="cpu").statevectors([qc])
    np.testing.assert_allclose(sv[0], Simulator(backend="cpu").statevector(qc), atol=1e-6)


def test_run_measure_unsorted_qubit_list_bit_order():
    """Regression: bit i must be qubits[i] for 3-cycle measure lists (the
    batched marginal used argsort(qubits) where the rank permutation is
    needed, like the single-circuit samplers)."""
    circuits = []
    for _ in range(3):
        qc = Circuit(3)
        qc.x(1)
        qc.x(2)  # |q0 q1 q2> = |011>
        qc.measure([1, 2, 0])
        circuits.append(qc)
    for counts in BatchedSimulator(backend="cpu", seed=0).run(circuits, shots=20):
        assert counts == Counter({"110": 20})
