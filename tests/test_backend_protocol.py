"""Tests for Backend Protocol conformance and the new abs2sum / expectation_pauli methods."""

from macquerel.backends import Backend
from macquerel.backends.cpu import CPUBackend
from macquerel.circuit import Circuit, Gate


def _ghz3_sv():
    """Statevector for 3-qubit GHZ state: (|000> + |111>) / sqrt(2)."""
    cpu = CPUBackend()
    qc = Circuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    sv = cpu.allocate(3)
    for op in qc.ops:
        if isinstance(op, Gate):
            sv = cpu.apply_matrix(sv, op.matrix, op.targets, op.controls or None)
    return sv


def _zero_sv(n=1):
    return CPUBackend().allocate(n)


def _plus_sv():
    """Single-qubit |+> = (|0> + |1>) / sqrt(2)."""
    cpu = CPUBackend()
    sv = cpu.allocate(1)
    import macquerel.gates as g

    return cpu.apply_matrix(sv, g.H(), [0])


# --- Protocol conformance ---


def test_cpu_backend_satisfies_protocol():
    assert isinstance(CPUBackend(), Backend)


# --- abs2sum ---


def test_abs2sum_ghz_qubit0():
    sv = _ghz3_sv()
    cpu = CPUBackend()
    result = cpu.abs2sum(sv, [0])
    # GHZ: marginal for qubit 0 is 50/50
    assert result.shape == (2,)
    assert abs(result[0] - 0.5) < 1e-5
    assert abs(result[1] - 0.5) < 1e-5


def test_abs2sum_zero_state():
    sv = _zero_sv(2)
    cpu = CPUBackend()
    result = cpu.abs2sum(sv, [0])
    assert abs(result[0] - 1.0) < 1e-5
    assert abs(result[1] - 0.0) < 1e-5


def test_abs2sum_sums_to_one():
    sv = _ghz3_sv()
    cpu = CPUBackend()
    for q in range(3):
        result = cpu.abs2sum(sv, [q])
        assert abs(result.sum() - 1.0) < 1e-5


# --- expectation_pauli ---


def test_expectation_Z_on_zero():
    sv = _zero_sv(1)
    cpu = CPUBackend()
    ev = cpu.expectation_pauli(sv, [(1.0, [("Z", 0)])])
    assert abs(ev[0] - 1.0) < 1e-5


def test_expectation_X_on_plus():
    sv = _plus_sv()
    cpu = CPUBackend()
    ev = cpu.expectation_pauli(sv, [(1.0, [("X", 0)])])
    assert abs(ev[0] - 1.0) < 1e-5


def test_expectation_Z_on_plus_is_zero():
    sv = _plus_sv()
    cpu = CPUBackend()
    ev = cpu.expectation_pauli(sv, [(1.0, [("Z", 0)])])
    assert abs(ev[0]) < 1e-5


def test_expectation_sum_of_terms():
    sv = _zero_sv(1)
    cpu = CPUBackend()
    # 0.5 * Z + 0.5 * I on |0>: 0.5*1 + 0.5*1 = 1.0
    ev = cpu.expectation_pauli(sv, [(0.5, [("Z", 0)]), (0.5, [("I", 0)])])
    assert abs(ev[0] - 0.5) < 1e-5
    assert abs(ev[1] - 0.5) < 1e-5
