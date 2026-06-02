"""Differential tests: every CPU backend circuit must agree with MLX backend to 1e-5."""

import numpy as np
import pytest

mlx = pytest.importorskip("mlx.core")

import macquerel.gates as g
from macquerel.backends.cpu import CPUBackend
from macquerel.backends.mlx_backend import MLXBackend


@pytest.fixture
def cpu():
    return CPUBackend()


@pytest.fixture
def mlx_backend():
    return MLXBackend()


def _apply_gates(backend, sv, gate_list):
    """Apply a list of (matrix, targets[, controls]) tuples."""
    for item in gate_list:
        matrix, targets = item[0], item[1]
        controls = item[2] if len(item) > 2 else None
        sv = backend.apply_matrix(sv, matrix, targets, controls)
    return sv


def _circuit_gates():
    return [
        # Bell
        [(g.H(), [0]), (g.CNOT(), [0, 1])],
        # GHZ
        [(g.H(), [0]), (g.CNOT(), [0, 1]), (g.CNOT(), [0, 2])],
        # diagonal gates
        [(g.H(), [0]), (g.Rz(0.5), [0]), (g.S(), [0]), (g.CZ(), [0, 1])],
        # mixed
        [(g.H(), [0]), (g.Rx(0.3), [1]), (g.SWAP(), [0, 1]), (g.Rz(0.7), [0])],
    ]


@pytest.mark.parametrize("gate_seq", _circuit_gates())
def test_differential(cpu, mlx_backend, gate_seq):
    n_qubits = max(q for item in gate_seq for q in item[1]) + 1
    sv_cpu = _apply_gates(cpu, cpu.allocate(n_qubits), gate_seq)
    sv_mlx = _apply_gates(mlx_backend, mlx_backend.allocate(n_qubits), gate_seq)
    sv_mlx = mlx_backend.to_numpy(sv_mlx)

    assert np.allclose(sv_cpu, sv_mlx, atol=1e-5), f"max diff: {np.max(np.abs(sv_cpu - sv_mlx))}"


def test_bell_state(mlx_backend):
    sv = mlx_backend.allocate(2)
    sv = mlx_backend.apply_matrix(sv, g.H(), [0])
    sv = mlx_backend.apply_matrix(sv, g.CNOT(), [0, 1])
    sv = mlx_backend.to_numpy(sv)
    expected = np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], dtype=np.complex64)
    assert np.allclose(sv, expected, atol=1e-5)


def test_ghz(mlx_backend):
    sv = mlx_backend.allocate(3)
    sv = mlx_backend.apply_matrix(sv, g.H(), [0])
    sv = mlx_backend.apply_matrix(sv, g.CNOT(), [0, 1])
    sv = mlx_backend.apply_matrix(sv, g.CNOT(), [0, 2])
    sv = mlx_backend.to_numpy(sv)
    inv_sqrt2 = 1 / np.sqrt(2)
    assert abs(sv[0] - inv_sqrt2) < 1e-5
    assert abs(sv[7] - inv_sqrt2) < 1e-5
    assert np.allclose(sv[1:7], 0, atol=1e-5)
