import numpy as np
import pytest

import macquerel.gates as g


def _is_unitary(matrix: np.ndarray, atol: float = 1e-5) -> bool:
    n = matrix.shape[0]
    return np.allclose(matrix @ matrix.conj().T, np.eye(n), atol=atol)


ALL_FIXED_GATES = [
    ("I", g.I()),
    ("H", g.H()),
    ("X", g.X()),
    ("Y", g.Y()),
    ("Z", g.Z()),
    ("S", g.S()),
    ("T", g.T()),
    ("CNOT", g.CNOT()),
    ("CZ", g.CZ()),
    ("SWAP", g.SWAP()),
]

ALL_PARAM_GATES = [
    ("Rx(0.3)", g.Rx(0.3)),
    ("Rx(pi)", g.Rx(np.pi)),
    ("Ry(0.7)", g.Ry(0.7)),
    ("Rz(1.2)", g.Rz(1.2)),
    ("P(pi/4)", g.P(np.pi / 4)),
    ("CP(pi/3)", g.CP(np.pi / 3)),
]


@pytest.mark.parametrize("name,mat", ALL_FIXED_GATES + ALL_PARAM_GATES)
def test_unitarity(name, mat):
    assert _is_unitary(mat), f"{name} is not unitary"


def test_H_squared_is_I():
    h = g.H()
    assert np.allclose(h @ h, np.eye(2), atol=1e-5)


def test_X_squared_is_I():
    x = g.X()
    assert np.allclose(x @ x, np.eye(2), atol=1e-5)


def test_S_squared_is_Z():
    assert np.allclose(g.S() @ g.S(), g.Z(), atol=1e-5)


def test_Rz_inverse():
    theta = 1.23
    assert np.allclose(g.Rz(theta) @ g.Rz(-theta), np.eye(2), atol=1e-5)


def test_classify_diagonal():
    assert g.classify(g.Z()) == "diagonal"
    assert g.classify(g.S()) == "diagonal"
    assert g.classify(g.T()) == "diagonal"
    assert g.classify(g.Rz(0.5)) == "diagonal"
    assert g.classify(g.CZ()) == "diagonal"
    assert g.classify(g.CP(0.5)) == "diagonal"


def test_classify_permutation():
    assert g.classify(g.X()) == "permutation"
    assert g.classify(g.SWAP()) == "permutation"
    assert g.classify(g.CNOT()) == "permutation"


def test_classify_dense():
    assert g.classify(g.H()) == "dense"
    assert g.classify(g.Rx(0.5)) == "dense"
    assert g.classify(g.Ry(0.5)) == "dense"


def test_classify_y_is_permutation():
    # Y = [[0,-i],[i,0]] is anti-diagonal with unit-magnitude entries — permutation structure
    assert g.classify(g.Y()) == "permutation"


def test_controlled_cnot():
    """controlled(X) should equal CNOT."""
    cx = g.controlled(g.X())
    assert np.allclose(cx, g.CNOT(), atol=1e-5)


def test_controlled_unitary():
    u = g.controlled(g.H())
    assert _is_unitary(u)
