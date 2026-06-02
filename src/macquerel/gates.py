from __future__ import annotations

from typing import Literal

import numpy as np

GateKind = Literal["diagonal", "permutation", "dense"]

_I2 = np.eye(2, dtype=complex)
_sqrt2_inv = 1.0 / np.sqrt(2.0)


def _c64(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.complex64)


def I() -> np.ndarray:
    return _c64(np.eye(2))


def H() -> np.ndarray:
    return _c64(np.array([[1, 1], [1, -1]]) * _sqrt2_inv)


def X() -> np.ndarray:
    return _c64(np.array([[0, 1], [1, 0]]))


def Y() -> np.ndarray:
    return _c64(np.array([[0, -1j], [1j, 0]]))


def Z() -> np.ndarray:
    return _c64(np.array([[1, 0], [0, -1]]))


def S() -> np.ndarray:
    return _c64(np.array([[1, 0], [0, 1j]]))


def T() -> np.ndarray:
    return _c64(np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]]))


def Rx(theta: float) -> np.ndarray:
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return _c64(np.array([[c, -1j * s], [-1j * s, c]]))


def Ry(theta: float) -> np.ndarray:
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return _c64(np.array([[c, -s], [s, c]]))


def Rz(theta: float) -> np.ndarray:
    return _c64(np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]]))


def P(lam: float) -> np.ndarray:
    return _c64(np.array([[1, 0], [0, np.exp(1j * lam)]]))


def CNOT() -> np.ndarray:
    return _c64(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]))


def CZ() -> np.ndarray:
    return _c64(np.diag([1, 1, 1, -1]))


def SWAP() -> np.ndarray:
    return _c64(np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]))


def CP(lam: float) -> np.ndarray:
    return _c64(np.diag([1, 1, 1, np.exp(1j * lam)]))


def controlled(U: np.ndarray) -> np.ndarray:
    """Lift a 1-qubit gate U to a 2-qubit controlled-U gate."""
    dim = U.shape[0]
    assert dim == 2, "controlled() expects a 2×2 matrix"
    mat = np.eye(4, dtype=np.complex128)
    mat[2:, 2:] = U
    return _c64(mat)


def classify(matrix: np.ndarray) -> GateKind:
    atol = 1e-6
    # diagonal: off-diagonal entries all ~0
    off_diag = matrix.copy()
    np.fill_diagonal(off_diag, 0)
    if np.allclose(off_diag, 0, atol=atol):
        return "diagonal"
    # permutation: each row/col has exactly one nonzero of magnitude ~1
    mags = np.abs(matrix)
    row_counts = np.sum(mags > atol, axis=1)
    col_counts = np.sum(mags > atol, axis=0)
    nonzero_mags = mags[mags > atol]
    if (
        np.all(row_counts == 1)
        and np.all(col_counts == 1)
        and np.allclose(nonzero_mags, 1.0, atol=atol)
    ):
        return "permutation"
    return "dense"
