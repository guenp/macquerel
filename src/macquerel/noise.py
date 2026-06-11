"""Kraus-operator noise channels (v0.3).

A channel is a list of Kraus operators ``[K_0, ..., K_m]`` satisfying the
completeness relation ``sum_k K_k^dagger K_k = I``, applied to a density
matrix as ``rho -> sum_k K_k rho K_k^dagger``. The factories below build the
standard single-qubit channels; arbitrary channels (including multi-qubit
ones) go through ``Circuit.kraus`` with ``validate_kraus`` as the gate.

The `DensityMatrixSimulator` applies a channel as one dense superoperator
``sum_k K_k (x) conj(K_k)`` acting on the vectorized density matrix — see
``macquerel.density`` for the doubled-space layout.
"""

from __future__ import annotations

import numpy as np

from macquerel.gates import I, X, Y, Z


def _check_prob(name: str, value: float) -> None:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def validate_kraus(operators: list[np.ndarray], n_qubits: int, atol: float = 1e-5) -> None:
    """Validate a Kraus decomposition for an ``n_qubits``-qubit channel.

    Checks that every operator is a square ``2**n x 2**n`` matrix and that the
    set satisfies the trace-preservation condition ``sum_k K_k^dagger K_k = I``
    (to ``atol``, loose enough for complex64 channel definitions).
    """
    if not operators:
        raise ValueError("a channel needs at least one Kraus operator")
    dim = 2**n_qubits
    acc = np.zeros((dim, dim), dtype=np.complex128)
    for k, op in enumerate(operators):
        mat = np.asarray(op)
        if mat.shape != (dim, dim):
            raise ValueError(
                f"Kraus operator {k} has shape {mat.shape}; expected ({dim}, {dim}) "
                f"for a {n_qubits}-qubit channel"
            )
        mat = mat.astype(np.complex128)
        acc += mat.conj().T @ mat
    if not np.allclose(acc, np.eye(dim), atol=atol):
        raise ValueError(
            "Kraus operators do not satisfy completeness (sum_k K_k^dagger K_k != I); "
            "the channel would not be trace-preserving"
        )


def bit_flip_kraus(p: float) -> list[np.ndarray]:
    """Bit flip: X applied with probability ``p``."""
    _check_prob("p", p)
    return [np.sqrt(1.0 - p) * I(), np.sqrt(p) * X()]


def phase_flip_kraus(p: float) -> list[np.ndarray]:
    """Phase flip: Z applied with probability ``p``."""
    _check_prob("p", p)
    return [np.sqrt(1.0 - p) * I(), np.sqrt(p) * Z()]


def depolarizing_kraus(p: float) -> list[np.ndarray]:
    """Depolarizing: each of X, Y, Z applied with probability ``p/3``."""
    _check_prob("p", p)
    return [
        np.sqrt(1.0 - p) * I(),
        np.sqrt(p / 3.0) * X(),
        np.sqrt(p / 3.0) * Y(),
        np.sqrt(p / 3.0) * Z(),
    ]


def amplitude_damping_kraus(gamma: float) -> list[np.ndarray]:
    """Amplitude damping: |1> decays to |0> with probability ``gamma``."""
    _check_prob("gamma", gamma)
    k0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - gamma)]], dtype=np.complex64)
    k1 = np.array([[0.0, np.sqrt(gamma)], [0.0, 0.0]], dtype=np.complex64)
    return [k0, k1]


def phase_damping_kraus(gamma: float) -> list[np.ndarray]:
    """Phase damping: coherence decays by ``sqrt(1-gamma)`` without population loss."""
    _check_prob("gamma", gamma)
    k0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - gamma)]], dtype=np.complex64)
    k1 = np.array([[0.0, 0.0], [0.0, np.sqrt(gamma)]], dtype=np.complex64)
    return [k0, k1]


def channel_superoperator(operators: list[np.ndarray]) -> np.ndarray:
    """The row-major-vec superoperator ``sum_k K_k (x) conj(K_k)``.

    With ``vec(rho)`` flattened row-major (ket index in the high bits), the
    channel acts as ``vec(rho') = S vec(rho)`` where
    ``S = sum_k kron(K_k, conj(K_k))`` — a single dense matrix on the channel's
    ket axes followed by its bra axes, which is exactly one backend
    ``apply_matrix`` call in the doubled space.
    """
    acc = None
    for op in operators:
        mat = np.asarray(op).astype(np.complex128)
        term = np.kron(mat, mat.conj())
        acc = term if acc is None else acc + term
    assert acc is not None
    return acc.astype(np.complex64)
