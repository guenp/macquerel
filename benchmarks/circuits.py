"""Shared circuit generators for benchmark scripts."""

from __future__ import annotations

import numpy as np

import macquerel.gates as g
from macquerel.circuit import Circuit


def build_qft(n: int) -> Circuit:
    qc = Circuit(n)
    for i in range(n):
        qc.h(i)
        for j in range(i + 1, n):
            qc.cp(i, j, np.pi / (2 ** (j - i)))
    for i in range(n // 2):
        qc.swap(i, n - 1 - i)
    return qc


def build_random(n: int, depth: int, seed: int = 42) -> Circuit:
    rng = np.random.default_rng(seed)
    qc = Circuit(n)
    gates_1q = [g.H, g.X, g.Z, lambda: g.Rz(rng.uniform(0, 2 * np.pi))]
    gates_2q = [g.CNOT, g.CZ, g.SWAP]
    for _ in range(depth):
        if n >= 2 and rng.random() < 0.4:
            pair = rng.choice(n, size=2, replace=False).tolist()
            qc._add("rand2q", gates_2q[rng.integers(len(gates_2q))](), pair)
        else:
            q = int(rng.integers(n))
            qc._add("rand1q", gates_1q[rng.integers(len(gates_1q))](), [q])
    return qc


def build_qaoa(n: int, layers: int = 3) -> Circuit:
    qc = Circuit(n)
    beta = 0.3
    for _ in range(layers):
        for i in range(n):
            qc.cz(i, (i + 1) % n)
        for i in range(n):
            qc.rx(i, 2 * beta)
    return qc


def _haar_special_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    z = (rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    ph = np.diagonal(r) / np.abs(np.diagonal(r))
    u = q * ph
    return u * np.linalg.det(u) ** (-1 / dim)


def build_quantum_volume(n: int, depth: int | None = None, seed: int = 23) -> Circuit:
    rng = np.random.default_rng(seed)
    d = depth if depth is not None else n
    qc = Circuit(n)
    for _ in range(d):
        perm = rng.permutation(n)
        for i in range(0, n - 1, 2):
            a, b = int(perm[i]), int(perm[i + 1])
            qc._add("su4", _haar_special_unitary(4, rng).astype(np.complex64), [a, b])
    return qc
