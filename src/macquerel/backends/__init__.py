from __future__ import annotations

from collections import Counter
from typing import Any, Protocol, runtime_checkable

import numpy as np

from macquerel.backends.cpu import CPUBackend
from macquerel.backends.metal_backend import MetalBackend
from macquerel.backends.mlx_backend import MLXBackend


@runtime_checkable
class Backend(Protocol):
    def allocate(self, n_qubits: int, dtype) -> Any: ...
    def apply_matrix(self, sv, matrix, targets, controls) -> Any: ...
    def measure(self, sv, qubits, *, collapse: bool) -> list[int]: ...
    def sample(self, sv, qubits, shots: int, batch_shots: int | str = ...) -> Counter: ...
    def to_numpy(self, sv) -> np.ndarray: ...
    def expectation_pauli(self, sv, pauli_strings) -> np.ndarray: ...
    def abs2sum(self, sv, qubits: list[int]) -> np.ndarray: ...


__all__ = ["Backend", "CPUBackend", "MLXBackend", "MetalBackend"]
