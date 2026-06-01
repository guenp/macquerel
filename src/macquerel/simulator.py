from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

from macquerel.circuit import Circuit, Gate, MeasureOp
from macquerel.compiler import fuse_gates

if TYPE_CHECKING:
    from macquerel.backends.cpu import CPUBackend


def _make_backend(name: str, dtype: str):
    if name == "cpu":
        from macquerel.backends.cpu import CPUBackend
        return CPUBackend()
    if name == "mlx":
        from macquerel.backends.mlx_backend import MLXBackend
        return MLXBackend()
    raise ValueError(f"Unknown backend: {name!r}. Choose 'cpu' or 'mlx'.")


class Simulator:
    def __init__(self, backend: str = "cpu", dtype: str = "complex64"):
        self.backend_name = backend
        self.dtype = dtype
        self._backend = _make_backend(backend, dtype)
        self._np_dtype = np.complex64 if dtype == "complex64" else np.complex128

    def statevector(self, circuit: Circuit) -> np.ndarray:
        sv = self._backend.allocate(circuit.n_qubits, self._np_dtype)
        fused = fuse_gates(circuit)
        for op in fused.ops:
            if isinstance(op, Gate):
                sv = self._backend.apply_matrix(sv, op.matrix, op.targets, op.controls or None)
            elif isinstance(op, MeasureOp):
                self._backend.measure(sv, op.qubits, collapse=True)
        return sv

    def run(self, circuit: Circuit, shots: int = 1000) -> Counter:
        fused = fuse_gates(circuit)

        # Split circuit at measurement boundaries:
        # collect gate segments and the measurement qubit lists
        segments: list[list[Gate]] = []
        measurements: list[list[int]] = []
        current_gates: list[Gate] = []

        for op in fused.ops:
            if isinstance(op, Gate):
                current_gates.append(op)
            elif isinstance(op, MeasureOp):
                segments.append(current_gates)
                measurements.append(op.qubits)
                current_gates = []
        # trailing gates after last measure (no measurement)
        if current_gates and not measurements:
            # no measurements at all — sample from final state
            sv = self._backend.allocate(circuit.n_qubits, self._np_dtype)
            for gate in current_gates:
                sv = self._backend.apply_matrix(sv, gate.matrix, gate.targets, gate.controls or None)
            return self._backend.sample(sv, list(range(circuit.n_qubits)), shots)

        sv = self._backend.allocate(circuit.n_qubits, self._np_dtype)
        outcome_bitstrings: list[Counter] = []

        for gates, meas_qubits in zip(segments, measurements):
            for gate in gates:
                sv = self._backend.apply_matrix(sv, gate.matrix, gate.targets, gate.controls or None)
            counts = self._backend.sample(sv, meas_qubits, shots)
            outcome_bitstrings.append(counts)

        # apply trailing gates
        for gate in current_gates:
            sv = self._backend.apply_matrix(sv, gate.matrix, gate.targets, gate.controls or None)

        # If only one measurement, return it directly
        if len(outcome_bitstrings) == 1:
            return outcome_bitstrings[0]

        # Multiple measurement points: return combined bitstrings
        result: Counter = Counter()
        for c in outcome_bitstrings:
            for k, v in c.items():
                result[k] += v
        return result
