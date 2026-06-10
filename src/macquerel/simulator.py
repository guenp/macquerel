from __future__ import annotations

from collections import Counter

import numpy as np

from macquerel.circuit import Circuit, Gate, MeasureOp
from macquerel.compiler import fuse_gates

try:
    import mlx.core as mx  # noqa: F401  # ty: ignore[unresolved-import]

    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False

try:
    from macquerel.backends.metal_backend import _METAL_AVAILABLE
except ImportError:  # pragma: no cover - module always importable; guard anyway
    _METAL_AVAILABLE = False


# Measured tier boundaries (benchmarks/data/large, 2026-06, M5 Max):
#   - CPU wins through ~16q: the state is only a few MB, so per-kernel GPU
#     dispatch latency dominates the compute.
#   - MLX wins 17-21q.
#   - Metal wins from 22q up (2.7-5x over MLX at 24-28q): its in-place
#     single-buffer updates avoid MLX's double-buffering + lazy-graph
#     temporaries, which thrash unified memory as the state grows. It is also
#     the only backend past 30q -- MLX's int32 ShapeElem rejects >=2**31
#     amplitudes (Gate 0, docs/plan_completed.md).
_CPU_MAX_QUBITS = 16
_METAL_MIN_QUBITS = 22
_MLX_MAX_QUBITS = 30


def _select_backend(n_qubits: int) -> str:
    if n_qubits <= _CPU_MAX_QUBITS:
        return "cpu"
    if _MLX_AVAILABLE and n_qubits < _METAL_MIN_QUBITS:
        return "mlx"
    if _METAL_AVAILABLE:
        return "metal"
    if _MLX_AVAILABLE and n_qubits <= _MLX_MAX_QUBITS:
        return "mlx"
    return "cpu"


def _make_backend(name: str, dtype: str, seed: int | None = None):
    if name == "cpu":
        from macquerel.backends.cpu import CPUBackend

        return CPUBackend(seed=seed)
    if name == "mlx":
        from macquerel.backends.mlx_backend import MLXBackend

        return MLXBackend(seed=seed)
    if name == "metal":
        from macquerel.backends.metal_backend import MetalBackend

        return MetalBackend(seed=seed)
    raise ValueError(f"Unknown backend: {name!r}. Choose 'cpu', 'mlx', 'metal', or 'auto'.")


class Simulator:
    def __init__(
        self,
        backend: str = "auto",
        dtype: str = "complex64",
        seed: int | None = None,
        batch_shots: int | str = "auto",
    ) -> None:
        self.backend_name = backend
        self.dtype = dtype
        self._seed = seed
        # Shot-batch size handed to the backend's sampler. "auto" lets a GPU
        # backend autotune the mx.random.categorical batch (Step 19); an int
        # pins it. Ignored by the host (NumPy) samplers.
        self.batch_shots = batch_shots
        self._np_dtype = np.complex64 if dtype == "complex64" else np.complex128
        self._backend = None if backend == "auto" else _make_backend(backend, dtype, seed)

    def _get_backend(self, n_qubits: int):
        if self._backend is not None:
            return self._backend
        name = _select_backend(n_qubits)
        return _make_backend(name, self.dtype, self._seed)

    def statevector(self, circuit: Circuit) -> np.ndarray:
        backend = self._get_backend(circuit.n_qubits)
        sv = backend.allocate(circuit.n_qubits, self._np_dtype)
        fused = fuse_gates(circuit)
        for op in fused.ops:
            if isinstance(op, Gate):
                sv = backend.apply_matrix(sv, op.matrix, op.targets, op.controls or None)
        return backend.to_numpy(sv)

    def run(self, circuit: Circuit, shots: int = 1000) -> Counter:
        backend = self._get_backend(circuit.n_qubits)
        fused = fuse_gates(circuit)

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

        if current_gates and not measurements:
            sv = backend.allocate(circuit.n_qubits, self._np_dtype)
            for gate in current_gates:
                sv = backend.apply_matrix(sv, gate.matrix, gate.targets, gate.controls or None)
            return backend.sample(
                sv, list(range(circuit.n_qubits)), shots, batch_shots=self.batch_shots
            )

        sv = backend.allocate(circuit.n_qubits, self._np_dtype)
        outcome_bitstrings: list[Counter] = []

        for gates, meas_qubits in zip(segments, measurements, strict=True):
            for gate in gates:
                sv = backend.apply_matrix(sv, gate.matrix, gate.targets, gate.controls or None)
            counts = backend.sample(sv, meas_qubits, shots, batch_shots=self.batch_shots)
            outcome_bitstrings.append(counts)

        for gate in current_gates:
            sv = backend.apply_matrix(sv, gate.matrix, gate.targets, gate.controls or None)

        if len(outcome_bitstrings) == 1:
            return outcome_bitstrings[0]

        result: Counter = Counter()
        for c in outcome_bitstrings:
            for k, v in c.items():
                result[k] += v
        return result
