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


def _select_backend(n_qubits: int) -> str:
    # CPU wins through ~16 qubits: the state vector is only a few MB, so per-kernel
    # GPU dispatch latency dominates the compute. Benchmarks (benchmarks/data) put
    # the crossover just above 16q, where MLX pulls ahead (2.4x at 18q, growing).
    if n_qubits <= 16:
        return "cpu"
    # MLX wins 17-30q, but its int32 ShapeElem rejects >=2**31 amplitudes, so it
    # caps at 30q (Gate 0, docs/plan.md). The Metal backend uses 64-bit indexing
    # and genuine in-place updates to reach 31-33q -- the only path past 30q.
    if _MLX_AVAILABLE and n_qubits <= 30:
        return "mlx"
    if _METAL_AVAILABLE:
        return "metal"
    if _MLX_AVAILABLE and n_qubits <= 30:
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
