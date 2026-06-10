from __future__ import annotations

import os
from collections import Counter

import numpy as np

from macquerel.circuit import Circuit, Gate, MeasureOp
from macquerel.compiler import fuse_gates, remap_qubits_with_perm

try:
    import mlx.core as mx  # noqa: F401  # ty: ignore[unresolved-import]

    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False

try:
    from macquerel.backends.metal_backend import _METAL_AVAILABLE
except ImportError:  # pragma: no cover - module always importable; guard anyway
    _METAL_AVAILABLE = False


# Measured tier boundaries (benchmarks/data/steps, 2026-06, M5 Max, after the
# Step 21-27 performance line):
#   - CPU wins through ~16q: the state is only a few MB, so per-kernel GPU
#     dispatch latency dominates the compute.
#   - Metal wins everywhere above that. Before Step 22 it paid a per-gate
#     commit + waitUntilCompleted that handed 17-21q to MLX; with batched
#     command-buffer encoding (Step 22) plus specialized kernels (Step 25)
#     that penalty is gone and Metal beats MLX at every measured count >=17
#     (e.g. 20q qft 21ms vs 33ms, 28q random 1.27s vs 2.75s). It is also the
#     only backend past 30q -- MLX's int32 ShapeElem rejects >=2**31
#     amplitudes (Gate 0, docs/plan_completed.md).
#   - MLX serves 17-30q only as the fallback when the Metal backend (pyobjc)
#     is not installed.
_CPU_MAX_QUBITS = 16
_MLX_MAX_QUBITS = 30


def _select_backend(n_qubits: int) -> str:
    if n_qubits <= _CPU_MAX_QUBITS:
        return "cpu"
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
        n = circuit.n_qubits
        sv = backend.allocate(n, self._np_dtype)
        fused, perm = self._compile(circuit)
        for op in fused.ops:
            if isinstance(op, Gate):
                sv = backend.apply_matrix(sv, op.matrix, op.targets, op.controls or None)
        out = backend.to_numpy(sv)
        if perm is not None:
            # Undo the Step 28 relabeling: logical qubit q lives on axis perm[q]
            # of the remapped state; transpose back to the caller's basis order.
            axes = [perm[q] for q in range(n)]
            out = np.ascontiguousarray(np.transpose(out.reshape((2,) * n), axes)).reshape(-1)
        return out

    def _compile(self, circuit: Circuit) -> tuple[Circuit, dict[int, int] | None]:
        """Fusion (+ optional Step 28 qubit remapping) for the hot path.

        Remapping relabels qubits so the hottest ones get the smallest strides.
        Counts need no fix-up — `remap_qubits` rewrites MeasureOp labels in list
        order, and `sample()` keys output bits by that order — but `statevector`
        readback must invert the permutation (see statevector()). Disabled by
        default pending the Step 28 A/B; enable with MACQUEREL_REMAP=1.
        """
        fused = fuse_gates(circuit)
        if os.environ.get("MACQUEREL_REMAP") != "1":
            return fused, None
        remapped, perm = remap_qubits_with_perm(fused)
        if all(perm[q] == q for q in perm):
            return fused, None
        return remapped, perm

    def run(self, circuit: Circuit, shots: int = 1000) -> Counter:
        backend = self._get_backend(circuit.n_qubits)
        fused, _ = self._compile(circuit)

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
