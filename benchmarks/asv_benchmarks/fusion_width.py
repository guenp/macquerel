from __future__ import annotations

import time

from asv_benchmarks.common import force_gc

from circuits import build_qaoa, build_qft, build_quantum_volume, build_random

WIDTHS = list(range(1, 7))

CIRCUITS = {
    "QFT": lambda n: build_qft(n),
    "random": lambda n: build_random(n, depth=50),
    "QAOA": lambda n: build_qaoa(n),
    "QV": lambda n: build_quantum_volume(n),
}


def _make_backend(name: str):
    try:
        if name == "cpu":
            from macquerel.backends.cpu import CPUBackend

            return CPUBackend()
        if name == "mlx":
            from macquerel.backends.mlx_backend import MLXBackend

            return MLXBackend()
        if name == "metal":
            from macquerel.backends.metal_backend import MetalBackend

            return MetalBackend()
    except Exception as exc:
        raise NotImplementedError(f"{name} backend unavailable") from exc
    raise NotImplementedError(f"unknown backend {name!r}")


def _time_fuse_and_apply(backend, circuit, width: int, n: int) -> float:
    from macquerel.circuit import Gate
    from macquerel.compiler import fuse_gates

    t0 = time.perf_counter()
    fused = fuse_gates(circuit, max_fused_qubits=width)
    sv = backend.allocate(n)
    for op in fused.ops:
        if isinstance(op, Gate):
            sv = backend.apply_matrix(sv, op.matrix, op.targets, op.controls or None)
    backend.to_numpy(sv)
    return time.perf_counter() - t0


class FusionWidthRuntime:
    """Fusion-width sweep from ``bench_fusion_width.py``."""

    params = [["cpu", "mlx", "metal"], list(CIRCUITS), [8, 12], WIDTHS]
    param_names = ["backend", "circuit", "qubits", "max_fused_qubits"]
    timeout = 600

    def setup(self, backend: str, circuit: str, qubits: int, max_fused_qubits: int) -> None:
        self.backend = _make_backend(backend)
        self.circuit_obj = CIRCUITS[circuit](qubits)

    def teardown(self, backend: str, circuit: str, qubits: int, max_fused_qubits: int) -> None:
        force_gc()

    def time_fuse_and_apply(
        self, backend: str, circuit: str, qubits: int, max_fused_qubits: int
    ) -> None:
        _time_fuse_and_apply(self.backend, self.circuit_obj, max_fused_qubits, qubits)
