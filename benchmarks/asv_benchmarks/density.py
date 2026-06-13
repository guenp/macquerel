from __future__ import annotations

from asv_benchmarks.common import force_gc, optional_backend

from bench_density import CIRCUITS, est_peak_gib


class DensityMatrixRuntime:
    """DensityMatrixSimulator runtime cases from ``bench_density.py``."""

    params = [CIRCUITS, ["cpu", "mlx", "metal"], [4, 8, 10]]
    param_names = ["circuit", "backend", "qubits"]
    timeout = 600

    def setup(self, circuit: str, backend: str, qubits: int) -> None:
        if est_peak_gib(backend, qubits) > 8.0:
            raise NotImplementedError("estimated cell memory exceeds ASV safety cap")
        from bench_density import build_circuit

        try:
            from macquerel.density import DensityMatrixSimulator
        except ModuleNotFoundError as exc:
            raise NotImplementedError("DensityMatrixSimulator is unavailable") from exc

        self.circuit_obj = build_circuit(circuit, qubits)
        self.sim = optional_backend(DensityMatrixSimulator, backend=backend)

    def teardown(self, circuit: str, backend: str, qubits: int) -> None:
        force_gc()

    def time_probabilities(self, circuit: str, backend: str, qubits: int) -> None:
        self.sim.probabilities(self.circuit_obj)
