from __future__ import annotations

import numpy as np

from asv_benchmarks.common import force_gc, optional_backend


def _ansatz(n: int, thetas: np.ndarray, layers: int = 3):
    from macquerel import Circuit

    qc = Circuit(n)
    t = iter(thetas)
    for _ in range(layers):
        for q in range(n):
            qc.ry(q, float(next(t)))
            qc.rz(q, float(next(t)))
        for q in range(n - 1):
            qc.cx(q, q + 1)
    return qc


def _sweep(n: int, batch: int, layers: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    return [
        _ansatz(n, rng.uniform(0, 2 * np.pi, size=2 * n * layers), layers) for _ in range(batch)
    ]


class BatchedSweepRuntime:
    """Batched simulator sweep from ``bench_batched.py``."""

    params = [[4, 8, 12], [32, 128], ["loop_auto", "batched_cpu", "batched_mlx"]]
    param_names = ["qubits", "batch", "engine"]
    timeout = 600

    def setup(self, qubits: int, batch: int, engine: str) -> None:
        from macquerel import Simulator

        self.circuits = _sweep(qubits, batch)
        if engine == "loop_auto":
            self.engine = Simulator()
        elif engine == "batched_cpu":
            try:
                from macquerel import BatchedSimulator
            except ImportError as exc:
                raise NotImplementedError("BatchedSimulator is unavailable") from exc

            self.engine = optional_backend(BatchedSimulator, backend="cpu")
        elif engine == "batched_mlx":
            try:
                from macquerel import BatchedSimulator
            except ImportError as exc:
                raise NotImplementedError("BatchedSimulator is unavailable") from exc

            self.engine = optional_backend(BatchedSimulator, backend="mlx")
        else:
            raise NotImplementedError(engine)

    def teardown(self, qubits: int, batch: int, engine: str) -> None:
        force_gc()

    def time_sweep(self, qubits: int, batch: int, engine: str) -> None:
        if engine == "loop_auto":
            for circuit in self.circuits:
                self.engine.statevector(circuit)
        else:
            self.engine.statevectors(self.circuits)
