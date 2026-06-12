from __future__ import annotations

import numpy as np

from asv_benchmarks.common import force_gc, optional_backend

from macquerel.simulator import _make_backend


def random_thetas(n: int, layers: int, seed: int = 7) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0, 2 * np.pi, size=2 * n * layers)


def ansatz(n: int, thetas: np.ndarray, layers: int):
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


def tfim_terms(n: int, j: float = 1.0, h: float = 1.0) -> list:
    terms = [(-j, [("Z", i), ("Z", i + 1)]) for i in range(n - 1)]
    terms += [(-h, [("X", i)]) for i in range(n)]
    return terms


def tfim_energy_host(states: np.ndarray, n: int, j: float = 1.0, h: float = 1.0) -> np.ndarray:
    b = states.shape[0]
    view = states.reshape((b,) + (2,) * n)
    probs = np.abs(view) ** 2
    sum_axes = tuple(range(1, n + 1))
    z = np.array([1.0, -1.0])
    e = np.zeros(b)
    for i in range(n - 1):
        shape_a = [1] * (n + 1)
        shape_a[1 + i] = 2
        shape_b = [1] * (n + 1)
        shape_b[2 + i] = 2
        e += -j * np.sum(probs * z.reshape(shape_a) * z.reshape(shape_b), axis=sum_axes)
    for i in range(n):
        flipped = np.flip(view, axis=1 + i)
        e += -h * np.real(np.sum(np.conj(view) * flipped, axis=sum_axes))
    return e


def shifted_circuits(n: int, layers: int, thetas: np.ndarray) -> list:
    circuits = []
    for k in range(len(thetas)):
        for sign in (1.0, -1.0):
            shifted = thetas.copy()
            shifted[k] += sign * np.pi / 2
            circuits.append(ansatz(n, shifted, layers))
    return circuits


def gradient_loop(sim, circuits: list, n: int) -> np.ndarray:
    energies = np.array([tfim_energy_host(sim.statevector(c)[None], n)[0] for c in circuits])
    return (energies[0::2] - energies[1::2]) / 2


def gradient_batched(bs, circuits: list, n: int) -> np.ndarray:
    energies = tfim_energy_host(bs.statevectors(circuits), n)
    return (energies[0::2] - energies[1::2]) / 2


class VQEEnergyRuntime:
    """VQE energy sweeps from ``bench_vqe.py``."""

    params = [["qubits", "depth"], ["cpu", "mlx", "metal"], [4, 8, 12]]
    param_names = ["sweep", "backend", "point"]
    timeout = 600

    def setup(self, sweep: str, backend: str, point: int) -> None:
        try:
            from macquerel import TrajectorySimulator
        except ImportError as exc:
            raise NotImplementedError("TrajectorySimulator is unavailable") from exc

        optional_backend(_make_backend, backend, "complex64", None)
        if sweep == "qubits":
            n = point
            layers = 4
        else:
            n = 12
            layers = point
        self.sim = TrajectorySimulator(backend=backend, trajectories=1)
        self.circuit = ansatz(n, random_thetas(n, layers), layers)
        self.ham = tfim_terms(n)

    def teardown(self, sweep: str, backend: str, point: int) -> None:
        force_gc()

    def time_energy(self, sweep: str, backend: str, point: int) -> None:
        self.sim.expectation_pauli(self.circuit, self.ham)


class VQEGradientRuntime:
    """Parameter-shift gradient sweep from ``bench_vqe.py``."""

    params = [[1, 2, 4], ["loop", "batched_cpu", "batched_mlx"]]
    param_names = ["layers", "engine"]
    timeout = 600

    def setup(self, layers: int, engine: str) -> None:
        from macquerel import Simulator

        self.n = 8
        thetas = random_thetas(self.n, layers)
        self.circuits = shifted_circuits(self.n, layers, thetas)
        if engine == "loop":
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

    def teardown(self, layers: int, engine: str) -> None:
        force_gc()

    def time_gradient(self, layers: int, engine: str) -> None:
        if engine == "loop":
            gradient_loop(self.engine, self.circuits, self.n)
        else:
            gradient_batched(self.engine, self.circuits, self.n)
