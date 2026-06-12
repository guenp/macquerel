from __future__ import annotations

import inspect
from argparse import Namespace

from asv_benchmarks.common import force_gc, optional_backend

from bench_mcm import MODES, rep_code_circuit, run_collapse, time_call as _time_call
from macquerel.circuit import Gate, MeasureOp
from macquerel.compiler import fuse_gates
from macquerel.simulator import _make_backend


def _fuse_gates(qc, backend_name: str):
    if "backend" in inspect.signature(fuse_gates).parameters:
        return fuse_gates(qc, backend=backend_name)
    return fuse_gates(qc)


def measure_cell(backend_name: str, mode: str, qc, args) -> float:
    reps = args.reps
    if mode == "sample":
        from macquerel import Simulator

        sim = Simulator(backend=backend_name)
        return _time_call(lambda: sim.run(qc, shots=args.shots), reps)

    backend = _make_backend(backend_name, "complex64", None)
    fused = _fuse_gates(qc, backend_name)
    ops = [op for op in fused.ops if isinstance(op, (Gate, MeasureOp))]
    secs = _time_call(lambda: run_collapse(backend, ops, qc.n_qubits, args.traj_shots), reps)
    return secs / args.traj_shots


class MidCircuitMeasurementRuntime:
    """MCM repetition-code cases from ``bench_mcm.py``."""

    params = [["rounds", "qubits"], ["cpu", "mlx", "metal"], MODES, [1, 8]]
    param_names = ["sweep", "backend", "mode", "point"]
    timeout = 600

    def setup(self, sweep: str, backend: str, mode: str, point: int) -> None:
        optional_backend(_make_backend, backend, "complex64", None)
        distance = 7 if sweep == "rounds" else point
        rounds = point if sweep == "rounds" else 8
        self.circuit = rep_code_circuit(distance, rounds)
        self.args = Namespace(shots=1000, traj_shots=8, reps=1)

    def teardown(self, sweep: str, backend: str, mode: str, point: int) -> None:
        force_gc()

    def time_mcm(self, sweep: str, backend: str, mode: str, point: int) -> None:
        measure_cell(backend, mode, self.circuit, self.args)
