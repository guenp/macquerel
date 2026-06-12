from __future__ import annotations

from asv_benchmarks.common import force_gc, optional_backend

from bench_statevector import (
    ALL_BACKENDS,
    GENERATORS,
    _system_ram_gib,
    est_peak_gib,
    make_backend,
)

# Same qubit policy as the per-step harness (run_step_bench.sh): the CPU
# backend stops at 22q (dense 24q+ sweeps take minutes per cell), everything
# else runs the full step grid up to 28q, gated by the same pessimistic
# memory budget bench_statevector.py uses (0.45x system RAM).
STEP_QUBITS = [6, 12, 16, 20, 22, 24, 26, 28]
QUBIT_CAP = {"macquerel-cpu": 22}
MEM_BUDGET_GIB = 0.45 * _system_ram_gib()


class StatevectorRuntime:
    """Framework comparison from ``bench_statevector.py``.

    ASV runs each (circuit, backend, qubits) cell in its own process, matching
    the subprocess isolation of ``bench_statevector.py --worker``.
    """

    params = [list(GENERATORS), ALL_BACKENDS, STEP_QUBITS]
    param_names = ["circuit", "backend", "qubits"]
    # Match bench_statevector.time_run(): one warm-up call (JIT / allocation /
    # lazy-graph realization), then time whole single calls, 3 reps in one
    # process. With `asv run --record-samples` the step exporter in
    # plot_steps.py takes min(samples), i.e. best-of-3.
    number = 1
    repeat = 3
    rounds = 1
    warmup_time = 0.1
    timeout = 1800

    def setup(self, circuit: str, backend: str, qubits: int) -> None:
        cap = QUBIT_CAP.get(backend, max(STEP_QUBITS))
        if qubits > cap:
            raise NotImplementedError(f"{backend} capped at {cap} qubits")
        if est_peak_gib(backend, qubits, double=False) > MEM_BUDGET_GIB:
            raise NotImplementedError("estimated cell memory exceeds the memory budget")
        self.ops = GENERATORS[circuit](qubits)
        self.fn = optional_backend(make_backend, backend, False)

    def teardown(self, circuit: str, backend: str, qubits: int) -> None:
        force_gc()

    def time_statevector(self, circuit: str, backend: str, qubits: int) -> None:
        self.fn(self.ops, qubits)
