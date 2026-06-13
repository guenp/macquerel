from __future__ import annotations

from asv_benchmarks import common  # noqa: F401

from bench_memory import theoretical_bytes, theoretical_dm_bytes


class MemoryFootprint:
    """Track deterministic memory-size baselines from ``bench_memory.py``.

    The measured-memory harness intentionally stays in ``bench_memory.py``
    because it depends on macOS ``/usr/bin/time -l`` process-ledger output.
    """

    params = [["statevector", "density_matrix"], [8, 12, 16, 20]]
    param_names = ["series", "qubits"]
    unit = "bytes"

    def track_theoretical_bytes(self, series: str, qubits: int) -> int:
        if series == "statevector":
            return theoretical_bytes(qubits)
        return theoretical_dm_bytes(qubits)
