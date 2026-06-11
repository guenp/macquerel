from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path

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
# Step 21-27 performance line; re-tuned after Step 34):
#   - CPU wins through ~15q: the state is only a few MB, so per-kernel GPU
#     dispatch latency dominates the compute. Step 34 (shared pipelines,
#     pooled buffers, fewer per-dispatch ObjC calls) moved this down from 16:
#     at 16q Metal now wins qft/random/qaoa (e.g. qft 5.6 ms vs cpu 9.2 ms)
#     and the Step 35 tier autotuner independently measures cpu_max=15 here.
#   - Metal wins everywhere above that. Before Step 22 it paid a per-gate
#     commit + waitUntilCompleted that handed 17-21q to MLX; with batched
#     command-buffer encoding (Step 22) plus specialized kernels (Step 25)
#     that penalty is gone and Metal beats MLX at every measured count >=17
#     (e.g. 20q qft 21ms vs 33ms, 28q random 1.27s vs 2.75s). It is also the
#     only backend past 30q -- MLX's int32 ShapeElem rejects >=2**31
#     amplitudes (Gate 0, docs/plan_completed.md).
#   - MLX serves 17-30q only as the fallback when the Metal backend (pyobjc)
#     is not installed.
_CPU_MAX_QUBITS = 15
_MLX_MAX_QUBITS = 30


# ---------------------------------------------------------------------------
# Step 35: per-chip tier boundary, with opt-in autotuning
# ---------------------------------------------------------------------------
# The 16q CPU/GPU crossover above is measured on an M5 Max; base M-series
# chips have different bandwidth/latency ratios, so the boundary is a chip
# property. Mirroring MACQUEREL_FUSION_WIDTH (Step 20):
#   1. MACQUEREL_BACKEND_TIERS=<int>  -> pin the CPU tier's max qubit count
#   2. MACQUEREL_BACKEND_TIERS=auto   -> measure the crossover on this chip
#      once and cache it (~/.cache/macquerel/backend_tiers.json + in-memory)
#   3. unset                          -> the measured default, no measuring
# Autotuning is opt-in because the measurement runs both backends across a
# span of qubit counts (~seconds), which has no place on the zero-config
# import path. Measurement failures fall back to the default, never raise.
_TIERS_CACHE: int | None = None
_TIER_SPAN = (10, 12, 14, 16, 18, 20)


def _tiers_cache_path() -> Path:
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "macquerel" / "backend_tiers.json"


def _gpu_backend_name() -> str | None:
    if _METAL_AVAILABLE:
        return "metal"
    if _MLX_AVAILABLE:
        return "mlx"
    return None


def _tier_circuits(n: int) -> list[Circuit]:
    """A QFT and a brickwork random circuit — the two regimes (diagonal-heavy
    and dense/scattered) whose crossovers bracket the realistic workloads."""
    from macquerel.compiler import _representative_circuit

    rng = np.random.default_rng(11)
    qc = Circuit(n)
    for d in range(8):
        for q in range(n):
            qc.rz(q, float(rng.uniform(0, 2 * np.pi)))
            qc.ry(q, float(rng.uniform(0, 2 * np.pi)))
        for q in range(d % 2, n - 1, 2):
            qc.cx(q, q + 1)
    return [_representative_circuit(n), qc]


def _time_statevector(backend_name: str, circuit: Circuit, reps: int = 3) -> float:
    """Min wall-clock seconds to fuse + apply + read back on `backend_name`."""
    backend = _make_backend(backend_name, "complex64")
    fused = fuse_gates(circuit, backend=backend_name)
    gates = [op for op in fused.ops if isinstance(op, Gate)]

    def once() -> float:
        t0 = time.perf_counter()
        sv = backend.allocate(circuit.n_qubits, np.complex64)
        for g in gates:
            sv = backend.apply_matrix(sv, g.matrix, g.targets, g.controls or None)
        backend.to_numpy(sv)
        return time.perf_counter() - t0

    once()  # warm-up: pipeline/kernel compilation, buffer pool fill
    return min(once() for _ in range(reps))


def _measure_cpu_max() -> int:
    """Measure the CPU/GPU crossover: the largest qubit count CPU should own.

    For each qubit count in the span, both backends run the same two circuits;
    the crossover is the smallest count where the GPU's total wins *and keeps
    winning* at every larger measured count (a single lucky cell must not move
    the boundary). If the GPU never sustainably wins, the CPU keeps the whole
    measured span.
    """
    gpu = _gpu_backend_name()
    if gpu is None:
        return _CPU_MAX_QUBITS
    gpu_wins: dict[int, bool] = {}
    for n in _TIER_SPAN:
        circuits = _tier_circuits(n)
        cpu_t = sum(_time_statevector("cpu", c) for c in circuits)
        gpu_t = sum(_time_statevector(gpu, c) for c in circuits)
        gpu_wins[n] = gpu_t < cpu_t
    for i, n in enumerate(_TIER_SPAN):
        if all(gpu_wins[m] for m in _TIER_SPAN[i:]):
            return n - 1
    return _TIER_SPAN[-1]


def autotune_backend_tiers(force: bool = False) -> int:
    """Measure and return this chip's CPU-tier boundary (opt-in, cached).

    Not invoked on the hot path — `_select_backend` defaults to the constant
    measured boundary. Call explicitly or set ``MACQUEREL_BACKEND_TIERS=auto``.
    The result is cached to disk and in-memory so the measurement runs at most
    once; `force=True` re-measures. Failures fall back to the default.
    """
    global _TIERS_CACHE

    if _TIERS_CACHE is not None and not force:
        return _TIERS_CACHE

    path = _tiers_cache_path()
    if not force and path.exists():
        try:
            cached = int(json.loads(path.read_text())["cpu_max_qubits"])
            if cached >= 1:
                _TIERS_CACHE = cached
                return _TIERS_CACHE
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass

    try:
        cpu_max = _measure_cpu_max()
    except Exception:
        cpu_max = _CPU_MAX_QUBITS

    _TIERS_CACHE = cpu_max
    try:  # best-effort persistence, like the fusion-width cache
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"cpu_max_qubits": cpu_max, "platform": os.uname().sysname}))
    except Exception:
        pass
    return cpu_max


def _resolve_cpu_max() -> int:
    env = os.environ.get("MACQUEREL_BACKEND_TIERS")
    if env:
        if env.strip().lower() == "auto":
            return autotune_backend_tiers()
        try:
            pinned = int(env)
            if pinned >= 0:
                return pinned
        except ValueError:
            pass
    return _CPU_MAX_QUBITS


def _select_backend(n_qubits: int) -> str:
    if n_qubits <= _resolve_cpu_max():
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
        # Step 34: in auto mode, reuse backend instances across calls instead
        # of constructing one per call — backend construction used to cost
        # ~7.5 ms for Metal (device + queue + pipeline compile; now also
        # cached process-wide) and reuse keeps the backend's buffer pool and
        # classify/pipeline caches warm. Seeded simulators keep the
        # fresh-backend-per-call behavior: each call must restart the RNG
        # stream so repeated runs stay bit-identical.
        self._auto_backends: dict[str, object] = {}

    def _backend_name_for(self, n_qubits: int) -> str:
        if self.backend_name != "auto":
            return self.backend_name
        return _select_backend(n_qubits)

    def _get_backend(self, n_qubits: int):
        if self._backend is not None:
            return self._backend
        name = self._backend_name_for(n_qubits)
        if self._seed is not None:
            return _make_backend(name, self.dtype, self._seed)
        backend = self._auto_backends.get(name)
        if backend is None:
            backend = _make_backend(name, self.dtype, None)
            self._auto_backends[name] = backend
        return backend

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

        The fusion width defaults per backend (Step 30), so fusion needs to
        know which backend this circuit will run on.
        """
        fused = fuse_gates(circuit, backend=self._backend_name_for(circuit.n_qubits))
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
