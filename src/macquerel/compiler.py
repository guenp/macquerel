from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

from macquerel.circuit import Circuit, Gate, MeasureOp
from macquerel.gates import classify

# ---------------------------------------------------------------------------
# Per-chip fusion-width autotuning (Step 20)
# ---------------------------------------------------------------------------
# The optimal max_fused_qubits is a per-chip property: wider fusion means fewer,
# larger gate applications (less Python/dispatch overhead) but a more expensive
# matrix composition and a larger dense apply. The "4 is optimal" figure is a
# default, not a law — we measure it once on the local machine and cache it.
#
# Resolution order for the default width:
#   1. MACQUEREL_FUSION_WIDTH env var (explicit override; never measures)
#   2. in-memory cache (per process)
#   3. on-disk cache (~/.cache/macquerel/fusion_width.json, written once)
#   4. measure on this chip, then persist
_DEFAULT_FUSION_WIDTH = 4
_FUSION_WIDTH_CACHE: int | None = None


def _fusion_cache_path() -> Path:
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "macquerel" / "fusion_width.json"


def _representative_circuit(n: int) -> Circuit:
    """A QFT — a fusion-rich mix of 1q gates and controlled phases."""
    qc = Circuit(n)
    for i in range(n):
        qc.h(i)
        for j in range(i + 1, n):
            qc.cp(i, j, float(np.pi / (2 ** (j - i))))
    for i in range(n // 2):
        qc.swap(i, n - 1 - i)
    return qc


def _time_fuse_and_apply(backend, circuit: Circuit, width: int, n: int) -> float:
    """Wall-clock seconds to fuse `circuit` at `width` and apply it on `backend`."""
    t0 = time.perf_counter()
    fused = fuse_gates(circuit, max_fused_qubits=width)
    sv = backend.allocate(n)
    for op in fused.ops:
        if isinstance(op, Gate):
            sv = backend.apply_matrix(sv, op.matrix, op.targets, op.controls or None)
    backend.to_numpy(sv)
    return time.perf_counter() - t0


def _autotune_backend() -> tuple[object, int]:
    """Pick the backend (and a representative qubit count) to measure on.

    Fusion width is a GPU-dispatch optimization: wider fusion trades a costlier
    host-side matrix composition for fewer kernel launches. That trade only pays
    off on a dispatch-bound backend, so we measure on MLX when it is available
    (the path that actually benefits, at a qubit count where it is selected) and
    fall back to the CPU reference otherwise.
    """
    try:
        from macquerel.backends.mlx_backend import MLXBackend

        return MLXBackend(), 18
    except Exception:
        from macquerel.backends.cpu import CPUBackend

        return CPUBackend(), 12


def _measure_fusion_width(reps: int = 3, candidates=(1, 2, 3, 4, 5, 6)) -> int:
    """Sweep candidate widths on a representative circuit; return the fastest."""
    backend, n = _autotune_backend()
    circuit = _representative_circuit(n)
    best_w, best_t = _DEFAULT_FUSION_WIDTH, float("inf")
    for w in candidates:
        _time_fuse_and_apply(backend, circuit, w, n)  # warm-up
        t = min(_time_fuse_and_apply(backend, circuit, w, n) for _ in range(reps))
        if t < best_t:
            best_t, best_w = t, w
    return best_w


def autotune_fusion_width(force: bool = False) -> int:
    """Return the max_fused_qubits that maximizes throughput on this chip.

    Measured once and cached to disk (and in-memory for the process). Set
    MACQUEREL_FUSION_WIDTH to pin a value and skip measuring. `force=True`
    re-measures and overwrites the cache. Any failure falls back to the
    hardcoded default rather than raising on the hot path.
    """
    global _FUSION_WIDTH_CACHE

    env = os.environ.get("MACQUEREL_FUSION_WIDTH")
    if env:
        try:
            return int(env)
        except ValueError:
            pass

    if _FUSION_WIDTH_CACHE is not None and not force:
        return _FUSION_WIDTH_CACHE

    path = _fusion_cache_path()
    if not force and path.exists():
        try:
            _FUSION_WIDTH_CACHE = int(json.loads(path.read_text())["max_fused_qubits"])
            return _FUSION_WIDTH_CACHE
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass

    try:
        width = _measure_fusion_width()
    except Exception:
        width = _DEFAULT_FUSION_WIDTH

    _FUSION_WIDTH_CACHE = width
    try:  # persistence is best-effort; never fail the hot path over a cache write
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"max_fused_qubits": width, "platform": os.uname().sysname}))
    except Exception:
        pass
    return width


def _compose_gates(group: list[Gate]) -> Gate:
    """Compose a list of gates into a single fused gate."""
    if len(group) == 1:
        return group[0]

    # Collect the ordered qubit set for the fused gate
    qubit_set: list[int] = []
    for g in group:
        for q in g.targets + g.controls:
            if q not in qubit_set:
                qubit_set.append(q)
    qubit_set.sort()
    k = len(qubit_set)
    dim = 2**k

    # Build the composed unitary in the qubit_set basis
    fused_matrix = np.eye(dim, dtype=np.complex128)

    for g in group:
        # Embed g.matrix into the full fused space
        # build full gate matrix (may need to embed controls)
        if g.controls:
            # already a 2-qubit (or larger) matrix including control semantics
            # g.matrix is the raw gate matrix (e.g. CNOT 4x4 already includes control row)
            full_mat = _embed(g.matrix, g.targets + g.controls, qubit_set)
        else:
            full_mat = _embed(g.matrix, g.targets, qubit_set)
        fused_matrix = full_mat @ fused_matrix

    fused_matrix = fused_matrix.astype(np.complex64)
    kind = classify(fused_matrix)
    name = "Fused(" + ",".join(g.name for g in group) + ")"
    return Gate(name=name, matrix=fused_matrix, targets=qubit_set, controls=[], kind=kind)


def _embed(matrix: np.ndarray, gate_qubits: list[int], full_qubits: list[int]) -> np.ndarray:
    """Embed a gate matrix acting on gate_qubits into the full_qubits space."""
    k_full = len(full_qubits)
    k_gate = len(gate_qubits)
    dim_full = 2**k_full
    dim_gate = 2**k_gate

    # Build the full unitary via tensor product with identity on remaining qubits
    # We use the standard trick: write the full unitary as a tensor, set entries.
    gate_mat = matrix.astype(np.complex128)

    # Map gate_qubits to their positions within full_qubits
    gate_pos = [full_qubits.index(q) for q in gate_qubits]

    # Rewrite using tensor indexing
    # full_mat[i, j] = gate_mat[i_gate, j_gate] if gate bits match, else delta
    # We use a direct construction: tensor reshape approach

    # Reshape gate_mat to (2,)*k_gate x (2,)*k_gate
    gate_t = gate_mat.reshape((2,) * (2 * k_gate))

    # Build output tensor
    out = np.eye(dim_full, dtype=np.complex128).reshape((2,) * (2 * k_full))

    # Iterate over all combinations of non-gate qubit indices
    non_gate_pos = [i for i in range(k_full) if i not in gate_pos]
    n_non_gate = len(non_gate_pos)
    dim_non_gate = 2**n_non_gate

    for env_idx in range(dim_non_gate):
        env_bits = [(env_idx >> (n_non_gate - 1 - i)) & 1 for i in range(n_non_gate)]
        for gi in range(dim_gate):
            for gj in range(dim_gate):
                gi_bits = [(gi >> (k_gate - 1 - b)) & 1 for b in range(k_gate)]
                gj_bits = [(gj >> (k_gate - 1 - b)) & 1 for b in range(k_gate)]
                # Build full row/col index tuples
                row_idx = [None] * k_full
                col_idx = [None] * k_full
                for pos, bit in zip(gate_pos, gi_bits, strict=True):
                    row_idx[pos] = bit
                for pos, bit in zip(gate_pos, gj_bits, strict=True):
                    col_idx[pos] = bit
                for pos, bit in zip(non_gate_pos, env_bits, strict=True):
                    row_idx[pos] = bit
                    col_idx[pos] = bit
                full_row = tuple(row_idx) + tuple(col_idx)
                # Reset to zero first (the identity initialisation handles diagonal)
                out[full_row] = gate_t[tuple(gi_bits + gj_bits)]

    return out.reshape(dim_full, dim_full)


def fuse_gates(circuit: Circuit, max_fused_qubits: int | None = None) -> Circuit:
    """Greedy gate fusion pass. Returns a new Circuit with fused gates.

    `max_fused_qubits=None` (the default) uses the per-chip autotuned width
    (Step 20); pass an int to pin it.
    """
    if max_fused_qubits is None:
        max_fused_qubits = autotune_fusion_width()
    result = Circuit(circuit.n_qubits)
    result.ops = []

    current_group: list[Gate] = []
    current_qubits: set[int] = set()

    def flush() -> None:
        if not current_group:
            return
        fused = _compose_gates(current_group)
        result.ops.append(fused)
        current_group.clear()
        current_qubits.clear()

    for op in circuit.ops:
        if isinstance(op, MeasureOp):
            flush()
            result.ops.append(op)
            continue

        assert isinstance(op, Gate)
        op_qubits = set(op.targets + op.controls)
        merged_qubits = current_qubits | op_qubits

        if current_group and len(merged_qubits) > max_fused_qubits:
            flush()

        current_group.append(op)
        current_qubits.update(op_qubits)

    flush()
    return result


def remap_qubits(circuit: Circuit) -> Circuit:
    """Relabel qubit indices so frequently-targeted qubits land on lowest indices.

    This is a single-window global relabeling (Doi-Horii technique). The returned
    circuit is logically equivalent: measurement qubit labels are updated to match,
    so that sampling produces the same distribution when labels are consistently
    applied via the returned permutation.

    Returns the remapped Circuit. The permutation applied is deterministic: the
    qubit with the highest gate-access count receives index 0, etc. Ties are broken
    by original qubit index (lower index wins).
    """
    freq: Counter[int] = Counter()
    for op in circuit.ops:
        if isinstance(op, Gate):
            for q in op.targets + op.controls:
                freq[q] += 1

    # Sort by descending frequency; ties broken by ascending original index.
    sorted_qubits = sorted(range(circuit.n_qubits), key=lambda q: (-freq[q], q))
    perm = {old: new for new, old in enumerate(sorted_qubits)}

    result = Circuit(circuit.n_qubits)
    for op in circuit.ops:
        if isinstance(op, Gate):
            result.ops.append(
                Gate(
                    name=op.name,
                    matrix=op.matrix,
                    targets=[perm[q] for q in op.targets],
                    controls=[perm[q] for q in op.controls],
                    kind=op.kind,
                )
            )
        elif isinstance(op, MeasureOp):
            result.ops.append(MeasureOp(qubits=[perm[q] for q in op.qubits]))
    return result
