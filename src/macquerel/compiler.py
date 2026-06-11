from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

from macquerel.circuit import ChannelOp, Circuit, Gate, MeasureOp
from macquerel.gates import classify

# ---------------------------------------------------------------------------
# Fusion width: per-backend defaults, with opt-in per-chip autotuning (Step 20/30)
# ---------------------------------------------------------------------------
# Wider fusion means fewer, larger gate applications (less Python/dispatch
# overhead) but a more expensive matrix composition and a larger dense apply.
# The optimum drifts with qubit count (small n is composition-bound, large n
# apply-bound) — and, since Steps 22/25, *with backend*: fusion amortizes
# per-gate overhead, and how much overhead a gate has is a backend property.
# Measured (Step 30): the width sweep benchmarks/data/fusion_width.json
# (widths 1-6 x {QFT, random, QAOA, QV} x 16-24q) picks the per-(backend, n)
# winners, and the step A/B in benchmarks/data/steps validates them end to end:
#   - metal: 2 up to ~22q, 4 above. Batched command buffers (Step 22) and
#     per-k specialized kernels (Step 25) removed most of the per-gate overhead
#     fusion used to amortize, so at small/mid n wide fusion mostly pays
#     host-side matrix composition (w2 is 1.3-2x faster than w4 at 6-22q). At
#     24q+ the apply dominates again and width 4 wins — a *flat* width 2
#     regressed metal random@24-28 by 2.7-3.7x in the step A/B, which is why
#     the default must be qubit-aware, not just per-backend.
#   - cpu: 3 up to ~18q, 4 above (w3 regressed cpu random@20-22 by ~2.3x).
#   - mlx: 4 everywhere (largest per-gate lazy-graph overhead of the three
#     backends, so it still rewards the widest fusion; the step A/B re-run
#     confirmed 1.00x geomean as the no-change control).
# See the ideal-width table on `fuse_gates` and the benchmark write-up:
#   https://github.com/guenp/macquerel/pull/8#issuecomment-4636543327
#
# Resolution order (see _resolve_fusion_width):
#   1. explicit max_fused_qubits arg to fuse_gates
#   2. MACQUEREL_FUSION_WIDTH=<int>   -> that fixed width (all backends, all n)
#   3. MACQUEREL_FUSION_WIDTH=auto    -> autotune_fusion_width() (measure+cache)
#   4. unset                          -> default_fusion_width(backend, n), no
#                                        measuring (4 if the backend is unknown)
_DEFAULT_FUSION_WIDTH = 4
# backend -> ((max_n, width), ...): the width for the smallest max_n >= n;
# qubit counts past the last bound (and unknown backends) use the default of 4.
_BACKEND_FUSION_WIDTH: dict[str, tuple[tuple[int, int], ...]] = {
    "metal": ((22, 2),),
    "cpu": ((18, 3),),
    "mlx": (),
}
_FUSION_WIDTH_CACHE: int | None = None


def default_fusion_width(backend: str | None, n_qubits: int | None) -> int:
    """The measured zero-config fusion width for (backend, qubit count)."""
    if backend is not None and n_qubits is not None:
        for max_n, width in _BACKEND_FUSION_WIDTH.get(backend, ()):
            if n_qubits <= max_n:
                return width
    return _DEFAULT_FUSION_WIDTH


def _fusion_cache_path() -> Path:
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "macquerel" / "fusion_width.json"


def _is_valid_fusion_width(width: int) -> bool:
    return width >= 1


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


def _autotune_backend() -> tuple[object, tuple[int, ...]]:
    """Pick the backend and the span of qubit counts to measure across.

    The optimal fusion width *drifts with qubit count*: at small n the one-time
    matrix-composition cost dominates the apply and rewards narrow fusion, but as
    n grows the apply (a full pass over the 2**n state) dominates and wider fusion
    wins by making fewer passes. Measuring at a single small n therefore picks a
    width that regresses the large-n path by up to ~2x (see benchmarks/data and
    the PR discussion). We instead measure across a *span* straddling the regime
    and pick the width that is best on aggregate, which is robust to that drift.

    MLX serves the 17-30q tier where wall-clock actually matters, so we measure
    there when available; otherwise we fall back to the CPU reference near the top
    of its <=16q auto-select tier.
    """
    try:
        from macquerel.backends.mlx_backend import MLXBackend

        return MLXBackend(), (20, 22)
    except Exception:
        from macquerel.backends.cpu import CPUBackend

        return CPUBackend(), (14, 16)


def _measure_fusion_width(reps: int = 2, candidates=(1, 2, 3, 4, 5, 6)) -> int:
    """Sweep candidate widths across the regime span; return the aggregate best.

    For each qubit count we time fuse+apply at every candidate width and normalize
    by that count's fastest width (so a slow large-n point doesn't simply dominate
    the sum). The width with the lowest total normalized time wins; ties within 2%
    break toward the documented default, since it is the most robust across n.
    """
    backend, n_span = _autotune_backend()
    score: dict[int, float] = dict.fromkeys(candidates, 0.0)
    for n in n_span:
        circuit = _representative_circuit(n)
        times: dict[int, float] = {}
        for w in candidates:
            _time_fuse_and_apply(backend, circuit, w, n)  # warm-up
            times[w] = min(_time_fuse_and_apply(backend, circuit, w, n) for _ in range(reps))
        fastest = min(times.values())
        for w in candidates:
            score[w] += times[w] / fastest

    best = min(score.values())
    # Among widths within 2% of the best aggregate score, prefer the one closest
    # to the documented default (4) — the choice that generalizes best across n.
    near_best = [w for w in candidates if score[w] <= best * 1.02]
    return min(near_best, key=lambda w: (abs(w - _DEFAULT_FUSION_WIDTH), w))


def autotune_fusion_width(force: bool = False) -> int:
    """Measure and return the best max_fused_qubits for this chip (opt-in).

    This is **not** invoked on the hot path — `fuse_gates` defaults to a fixed
    width of 4 (see `_resolve_fusion_width`). Call this explicitly, or set
    ``MACQUEREL_FUSION_WIDTH=auto``, to opt into per-chip tuning. The result is
    cached to disk (`~/.cache/macquerel/fusion_width.json`) and in-memory, so the
    measurement runs at most once. `force=True` re-measures and overwrites the
    cache. Any measurement failure falls back to the default rather than raising.
    """
    global _FUSION_WIDTH_CACHE

    if _FUSION_WIDTH_CACHE is not None and not force:
        return _FUSION_WIDTH_CACHE

    path = _fusion_cache_path()
    if not force and path.exists():
        try:
            cached = int(json.loads(path.read_text())["max_fused_qubits"])
            if _is_valid_fusion_width(cached):
                _FUSION_WIDTH_CACHE = cached
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


def _resolve_fusion_width(backend: str | None = None, n_qubits: int | None = None) -> int:
    """Resolve the default fusion width when fuse_gates is called without one.

    Zero-config default is the per-(backend, qubit-count) table
    (`default_fusion_width`), or 4 when the backend is unknown.
    ``MACQUEREL_FUSION_WIDTH`` opts out: a positive int pins the width for
    every backend and qubit count; ``auto`` runs the per-chip autotuner.
    Unparseable or non-positive values are ignored in favor of the default.
    """
    env = os.environ.get("MACQUEREL_FUSION_WIDTH")
    if env:
        if env.strip().lower() == "auto":
            return autotune_fusion_width()
        try:
            width = int(env)
            if _is_valid_fusion_width(width):
                return width
        except ValueError:
            pass
    return default_fusion_width(backend, n_qubits)


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


# ---------------------------------------------------------------------------
# Step 26: diagonal-run wide fusion
# ---------------------------------------------------------------------------
# Diagonal x diagonal = diagonal: composition is an O(2**k) elementwise product
# (not an O(8**k) matrix product) and *applying* a diagonal is one elementwise
# pass over the state regardless of k. So adjacent diagonal gates can fuse far
# wider than the dense width-4 limit. This runs as a second pass after the
# general fusion pass — deliberately, because general fusion is what composes
# e.g. QAOA's CX-Rz-CX into a diagonal ZZ block in the first place; the second
# pass then merges those blocks (and QFT's CP cascades) into wide diagonals.
# The cap bounds the dense 2**k x 2**k matrix the Gate still carries for the
# backends. Measured sweep (QFT@22-26, all three backends): widths 7-8 win;
# width 10 *regresses* because materializing/classifying a 1024x1024 dense
# matrix (8 MiB memset at fuse time + an 8 MiB copy in every backend's
# classify per apply) costs more than the saved passes. 8 (256x256, 512 KiB)
# is the aggregate winner.
_DIAG_FUSION_WIDTH = 8


def _resolve_diag_fusion_width() -> int:
    """MACQUEREL_DIAG_FUSION_WIDTH overrides; <=1 disables the pass."""
    env = os.environ.get("MACQUEREL_DIAG_FUSION_WIDTH")
    if env:
        try:
            return int(env)
        except ValueError:
            pass
    return _DIAG_FUSION_WIDTH


def _compose_diagonal_run(group: list[Gate]) -> Gate:
    """Compose adjacent diagonal gates via an elementwise diagonal product."""
    if len(group) == 1:
        return group[0]
    qubit_set: list[int] = sorted({q for g in group for q in g.targets})
    k = len(qubit_set)
    pos = {q: i for i, q in enumerate(qubit_set)}
    idx = np.arange(2**k)
    diag = np.ones(2**k, dtype=np.complex128)
    for g in group:
        kg = len(g.targets)
        gd = np.diag(g.matrix).astype(np.complex128)
        sub = np.zeros(2**k, dtype=np.int64)
        for j, q in enumerate(g.targets):
            bit = (idx >> (k - 1 - pos[q])) & 1
            sub |= bit << (kg - 1 - j)
        diag = diag * gd[sub]
    matrix = np.diag(diag).astype(np.complex64)
    name = "DiagFused(" + ",".join(g.name for g in group) + ")"
    return Gate(name=name, matrix=matrix, targets=qubit_set, controls=[], kind="diagonal")


def _merge_diagonal_runs(circuit: Circuit, max_width: int) -> Circuit:
    """Merge maximal runs of adjacent diagonal gates up to `max_width` qubits.

    Diagonal gates all commute with each other, but this pass does not rely on
    that: it only merges *adjacent* diagonal gates, so any interleaved dense or
    permutation gate (which may not commute) acts as a barrier.
    """
    result = Circuit(circuit.n_qubits)
    result.ops = []
    run: list[Gate] = []
    run_qubits: set[int] = set()

    def flush() -> None:
        if not run:
            return
        result.ops.append(_compose_diagonal_run(run))
        run.clear()
        run_qubits.clear()

    for op in circuit.ops:
        is_diag = isinstance(op, Gate) and not op.controls and classify(op.matrix) == "diagonal"
        if not is_diag:
            flush()
            result.ops.append(op)
            continue
        if run and len(run_qubits | set(op.targets)) > max_width:
            flush()
        run.append(op)
        run_qubits.update(op.targets)
    flush()
    return result


def fuse_gates(
    circuit: Circuit,
    max_fused_qubits: int | None = None,
    backend: str | None = None,
) -> Circuit:
    """Greedy gate fusion pass. Returns a new Circuit with fused gates.

    `max_fused_qubits=None` (the default) resolves to the measured
    **per-(backend, qubit-count)** default — metal 2 up to 22q, cpu 3 up to
    18q, otherwise 4 (Step 30; see `default_fusion_width`). `backend` is only
    consulted for that resolution; the Simulator passes the backend it
    selected.

    Why per-backend? The ideal width depends on qubit count, because fusion
    trades a one-time matrix-composition cost against the per-apply cost of a
    full pass over the ``2**n`` state — and on how much per-gate dispatch
    overhead the backend has left to amortize. Benchmarked on an M5 Max
    (fuse+apply, MLX backend):

        | qubits n   | ideal max_fused_qubits | why                          |
        |------------|------------------------|------------------------------|
        | <= ~16     | 1-2 (immaterial)       | apply is sub-ms; composition |
        |            |                        | overhead dominates           |
        | ~20        | 3                      | apply starting to dominate   |
        | ~22        | 4-5                    |                              |
        | ~24        | 5-6                    | apply-bound; wider = fewer   |
        |            |                        | passes over the big state    |
        | aggregate  | **4**                  | normalized winner across     |
        | 17-30q     |                        | the measured regime          |

    Metal's per-gate overhead mostly vanished with batched command buffers and
    per-k specialized kernels (Steps 22/25), shifting its small/mid-n winner
    down to 2 (its large-n winner stays 4 — the apply still dominates there);
    the CPU backend's NumPy dispatch sits in between. A naive single-small-n
    autotuner instead picks 2 for MLX and regresses its large-n path by up to
    ~2x — see the benchmark write-up:
    https://github.com/guenp/macquerel/pull/8#issuecomment-4636543327

    Pass a positive int to pin the width, or set ``MACQUEREL_FUSION_WIDTH`` (a
    positive int, or ``auto`` to opt into the cached per-chip autotuner; see
    `autotune_fusion_width`).
    """
    if max_fused_qubits is None:
        max_fused_qubits = _resolve_fusion_width(backend, circuit.n_qubits)
    elif not _is_valid_fusion_width(max_fused_qubits):
        raise ValueError("max_fused_qubits must be >= 1")
    result = Circuit(circuit.n_qubits)
    result.ops = []

    # Step 27: commutation-aware grouping. Gates on disjoint qubit sets
    # commute, so instead of one greedy in-order group (where a gate on an
    # unrelated qubit inflates the union and forces an early flush), keep
    # several open groups and route each gate to the best one:
    #   - a gate must land in, or after, the *latest* open group it shares a
    #     qubit with (that is the only real ordering constraint);
    #   - within that freedom, join the first group with capacity;
    #   - otherwise open a new group.
    # Groups are emitted in opening order, so all shared-qubit ordering is
    # preserved; everything that gets reordered across groups is disjoint.
    open_groups: list[tuple[list[Gate], set[int]]] = []
    _MAX_OPEN_GROUPS = 8  # bounds the per-gate scan; oldest is flushed beyond this

    def flush_all() -> None:
        for gates, _ in open_groups:
            result.ops.append(_compose_gates(gates))
        open_groups.clear()

    for op in circuit.ops:
        if not isinstance(op, Gate):
            # MeasureOp / ChannelOp: a non-unitary barrier. Channels do not
            # commute with the gates around them, so every open group flushes
            # and the op passes through in place.
            flush_all()
            result.ops.append(op)
            continue

        op_qubits = set(op.targets + op.controls)

        # Latest open group sharing a qubit with this gate (ordering barrier).
        barrier = -1
        for i in range(len(open_groups) - 1, -1, -1):
            if open_groups[i][1] & op_qubits:
                barrier = i
                break

        placed = False
        for i in range(max(barrier, 0), len(open_groups)):
            gates, qubits = open_groups[i]
            if len(qubits | op_qubits) <= max_fused_qubits:
                gates.append(op)
                qubits.update(op_qubits)
                placed = True
                break
        if not placed:
            open_groups.append(([op], set(op_qubits)))
            if len(open_groups) > _MAX_OPEN_GROUPS:
                gates, _ = open_groups.pop(0)
                result.ops.append(_compose_gates(gates))

    flush_all()

    # Step 26: second pass — merge adjacent diagonal gates (including diagonal
    # composites produced above, e.g. CX·Rz·CX -> ZZ block) into wide diagonals.
    diag_width = _resolve_diag_fusion_width()
    if diag_width > max_fused_qubits:
        result = _merge_diagonal_runs(result, diag_width)
    return result


def remap_qubits(circuit: Circuit) -> Circuit:
    """Relabel qubit indices so frequently-targeted qubits land on lowest indices.

    This is a single-window global relabeling (Doi-Horii technique). The returned
    circuit is logically equivalent: measurement qubit labels are updated to match,
    so that sampling produces the same distribution when labels are consistently
    applied via the returned permutation.

    Returns the remapped Circuit. The permutation applied is deterministic: the
    qubit with the highest gate-access count receives index 0, etc. Ties are broken
    by original qubit index (lower index wins). Use `remap_qubits_with_perm` when
    the caller needs the permutation to invert the relabeling (Step 28).
    """
    return remap_qubits_with_perm(circuit)[0]


def remap_qubits_with_perm(circuit: Circuit) -> tuple[Circuit, dict[int, int]]:
    """Like `remap_qubits`, but also returns the applied permutation.

    The permutation maps ``old qubit -> new qubit``: gate/measure label ``q`` in
    the input circuit appears as ``perm[q]`` in the returned circuit.
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
        elif isinstance(op, ChannelOp):
            result.ops.append(
                ChannelOp(name=op.name, kraus=op.kraus, qubits=[perm[q] for q in op.qubits])
            )
    return result, perm
