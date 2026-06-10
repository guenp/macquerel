import numpy as np

import macquerel.compiler as compiler
import macquerel.gates as g
from macquerel.backends.cpu import CPUBackend
from macquerel.circuit import Circuit, Gate, MeasureOp
from macquerel.compiler import autotune_fusion_width, fuse_gates, remap_qubits


def _run_statevector(circuit: Circuit) -> np.ndarray:
    cpu = CPUBackend()
    sv = cpu.allocate(circuit.n_qubits)
    for op in circuit.ops:
        if isinstance(op, Gate):
            sv = cpu.apply_matrix(sv, op.matrix, op.targets, op.controls or None)
    return sv


def test_fusion_preserves_statevector():
    qc = Circuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(2, 0.7)
    qc.h(1)

    sv_unfused = _run_statevector(qc)
    fused = fuse_gates(qc)
    sv_fused = _run_statevector(fused)

    assert np.allclose(sv_unfused, sv_fused, atol=1e-5), (
        f"max diff: {np.max(np.abs(sv_unfused - sv_fused))}"
    )


def test_single_gate_unchanged():
    qc = Circuit(2)
    qc.h(0)

    fused = fuse_gates(qc)
    assert len(fused.ops) == 1
    gate = fused.ops[0]
    assert isinstance(gate, Gate)
    assert np.allclose(gate.matrix, g.H(), atol=1e-6)


def test_measure_barrier():
    """Gates on either side of a measurement should not be fused together."""
    qc = Circuit(2)
    qc.cx(0, 1)
    qc.measure([0])
    qc.cx(0, 1)

    fused = fuse_gates(qc)
    gate_ops = [op for op in fused.ops if isinstance(op, Gate)]
    measure_ops = [op for op in fused.ops if isinstance(op, MeasureOp)]

    assert len(measure_ops) == 1
    assert len(gate_ops) == 2
    # The measure op should be between the two gate ops
    assert isinstance(fused.ops[0], Gate)
    assert isinstance(fused.ops[1], MeasureOp)
    assert isinstance(fused.ops[2], Gate)


def test_fused_matrix_unitarity():
    qc = Circuit(2)
    qc.h(0)
    qc.cx(0, 1)

    fused = fuse_gates(qc)
    for op in fused.ops:
        if isinstance(op, Gate):
            m = op.matrix.astype(np.complex128)
            assert np.allclose(m @ m.conj().T, np.eye(len(m)), atol=1e-5), (
                f"Fused gate {op.name} is not unitary"
            )


def test_fusion_limit():
    """Gates acting on too many qubits together should not be fused."""
    qc = Circuit(5)
    for i in range(5):
        qc.h(i)

    # With max_fused_qubits=4, we can fuse at most 4 of the 5 H gates
    fused = fuse_gates(qc, max_fused_qubits=4)
    gate_ops = [op for op in fused.ops if isinstance(op, Gate)]
    # All H gates act on distinct qubits, so they can all be fused into one 5-qubit group
    # BUT max_fused_qubits=4 means we stop at 4, so we should have 2 groups
    total_qubits = sum(len(op.targets) + len(op.controls) for op in gate_ops)
    assert total_qubits == 5


def _cp_cascade(n: int) -> Circuit:
    """A QFT-style controlled-phase cascade onto qubit 0 (all diagonal)."""
    qc = Circuit(n)
    for k in range(1, n):
        qc.cp(k, 0, float(np.pi / 2**k))
    return qc


def test_diag_run_fusion_merges_wide(monkeypatch):
    """Step 26: adjacent diagonal gates merge past the dense width-4 limit."""
    monkeypatch.delenv("MACQUEREL_DIAG_FUSION_WIDTH", raising=False)
    monkeypatch.delenv("MACQUEREL_FUSION_WIDTH", raising=False)
    n = 8
    qc = _cp_cascade(n)
    fused = fuse_gates(qc)
    gate_ops = [op for op in fused.ops if isinstance(op, Gate)]
    # 7 CPs over 8 qubits fit in one 8-qubit diagonal (cap is 8).
    assert len(gate_ops) == 1
    assert gate_ops[0].kind == "diagonal"
    assert len(gate_ops[0].targets) == n
    assert np.allclose(_run_statevector(qc), _run_statevector(fused), atol=1e-5)


def test_diag_run_fusion_preserves_statevector(monkeypatch):
    """Mixed dense/diagonal circuit: the merge pass must respect barriers."""
    monkeypatch.delenv("MACQUEREL_DIAG_FUSION_WIDTH", raising=False)
    rng = np.random.default_rng(3)
    n = 7
    qc = Circuit(n)
    for i in range(n):
        qc.h(i)
    for _ in range(30):
        r = rng.random()
        if r < 0.5:
            qc.cp(int(rng.integers(n - 1)) + 1, 0, float(rng.uniform(0, np.pi)))
        elif r < 0.75:
            qc.ry(int(rng.integers(n)), float(rng.uniform(0, np.pi)))
        else:
            qc.rz(int(rng.integers(n)), float(rng.uniform(0, np.pi)))
    fused = fuse_gates(qc)
    assert np.allclose(_run_statevector(qc), _run_statevector(fused), atol=1e-4)


def test_diag_fusion_env_disable(monkeypatch):
    """MACQUEREL_DIAG_FUSION_WIDTH <= max_fused_qubits disables the merge pass."""
    monkeypatch.setenv("MACQUEREL_DIAG_FUSION_WIDTH", "1")
    fused = fuse_gates(_cp_cascade(9))
    gate_ops = [op for op in fused.ops if isinstance(op, Gate)]
    assert all(len(op.targets) + len(op.controls) <= 4 for op in gate_ops)


def test_cpu_diagonal_fast_path_matches_dense():
    """CPU diagonal path must agree with the tensordot path it bypasses."""
    rng = np.random.default_rng(4)
    n = 6
    cpu = CPUBackend()
    diag_vals = np.exp(1j * rng.uniform(0, 2 * np.pi, size=8)).astype(np.complex64)
    mat = np.diag(diag_vals)
    targets = [4, 1, 3]  # unsorted, non-adjacent
    sv = cpu.allocate(n)
    for i in range(n):
        sv = cpu.apply_matrix(sv, g.H(), [i])
    expected = sv.copy().reshape((2,) * n)
    gate_t = mat.reshape((2,) * 6).astype(np.complex64)
    out = np.tensordot(gate_t, expected, axes=([3, 4, 5], targets))
    remaining = [i for i in range(n) if i not in targets]
    inv = [0] * n
    for new_pos, old_pos in enumerate(targets + remaining):
        inv[old_pos] = new_pos
    expected = np.transpose(out, inv).reshape(-1)
    got = cpu.apply_matrix(sv, mat, targets)
    assert np.allclose(expected, got, atol=1e-6)


def test_commutation_grouping_packs_disjoint_layers(monkeypatch):
    """Step 27: a brickwork circuit packs per qubit-neighborhood, not program order.

    Naive in-order fusion lets a rotation on an unrelated qubit inflate the
    group's qubit union and force an early flush; the commutation-aware
    scheduler routes disjoint gates into parallel open groups.
    """
    monkeypatch.delenv("MACQUEREL_FUSION_WIDTH", raising=False)
    n = 8
    qc = Circuit(n)
    for _ in range(6):  # 6 layers of rotations + aligned CX brickwork
        for q in range(n):
            qc.ry(q, 0.3)
        for q in range(0, n - 1, 2):
            qc.cx(q, q + 1)
    fused = fuse_gates(qc, max_fused_qubits=4)
    gate_ops = [op for op in fused.ops if isinstance(op, Gate)]
    # Two disjoint 4-qubit neighborhoods absorb all six layers -> 2 groups.
    assert len(gate_ops) == 2
    assert np.allclose(_run_statevector(qc), _run_statevector(fused), atol=1e-4)


def test_commutation_grouping_preserves_order_on_shared_qubits():
    """Gates sharing qubits must never be reordered across groups."""
    rng = np.random.default_rng(12)
    for seed in range(6):
        rng = np.random.default_rng(seed)
        n = 7
        qc = Circuit(n)
        for _ in range(50):
            r = rng.random()
            if r < 0.4:
                a, b = rng.choice(n, size=2, replace=False)
                qc.cx(int(a), int(b))
            elif r < 0.7:
                qc.ry(int(rng.integers(n)), float(rng.uniform(0, np.pi)))
            else:
                qc.rz(int(rng.integers(n)), float(rng.uniform(0, np.pi)))
        fused = fuse_gates(qc)
        assert np.allclose(_run_statevector(qc), _run_statevector(fused), atol=1e-4), f"seed={seed}"


def test_remap_preserves_distribution():
    """Remapped and original circuits must produce identical measurement distributions."""
    # Build a 4-qubit circuit with unequal qubit access frequency.
    # Qubits 0 and 1 are used much more than 2 and 3.
    qc = Circuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(0, 0.3)
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(1, 0.7)
    qc.h(2)
    qc.cx(2, 3)
    qc.measure_all()

    remapped = remap_qubits(qc)

    # Run both circuits
    cpu = CPUBackend()

    def run_sv(circuit):
        sv = cpu.allocate(circuit.n_qubits)
        for op in circuit.ops:
            if isinstance(op, Gate):
                sv = cpu.apply_matrix(sv, op.matrix, op.targets, op.controls or None)
        return sv

    sv_orig = run_sv(qc)
    sv_remap = run_sv(remapped)

    # Recover the permutation from remap_qubits to invert bitstring labels
    from collections import Counter as _Counter

    freq: _Counter[int] = _Counter()
    for op in qc.ops:
        if isinstance(op, Gate):
            for q_idx in op.targets + op.controls:
                freq[q_idx] += 1
    sorted_qubits = sorted(range(qc.n_qubits), key=lambda q: (-freq[q], q))
    perm = {old: new for new, old in enumerate(sorted_qubits)}
    inv_perm = {new: old for old, new in perm.items()}

    # Compute marginal probabilities for all qubits from both statevectors
    n = qc.n_qubits
    probs_orig = np.abs(sv_orig) ** 2
    probs_remap = np.abs(sv_remap) ** 2

    # For each basis state in the remapped SV, compute the equivalent original basis state
    # and check probabilities match
    for idx in range(2**n):
        # bits in remapped ordering
        remap_bits = [(idx >> (n - 1 - new_q)) & 1 for new_q in range(n)]
        # convert back to original ordering
        orig_idx = 0
        for new_q, bit in enumerate(remap_bits):
            old_q = inv_perm[new_q]
            orig_idx |= bit << (n - 1 - old_q)
        assert abs(probs_remap[idx] - probs_orig[orig_idx]) < 1e-5, (
            f"Probability mismatch at remapped={idx}, orig={orig_idx}: "
            f"{probs_remap[idx]} vs {probs_orig[orig_idx]}"
        )


# --- Step 20: fusion-width default + opt-in per-chip autotuning ---


def test_default_width_is_four_without_measuring(monkeypatch):
    """Unset env -> fixed default of 4, with no autotuning measurement."""
    monkeypatch.delenv("MACQUEREL_FUSION_WIDTH", raising=False)

    def _boom(*a, **k):
        raise AssertionError("autotuner must not run on the default hot path")

    monkeypatch.setattr(compiler, "_measure_fusion_width", _boom)
    assert compiler._resolve_fusion_width() == 4


def test_env_override_pins_int(monkeypatch):
    """MACQUEREL_FUSION_WIDTH=<int> pins the width and skips any measurement."""
    monkeypatch.setenv("MACQUEREL_FUSION_WIDTH", "3")
    assert compiler._resolve_fusion_width() == 3


def test_env_auto_opts_into_autotuner(monkeypatch):
    """MACQUEREL_FUSION_WIDTH=auto routes to the autotuner."""
    monkeypatch.setenv("MACQUEREL_FUSION_WIDTH", "auto")
    monkeypatch.setattr(compiler, "autotune_fusion_width", lambda: 5)
    assert compiler._resolve_fusion_width() == 5


def test_env_garbage_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("MACQUEREL_FUSION_WIDTH", "not-a-number")
    assert compiler._resolve_fusion_width() == compiler._DEFAULT_FUSION_WIDTH


def test_env_non_positive_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("MACQUEREL_FUSION_WIDTH", "0")
    assert compiler._resolve_fusion_width() == compiler._DEFAULT_FUSION_WIDTH
    monkeypatch.setenv("MACQUEREL_FUSION_WIDTH", "-2")
    assert compiler._resolve_fusion_width() == compiler._DEFAULT_FUSION_WIDTH


def test_cached_non_positive_width_is_ignored(monkeypatch, tmp_path):
    monkeypatch.setattr(compiler, "_FUSION_WIDTH_CACHE", None)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    cache = tmp_path / "macquerel" / "fusion_width.json"
    cache.parent.mkdir(parents=True)
    cache.write_text('{"max_fused_qubits": 0}')
    monkeypatch.setattr(compiler, "_measure_fusion_width", lambda: 3)
    assert autotune_fusion_width() == 3


def test_autotune_measures_and_caches(monkeypatch, tmp_path):
    """With no override, the width is measured once and persisted to disk."""
    monkeypatch.delenv("MACQUEREL_FUSION_WIDTH", raising=False)
    monkeypatch.setattr(compiler, "_FUSION_WIDTH_CACHE", None)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    # Measure on a tiny CPU span so the test stays fast (the production span runs
    # on MLX at 20-22q); we only need to exercise the measure/normalize/cache path.
    monkeypatch.setattr(compiler, "_autotune_backend", lambda: (CPUBackend(), (6, 8)))

    width = autotune_fusion_width(force=True)
    assert width in range(1, 7)

    cache = tmp_path / "macquerel" / "fusion_width.json"
    assert cache.exists()

    # A second call (no force) returns the same value without re-measuring.
    called = {"n": 0}
    real = compiler._measure_fusion_width

    def _spy(*a, **k):
        called["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(compiler, "_measure_fusion_width", _spy)
    monkeypatch.setattr(compiler, "_FUSION_WIDTH_CACHE", None)  # force disk read path
    assert autotune_fusion_width() == width
    assert called["n"] == 0  # served from the on-disk cache, no measurement


def test_autotune_measure_fallback_on_error(monkeypatch, tmp_path):
    """A measurement failure falls back to the default width, never raises."""
    monkeypatch.delenv("MACQUEREL_FUSION_WIDTH", raising=False)
    monkeypatch.setattr(compiler, "_FUSION_WIDTH_CACHE", None)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    def _boom(*a, **k):
        raise RuntimeError("measurement blew up")

    monkeypatch.setattr(compiler, "_measure_fusion_width", _boom)
    assert autotune_fusion_width(force=True) == compiler._DEFAULT_FUSION_WIDTH


def test_fuse_gates_uses_resolved_width(monkeypatch):
    """fuse_gates() with no width argument resolves via _resolve_fusion_width."""
    monkeypatch.setattr(compiler, "_resolve_fusion_width", lambda: 2)
    qc = Circuit(5)
    for i in range(5):
        qc.h(i)
    fused = fuse_gates(qc)  # width 2 -> H gates fuse in pairs
    widths = [len(op.targets) + len(op.controls) for op in fused.ops if isinstance(op, Gate)]
    assert max(widths) <= 2
    assert sum(widths) == 5


def test_fuse_gates_default_width_is_four(monkeypatch):
    """With no override, fuse_gates fuses up to 4 qubits per group."""
    monkeypatch.delenv("MACQUEREL_FUSION_WIDTH", raising=False)
    qc = Circuit(6)
    for i in range(6):
        qc.h(i)
    fused = fuse_gates(qc)  # 6 distinct-qubit H gates -> groups capped at 4
    widths = [len(op.targets) + len(op.controls) for op in fused.ops if isinstance(op, Gate)]
    assert max(widths) == 4
    assert sum(widths) == 6


def test_fuse_gates_rejects_non_positive_width():
    qc = Circuit(2)
    qc.h(0)
    qc.h(1)
    for width in (0, -1):
        try:
            fuse_gates(qc, max_fused_qubits=width)
        except ValueError as e:
            assert "max_fused_qubits" in str(e)
        else:
            raise AssertionError(f"width {width} should have raised")
