"""Differential tests: every CPU backend circuit must agree with MLX backend to 1e-5."""

import numpy as np
import pytest

mlx = pytest.importorskip("mlx.core")

import macquerel.gates as g
from macquerel.backends.cpu import CPUBackend
from macquerel.backends.mlx_backend import MLXBackend


@pytest.fixture
def cpu():
    return CPUBackend()


@pytest.fixture
def mlx_backend():
    return MLXBackend()


def _apply_gates(backend, sv, gate_list):
    """Apply a list of (matrix, targets[, controls]) tuples."""
    for item in gate_list:
        matrix, targets = item[0], item[1]
        controls = item[2] if len(item) > 2 else None
        sv = backend.apply_matrix(sv, matrix, targets, controls)
    return sv


def _circuit_gates():
    return [
        # Bell
        [(g.H(), [0]), (g.CNOT(), [0, 1])],
        # GHZ
        [(g.H(), [0]), (g.CNOT(), [0, 1]), (g.CNOT(), [0, 2])],
        # diagonal gates
        [(g.H(), [0]), (g.Rz(0.5), [0]), (g.S(), [0]), (g.CZ(), [0, 1])],
        # mixed
        [(g.H(), [0]), (g.Rx(0.3), [1]), (g.SWAP(), [0, 1]), (g.Rz(0.7), [0])],
    ]


@pytest.mark.parametrize("gate_seq", _circuit_gates())
def test_differential(cpu, mlx_backend, gate_seq):
    n_qubits = max(q for item in gate_seq for q in item[1]) + 1
    sv_cpu = _apply_gates(cpu, cpu.allocate(n_qubits), gate_seq)
    sv_mlx = _apply_gates(mlx_backend, mlx_backend.allocate(n_qubits), gate_seq)
    sv_mlx = mlx_backend.to_numpy(sv_mlx)

    assert np.allclose(sv_cpu, sv_mlx, atol=1e-5), f"max diff: {np.max(np.abs(sv_cpu - sv_mlx))}"


def test_bell_state(mlx_backend):
    sv = mlx_backend.allocate(2)
    sv = mlx_backend.apply_matrix(sv, g.H(), [0])
    sv = mlx_backend.apply_matrix(sv, g.CNOT(), [0, 1])
    sv = mlx_backend.to_numpy(sv)
    expected = np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], dtype=np.complex64)
    assert np.allclose(sv, expected, atol=1e-5)


def test_ghz(mlx_backend):
    sv = mlx_backend.allocate(3)
    sv = mlx_backend.apply_matrix(sv, g.H(), [0])
    sv = mlx_backend.apply_matrix(sv, g.CNOT(), [0, 1])
    sv = mlx_backend.apply_matrix(sv, g.CNOT(), [0, 2])
    sv = mlx_backend.to_numpy(sv)
    inv_sqrt2 = 1 / np.sqrt(2)
    assert abs(sv[0] - inv_sqrt2) < 1e-5
    assert abs(sv[7] - inv_sqrt2) < 1e-5
    assert np.allclose(sv[1:7], 0, atol=1e-5)


def _monomial(perm, phases):
    """Build a (2^k x 2^k) monomial matrix: row j has its single nonzero
    (value phases[j]) in column perm[j]."""
    d = len(perm)
    m = np.zeros((d, d), dtype=np.complex64)
    for j in range(d):
        m[j, perm[j]] = phases[j]
    return m


def test_permutation_non_involution(cpu, mlx_backend):
    """A composed permutation (CX then CX) is NOT self-inverse. The gather path
    must use the forward source map, not its inverse — regression for a bug where
    only involutions (X/SWAP/CNOT) were handled correctly."""
    cx01 = np.kron(g.CNOT(), np.eye(2, dtype=np.complex64))  # CX on (0,1) of 3q
    cx12 = np.kron(np.eye(2, dtype=np.complex64), g.CNOT())  # CX on (1,2) of 3q
    fused = (cx12 @ cx01).astype(np.complex64)
    assert g.classify(fused) == "permutation"
    assert not np.allclose(fused @ fused, np.eye(8), atol=1e-6)  # not an involution

    for targets in ([0, 1, 2], [2, 3, 4]):
        n = max(targets) + 1
        rng = np.random.default_rng(0)
        psi = (rng.standard_normal(2**n) + 1j * rng.standard_normal(2**n)).astype(np.complex64)
        psi /= np.linalg.norm(psi)
        sc = cpu.allocate(n)
        sc[:] = psi
        ref = np.asarray(cpu.apply_matrix(sc, fused, targets, None)).ravel()
        sm = mlx_backend.allocate(n)
        sm.data = mlx.array(psi)
        got = mlx_backend.to_numpy(mlx_backend.apply_matrix(sm, fused, targets, None)).ravel()
        diff = np.max(np.abs(ref - got))
        assert np.allclose(ref, got, atol=1e-5), f"targets={targets} diff={diff}"


def test_permutation_with_phase(cpu, mlx_backend):
    """A phased monomial matrix (e.g. fused CX·Rz) classifies as 'permutation'
    but its nonzero entries carry phase — the fast path must apply that phase,
    not just gather. Regression for dropped phases on fused diagonal+perm gates."""
    perm = [0, 1, 3, 2]  # CX-like swap of last two basis states
    phases = np.exp(1j * np.array([0.0, 0.3, -0.7, 1.1])).astype(np.complex64)
    mono = _monomial(perm, phases)
    assert g.classify(mono) == "permutation"

    n = 4
    targets = [1, 2]
    rng = np.random.default_rng(1)
    psi = (rng.standard_normal(2**n) + 1j * rng.standard_normal(2**n)).astype(np.complex64)
    psi /= np.linalg.norm(psi)
    sc = cpu.allocate(n)
    sc[:] = psi
    ref = np.asarray(cpu.apply_matrix(sc, mono, targets, None)).ravel()
    sm = mlx_backend.allocate(n)
    sm.data = mlx.array(psi)
    got = mlx_backend.to_numpy(mlx_backend.apply_matrix(sm, mono, targets, None)).ravel()
    assert np.allclose(ref, got, atol=1e-5), f"diff={np.max(np.abs(ref - got))}"


def test_fused_random_circuit_differential(cpu, mlx_backend):
    """End-to-end: random circuits run through the fusion compiler must agree
    CPU vs MLX. Fusion produces composed/phased permutation gates that the
    per-gate tests above target in isolation."""
    from macquerel import Circuit
    from macquerel.compiler import fuse_gates

    for seed in range(5):
        rng = np.random.default_rng(seed)
        n = 6
        c = Circuit(n)
        for _ in range(20):
            for q in range(n):
                gate = rng.choice(["rx", "ry", "rz"])
                getattr(c, gate)(q, float(rng.uniform(0, 2 * np.pi)))
            for q in range(0, n - 1, 2):
                c.cx(q, q + 1)
        fc = fuse_gates(c)
        sc = cpu.allocate(n)
        sm = mlx_backend.allocate(n)
        for gate in fc.ops:
            ctrls = getattr(gate, "controls", None) or None
            sc = cpu.apply_matrix(sc, gate.matrix, gate.targets, ctrls)
            sm = mlx_backend.apply_matrix(sm, gate.matrix, gate.targets, ctrls)
        ref = np.asarray(sc).ravel()
        got = mlx_backend.to_numpy(sm).ravel()
        assert np.allclose(ref, got, atol=1e-5), f"seed={seed} diff={np.max(np.abs(ref - got))}"


# --- Step 23: axis-order tracking (deferred dense-gate transpose) ---


@pytest.mark.parametrize("seed", range(8))
def test_mixed_kind_fuzz_differential(cpu, mlx_backend, seed):
    """Interleaved dense/diagonal/permutation/controlled gates vs the CPU oracle.

    Step 23 defers the dense-gate axis permutation, so every other gate kind
    (and the controlled path) must correctly translate its targets through a
    non-trivial axis map. This fuzz drives all four paths against each other.
    """
    rng = np.random.default_rng(seed)
    n = int(rng.integers(4, 8))
    sc, sm = cpu.allocate(n), mlx_backend.allocate(n)
    for _ in range(60):
        r = rng.random()
        if r < 0.25:  # dense 1q (scrambles the axis order)
            mat, tgts, ctl = g.Ry(float(rng.uniform(0, 6.28))), [int(rng.integers(n))], None
        elif r < 0.4:  # dense 2q (Haar-ish via composed rotations is overkill; H⊗Ry)
            q = rng.choice(n, size=2, replace=False)
            mat = np.kron(g.H(), g.Ry(float(rng.uniform(0, 6.28)))).astype(np.complex64)
            tgts, ctl = [int(q[0]), int(q[1])], None
        elif r < 0.6:  # diagonal
            mat, tgts, ctl = g.Rz(float(rng.uniform(0, 6.28))), [int(rng.integers(n))], None
        elif r < 0.8:  # permutation
            q = rng.choice(n, size=2, replace=False)
            mat, tgts, ctl = (g.CNOT(), [int(q[0]), int(q[1])], None)
        else:  # controlled dense
            q = rng.choice(n, size=2, replace=False)
            mat, tgts, ctl = g.Rx(float(rng.uniform(0, 6.28))), [int(q[0])], [int(q[1])]
        sc = cpu.apply_matrix(sc, mat, tgts, ctl)
        sm = mlx_backend.apply_matrix(sm, mat, tgts, ctl)
    ref, got = np.asarray(sc).ravel(), mlx_backend.to_numpy(sm).ravel()
    assert np.allclose(ref, got, atol=1e-4), f"n={n} seed={seed} diff={np.max(np.abs(ref - got))}"


def test_readback_order_after_dense_gate(cpu, mlx_backend):
    """A dense gate on a middle qubit leaves a non-trivial axis permutation;
    to_numpy must still return canonical basis order."""
    n = 4
    sc, sm = cpu.allocate(n), mlx_backend.allocate(n)
    ops = [(g.H(), [2]), (g.Ry(0.9), [1]), (g.H(), [0])]
    for mat, tgts in ops:
        sc = cpu.apply_matrix(sc, mat, tgts)
        sm = mlx_backend.apply_matrix(sm, mat, tgts)
    assert np.allclose(np.asarray(sc), mlx_backend.to_numpy(sm), atol=1e-5)


def test_partial_abs2sum_with_pending_permutation(cpu, mlx_backend):
    """abs2sum over a qubit subset must canonicalize the deferred layout first."""
    n = 5
    sc, sm = cpu.allocate(n), mlx_backend.allocate(n)
    rng = np.random.default_rng(5)
    for _ in range(12):
        q = int(rng.integers(n))
        mat = g.Ry(float(rng.uniform(0, 6.28)))
        sc = cpu.apply_matrix(sc, mat, [q])
        sm = mlx_backend.apply_matrix(sm, mat, [q])
        q2 = rng.choice(n, size=2, replace=False)
        sc = cpu.apply_matrix(sc, g.CNOT(), [int(q2[0]), int(q2[1])])
        sm = mlx_backend.apply_matrix(sm, g.CNOT(), [int(q2[0]), int(q2[1])])
    for subset in ([0], [1, 3], [4, 0, 2]):
        assert np.allclose(cpu.abs2sum(sc, subset), mlx_backend.abs2sum(sm, subset), atol=1e-5)


def test_measure_collapse_with_pending_permutation(mlx_backend):
    """measure() after dense gates (pending axis map) collapses correctly and
    later gates see the collapsed, canonical state."""
    backend = type(mlx_backend)(seed=9)
    sv = backend.allocate(3)
    sv = backend.apply_matrix(sv, g.H(), [1])  # dense -> perm (1,0,2)
    sv = backend.apply_matrix(sv, g.CNOT(), [1, 2])
    (outcome,) = backend.measure(sv, [1], collapse=True)
    sv = backend.apply_matrix(sv, g.X(), [1])
    probs = backend.abs2sum(sv, [1])
    assert np.isclose(probs[1 - outcome], 1.0, atol=1e-5)


def test_async_eval_cadence_preserves_state(cpu, mlx_backend, monkeypatch):
    """Step 24: periodic mx.async_eval mid-circuit must not change results.

    The cadence only fires at >= _ASYNC_EVAL_MIN_QUBITS in production; force it
    on for a small state so CI exercises the interleaved async_eval path.
    """
    import macquerel.backends.mlx_backend as mb

    monkeypatch.setattr(mb, "_ASYNC_EVAL_MIN_QUBITS", 1)
    monkeypatch.setattr(mb, "_ASYNC_EVAL_INTERVAL", 3)
    rng = np.random.default_rng(2)
    n = 5
    ops = []
    for _ in range(40):
        r = rng.random()
        if r < 0.3:
            ops.append((g.CNOT(), [int(q) for q in rng.choice(n, 2, replace=False)]))
        elif r < 0.6:
            ops.append((g.Rz(float(rng.uniform(0, 6.28))), [int(rng.integers(n))]))
        else:
            ops.append((g.Ry(float(rng.uniform(0, 6.28))), [int(rng.integers(n))]))
    sc, sm = cpu.allocate(n), mlx_backend.allocate(n)
    for mat, tgts in ops:
        sc = cpu.apply_matrix(sc, mat, tgts)
        sm = mlx_backend.apply_matrix(sm, mat, tgts)
    assert np.allclose(np.asarray(sc), mlx_backend.to_numpy(sm), atol=1e-5)


# --- Step 19: shot-batch autotuning / batched sampling ---


def _bell_state(backend):
    sv = backend.allocate(2)
    sv = backend.apply_matrix(sv, g.H(), [0])
    sv = backend.apply_matrix(sv, g.CNOT(), [0, 1])
    return sv


def test_mlx_sample_autotune_distribution(mlx_backend):
    """Unseeded 'auto' batching draws the right total and a ~50/50 Bell split."""
    sv = _bell_state(mlx_backend)
    shots = 8000
    counts = mlx_backend.sample(sv, [0, 1], shots, batch_shots="auto")
    assert sum(counts.values()) == shots
    assert set(counts) <= {"00", "11"}
    assert 0.4 < counts["00"] / shots < 0.6
    # The tuned batch size is memoized for this category count (2**2 = 4).
    assert mlx_backend._tuned_batch.get(4)


def test_mlx_sample_explicit_batch_chunks(mlx_backend):
    """An explicit batch smaller than shots forces multiple chunks; total holds."""
    sv = _bell_state(mlx_backend)
    shots = 5000
    counts = mlx_backend.sample(sv, [0, 1], shots, batch_shots=512)
    assert sum(counts.values()) == shots
    assert set(counts) <= {"00", "11"}


def test_mlx_sample_seed_reproducible():
    """Seeded MLX sampling is reproducible for a fixed batch_shots setting."""
    # Default "auto" path: single deterministic pass when seeded.
    a = MLXBackend(seed=5)
    b = MLXBackend(seed=5)
    assert a.sample(_bell_state(a), [0, 1], 1000) == b.sample(_bell_state(b), [0, 1], 1000)

    # Explicit chunked batch is also reproducible (deterministic per-chunk subkeys).
    c = MLXBackend(seed=5)
    d = MLXBackend(seed=5)
    assert c.sample(_bell_state(c), [0, 1], 1000, batch_shots=128) == d.sample(
        _bell_state(d), [0, 1], 1000, batch_shots=128
    )
