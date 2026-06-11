"""Differential tests: the Metal backend must agree with the CPU backend to 1e-5.

Skipped wholesale unless pyobjc-framework-Metal is installed and a Metal device
is present (i.e. Apple Silicon), like the MLX tests. The Metal backend reaches
the >31-qubit regime via 64-bit GPU indexing; here we pin its gate semantics
against the NumPy reference on small circuits where a full readback is cheap.
"""

import numpy as np
import pytest

pytest.importorskip("Metal")

from macquerel.backends.metal_backend import _METAL_AVAILABLE, MetalBackend

if not _METAL_AVAILABLE:  # importable but no GPU device (e.g. headless CI)
    pytest.skip("no Metal device available", allow_module_level=True)

import macquerel.gates as g
from macquerel.backends.cpu import CPUBackend
from macquerel.circuit import Gate


@pytest.fixture
def cpu():
    return CPUBackend()


@pytest.fixture
def metal():
    return MetalBackend()


def _apply_gates(backend, sv, gate_list):
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
        # diagonal gates (diagonal fast path)
        [(g.H(), [0]), (g.Rz(0.5), [0]), (g.S(), [0]), (g.CZ(), [0, 1])],
        # permutation gates (SWAP / X)
        [(g.H(), [0]), (g.X(), [2]), (g.SWAP(), [0, 2])],
        # explicit controlled gate (control mask path)
        [(g.H(), [0]), (g.X(), [1]), (g.X(), [2], [0, 1])],
        # mixed multi-qubit, non-adjacent targets
        [(g.H(), [0]), (g.Rx(0.3), [1]), (g.SWAP(), [0, 3]), (g.Rz(0.7), [2]), (g.CNOT(), [1, 2])],
    ]


@pytest.mark.parametrize("gate_seq", _circuit_gates())
def test_differential(cpu, metal, gate_seq):
    n_qubits = (
        max(q for item in gate_seq for q in (item[1] + (item[2] if len(item) > 2 else []))) + 1
    )
    sv_cpu = _apply_gates(cpu, cpu.allocate(n_qubits), gate_seq)
    sv_metal = metal.to_numpy(_apply_gates(metal, metal.allocate(n_qubits), gate_seq))
    assert np.allclose(sv_cpu, sv_metal, atol=1e-5), (
        f"max diff: {np.max(np.abs(sv_cpu - sv_metal))}"
    )


_G1 = [g.H, g.X, g.Z, g.S, g.T, lambda: g.Rz(0.7), lambda: g.Rx(0.4), lambda: g.Ry(1.1)]
_G2 = [g.CNOT, g.CZ, g.SWAP]


def _random_ops(n, depth, rng):
    ops = []
    for _ in range(depth):
        if n >= 2 and rng.random() < 0.45:
            q = rng.choice(n, size=2, replace=False).tolist()
            ops.append((_G2[rng.integers(len(_G2))](), q))
        else:
            ops.append((_G1[rng.integers(len(_G1))](), [int(rng.integers(n))]))
    return ops


@pytest.mark.parametrize("seed", range(12))
def test_fuzz_differential(cpu, metal, seed):
    """Fuzzed random circuits must match the CPU reference (catches index/stride bugs)."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 8))
    ops = _random_ops(n, 40, rng)
    sv_cpu = cpu.to_numpy(_apply_gates(cpu, cpu.allocate(n), ops))
    sv_metal = metal.to_numpy(_apply_gates(metal, metal.allocate(n), ops))
    assert np.allclose(sv_cpu, sv_metal, atol=1e-5), (
        f"n={n} seed={seed} max diff: {np.max(np.abs(sv_cpu - sv_metal))}"
    )


def test_fused_random_circuit_differential(cpu, metal):
    """Random circuits through the fusion compiler must agree CPU vs Metal.

    Fusion produces composed monomial matrices that are *not* involutions and
    carry non-trivial row phases — the exact class the Step 25 monomial kernel
    must get right (gather direction + phase application).
    """
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
        sm = metal.allocate(n)
        for gate in fc.ops:
            assert isinstance(gate, Gate)
            ctrls = getattr(gate, "controls", None) or None
            sc = cpu.apply_matrix(sc, gate.matrix, gate.targets, ctrls)
            sm = metal.apply_matrix(sm, gate.matrix, gate.targets, ctrls)
        ref = np.asarray(sc).ravel()
        got = metal.to_numpy(sm).ravel()
        assert np.allclose(ref, got, atol=1e-4), f"seed={seed} diff={np.max(np.abs(ref - got))}"


def test_deep_circuit_crosses_flush_cap(cpu, metal):
    """A gate run longer than _FLUSH_EVERY spans several command buffers (Step 22).

    Exercises the batched-encoding path: encode > _FLUSH_EVERY dispatches so at
    least one mid-run flush happens, and verify the final state still matches
    the CPU oracle (ordering across command-buffer boundaries is preserved).
    """
    from macquerel.backends.metal_backend import _FLUSH_EVERY

    rng = np.random.default_rng(7)
    n = 5
    ops = _random_ops(n, _FLUSH_EVERY + 40, rng)
    sv_cpu = cpu.to_numpy(_apply_gates(cpu, cpu.allocate(n), ops))
    sv_metal = metal.to_numpy(_apply_gates(metal, metal.allocate(n), ops))
    assert np.allclose(sv_cpu, sv_metal, atol=1e-4), (
        f"max diff: {np.max(np.abs(sv_cpu - sv_metal))}"
    )


def test_const_cache_reset_mid_encoding(cpu, metal, monkeypatch):
    """Const buffers must survive a cache reset while dispatches are in flight.

    With deferred submission (Step 22), an encoded dispatch references its
    matrix/index MTLBuffers until the command buffer completes. Shrink the cache
    cap so eviction resets happen repeatedly mid-encoding; correctness then
    rests on the command buffer's own retention of referenced resources.
    """
    import macquerel.backends.metal_backend as mb

    monkeypatch.setattr(mb, "_CONST_CACHE_MAX", 2)
    rng = np.random.default_rng(11)
    n = 4
    # Distinct rotation angles -> distinct matrices -> constant cache churn.
    ops = [(g.Rz(float(rng.uniform(0, 6.28))), [int(rng.integers(n))]) for _ in range(50)]
    ops += [(g.Rx(float(rng.uniform(0, 6.28))), [int(rng.integers(n))]) for _ in range(50)]
    sv_cpu = cpu.to_numpy(_apply_gates(cpu, cpu.allocate(n), ops))
    sv_metal = metal.to_numpy(_apply_gates(metal, metal.allocate(n), ops))
    assert np.allclose(sv_cpu, sv_metal, atol=1e-4), (
        f"max diff: {np.max(np.abs(sv_cpu - sv_metal))}"
    )


def test_gates_after_measure_see_collapsed_state(metal):
    """Host-side collapse writes must be ordered against batched GPU dispatches."""
    backend = MetalBackend(seed=3)
    sv = backend.allocate(2)
    sv = backend.apply_matrix(sv, g.H(), [0])
    (outcome,) = backend.measure(sv, [0], collapse=True)
    # After collapse, qubit 0 is a basis state; X flips it deterministically.
    sv = backend.apply_matrix(sv, g.X(), [0])
    final = backend.to_numpy(sv)
    expected_idx = (1 - outcome) << 1
    assert np.isclose(abs(final[expected_idx]), 1.0, atol=1e-5)


def test_bell_state(metal):
    sv = metal.allocate(2)
    sv = metal.apply_matrix(sv, g.H(), [0])
    sv = metal.apply_matrix(sv, g.CNOT(), [0, 1])
    sv = metal.to_numpy(sv)
    expected = np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], dtype=np.complex64)
    assert np.allclose(sv, expected, atol=1e-5)


def test_ghz(metal):
    sv = metal.allocate(3)
    for m, t in [(g.H(), [0]), (g.CNOT(), [0, 1]), (g.CNOT(), [0, 2])]:
        sv = metal.apply_matrix(sv, m, t)
    sv = metal.to_numpy(sv)
    inv_sqrt2 = 1 / np.sqrt(2)
    assert abs(sv[0] - inv_sqrt2) < 1e-5
    assert abs(sv[7] - inv_sqrt2) < 1e-5
    assert np.allclose(sv[1:7], 0, atol=1e-5)


def test_allocate_is_basis_state(metal):
    for n in (1, 2, 5):
        sv = metal.to_numpy(metal.allocate(n))
        assert sv[0] == 1.0
        assert np.allclose(sv[1:], 0)


def test_identity_leaves_state_unchanged(cpu, metal):
    ops = [(g.H(), [0]), (g.Ry(0.6), [1]), (g.CNOT(), [0, 1])]
    sv = _apply_gates(metal, metal.allocate(2), ops)
    before = metal.to_numpy(sv)
    sv = metal.apply_matrix(sv, g.I(), [0])
    sv = metal.apply_matrix(sv, g.I(), [1])
    assert np.allclose(before, metal.to_numpy(sv), atol=1e-6)


def test_abs2sum_matches_cpu(cpu, metal):
    ops = [(g.H(), [0]), (g.CNOT(), [0, 1]), (g.Ry(0.9), [2])]
    sv_c = _apply_gates(cpu, cpu.allocate(3), ops)
    sv_m = _apply_gates(metal, metal.allocate(3), ops)
    assert np.allclose(cpu.abs2sum(sv_c, [0, 2]), metal.abs2sum(sv_m, [0, 2]), atol=1e-6)


def test_expectation_pauli_matches_cpu(cpu, metal):
    ops = [(g.H(), [0]), (g.CNOT(), [0, 1]), (g.Ry(0.9), [2])]
    sv_c = _apply_gates(cpu, cpu.allocate(3), ops)
    sv_m = _apply_gates(metal, metal.allocate(3), ops)
    ps = [(1.0, [("Z", 0)]), (0.5, [("X", 1), ("Z", 2)])]
    assert np.allclose(
        cpu.expectation_pauli(sv_c, ps), metal.expectation_pauli(sv_m, ps), atol=1e-5
    )


def test_sample_bell_only_correlated_outcomes():
    metal = MetalBackend(seed=1)
    sv = metal.allocate(2)
    sv = metal.apply_matrix(sv, g.H(), [0])
    sv = metal.apply_matrix(sv, g.CNOT(), [0, 1])
    counts = metal.sample(sv, [0, 1], 2000)
    assert set(counts) <= {"00", "11"}
    assert counts["00"] > 0 and counts["11"] > 0


def test_measure_collapses_to_basis_state():
    metal = MetalBackend(seed=2)
    sv = metal.allocate(2)
    sv = metal.apply_matrix(sv, g.H(), [0])
    sv = metal.apply_matrix(sv, g.CNOT(), [0, 1])
    out = metal.measure(sv, [0], collapse=True)
    post = metal.to_numpy(sv)
    assert np.isclose(np.linalg.norm(post), 1.0, atol=1e-5)
    nz = np.flatnonzero(np.abs(post) > 1e-6).tolist()
    assert nz == ([0] if out[0] == 0 else [3])  # |00> or |11>


def test_buffer_pool_defers_inflight_buffers(metal):
    """Step 34: a state buffer with encoded-but-unsubmitted gates must not be
    recycled until the open command buffer is flushed; afterwards it is."""
    sv = metal.allocate(8)
    sv = metal.apply_matrix(sv, g.H(), [0])  # leaves the command buffer open
    buf = sv.buf
    del sv
    assert any(b is buf for _, b in metal._deferred_bufs)
    fresh = metal.allocate(8)  # same size: must NOT get the in-flight buffer
    assert fresh.buf is not buf
    out = metal.to_numpy(fresh)
    assert out[0] == 1.0 and np.isclose(np.abs(out).sum(), 1.0)
    # allocate() flushed the open command buffer, so `buf` is poolable now.
    again = metal.allocate(8)
    assert again.buf is buf


def test_shared_pipelines_across_instances():
    """Step 34: Metal pipelines are process-wide; instances share them."""
    a, b = MetalBackend(), MetalBackend()
    assert a._pipeline("dense", 1) is b._pipeline("dense", 1)
