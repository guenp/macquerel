from collections import Counter

import numpy as np
import pytest

from macquerel.circuit import Circuit
from macquerel.simulator import Simulator


def test_basic_simulation():
    circuit = Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.rz(1, 0.3)
    circuit.measure_all()

    sim = Simulator(backend="cpu")
    result = sim.run(circuit, shots=1000)

    assert isinstance(result, Counter)
    assert all(isinstance(k, str) for k in result)
    assert all(isinstance(v, int) for v in result.values())
    assert sum(result.values()) == 1000


def test_statevector_bell():

    circuit = Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)

    sim = Simulator(backend="cpu")
    sv = sim.statevector(circuit)

    expected = np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], dtype=np.complex64)
    assert np.allclose(np.abs(sv), np.abs(expected), atol=1e-5)


def test_statevector_ignores_measurements():
    circuit = Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()

    sim = Simulator(backend="cpu")
    sv = sim.statevector(circuit)

    expected = np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], dtype=np.complex64)
    assert np.allclose(np.abs(sv), np.abs(expected), atol=1e-5)


def test_no_measurement_run():
    circuit = Circuit(1)
    circuit.h(0)

    sim = Simulator(backend="cpu")
    result = sim.run(circuit, shots=500)

    assert isinstance(result, Counter)
    assert sum(result.values()) == 500


def test_seed_reproducibility():
    qc = Circuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    s1 = Simulator(backend="cpu", seed=42).run(qc, shots=200)
    s2 = Simulator(backend="cpu", seed=42).run(qc, shots=200)
    assert s1 == s2, f"Seeded runs differ: {s1} vs {s2}"


def test_batch_shots_default_is_auto():
    assert Simulator(backend="cpu").batch_shots == "auto"


def test_batch_shots_explicit_int_runs_correctly():
    """An explicit batch_shots must not change the total shot count or validity."""
    qc = Circuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure_all()

    sim = Simulator(backend="cpu", batch_shots=128)
    result = sim.run(qc, shots=1000)
    assert sum(result.values()) == 1000
    assert set(result) <= {"000", "111"}


def test_batch_shots_seed_reproducible():
    """Seeded runs stay reproducible across batch_shots settings (single-pass path)."""
    qc = Circuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    a = Simulator(backend="cpu", seed=7, batch_shots="auto").run(qc, shots=300)
    b = Simulator(backend="cpu", seed=7, batch_shots=64).run(qc, shots=300)
    # CPU draws all shots in one np.random.choice regardless of batch_shots,
    # so a fixed seed yields identical counts either way.
    assert a == b


def test_auto_backend_default():
    circuit = Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()

    sim = Simulator()  # default is 'auto'
    result = sim.run(circuit, shots=500)
    assert sum(result.values()) == 500


def test_remap_env_statevector_matches(monkeypatch):
    """Step 28: MACQUEREL_REMAP=1 must return the identical statevector."""
    import numpy as np

    rng = np.random.default_rng(6)
    n = 6
    circuit = Circuit(n)
    for _ in range(25):
        r = rng.random()
        if r < 0.4:
            a, b = rng.choice(n, size=2, replace=False)
            circuit.cx(int(a), int(b))
        else:
            circuit.ry(int(rng.integers(n)), float(rng.uniform(0, 3.14)))
    monkeypatch.delenv("MACQUEREL_REMAP", raising=False)
    base = Simulator(backend="cpu").statevector(circuit)
    monkeypatch.setenv("MACQUEREL_REMAP", "1")
    remapped = Simulator(backend="cpu").statevector(circuit)
    assert np.allclose(base, remapped, atol=1e-5)


def test_remap_env_counts_match(monkeypatch):
    """Step 28: remapped sampling must report counts in caller qubit order."""
    circuit = Circuit(3)
    circuit.h(2)
    circuit.cx(2, 0)
    circuit.measure_all()
    monkeypatch.setenv("MACQUEREL_REMAP", "1")
    counts = Simulator(backend="cpu", seed=1).run(circuit, shots=400)
    # Bell pair on qubits (2, 0): only '000' and '101' may appear.
    assert set(counts) <= {"000", "101"}
    assert sum(counts.values()) == 400


def test_select_backend_tiers(monkeypatch):
    """CPU <=15q, Metal 16q+ (post-Step-34 tiers); MLX is the no-Metal fallback."""
    import macquerel.simulator as sim

    monkeypatch.setattr(sim, "_MLX_AVAILABLE", True)
    monkeypatch.setattr(sim, "_METAL_AVAILABLE", True)
    assert sim._select_backend(15) == "cpu"
    assert sim._select_backend(16) == "metal"
    assert sim._select_backend(17) == "metal"
    assert sim._select_backend(22) == "metal"
    assert sim._select_backend(30) == "metal"
    assert sim._select_backend(33) == "metal"


def test_select_backend_fallbacks(monkeypatch):
    import macquerel.simulator as sim

    # No Metal: MLX still serves its full 17-30q range. 31q+ has no working
    # backend (MLX would crash on the int32 ceiling), so we fall back to CPU
    # rather than route there.
    monkeypatch.setattr(sim, "_MLX_AVAILABLE", True)
    monkeypatch.setattr(sim, "_METAL_AVAILABLE", False)
    assert sim._select_backend(25) == "mlx"
    assert sim._select_backend(30) == "mlx"
    assert sim._select_backend(31) == "cpu"

    # No MLX: Metal serves everything above the CPU tier.
    monkeypatch.setattr(sim, "_MLX_AVAILABLE", False)
    monkeypatch.setattr(sim, "_METAL_AVAILABLE", True)
    assert sim._select_backend(20) == "metal"
    assert sim._select_backend(31) == "metal"

    # Neither: CPU everywhere.
    monkeypatch.setattr(sim, "_METAL_AVAILABLE", False)
    assert sim._select_backend(20) == "cpu"


def test_backend_tiers_env_pins_boundary(monkeypatch):
    """MACQUEREL_BACKEND_TIERS=<int> pins the CPU tier without measuring."""
    import macquerel.simulator as sim

    monkeypatch.setenv("MACQUEREL_BACKEND_TIERS", "10")
    monkeypatch.setattr(sim, "_METAL_AVAILABLE", True)
    monkeypatch.setattr(
        sim, "autotune_backend_tiers", lambda: (_ for _ in ()).throw(AssertionError)
    )
    assert sim._select_backend(10) == "cpu"
    assert sim._select_backend(11) == "metal"


def test_backend_tiers_default_never_measures(monkeypatch):
    """The zero-config path must not run the tier measurement."""
    import macquerel.simulator as sim

    monkeypatch.delenv("MACQUEREL_BACKEND_TIERS", raising=False)

    def boom(*a, **k):
        raise AssertionError("tier autotuner must not run on the default path")

    monkeypatch.setattr(sim, "autotune_backend_tiers", boom)
    monkeypatch.setattr(sim, "_measure_cpu_max", boom)
    assert sim._select_backend(8) == "cpu"


def test_backend_tiers_env_auto_routes_to_autotuner(monkeypatch):
    """MACQUEREL_BACKEND_TIERS=auto consults the (cached) autotuner."""
    import macquerel.simulator as sim

    monkeypatch.setenv("MACQUEREL_BACKEND_TIERS", "auto")
    monkeypatch.setattr(sim, "autotune_backend_tiers", lambda: 12)
    monkeypatch.setattr(sim, "_METAL_AVAILABLE", True)
    assert sim._select_backend(12) == "cpu"
    assert sim._select_backend(13) == "metal"


def test_backend_tiers_autotune_caches_to_disk(monkeypatch, tmp_path):
    """The measured boundary is persisted and re-read without re-measuring."""
    import macquerel.simulator as sim

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(sim, "_TIERS_CACHE", None)
    calls = []

    def fake_measure():
        calls.append(1)
        return 14

    monkeypatch.setattr(sim, "_measure_cpu_max", fake_measure)
    assert sim.autotune_backend_tiers(force=True) == 14
    assert calls == [1]
    # A fresh in-memory state must hit the disk cache, not re-measure.
    monkeypatch.setattr(sim, "_TIERS_CACHE", None)
    assert sim.autotune_backend_tiers() == 14
    assert calls == [1]


def test_backend_tiers_measure_failure_falls_back(monkeypatch, tmp_path):
    """A measurement failure falls back to the default and never raises."""
    import macquerel.simulator as sim

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(sim, "_TIERS_CACHE", None)

    def boom():
        raise RuntimeError("no GPU today")

    monkeypatch.setattr(sim, "_measure_cpu_max", boom)
    assert sim.autotune_backend_tiers(force=True) == sim._CPU_MAX_QUBITS


# --- measurement bit order (regression: argsort vs rank in sample()) ---

try:
    from macquerel.backends.metal_backend import _METAL_AVAILABLE as _METAL_OK
except ImportError:  # pragma: no cover
    _METAL_OK = False
try:
    import mlx.core  # noqa: F401

    _MLX_OK = True
except ImportError:
    _MLX_OK = False

_SAMPLE_BACKENDS = ["cpu"] + (["metal"] if _METAL_OK else []) + (["mlx"] if _MLX_OK else [])


@pytest.mark.parametrize("backend", _SAMPLE_BACKENDS)
@pytest.mark.parametrize("order", [[1, 2, 0], [2, 0, 1], [3, 1, 0, 2]])
def test_run_measure_unsorted_qubit_list_bit_order(backend, order):
    """Bit i of each sampled bitstring must be qubits[i], for *any* qubit order.

    sample() used to transpose the marginal with argsort(qubits) where the rank
    permutation (its inverse) is needed; the two first diverge on 3-cycles, so
    sorted lists, measure_all, and 2-qubit swaps all hid the bug.
    """
    n = max(order) + 1
    excited = {1, 2}  # deterministic |q0 q1 q2 ...> with X on qubits 1 and 2
    circuit = Circuit(n)
    for q in sorted(excited):
        circuit.x(q)
    circuit.measure(order)
    expected = "".join("1" if q in excited else "0" for q in order)
    counts = Simulator(backend=backend, seed=0).run(circuit, shots=20)
    assert counts == Counter({expected: 20})
