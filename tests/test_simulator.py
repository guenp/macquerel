from collections import Counter

import numpy as np

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
    """CPU <=16q, MLX 17-21q, Metal 22q+ (measured tiers; MLX caps at 30q)."""
    import macquerel.simulator as sim

    monkeypatch.setattr(sim, "_MLX_AVAILABLE", True)
    monkeypatch.setattr(sim, "_METAL_AVAILABLE", True)
    assert sim._select_backend(16) == "cpu"
    assert sim._select_backend(17) == "mlx"
    assert sim._select_backend(21) == "mlx"
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
