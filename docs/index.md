# macquerel

<p align="center">
  <img src="assets/logo.png" alt="macquerel logo" width="200" />
</p>

A quantum state-vector simulator targeting Apple Silicon's unified-memory architecture.

## Install

```sh
pip install macquerel          # CPU backend
pip install "macquerel[mlx]"   # + Apple Silicon GPU backend
```

## Quickstart

```python
from macquerel import Circuit, Simulator

circuit = Circuit(2)
circuit.h(0).cx(0, 1).measure_all()

sim = Simulator()                  # auto-selects a backend
counts = sim.run(circuit, shots=1000)
print(counts)  # Counter({'00': ~500, '11': ~500})
```

## Backends

| Backend | Range | Notes |
|---|---|---|
| CPU (NumPy) | ≤15q | Reference implementation |
| Metal | 16–33q | PyObjC kernels, 64-bit indexing, in-place |
| MLX | 16–30q | Apple Silicon GPU; fallback when Metal is unavailable |

`Simulator()` selects automatically by qubit count; pass `backend="cpu" / "mlx" / "metal"`
to force one.

![CPU vs MLX vs Metal circuit time across qubit counts](assets/backend_runtimes.png)

*Circuit time by backend (log scale) on an Apple M5 Max. Past 30 qubits only the
Metal backend can allocate the state.*

Each backend wins a different regime:

- **CPU (NumPy)** — the portable reference. Fastest at **≤15q**, where the state
  vector is only a few MB and per-gate GPU dispatch latency would dominate. Time and
  memory become impractical beyond ~24q. *Pro:* runs anywhere, no GPU. *Con:* doesn't
  scale.
- **Metal** — custom PyObjC kernels with 64-bit indexing and genuine in-place updates.
  The fastest backend **everywhere ≥16q** (batched command buffers and specialized
  kernels removed the per-gate sync that used to hold it back at mid sizes), and the
  **only** one past 30q, reaching **31–33q** (16/32/64 GiB states) at ~2× per added
  qubit — the bandwidth-bound ideal. At 24q it beats CPU by up to **28×**.
- **MLX** — Apple Silicon GPU via a fused lazy graph. The fallback for **17–30q**
  when PyObjC-Metal isn't installed: far ahead of CPU (random circuit @24q **17×**),
  within 1.0–2× of Metal on nearest-neighbor circuits like QAOA, further behind on
  diagonal-heavy ones like QFT. *Cons:* double-buffers every gate (≈2× memory), and
  its `int32` indexing hard-caps it at **30 qubits** (2³¹ amplitudes).

See [Backends](backends.md) for the measurements behind these tiers, why the GPU
loses below 16q, how circuit structure moves the MLX/Metal gap, and tuning tips.
For a guided tour of the internals — what statevector simulation is, how each
backend applies gates, and how the optimizations work — see
[How it works](how-it-works/index.md).
For parameter sweeps of many small circuits, see `BatchedSimulator` — it packs the
whole sweep into batched array ops (up to 47× over a per-circuit loop).

## Noise simulation

Circuits accept Kraus-operator noise channels (`.depolarizing(q, p)`,
`.amplitude_damping(q, gamma)`, …) and run on the `DensityMatrixSimulator`, which
carries the density matrix as a vectorized 2n-qubit state over the same three
backends. An n-qubit noisy simulation therefore costs what a 2n-qubit statevector
costs — up to 15 qubits on MLX and 16 on Metal (a 32 GiB state). See
[Getting Started](getting-started.md#simulating-noise) and the
[API reference](reference/api.md#densitymatrixsimulatorbackendauto-dtypecomplex64-seednone).
