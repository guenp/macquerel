# macquerel Implementation Plan — Remaining Work

macquerel is a quantum state-vector simulator targeting Apple Silicon's unified-memory
architecture. The v0.1 → v0.3 milestones (CPU/MLX/Metal backends, gate fusion + qubit
remapping, expectation values, automatic backend selection, Cirq/Qiskit adapters) and the
core benchmarking suite are **complete** — see [`plan_completed.md`](plan_completed.md) for
the shipped record, including the MLX/Metal performance findings.

This document tracks only work that has **not** been implemented yet. Steps keep their
original numbering for continuity with the completed record.

---

## v1.0

### Step 18 (remaining): Benchmarking suite gaps (`benchmarks/`)

The microbenchmarks, circuit macrobenchmarks (QFT / random / QAOA), the
`max_fused_qubits ∈ {1..6}` sweep, and the Aer/qulacs comparison harness are done. Still
outstanding from the §9 plan:

- **Quantum Volume macrobenchmark** — add QV circuits to `bench_circuits.py`, swept across
  the 20–32 qubit range alongside the existing QFT/random/QAOA workloads.
- **qsim CPU comparison** — add a qsim statevector backend to the `bench_statevector.py`
  comparison harness (currently only Qiskit Aer and qulacs).

Companion test gap (from the v0.2 test additions): add Quantum Volume circuits and
random-circuit-sampling spot checks to `tests/test_known_circuits.py`.

### Step 19: Shot batch-size autotuning (`src/macquerel/simulator.py`)

Autotune the shot batch passed to `mx.random.categorical` by doubling until throughput
plateaus (the Tsim approach). Expose a `batch_shots` parameter on `Simulator` with an
`'auto'` default.

### Step 20: Per-chip fusion-width autotuning (`src/macquerel/compiler.py`)

At install time (or first run), measure the bandwidth/FLOP ratio on the local chip and pick
the `max_fused_qubits` value that maximises throughput, rather than hardcoding 4.

---

## v2

- **Noise channels / density matrices** — `DensityMatrixSimulator` with Kraus-operator
  channels.
- **Memory-mapped out-of-core backend** — state vector backed by an NVMe file via
  `np.memmap`, for single large runs past DRAM capacity.
- **Batched small-circuit simulation** — `BatchedSimulator` packing many small circuits
  (QML/VQE parameter sweeps) into one kernel launch.
- **Multi-Mac over Thunderbolt** — distributed state vector using index-bit partitioning
  across machines.

---

## Verification

After each step, run `uv run pytest tests/ -x -q` and confirm the new tests pass before
moving to the next step. Final verification:

```bash
uv run pytest tests/ -v           # full suite green
python -c "
import macquerel as mq
qc = mq.Circuit(n_qubits=3)
qc.h(0); qc.cx(0, 1); qc.cx(0, 2); qc.measure_all()
result = mq.Simulator(backend='cpu').run(qc, shots=1000)
print(result)   # should show ~500 '000' and ~500 '111'
"
```
