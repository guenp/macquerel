# macquerel Implementation Plan — Remaining Work

macquerel is a quantum state-vector simulator targeting Apple Silicon's unified-memory
architecture. The v0.1 and v0.2 lines (CPU/MLX/Metal backends, gate fusion + qubit
remapping, expectation values, automatic backend selection, Cirq/Qiskit adapters, the full
benchmarking suite, and the shot-batch / fusion-width autotuners) are **complete** — see
[`plan_completed.md`](plan_completed.md) for the shipped record, including the MLX/Metal
performance findings.

This document tracks only work that has **not** been implemented yet. Steps keep their
original numbering for continuity with the completed record.

---

## v0.3

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
