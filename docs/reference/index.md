# Reference

Reference material for `macquerel`. See the [API](api.md) page for the public surface.

The public API is re-exported from the top-level `macquerel` package:

- `Circuit` — gate/measurement/noise-channel builder
- `Simulator` — runs circuits (`run`, `statevector`)
- `DensityMatrixSimulator` — runs noisy circuits (`run`, `density_matrix`, `probabilities`, `expectation_pauli`, `purity`)
- `BatchedSimulator` — runs many same-width circuits as one batched evolution
- `Gate`, `MeasureOp`, `ChannelOp` — circuit operation dataclasses
- `from_cirq`, `from_qiskit` — adapters (available when the matching extra is installed)
