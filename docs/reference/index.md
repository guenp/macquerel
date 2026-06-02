# Reference

Reference material for `macquerel`. See the [API](api.md) page for the public surface.

The public API is re-exported from the top-level `macquerel` package:

- `Circuit` — gate/measurement builder
- `Simulator` — runs circuits (`run`, `statevector`)
- `Gate`, `MeasureOp` — circuit operation dataclasses
- `from_cirq`, `from_qiskit` — adapters (available when the matching extra is installed)
