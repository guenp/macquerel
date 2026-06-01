"""Converter from cirq.Circuit to macquerel.Circuit."""
from __future__ import annotations

import numpy as np


def from_cirq(cirq_circuit) -> "macquerel.Circuit":
    """Convert a cirq.Circuit to a macquerel.Circuit.

    Supports H, X, Y, Z, S, T, Rx, Ry, Rz, CX/CNOT, CZ, SWAP, and measure operations.
    Raises NotImplementedError for unsupported gate types.

    Requires cirq-core to be installed.
    """
    try:
        import cirq
    except ImportError as e:
        raise ImportError(
            "cirq-core is required for from_cirq(). Install with: pip install cirq-core"
        ) from e

    import macquerel
    import macquerel.gates as g

    all_qubits = sorted(cirq_circuit.all_qubits())
    qubit_index = {q: i for i, q in enumerate(all_qubits)}
    n = len(all_qubits)
    qc = macquerel.Circuit(n)

    for moment in cirq_circuit:
        for op in moment.operations:
            qubits = [qubit_index[q] for q in op.qubits]
            gate = op.gate

            if isinstance(gate, cirq.MeasurementGate):
                qc.measure(qubits)
            elif isinstance(gate, cirq.HPowGate) and gate.exponent == 1:
                qc.h(qubits[0])
            elif isinstance(gate, cirq.XPowGate) and gate.exponent == 1:
                qc.x(qubits[0])
            elif isinstance(gate, cirq.YPowGate) and gate.exponent == 1:
                qc.y(qubits[0])
            elif isinstance(gate, cirq.ZPowGate) and gate.exponent == 1:
                qc.z(qubits[0])
            elif isinstance(gate, cirq.ZPowGate) and gate.exponent == 0.5:
                qc.s(qubits[0])
            elif isinstance(gate, cirq.ZPowGate) and gate.exponent == 0.25:
                qc.t(qubits[0])
            elif isinstance(gate, cirq.rx):
                qc.rx(qubits[0], gate.rads)
            elif isinstance(gate, cirq.ry):
                qc.ry(qubits[0], gate.rads)
            elif isinstance(gate, cirq.rz):
                qc.rz(qubits[0], gate.rads)
            elif isinstance(gate, cirq.CXPowGate) and gate.exponent == 1:
                qc.cx(qubits[0], qubits[1])
            elif isinstance(gate, cirq.CZPowGate) and gate.exponent == 1:
                qc.cz(qubits[0], qubits[1])
            elif isinstance(gate, cirq.SWAP):
                qc.swap(qubits[0], qubits[1])
            elif isinstance(gate, cirq.SwapPowGate) and gate.exponent == 1:
                qc.swap(qubits[0], qubits[1])
            else:
                raise NotImplementedError(
                    f"Unsupported cirq gate: {type(gate).__name__} ({gate!r}). "
                    "Convert it to a matrix gate manually or decompose it first."
                )

    return qc
