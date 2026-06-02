"""Converter from qiskit.QuantumCircuit to macquerel.Circuit."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from macquerel.circuit import Circuit


def from_qiskit(qiskit_circuit) -> Circuit:
    """Convert a qiskit.QuantumCircuit to a macquerel.Circuit.

    Supports h, x, y, z, s, t, rx, ry, rz, cx, cz, swap, cp, p, and measure operations.
    Raises NotImplementedError for unsupported instructions.

    Requires qiskit to be installed.
    """
    try:
        import qiskit  # noqa: F401  # ty: ignore[unresolved-import]
    except ImportError as e:
        raise ImportError(
            "qiskit is required for from_qiskit(). Install with: pip install qiskit"
        ) from e

    import macquerel

    n = qiskit_circuit.num_qubits
    qc = macquerel.Circuit(n)

    # qiskit's measure_all() emits one single-qubit `measure` per qubit; collect
    # them into a single measurement so outcomes are full n-bit strings.
    measured: list[int] = []

    for instruction in qiskit_circuit.data:
        op = instruction.operation
        qubits = [qiskit_circuit.find_bit(q).index for q in instruction.qubits]
        name = op.name.lower()

        if name == "measure":
            measured.extend(qubits)
        elif name == "h":
            qc.h(qubits[0])
        elif name == "x":
            qc.x(qubits[0])
        elif name == "y":
            qc.y(qubits[0])
        elif name == "z":
            qc.z(qubits[0])
        elif name == "s":
            qc.s(qubits[0])
        elif name == "t":
            qc.t(qubits[0])
        elif name == "rx":
            qc.rx(qubits[0], float(op.params[0]))
        elif name == "ry":
            qc.ry(qubits[0], float(op.params[0]))
        elif name == "rz":
            qc.rz(qubits[0], float(op.params[0]))
        elif name == "p":
            qc.p(qubits[0], float(op.params[0]))
        elif name in ("cx", "cnot"):
            qc.cx(qubits[0], qubits[1])
        elif name == "cz":
            qc.cz(qubits[0], qubits[1])
        elif name == "swap":
            qc.swap(qubits[0], qubits[1])
        elif name == "cp":
            qc.cp(qubits[0], qubits[1], float(op.params[0]))
        elif name == "barrier":
            pass  # barriers are purely visual, ignore
        else:
            raise NotImplementedError(
                f"Unsupported qiskit gate: '{op.name}'. Decompose it to supported primitives first."
            )

    if measured:
        qc.measure(sorted(set(measured)))

    return qc
