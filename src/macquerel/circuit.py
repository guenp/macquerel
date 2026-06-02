from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from macquerel.gates import (
    CNOT,
    CP,
    CZ,
    SWAP,
    GateKind,
    H,
    I,
    P,
    Rx,
    Ry,
    Rz,
    S,
    T,
    X,
    Y,
    Z,
    classify,
)


@dataclass
class Gate:
    name: str
    matrix: np.ndarray
    targets: list[int]
    controls: list[int] = field(default_factory=list)
    kind: GateKind = "dense"


@dataclass
class MeasureOp:
    qubits: list[int]


class Circuit:
    def __init__(self, n_qubits: int):
        if n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
        self.n_qubits = n_qubits
        self.ops: list[Gate | MeasureOp] = []

    def _check(self, *qubits: int) -> None:
        seen: set[int] = set()
        for q in qubits:
            if not (0 <= q < self.n_qubits):
                raise ValueError(f"Qubit index {q} out of range [0, {self.n_qubits})")
            if q in seen:
                raise ValueError(f"Duplicate qubit index {q}")
            seen.add(q)

    def _add(
        self, name: str, matrix: np.ndarray, targets: list[int], controls: list[int] | None = None
    ) -> None:
        ctrls = controls or []
        mat = matrix.astype(np.complex64)
        self.ops.append(
            Gate(name=name, matrix=mat, targets=targets, controls=ctrls, kind=classify(mat))
        )

    def i(self, qubit: int) -> Circuit:
        self._check(qubit)
        self._add("I", I(), [qubit])
        return self

    def h(self, qubit: int) -> Circuit:
        self._check(qubit)
        self._add("H", H(), [qubit])
        return self

    def x(self, qubit: int) -> Circuit:
        self._check(qubit)
        self._add("X", X(), [qubit])
        return self

    def y(self, qubit: int) -> Circuit:
        self._check(qubit)
        self._add("Y", Y(), [qubit])
        return self

    def z(self, qubit: int) -> Circuit:
        self._check(qubit)
        self._add("Z", Z(), [qubit])
        return self

    def s(self, qubit: int) -> Circuit:
        self._check(qubit)
        self._add("S", S(), [qubit])
        return self

    def t(self, qubit: int) -> Circuit:
        self._check(qubit)
        self._add("T", T(), [qubit])
        return self

    def rx(self, qubit: int, theta: float) -> Circuit:
        self._check(qubit)
        self._add("Rx", Rx(theta), [qubit])
        return self

    def ry(self, qubit: int, theta: float) -> Circuit:
        self._check(qubit)
        self._add("Ry", Ry(theta), [qubit])
        return self

    def rz(self, qubit: int, theta: float) -> Circuit:
        self._check(qubit)
        self._add("Rz", Rz(theta), [qubit])
        return self

    def p(self, qubit: int, lam: float) -> Circuit:
        self._check(qubit)
        self._add("P", P(lam), [qubit])
        return self

    def cx(self, control: int, target: int) -> Circuit:
        self._check(control, target)
        self._add("CX", CNOT(), [control, target])
        return self

    def cz(self, control: int, target: int) -> Circuit:
        self._check(control, target)
        self._add("CZ", CZ(), [control, target])
        return self

    def swap(self, q0: int, q1: int) -> Circuit:
        self._check(q0, q1)
        self._add("SWAP", SWAP(), [q0, q1])
        return self

    def cp(self, control: int, target: int, lam: float) -> Circuit:
        self._check(control, target)
        self._add("CP", CP(lam), [control, target])
        return self

    def measure(self, qubits: list[int]) -> Circuit:
        self._check(*qubits)
        self.ops.append(MeasureOp(qubits=list(qubits)))
        return self

    def measure_all(self) -> Circuit:
        self.ops.append(MeasureOp(qubits=list(range(self.n_qubits))))
        return self
