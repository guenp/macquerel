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


@dataclass
class ChannelOp:
    """A Kraus-operator noise channel on `qubits`.

    Only the `DensityMatrixSimulator` can execute these; the statevector
    `Simulator` rejects circuits containing them. Kraus operators are stored
    as given (already validated by the builder); each is a
    ``2**len(qubits) x 2**len(qubits)`` complex matrix.
    """

    name: str
    kraus: list[np.ndarray]
    qubits: list[int]


class Circuit:
    def __init__(self, n_qubits: int):
        if n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
        self.n_qubits = n_qubits
        self.ops: list[Gate | MeasureOp | ChannelOp] = []

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

    # ---- noise channels (Kraus operators; executed by DensityMatrixSimulator) ----

    def kraus(self, qubits: list[int], operators: list[np.ndarray], name: str = "Kraus") -> Circuit:
        """Append an arbitrary Kraus channel on `qubits`.

        `operators` must be ``2**len(qubits)``-dimensional square matrices
        satisfying ``sum_k K_k^dagger K_k = I`` (validated here). Qubit order
        matters: the operators' most-significant bit acts on ``qubits[0]``,
        matching the gate-matrix convention.
        """
        from macquerel.noise import validate_kraus

        self._check(*qubits)
        validate_kraus(operators, len(qubits))
        ops = [np.asarray(op).astype(np.complex64) for op in operators]
        self.ops.append(ChannelOp(name=name, kraus=ops, qubits=list(qubits)))
        return self

    def bit_flip(self, qubit: int, p: float) -> Circuit:
        from macquerel.noise import bit_flip_kraus

        return self.kraus([qubit], bit_flip_kraus(p), name=f"BitFlip({p})")

    def phase_flip(self, qubit: int, p: float) -> Circuit:
        from macquerel.noise import phase_flip_kraus

        return self.kraus([qubit], phase_flip_kraus(p), name=f"PhaseFlip({p})")

    def depolarizing(self, qubit: int, p: float) -> Circuit:
        from macquerel.noise import depolarizing_kraus

        return self.kraus([qubit], depolarizing_kraus(p), name=f"Depolarizing({p})")

    def amplitude_damping(self, qubit: int, gamma: float) -> Circuit:
        from macquerel.noise import amplitude_damping_kraus

        return self.kraus([qubit], amplitude_damping_kraus(gamma), name=f"AmpDamp({gamma})")

    def phase_damping(self, qubit: int, gamma: float) -> Circuit:
        from macquerel.noise import phase_damping_kraus

        return self.kraus([qubit], phase_damping_kraus(gamma), name=f"PhaseDamp({gamma})")

    def measure(self, qubits: list[int]) -> Circuit:
        self._check(*qubits)
        self.ops.append(MeasureOp(qubits=list(qubits)))
        return self

    def measure_all(self) -> Circuit:
        self.ops.append(MeasureOp(qubits=list(range(self.n_qubits))))
        return self
