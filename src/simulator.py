from typing import Protocol, runtime_checkable
import mlx.core as mx

@runtime_checkable
class Backend(Protocol):
    def allocate(self, n_qubits: int, dtype) -> mx.array: ...
    def apply_matrix(self, sv: mx.array, matrix: mx.array, targets: list[int], controls: list[int] = None) -> None: ...
    def measure(self, sv: mx.array, qubits: list[int], *, collapse: bool = True) -> list[int]: ...
    def sample(self, sv: mx.array, qubits: list[int], shots: int) -> dict[int, int]: ...
    def expectation_pauli(self, sv: mx.array, pauli_strings: list[str]) -> mx.array: ...


class Circuit:
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.gates = []

    def h(self, qubit: int) -> None:
        self.gates.append(('H', [qubit]))

    def cx(self, control: int, target: int) -> None:
        self.gates.append(('CNOT', [control, target]))

    def rz(self, qubit: int, theta: float) -> None:
        self.gates.append(('RZ', [qubit, theta]))

    def measure_all(self) -> None:
        self.gates.append(('MEASURE', []))


class Simulator:
    def __init__(self, backend: str = 'mlx', dtype: str = 'complex64'):
        self.backend = backend
        self.dtype = dtype
        self.state = None

    def run(self, circuit: Circuit, shots: int = 1000) -> dict[int, int]:
        self.state = mx.random.normal((2**circuit.n_qubits,))
        for gate in circuit.gates:
            if gate[0] == 'H':
                self._apply_hadamard(gate[1][0])
            elif gate[0] == 'CNOT':
                self._apply_cnot(gate[1][0], gate[1][1])
            elif gate[0] == 'RZ':
                self._apply_rz(gate[1][0], gate[1][1])
        return self._sample(shots)

    def _apply_hadamard(self, qubit: int) -> None:
        # Implementation details would go here
        pass

    def _apply_cnot(self, control: int, target: int) -> None:
        # Implementation details would go here
        pass

    def _apply_rz(self, qubit: int, theta: float) -> None:
        # Implementation details would go here
        pass

    def _sample(self, shots: int) -> dict[int, int]:
        # Implementation details would go here
        return {0: shots // 2, 1: shots // 2}

    def statevector(self, circuit: Circuit) -> mx.array:
        # Return the state vector
        return self.state