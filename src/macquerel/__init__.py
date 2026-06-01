from macquerel.circuit import Circuit, Gate, MeasureOp
from macquerel.simulator import Simulator

__all__ = ["Circuit", "Gate", "MeasureOp", "Simulator"]

try:
    from macquerel.adapters.cirq import from_cirq
    __all__ += ["from_cirq"]
except ImportError:
    pass

try:
    from macquerel.adapters.qiskit import from_qiskit
    __all__ += ["from_qiskit"]
except ImportError:
    pass
