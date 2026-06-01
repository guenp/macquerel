from __future__ import annotations

import numpy as np

from macquerel.circuit import Circuit, Gate, MeasureOp
from macquerel.gates import classify


def _compose_gates(group: list[Gate]) -> Gate:
    """Compose a list of gates into a single fused gate."""
    if len(group) == 1:
        return group[0]

    # Collect the ordered qubit set for the fused gate
    qubit_set: list[int] = []
    for g in group:
        for q in g.targets + g.controls:
            if q not in qubit_set:
                qubit_set.append(q)
    qubit_set.sort()
    k = len(qubit_set)
    dim = 2**k

    # Build the composed unitary in the qubit_set basis
    fused_matrix = np.eye(dim, dtype=np.complex128)

    for g in group:
        # Embed g.matrix into the full fused space
        all_qubits = g.targets + g.controls
        gate_qubits = [q for q in qubit_set if q in all_qubits]
        # build full gate matrix (may need to embed controls)
        if g.controls:
            # already a 2-qubit (or larger) matrix including control semantics
            # g.matrix is the raw gate matrix (e.g. CNOT 4x4 already includes control row)
            full_mat = _embed(g.matrix, g.targets + g.controls, qubit_set)
        else:
            full_mat = _embed(g.matrix, g.targets, qubit_set)
        fused_matrix = full_mat @ fused_matrix

    fused_matrix = fused_matrix.astype(np.complex64)
    kind = classify(fused_matrix)
    name = "Fused(" + ",".join(g.name for g in group) + ")"
    return Gate(name=name, matrix=fused_matrix, targets=qubit_set, controls=[], kind=kind)


def _embed(matrix: np.ndarray, gate_qubits: list[int], full_qubits: list[int]) -> np.ndarray:
    """Embed a gate matrix acting on gate_qubits into the full_qubits space."""
    k_full = len(full_qubits)
    k_gate = len(gate_qubits)
    dim_full = 2**k_full
    dim_gate = 2**k_gate

    # Build the full unitary via tensor product with identity on remaining qubits
    # We use the standard trick: write the full unitary as a tensor, set entries.
    full_mat = np.eye(dim_full, dtype=np.complex128)
    gate_mat = matrix.astype(np.complex128)

    # Map gate_qubits to their positions within full_qubits
    gate_pos = [full_qubits.index(q) for q in gate_qubits]

    # Rewrite using tensor indexing
    # full_mat[i, j] = gate_mat[i_gate, j_gate] if gate bits match, else delta
    # We use a direct construction: tensor reshape approach

    # Reshape full_mat to (2,)*k_full x (2,)*k_full
    full_t = full_mat.reshape((2,) * (2 * k_full))
    gate_t = gate_mat.reshape((2,) * (2 * k_gate))

    # Build output tensor
    out = np.eye(dim_full, dtype=np.complex128).reshape((2,) * (2 * k_full))

    # Iterate over all combinations of non-gate qubit indices
    non_gate_pos = [i for i in range(k_full) if i not in gate_pos]
    n_non_gate = len(non_gate_pos)
    dim_non_gate = 2**n_non_gate

    for env_idx in range(dim_non_gate):
        env_bits = [(env_idx >> (n_non_gate - 1 - i)) & 1 for i in range(n_non_gate)]
        for gi in range(dim_gate):
            for gj in range(dim_gate):
                gi_bits = [(gi >> (k_gate - 1 - b)) & 1 for b in range(k_gate)]
                gj_bits = [(gj >> (k_gate - 1 - b)) & 1 for b in range(k_gate)]
                # Build full row/col index tuples
                row_idx = [None] * k_full
                col_idx = [None] * k_full
                for pos, bit in zip(gate_pos, gi_bits):
                    row_idx[pos] = bit
                for pos, bit in zip(gate_pos, gj_bits):
                    col_idx[pos] = bit
                for pos, bit in zip(non_gate_pos, env_bits):
                    row_idx[pos] = bit
                    col_idx[pos] = bit
                full_row = tuple(row_idx) + tuple(col_idx)
                # Reset to zero first (the identity initialisation handles diagonal)
                out[full_row] = gate_t[tuple(gi_bits + gj_bits)]

    return out.reshape(dim_full, dim_full)


def fuse_gates(circuit: Circuit, max_fused_qubits: int = 4) -> Circuit:
    """Greedy gate fusion pass. Returns a new Circuit with fused gates."""
    result = Circuit(circuit.n_qubits)
    result.ops = []

    current_group: list[Gate] = []
    current_qubits: set[int] = set()

    def flush() -> None:
        if not current_group:
            return
        fused = _compose_gates(current_group)
        result.ops.append(fused)
        current_group.clear()
        current_qubits.clear()

    for op in circuit.ops:
        if isinstance(op, MeasureOp):
            flush()
            result.ops.append(op)
            continue

        assert isinstance(op, Gate)
        op_qubits = set(op.targets + op.controls)
        merged_qubits = current_qubits | op_qubits

        if current_group and len(merged_qubits) > max_fused_qubits:
            flush()

        current_group.append(op)
        current_qubits.update(op_qubits)

    flush()
    return result
