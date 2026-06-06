from __future__ import annotations

from collections import Counter

import numpy as np


class CPUBackend:
    """NumPy reference backend. Correctness over speed."""

    def __init__(self, seed: int | None = None) -> None:
        if seed is not None:
            np.random.seed(seed)

    def allocate(self, n_qubits: int, dtype=np.complex64) -> np.ndarray:
        sv = np.zeros(2**n_qubits, dtype=dtype)
        sv[0] = 1.0
        return sv

    def to_numpy(self, sv: np.ndarray) -> np.ndarray:
        return sv

    def apply_matrix(
        self,
        sv: np.ndarray,
        matrix: np.ndarray,
        targets: list[int],
        controls: list[int] | None = None,
    ) -> np.ndarray:
        n = int(np.log2(len(sv)))
        k = len(targets)

        if controls:
            return self._apply_controlled(sv, matrix, targets, controls)

        state = sv.reshape((2,) * n)
        gate_t = matrix.astype(sv.dtype).reshape((2,) * (2 * k))
        out = np.tensordot(gate_t, state, axes=(list(range(k, 2 * k)), targets))
        remaining = [i for i in range(n) if i not in targets]
        dest = targets + remaining
        inv_perm = [0] * n
        for new_pos, old_pos in enumerate(dest):
            inv_perm[old_pos] = new_pos
        out = np.transpose(out, inv_perm)
        sv[:] = out.reshape(-1)
        return sv

    def _apply_controlled(
        self,
        sv: np.ndarray,
        matrix: np.ndarray,
        targets: list[int],
        controls: list[int],
    ) -> np.ndarray:
        """Apply matrix conditioned on all control qubits being |1⟩."""
        n = int(np.log2(len(sv)))
        k = len(targets)
        state = sv.reshape((2,) * n)

        gate_t = matrix.astype(sv.dtype).reshape((2,) * (2 * k))

        ctrl_idx = tuple(1 if i in controls else slice(None) for i in range(n))
        sub = state[ctrl_idx]
        free_axes = [i for i in range(n) if i not in controls]
        local_targets = [free_axes.index(t) for t in targets]
        n_free = len(free_axes)

        out_sub = np.tensordot(gate_t, sub, axes=(list(range(k, 2 * k)), local_targets))
        remaining = [i for i in range(n_free) if i not in local_targets]
        dest = local_targets + remaining
        inv_perm = [0] * n_free
        for new_pos, old_pos in enumerate(dest):
            inv_perm[old_pos] = new_pos
        out_sub = np.transpose(out_sub, inv_perm)

        state[ctrl_idx] = out_sub
        sv[:] = state.reshape(-1)
        return sv

    def measure(
        self,
        sv: np.ndarray,
        qubits: list[int],
        *,
        collapse: bool = True,
    ) -> list[int]:
        n = int(np.log2(len(sv)))
        state = sv.reshape((2,) * n)
        probs2 = np.abs(state) ** 2

        outcomes = []
        for q in qubits:
            all_axes = list(range(n))
            sum_axes = tuple(i for i in all_axes if i != q)
            marginal = np.sum(probs2, axis=sum_axes)
            p0 = float(np.real(marginal[0]))
            p1 = float(np.real(marginal[1]))
            total = p0 + p1
            if total < 1e-15:
                outcome = 0
            else:
                outcome = int(np.random.choice([0, 1], p=[p0 / total, p1 / total]))
            outcomes.append(outcome)

            if collapse:
                idx: list = [slice(None)] * n
                idx[q] = 1 - outcome
                state[tuple(idx)] = 0.0
                norm = np.sqrt(np.sum(np.abs(state) ** 2))
                if norm > 1e-15:
                    state /= norm
                probs2 = np.abs(state) ** 2

        sv[:] = state.reshape(-1)
        return outcomes

    def sample(
        self,
        sv: np.ndarray,
        qubits: list[int],
        shots: int,
        batch_shots: int | str = "auto",
    ) -> Counter:
        # batch_shots is accepted for interface parity with the GPU backends
        # (Step 19). NumPy draws all shots in one np.random.choice call, so
        # there is no per-launch overhead to amortize and nothing to tune.
        n = int(np.log2(len(sv)))
        state = sv.reshape((2,) * n)
        probs2 = np.abs(state) ** 2

        sum_axes = tuple(i for i in range(n) if i not in qubits)
        joint = np.sum(probs2, axis=sum_axes)
        qubits_in_state_order = sorted(range(len(qubits)), key=lambda i: qubits[i])
        joint = np.transpose(joint, qubits_in_state_order)
        flat_probs = joint.reshape(-1)
        flat_probs = flat_probs / flat_probs.sum()

        num_states = 2 ** len(qubits)
        indices = np.random.choice(num_states, size=shots, p=flat_probs)

        result: Counter = Counter()
        for idx in indices:
            bits = format(idx, f"0{len(qubits)}b")
            result[bits] += 1
        return result

    def abs2sum(self, sv: np.ndarray, qubits: list[int]) -> np.ndarray:
        n = int(np.log2(len(sv)))
        probs = np.abs(sv.reshape((2,) * n)) ** 2
        sum_axes = tuple(i for i in range(n) if i not in qubits)
        return np.sum(probs, axis=sum_axes).reshape(-1)

    def expectation_pauli(self, sv: np.ndarray, pauli_strings) -> np.ndarray:
        from macquerel.gates import I as I_gate
        from macquerel.gates import X, Y, Z

        PAULI_MAP = {"X": X(), "Y": Y(), "Z": Z(), "I": I_gate()}
        results = []
        for coeff, terms in pauli_strings:
            psi_p = sv.copy()
            for pauli_char, qubit in terms:
                psi_p = self.apply_matrix(psi_p, PAULI_MAP[pauli_char], [qubit])
            results.append(coeff * float(np.real(np.dot(sv.conj(), psi_p))))
        return np.array(results)
