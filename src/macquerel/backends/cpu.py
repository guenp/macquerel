from __future__ import annotations

from collections import Counter

import numpy as np


class CPUBackend:
    """NumPy reference backend. Correctness over speed."""

    def allocate(self, n_qubits: int, dtype=np.complex64) -> np.ndarray:
        sv = np.zeros(2**n_qubits, dtype=dtype)
        sv[0] = 1.0
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
            # verify that all control qubits are |1⟩ across all amplitudes —
            # handled per-amplitude inside the tensor trick by restricting axes.
            # Instead we build the full controlled matrix explicitly.
            # For the CPU reference we delegate to _apply_controlled.
            return self._apply_controlled(sv, matrix, targets, controls)

        state = sv.reshape((2,) * n)
        # reshape gate to tensor with 2*k legs
        gate_t = matrix.astype(sv.dtype).reshape((2,) * (2 * k))
        # contract gate output legs (first k) against target axes of state
        # result has shape with target axes replaced, then moved to front
        out = np.tensordot(gate_t, state, axes=(list(range(k, 2 * k)), targets))
        # tensordot puts the new target axes first, then the remaining state axes
        # we need to move them back to positions `targets`
        remaining = [i for i in range(n) if i not in targets]
        # out shape: (2,)*k + remaining axes in their original relative order
        # build inverse permutation
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
        # We'll iterate over all control combinations.
        # The unitary applies only where all control bits == 1.
        # Build a mask array of shape (2,)*n that is True where controls are all 1.
        mask = np.ones((2,) * n, dtype=bool)
        for c in controls:
            # zero out all slices where control qubit c == 0
            idx = [slice(None)] * n
            idx[c] = 0
            mask[tuple(idx)] = False

        # Apply the gate to the masked slice:
        # extract the sub-tensor where controls are 1, apply gate, write back.
        ctrl_idx = tuple(1 if i in controls else slice(None) for i in range(n))
        sub = state[ctrl_idx]  # shape: (2,)*len(free_axes)
        free_axes = [i for i in range(n) if i not in controls]
        # targets within free_axes (find local indices)
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
            complementary = tuple(i for i in range(n) if i != q and i not in qubits[:qubits.index(q)])
            # marginal over already-decided qubits: sum over all other axes
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
                # zero amplitudes inconsistent with outcome, then renormalize
                idx = [slice(None)] * n
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
    ) -> Counter:
        n = int(np.log2(len(sv)))
        state = sv.reshape((2,) * n)
        probs2 = np.abs(state) ** 2

        # marginalise to the joint distribution over `qubits`
        sum_axes = tuple(i for i in range(n) if i not in qubits)
        joint = np.sum(probs2, axis=sum_axes)
        # joint now has shape (2,)*len(qubits) with axes in original order
        # reorder so that qubits appear in the given order
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
