from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

try:
    import mlx.core as mx  # ty: ignore[unresolved-import]

    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False


@dataclass
class MLXState:
    """Statevector held as a single complex64 mx.array (P4: native complex storage)."""

    data: mx.array  # shape (2**n,), complex64
    n_qubits: int


def _diag_phase_kernel(data, diag, gate_idx):
    """Diagonal gate: gather the per-amplitude complex phase and multiply.

    Pure-functional so mx.compile can fuse the gather + multiply into one kernel."""
    return diag[gate_idx] * data


def _perm_gather_kernel(data, src):
    """Permutation gate: gather amplitudes by source index."""
    return data[src]


class MLXBackend:
    """
    MLX-accelerated backend for Apple Silicon. State is stored as a single
    complex64 mx.array between gate calls. On non-Apple platforms where mlx is
    not installed, raises ImportError.
    """

    def __init__(self, seed: int | None = None) -> None:
        if not _MLX_AVAILABLE:
            raise ImportError(
                "mlx is not installed. Install it with: pip install mlx\n"
                "Note: mlx requires macOS on Apple Silicon (M1 or later)."
            )
        self._rng_key = mx.random.key(seed) if seed is not None else None
        # Per-gate device constants reused across calls (cause (3)/(7) in the
        # perf plan). The arange index is reused by every diagonal/permutation
        # gate and bounded by the number of distinct qubit counts; `_one` avoids
        # rebuilding the scalar uint32 mask each loop iteration; the classify
        # cache avoids re-scanning identical gate matrices.
        self._one = mx.array(1, dtype=mx.uint32)
        self._arange_cache: dict[int, mx.array] = {}
        self._classify_cache: dict[tuple, str] = {}
        # P8: compile the hot elementwise kernels so MLX fuses the gather +
        # arithmetic into one kernel and caches the trace per input shape.
        self._diag_kernel = mx.compile(_diag_phase_kernel)
        self._perm_kernel = mx.compile(_perm_gather_kernel)

    def _arange(self, n: int) -> mx.array:
        """Cached, evaluated uint32 [0, 2**n) index vector (one per qubit count)."""
        a = self._arange_cache.get(n)
        if a is None:
            a = mx.arange(2**n, dtype=mx.uint32)
            mx.eval(a)
            self._arange_cache[n] = a
        return a

    def _classify(self, mat: np.ndarray) -> str:
        """classify(mat) memoized by matrix bytes (identical gates share a result)."""
        key = (mat.shape, mat.tobytes())
        kind = self._classify_cache.get(key)
        if kind is None:
            from macquerel.gates import classify

            kind = classify(mat)
            self._classify_cache[key] = kind
        return kind

    def allocate(self, n_qubits: int, dtype=np.complex64) -> MLXState:
        size = 2**n_qubits
        arr = np.zeros(size, dtype=np.complex64)
        arr[0] = 1.0
        data = mx.array(arr)
        mx.eval(data)
        return MLXState(data=data, n_qubits=n_qubits)

    def to_numpy(self, sv: MLXState) -> np.ndarray:
        mx.eval(sv.data)
        return np.array(sv.data).astype(np.complex64)

    def apply_matrix(
        self,
        sv: MLXState,
        matrix: np.ndarray,
        targets: list[int],
        controls: list[int] | None = None,
    ) -> MLXState:
        mat = matrix.astype(np.complex64)
        kind = self._classify(mat)

        if kind == "diagonal" and not controls:
            return self._apply_diagonal(sv, mat, targets)
        if kind == "permutation" and not controls:
            return self._apply_permutation(sv, mat, targets)
        return self._apply_general(sv, mat, targets, controls)

    def _gate_index(self, targets: list[int], n: int) -> mx.array:
        """Pack the k target bits of every basis index into a gate-row index."""
        k = len(targets)
        indices = self._arange(n)
        gate_idx = mx.zeros(2**n, dtype=mx.uint32)
        for bit_pos, q in enumerate(targets):
            bit = (indices >> (n - 1 - q)) & self._one
            gate_idx = gate_idx | (bit << (k - 1 - bit_pos))
        return gate_idx

    def _apply_diagonal(self, sv: MLXState, matrix: np.ndarray, targets: list[int]) -> MLXState:
        """Diagonal gate: elementwise complex phase multiply."""
        n = sv.n_qubits
        diag_mx = mx.array(np.diag(matrix).astype(np.complex64))
        gate_idx = self._gate_index(targets, n)
        new_data = self._diag_kernel(sv.data, diag_mx, gate_idx)
        return MLXState(data=new_data, n_qubits=n)

    def _apply_permutation(self, sv: MLXState, matrix: np.ndarray, targets: list[int]) -> MLXState:
        """Permutation gate: pure gather — no multiply/add needed.

        The gather index array is built entirely on-device with mx.arange +
        bitwise ops, so there is no host-side O(2**n) NumPy table build and no
        host->device copy per gate. The only host work is the tiny 2**k
        inverse-permutation lookup over the target subspace.
        """
        n = sv.n_qubits
        size = 2**n
        k = len(targets)

        # Small (2**k)-element permutation on the target subspace:
        # gate_perm[input_row] -> output_row (matches the dense argmax mapping).
        gate_perm = np.array(
            [int(np.argmax(np.abs(matrix[r]))) for r in range(2**k)],
            dtype=np.int64,
        )
        # Gather needs the inverse: which input row feeds a given output row.
        inv_gate_perm = np.empty(2**k, dtype=np.uint32)
        inv_gate_perm[gate_perm] = np.arange(2**k, dtype=np.uint32)

        indices = self._arange(n)
        one = self._one

        # Extract the k target bits of every output index into a gate-row index.
        out_row = mx.zeros(size, dtype=mx.uint32)
        for j, t in enumerate(targets):
            bit = (indices >> (n - 1 - t)) & one
            out_row = out_row | (bit << (k - 1 - j))

        # Map each output row to its source row via the inverse permutation.
        in_row = mx.array(inv_gate_perm)[out_row]

        # Source index = output index with its target bits rewritten to in_row.
        src = indices
        for j, t in enumerate(targets):
            shift = n - 1 - t
            clear_mask = mx.array(((1 << n) - 1) ^ (1 << shift), dtype=mx.uint32)
            bit = (in_row >> (k - 1 - j)) & one
            src = (src & clear_mask) | (bit << shift)

        new_data = self._perm_kernel(sv.data, src)
        return MLXState(data=new_data, n_qubits=n)

    def _dense_apply(
        self,
        data: mx.array,
        mat: np.ndarray,
        targets: list[int],
        n: int,
    ) -> mx.array:
        """Apply a dense k-qubit matrix to `targets` over the full state.

        Single complex64 tensordot (vs four real tensordots in the SoA layout),
        then transpose the contracted axes back into canonical qubit order.
        """
        k = len(targets)
        mat_mx = mx.array(mat.astype(np.complex64)).reshape((2,) * (2 * k))
        state = data.reshape((2,) * n)
        input_axes = list(range(k, 2 * k))

        out = mx.tensordot(mat_mx, state, axes=[input_axes, targets])

        remaining = [i for i in range(n) if i not in targets]
        dest = targets + remaining
        inv_perm = [0] * n
        for new_pos, old_pos in enumerate(dest):
            inv_perm[old_pos] = new_pos
        return mx.transpose(out, inv_perm).reshape(-1)

    def _apply_controlled(
        self,
        sv: MLXState,
        matrix: np.ndarray,
        targets: list[int],
        controls: list[int],
    ) -> MLXState:
        """Controlled gate, computed natively on the GPU.

        Applies the matrix to the targets across the whole state, then selects
        between the gated and original amplitudes with a control mask (all
        control bits set) via mx.where. No CPU fallback / numpy round-trip.
        """
        n = sv.n_qubits
        gated = self._dense_apply(sv.data, matrix, targets, n)

        indices = self._arange(n)
        one = self._one
        mask = None
        for c in controls:
            bit = (indices >> (n - 1 - c)) & one
            cond = bit == one
            mask = cond if mask is None else (mask & cond)  # ty: ignore[unsupported-operator]

        new_data = mx.where(mask, gated, sv.data)  # ty: ignore[invalid-argument-type]
        return MLXState(data=new_data, n_qubits=n)

    def _apply_general(
        self,
        sv: MLXState,
        matrix: np.ndarray,
        targets: list[int],
        controls: list[int] | None,
    ) -> MLXState:
        """General dense gate, kept on the GPU via mx.tensordot."""
        n = sv.n_qubits
        mat = matrix.astype(np.complex64)

        if controls:
            return self._apply_controlled(sv, mat, targets, controls)

        out = self._dense_apply(sv.data, mat, targets, n)
        return MLXState(data=out, n_qubits=n)

    def measure(
        self,
        sv: MLXState,
        qubits: list[int],
        *,
        collapse: bool = True,
    ) -> list[int]:
        from macquerel.backends.cpu import CPUBackend

        sv_np = self.to_numpy(sv)
        outcomes = CPUBackend().measure(sv_np, qubits, collapse=collapse)
        if collapse:
            sv.data = mx.array(sv_np.astype(np.complex64))
            mx.eval(sv.data)
        return outcomes

    def sample(
        self,
        sv: MLXState,
        qubits: list[int],
        shots: int,
    ) -> Counter:
        n = sv.n_qubits
        mx.eval(sv.data)
        state = np.array(sv.data).reshape((2,) * n)
        probs2 = np.abs(state) ** 2

        sum_axes = tuple(i for i in range(n) if i not in qubits)
        joint = np.sum(probs2, axis=sum_axes)
        qubits_in_state_order = sorted(range(len(qubits)), key=lambda i: qubits[i])
        joint = np.transpose(joint, qubits_in_state_order)
        flat_probs = joint.reshape(-1)
        flat_probs = flat_probs / flat_probs.sum()

        log_probs = mx.log(mx.array(flat_probs.astype(np.float32)) + 1e-38)
        if self._rng_key is not None:
            samples = mx.random.categorical(log_probs, num_samples=shots, key=self._rng_key)
        else:
            samples = mx.random.categorical(log_probs, num_samples=shots)
        mx.eval(samples)
        indices = np.array(samples)

        result: Counter = Counter()
        for idx in indices:
            bits = format(int(idx), f"0{len(qubits)}b")
            result[bits] += 1
        return result

    def abs2sum(self, sv: MLXState, qubits: list[int]) -> np.ndarray:
        n = sv.n_qubits
        mx.eval(sv.data)
        arr = np.array(sv.data)
        probs = (np.abs(arr) ** 2).reshape((2,) * n)
        sum_axes = tuple(i for i in range(n) if i not in qubits)
        return np.sum(probs, axis=sum_axes).reshape(-1)

    def expectation_pauli(self, sv: MLXState, pauli_strings) -> np.ndarray:
        from macquerel.gates import I as I_gate
        from macquerel.gates import X, Y, Z

        PAULI_MAP = {"X": X(), "Y": Y(), "Z": Z(), "I": I_gate()}
        sv_np = self.to_numpy(sv)
        results = []
        for coeff, terms in pauli_strings:
            psi_p = sv_np.copy()
            from macquerel.backends.cpu import CPUBackend

            cpu = CPUBackend()
            for pauli_char, qubit in terms:
                psi_p = cpu.apply_matrix(psi_p, PAULI_MAP[pauli_char], [qubit])
            results.append(coeff * float(np.real(np.dot(sv_np.conj(), psi_p))))
        return np.array(results)
