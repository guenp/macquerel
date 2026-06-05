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
        # Step 19: autotuned shot-batch size for mx.random.categorical, memoized
        # by category count (2**len(qubits)); see _autotune_batch / sample.
        self._tuned_batch: dict[int, int] = {}
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
        """Monomial (generalized-permutation) gate: gather + per-row phase.

        `classify` labels any matrix with exactly one magnitude-1 nonzero per
        row/col as "permutation" — this includes *phased* monomial matrices such
        as a fused CX·(Rz⊗Rz), whose nonzero entries are complex phases, not 1.
        So the gather alone is not enough: each gathered amplitude must also be
        multiplied by the value of its nonzero matrix entry. (A pure permutation
        like X/SWAP/CNOT has all-1 phases, so the multiply is a no-op there.)

        The gather index array is built entirely on-device with mx.arange +
        bitwise ops, so there is no host-side O(2**n) NumPy table build and no
        host->device copy per gate. The only host work is over the tiny 2**k
        target subspace.
        """
        n = sv.n_qubits
        size = 2**n
        k = len(targets)

        # Per-output-row source map on the (2**k) target subspace. For a monomial
        # matrix M, out[j] = M[j, i] * in[i] where i = argmax(|M[j]|) is the one
        # nonzero column of row j. The gather kernel does new[idx] = old[src[idx]],
        # so this row map is used directly — no inversion. (Inverting only happened
        # to match for involutions like X / SWAP / CNOT, which are self-inverse;
        # a fused CX·CX is not, which is why composed permutations were wrong.)
        gate_perm = np.array(
            [int(np.argmax(np.abs(matrix[r]))) for r in range(2**k)],
            dtype=np.uint32,
        )
        # The nonzero entry of each row — the phase to apply after the gather.
        row_phase = matrix[np.arange(2**k), gate_perm].astype(np.complex64)
        has_phase = not np.allclose(row_phase, 1.0, atol=1e-6)

        indices = self._arange(n)
        one = self._one

        # Extract the k target bits of every output index into a gate-row index.
        out_row = mx.zeros(size, dtype=mx.uint32)
        for j, t in enumerate(targets):
            bit = (indices >> (n - 1 - t)) & one
            out_row = out_row | (bit << (k - 1 - j))

        # Map each output row to its source input row.
        in_row = mx.array(gate_perm)[out_row]

        # Source index = output index with its target bits rewritten to in_row.
        src = indices
        for j, t in enumerate(targets):
            shift = n - 1 - t
            clear_mask = mx.array(((1 << n) - 1) ^ (1 << shift), dtype=mx.uint32)
            bit = (in_row >> (k - 1 - j)) & one
            src = (src & clear_mask) | (bit << shift)

        new_data = self._perm_kernel(sv.data, src)
        if has_phase:
            # Multiply each amplitude by its row's nonzero entry (phase gather).
            new_data = mx.array(row_phase)[out_row] * new_data
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

    def _autotune_batch(self, log_probs: mx.array, shots: int) -> int:
        """Pick the mx.random.categorical batch size that maximizes throughput.

        Doubles the batch from a 1024-shot base and times one draw at each size,
        stopping when shots/sec stops improving (the Tsim doubling heuristic).
        GPU sampling is dispatch-bound for tiny batches, so throughput climbs
        with batch size and then plateaus once the launch cost is amortized.
        """
        import time

        bs = 1024
        cap = max(shots, 1024)
        best_bs, best_tput = bs, 0.0
        while bs <= cap:
            # Warm draw first so kernel compilation is not charged to the timing.
            warm = mx.random.categorical(log_probs, num_samples=bs)
            mx.eval(warm)
            t0 = time.perf_counter()
            s = mx.random.categorical(log_probs, num_samples=bs)
            mx.eval(s)
            dt = time.perf_counter() - t0
            tput = bs / dt if dt > 0 else float("inf")
            if tput >= best_tput * 1.05:  # >5% gain counts as still improving
                best_tput, best_bs = tput, bs
                bs *= 2
            else:
                break
        return best_bs

    def _resolve_batch(
        self, batch_shots, log_probs: mx.array, num_categories: int, shots: int
    ) -> int:
        if batch_shots != "auto":
            return max(1, int(batch_shots))
        # A seeded run draws in a single deterministic pass so results are
        # reproducible regardless of the (timing-dependent) tuned batch size.
        if self._rng_key is not None:
            return max(1, shots)
        cached = self._tuned_batch.get(num_categories)
        if cached is None:
            cached = self._autotune_batch(log_probs, shots)
            self._tuned_batch[num_categories] = cached
        return cached

    def sample(
        self,
        sv: MLXState,
        qubits: list[int],
        shots: int,
        batch_shots: int | str = "auto",
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
        num_categories = flat_probs.size

        log_probs = mx.log(mx.array(flat_probs.astype(np.float32)) + 1e-38)
        batch = self._resolve_batch(batch_shots, log_probs, num_categories, shots)

        num_chunks = (shots + batch - 1) // batch
        if self._rng_key is not None:
            # One chunk reuses the base key (so single-pass output is unchanged);
            # multiple chunks get deterministic per-chunk subkeys.
            subkeys = (
                [self._rng_key]
                if num_chunks == 1
                else list(mx.random.split(self._rng_key, num_chunks))
            )
        else:
            subkeys = [None] * num_chunks

        width = len(qubits)
        result: Counter = Counter()
        remaining = shots
        for i in range(num_chunks):
            b = min(batch, remaining)
            if subkeys[i] is not None:
                samples = mx.random.categorical(log_probs, num_samples=b, key=subkeys[i])
            else:
                samples = mx.random.categorical(log_probs, num_samples=b)
            mx.eval(samples)
            for idx in np.array(samples):
                result[format(int(idx), f"0{width}b")] += 1
            remaining -= b
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
