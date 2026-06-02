from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

try:
    import mlx.core as mx
    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False

_METAL_KERNEL = None


def _try_register_metal_kernel() -> None:
    global _METAL_KERNEL
    if not _MLX_AVAILABLE:
        return
    try:
        import mlx.core.metal  # noqa: F401 — only present on Apple Silicon
        source = r"""
        uint p = thread_position_in_grid.x;
        uint k = targets[0];
        uint low = (1u << k) - 1u;
        uint i0 = ((p & ~low) << 1) | (p & low);
        uint i1 = i0 | (1u << k);
        float a0r = real_in[i0], a0i = imag_in[i0];
        float a1r = real_in[i1], a1i = imag_in[i1];
        real_out[i0] = mat[0]*a0r - mat[1]*a0i + mat[2]*a1r - mat[3]*a1i;
        imag_out[i0] = mat[0]*a0i + mat[1]*a0r + mat[2]*a1i + mat[3]*a1r;
        real_out[i1] = mat[4]*a0r - mat[5]*a0i + mat[6]*a1r - mat[7]*a1i;
        imag_out[i1] = mat[4]*a0i + mat[5]*a0r + mat[6]*a1i + mat[7]*a1r;
        """
        _METAL_KERNEL = mx.fast.metal_kernel(
            name="apply_1q_gate",
            input_names=["real_in", "imag_in", "mat", "targets"],
            output_names=["real_out", "imag_out"],
            source=source,
        )
    except (ImportError, AttributeError):
        pass


_try_register_metal_kernel()


@dataclass
class MLXState:
    """SoA statevector: two float32 mx.arrays. Avoids repeated interleaved↔SoA conversion."""
    real: "mx.array"  # shape (2**n,), float32
    imag: "mx.array"  # shape (2**n,), float32
    n_qubits: int


def _diag_phase_kernel(real, imag, diag_r, diag_i, gate_idx):
    """Diagonal gate: gather the per-amplitude phase, then complex-multiply.

    Pure-functional so mx.compile can fuse the gather and the six elementwise
    ops into a single kernel (cached per input shape)."""
    pr = diag_r[gate_idx]
    pi = diag_i[gate_idx]
    return pr * real - pi * imag, pr * imag + pi * real


def _perm_gather_kernel(real, imag, src):
    """Permutation gate: gather both SoA components by source index."""
    return real[src], imag[src]


class MLXBackend:
    """
    MLX-accelerated backend for Apple Silicon. State is stored in SoA form
    (two float32 mx.arrays) between gate calls. On non-Apple platforms where
    mlx is not installed, raises ImportError.
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
        self._arange_cache: dict[int, "mx.array"] = {}
        self._classify_cache: dict[tuple, str] = {}
        # P8: compile the hot elementwise kernels so MLX fuses the gather +
        # arithmetic into one kernel and caches the trace per input shape.
        self._diag_kernel = mx.compile(_diag_phase_kernel)
        self._perm_kernel = mx.compile(_perm_gather_kernel)

    def _arange(self, n: int) -> "mx.array":
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
        real_np = np.zeros(size, dtype=np.float32)
        real_np[0] = 1.0
        real = mx.array(real_np)
        imag = mx.zeros(size, dtype=mx.float32)
        mx.eval(real, imag)
        return MLXState(real=real, imag=imag, n_qubits=n_qubits)

    def to_numpy(self, sv: MLXState) -> np.ndarray:
        mx.eval(sv.real, sv.imag)
        r = np.array(sv.real, dtype=np.float32)
        i = np.array(sv.imag, dtype=np.float32)
        return (r + 1j * i).astype(np.complex64)

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

    def _apply_diagonal(self, sv: MLXState, matrix: np.ndarray, targets: list[int]) -> MLXState:
        """Diagonal gate: elementwise phase multiply."""
        n = sv.n_qubits
        k = len(targets)
        diag = np.diag(matrix).astype(np.complex64)
        diag_r = mx.array(diag.real.astype(np.float32))  # zero-copy
        diag_i = mx.array(diag.imag.astype(np.float32))  # zero-copy

        size = 2**n
        indices = self._arange(n)
        gate_idx = mx.zeros(size, dtype=mx.uint32)
        for bit_pos, q in enumerate(targets):
            bit = (indices >> (n - 1 - q)) & self._one
            gate_idx = gate_idx | (bit << (k - 1 - bit_pos))

        # Compiled gather + complex multiply (kept lazy; eval at segment boundary).
        new_real, new_imag = self._diag_kernel(sv.real, sv.imag, diag_r, diag_i, gate_idx)
        return MLXState(real=new_real, imag=new_imag, n_qubits=n)

    def _apply_permutation(self, sv: MLXState, matrix: np.ndarray, targets: list[int]) -> MLXState:
        """Permutation gate: pure gather — no multiply/add needed.

        The gather index array is built entirely on-device with mx.arange +
        bitwise ops (mirroring _apply_diagonal), so there is no host-side O(2**n)
        NumPy table build and no host->device copy per gate. The only host work
        is the tiny 2**k inverse-permutation lookup over the target subspace.
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

        new_real, new_imag = self._perm_kernel(sv.real, sv.imag, src)
        return MLXState(real=new_real, imag=new_imag, n_qubits=n)

    def _apply_metal_kernel_1q(self, sv: MLXState, matrix: np.ndarray, targets: list[int]) -> MLXState:
        """Single-qubit dense gate via custom Metal kernel (Apple Silicon only)."""
        n = sv.n_qubits
        mat_flat = np.zeros(8, dtype=np.float32)
        m = matrix.astype(np.complex64)
        mat_flat[0] = m[0, 0].real; mat_flat[1] = m[0, 0].imag
        mat_flat[2] = m[0, 1].real; mat_flat[3] = m[0, 1].imag
        mat_flat[4] = m[1, 0].real; mat_flat[5] = m[1, 0].imag
        mat_flat[6] = m[1, 1].real; mat_flat[7] = m[1, 1].imag
        mx_mat = mx.array(mat_flat)
        mx_targets = mx.array(np.array([n - 1 - targets[0]], dtype=np.uint32))
        threads = 2 ** (n - 1)
        out_real, out_imag = _METAL_KERNEL(
            inputs=[sv.real, sv.imag, mx_mat, mx_targets],
            output_shapes=[(2**n,), (2**n,)],
            output_dtypes=[mx.float32, mx.float32],
            grid=(threads, 1, 1),
            threadgroup=(min(threads, 256), 1, 1),
        )
        return MLXState(real=out_real, imag=out_imag, n_qubits=n)

    def _dense_apply(
        self,
        real: "mx.array",
        imag: "mx.array",
        mat: np.ndarray,
        targets: list[int],
        n: int,
    ) -> tuple["mx.array", "mx.array"]:
        """Apply a dense k-qubit matrix to `targets` over the full state.

        Returns flattened (real, imag) MLX arrays. Stays entirely on the GPU
        via mx.tensordot — no numpy round-trip.
        """
        k = len(targets)
        mat_r = mx.array(mat.real.astype(np.float32)).reshape((2,) * (2 * k))
        mat_i = mx.array(mat.imag.astype(np.float32)).reshape((2,) * (2 * k))

        state_r = real.reshape((2,) * n)
        state_i = imag.reshape((2,) * n)

        input_axes = list(range(k, 2 * k))

        out_rr = mx.tensordot(mat_r, state_r, axes=[input_axes, targets])
        out_ri = mx.tensordot(mat_r, state_i, axes=[input_axes, targets])
        out_ir = mx.tensordot(mat_i, state_r, axes=[input_axes, targets])
        out_ii = mx.tensordot(mat_i, state_i, axes=[input_axes, targets])

        out_r = out_rr - out_ii
        out_i = out_ri + out_ir

        remaining = [i for i in range(n) if i not in targets]
        dest = targets + remaining
        inv_perm = [0] * n
        for new_pos, old_pos in enumerate(dest):
            inv_perm[old_pos] = new_pos
        out_r = mx.transpose(out_r, inv_perm).reshape(-1)
        out_i = mx.transpose(out_i, inv_perm).reshape(-1)
        return out_r, out_i

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
        size = 2**n

        gated_r, gated_i = self._dense_apply(sv.real, sv.imag, matrix, targets, n)

        indices = mx.arange(size, dtype=mx.uint32)
        one = mx.array(1, dtype=mx.uint32)
        mask = None
        for c in controls:
            bit = (indices >> (n - 1 - c)) & one
            cond = bit == one
            mask = cond if mask is None else (mask & cond)

        new_real = mx.where(mask, gated_r, sv.real)
        new_imag = mx.where(mask, gated_i, sv.imag)
        return MLXState(real=new_real, imag=new_imag, n_qubits=n)

    def _apply_general(
        self,
        sv: MLXState,
        matrix: np.ndarray,
        targets: list[int],
        controls: list[int] | None,
    ) -> MLXState:
        """General dense gate, kept on the GPU via mx.tensordot."""
        n = sv.n_qubits
        k = len(targets)
        mat = matrix.astype(np.complex64)

        if _METAL_KERNEL is not None and k == 1 and not controls:
            return self._apply_metal_kernel_1q(sv, mat, targets)

        if controls:
            return self._apply_controlled(sv, mat, targets, controls)

        out_r, out_i = self._dense_apply(sv.real, sv.imag, mat, targets, n)
        return MLXState(real=out_r, imag=out_i, n_qubits=n)

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
            mx.eval(sv.real, sv.imag)
            sv.real = mx.array(sv_np.real.astype(np.float32))
            sv.imag = mx.array(sv_np.imag.astype(np.float32))
            mx.eval(sv.real, sv.imag)
        return outcomes

    def sample(
        self,
        sv: MLXState,
        qubits: list[int],
        shots: int,
    ) -> Counter:
        n = sv.n_qubits
        mx.eval(sv.real, sv.imag)
        real_np = np.array(sv.real, dtype=np.float32)
        imag_np = np.array(sv.imag, dtype=np.float32)
        state = (real_np + 1j * imag_np).reshape((2,) * n)
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
        mx.eval(sv.real, sv.imag)
        real_np = np.array(sv.real, dtype=np.float32)
        imag_np = np.array(sv.imag, dtype=np.float32)
        probs = (real_np**2 + imag_np**2).reshape((2,) * n)
        sum_axes = tuple(i for i in range(n) if i not in qubits)
        return np.sum(probs, axis=sum_axes).reshape(-1)

    def expectation_pauli(self, sv: MLXState, pauli_strings) -> np.ndarray:
        from macquerel.gates import X, Y, Z, I as I_gate
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
