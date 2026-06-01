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
        from macquerel.gates import classify
        mat = matrix.astype(np.complex64)
        kind = classify(mat)

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
        indices = mx.arange(size, dtype=mx.uint32)
        gate_idx = mx.zeros(size, dtype=mx.uint32)
        for bit_pos, q in enumerate(targets):
            bit = (indices >> (n - 1 - q)) & mx.array(1, dtype=mx.uint32)
            gate_idx = gate_idx | (bit << (k - 1 - bit_pos))

        pr = diag_r[gate_idx]
        pi = diag_i[gate_idx]

        new_real = pr * sv.real - pi * sv.imag
        new_imag = pr * sv.imag + pi * sv.real
        mx.eval(new_real, new_imag)
        return MLXState(real=new_real, imag=new_imag, n_qubits=n)

    def _apply_permutation(self, sv: MLXState, matrix: np.ndarray, targets: list[int]) -> MLXState:
        """Permutation gate: pure gather — no multiply/add needed."""
        n = sv.n_qubits
        size = 2**n
        k = len(targets)
        perm = np.arange(size, dtype=np.int32)
        for i in range(size):
            bits = [(i >> (n - 1 - t)) & 1 for t in targets]
            gate_row = sum(b << (k - 1 - j) for j, b in enumerate(bits))
            out_row = int(np.argmax(np.abs(matrix[gate_row])))
            new_i = i
            for j, t in enumerate(targets):
                bit = (out_row >> (k - 1 - j)) & 1
                if bit:
                    new_i |= 1 << (n - 1 - t)
                else:
                    new_i &= ~(1 << (n - 1 - t))
            perm[new_i] = i
        mx_perm = mx.array(perm)
        new_real = sv.real[mx_perm]
        new_imag = sv.imag[mx_perm]
        mx.eval(new_real, new_imag)
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
        mx.eval(out_real, out_imag)
        return MLXState(real=out_real, imag=out_imag, n_qubits=n)

    def _apply_general(
        self,
        sv: MLXState,
        matrix: np.ndarray,
        targets: list[int],
        controls: list[int] | None,
    ) -> MLXState:
        """General gate: SoA components, tensordot on CPU for correctness."""
        n = sv.n_qubits
        k = len(targets)
        mat = matrix.astype(np.complex64)

        if _METAL_KERNEL is not None and k == 1 and not controls:
            return self._apply_metal_kernel_1q(sv, mat, targets)

        if controls:
            from macquerel.backends.cpu import CPUBackend
            mx.eval(sv.real, sv.imag)
            real_np = np.array(sv.real, dtype=np.float32)
            imag_np = np.array(sv.imag, dtype=np.float32)
            sv_complex = (real_np + 1j * imag_np).astype(np.complex64)
            sv_complex = CPUBackend()._apply_controlled(sv_complex, mat, targets, controls)
            new_real = mx.array(sv_complex.real.astype(np.float32))
            new_imag = mx.array(sv_complex.imag.astype(np.float32))
            mx.eval(new_real, new_imag)
            return MLXState(real=new_real, imag=new_imag, n_qubits=n)

        # Use MLX tensordot to keep state on GPU — no numpy round-trip
        mat_r = mx.array(mat.real.astype(np.float32)).reshape((2,) * (2 * k))
        mat_i = mx.array(mat.imag.astype(np.float32)).reshape((2,) * (2 * k))

        state_r = sv.real.reshape((2,) * n)
        state_i = sv.imag.reshape((2,) * n)

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

        mx.eval(out_r, out_i)
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
