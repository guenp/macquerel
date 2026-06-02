"""Native Metal backend for the >31-qubit regime (Step 15).

Released MLX caps at 30 qubits: its ``ShapeElem`` is ``int32``, so any array of
``2**31`` or more elements is rejected before allocation (Gate 0, see
``docs/plan.md``). This backend reaches 31-33q by driving Metal directly through
PyObjC: amplitudes live in a single unified-memory ``MTLBuffer`` addressed with
64-bit indices, and gate kernels are dispatched over a 3D grid whose linear
``uint64`` index is reconstructed on the GPU. Updates are genuinely in-place
(each thread owns a disjoint set of amplitudes), so there is no double-buffering
and a 33q (64 GiB) state fits this 128 GiB machine with room to spare.

Shaders are compiled at runtime from source via ``newLibraryWithSource`` -- this
needs only the Metal *framework* (always present), not the offline ``metallib``
toolchain. The package stays pure-Python; ``pip install macquerel[metal]`` adds
only ``pyobjc-framework-Metal``. On non-Apple platforms the import fails and the
simulator falls back to MLX/CPU, exactly like the optional ``mlx`` import.

State layout: one ``MTLBuffer`` viewed as ``float2[2**n]`` -- bit-for-bit
identical to a NumPy ``complex64`` array (real, imag interleaved), so readback is
a zero-copy view. Qubit ``q`` occupies bit ``n-1-q`` of the linear index (axis 0
is the most-significant bit), matching the CPU/MLX backends.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

try:
    import Metal  # pyobjc-framework-Metal  # ty: ignore[unresolved-import]

    _METAL_AVAILABLE = Metal.MTLCreateSystemDefaultDevice() is not None
except Exception:  # pragma: no cover - import guard for non-Apple / no-GPU
    _METAL_AVAILABLE = False


# Metal Shading Language source, compiled once at backend construction.
#
# The linear amplitude index is reconstructed from a 3D grid as
#   i = (gid.z * gdim.y + gid.y) * gdim.x + gid.x
# with `gdim = threads_per_grid`; the Python side factors the thread count into
# powers of two so the product is exact and every index is covered once. All
# index math is uint64 (`ulong`), which is what lifts MLX's 2**31 ceiling.
_KERNEL_SRC = r"""
#include <metal_stdlib>
using namespace metal;

static inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Diagonal gate: out[i] = diag[gate_row(i)] * in[i], in place, one thread/amp.
kernel void diagonal(device float2*   state [[buffer(0)]],
                     constant float2* diag  [[buffer(1)]],
                     constant uint*   tpos  [[buffer(2)]],  // target bit positions
                     constant uint&   k     [[buffer(3)]],
                     uint3 gid  [[thread_position_in_grid]],
                     uint3 gdim [[threads_per_grid]]) {
    ulong i = ((ulong)gid.z * gdim.y + gid.y) * gdim.x + gid.x;
    uint row = 0u;
    for (uint j = 0u; j < k; ++j) {
        uint bit = (uint)((i >> tpos[j]) & 1u);
        row |= bit << (k - 1u - j);
    }
    state[i] = cmul(diag[row], state[i]);
}

// General k-qubit dense gate (also handles permutation and controlled gates).
// One thread owns a group: the 2^k amplitudes sharing the same non-target bits.
// It loads them, applies the 2^k x 2^k matrix, and writes them back -- disjoint
// across groups, so the in-place update is race-free without double-buffering.
kernel void dense(device float2*   state       [[buffer(0)]],
                  constant float2* M           [[buffer(1)]],  // 2^k x 2^k row-major
                  constant uint*   tpos        [[buffer(2)]],  // target bits, gate order
                  constant uint*   tpos_sorted [[buffer(3)]],  // target bits, ascending
                  constant uint*   cpos        [[buffer(4)]],  // control bit positions
                  constant uint&   k           [[buffer(5)]],
                  constant uint&   nc          [[buffer(6)]],
                  uint3 gid  [[thread_position_in_grid]],
                  uint3 gdim [[threads_per_grid]]) {
    ulong g = ((ulong)gid.z * gdim.y + gid.y) * gdim.x + gid.x;

    // Expand the group id into a base index with 0 at every target bit, by
    // inserting a zero bit at each target position (ascending so higher bits
    // shift up correctly).
    ulong base = g;
    for (uint j = 0u; j < k; ++j) {
        ulong mask = ((ulong)1 << tpos_sorted[j]) - 1;
        base = ((base & ~mask) << 1) | (base & mask);
    }

    // Controlled gate: the group acts only if all control bits are set. Control
    // qubits are non-target, so the bit is constant across the group.
    for (uint c = 0u; c < nc; ++c) {
        if (((base >> cpos[c]) & 1u) == 0u) return;
    }

    uint K = 1u << k;
    float2 amp[16];           // k <= 4 (fused) -> 2^k <= 16
    ulong idx[16];
    for (uint m = 0u; m < K; ++m) {
        ulong id = base;
        for (uint j = 0u; j < k; ++j) {
            uint v = (m >> (k - 1u - j)) & 1u;
            id |= (ulong)v << tpos[j];
        }
        idx[m] = id;
        amp[m] = state[id];
    }
    for (uint m = 0u; m < K; ++m) {
        float2 acc = float2(0.0, 0.0);
        for (uint c = 0u; c < K; ++c) {
            acc += cmul(M[m * K + c], amp[c]);
        }
        state[idx[m]] = acc;
    }
}
"""

_MAX_DIM = 1 << 15  # per-grid-dimension thread cap used when factoring the grid


@dataclass
class MetalState:
    """Statevector held in a single unified-memory MTLBuffer (complex64)."""

    buf: object  # MTLBuffer, length 2**n_qubits * 8 bytes
    n_qubits: int


class MetalBackend:
    """Metal-accelerated backend reaching 31-33 qubits via 64-bit indexing.

    Mirrors the CPU/MLX gate semantics exactly (differential-tested to 1e-5).
    Aimed at the large-n regime MLX cannot reach; at equal qubit counts it is
    not expected to beat MLX (both are memory-bandwidth bound).
    """

    def __init__(self, seed: int | None = None) -> None:
        if not _METAL_AVAILABLE:
            raise ImportError(
                "Metal backend unavailable. Install it with: "
                "pip install 'macquerel[metal]'\n"
                "Note: requires macOS on Apple Silicon with a Metal GPU."
            )
        self._rng = np.random.default_rng(seed)
        self._dev = Metal.MTLCreateSystemDefaultDevice()
        self._queue = self._dev.newCommandQueue()

        lib, err = self._dev.newLibraryWithSource_options_error_(_KERNEL_SRC, None, None)
        if lib is None:
            raise RuntimeError(f"Metal kernel compilation failed: {err}")
        self._pipelines = {}
        for name in ("diagonal", "dense"):
            fn = lib.newFunctionWithName_(name)
            pipe, perr = self._dev.newComputePipelineStateWithFunction_error_(fn, None)
            if pipe is None:
                raise RuntimeError(f"Pipeline build failed for {name!r}: {perr}")
            self._pipelines[name] = pipe

        self._classify_cache: dict[tuple, str] = {}

    # ---- buffer / readback helpers ---------------------------------------

    def _view(self, sv: MetalState) -> np.ndarray:
        """Zero-copy, writable complex64 NumPy view over the buffer contents."""
        nbytes = (2**sv.n_qubits) * 8
        mv = sv.buf.contents().as_buffer(nbytes)  # ty: ignore[unresolved-attribute]
        return np.frombuffer(mv, dtype=np.complex64)

    def _const(self, arr: np.ndarray):
        """A shared MTLBuffer holding a small constant array (matrix/index data)."""
        data = np.ascontiguousarray(arr).tobytes()
        # length must be > 0; callers pass at least one element.
        return self._dev.newBufferWithBytes_length_options_(
            data, len(data), Metal.MTLResourceStorageModeShared
        )

    def _grid(self, total: int):
        """Factor `total` (a power of two) into a 3D grid + threadgroup."""
        gx = min(total, _MAX_DIM)
        rem = total // gx
        gy = min(rem, _MAX_DIM)
        gz = rem // gy
        tg = min(gx, 256)
        return (gx, gy, gz), tg

    def _dispatch(self, name: str, total: int, buffers, scalars):
        """Encode + run one compute pass over `total` threads.

        `buffers`: list of (mtl_buffer, index). `scalars`: list of (uint32, index).
        """
        (gx, gy, gz), tg = self._grid(total)
        cb = self._queue.commandBuffer()
        enc = cb.computeCommandEncoder()
        enc.setComputePipelineState_(self._pipelines[name])
        for buf, idx in buffers:
            enc.setBuffer_offset_atIndex_(buf, 0, idx)
        for val, idx in scalars:
            enc.setBytes_length_atIndex_(np.array([val], dtype=np.uint32).tobytes(), 4, idx)
        enc.dispatchThreads_threadsPerThreadgroup_(
            Metal.MTLSizeMake(gx, gy, gz), Metal.MTLSizeMake(tg, 1, 1)
        )
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

    # ---- Backend protocol ------------------------------------------------

    def allocate(self, n_qubits: int, dtype=np.complex64) -> MetalState:
        nbytes = (2**n_qubits) * 8
        buf = self._dev.newBufferWithLength_options_(nbytes, Metal.MTLResourceStorageModeShared)
        if buf is None:
            raise MemoryError(f"Metal buffer allocation failed for {n_qubits} qubits")
        state = MetalState(buf=buf, n_qubits=n_qubits)
        view = self._view(state)
        view[:] = 0
        view[0] = 1.0
        return state

    def to_numpy(self, sv: MetalState) -> np.ndarray:
        return np.array(self._view(sv), dtype=np.complex64)

    def _classify(self, mat: np.ndarray) -> str:
        key = (mat.shape, mat.tobytes())
        kind = self._classify_cache.get(key)
        if kind is None:
            from macquerel.gates import classify

            kind = classify(mat)
            self._classify_cache[key] = kind
        return kind

    def apply_matrix(
        self,
        sv: MetalState,
        matrix: np.ndarray,
        targets: list[int],
        controls: list[int] | None = None,
    ) -> MetalState:
        mat = matrix.astype(np.complex64)
        n = sv.n_qubits
        k = len(targets)
        tpos = np.array([n - 1 - t for t in targets], dtype=np.uint32)

        if self._classify(mat) == "diagonal" and not controls:
            diag = np.ascontiguousarray(np.diag(mat).astype(np.complex64))
            self._dispatch(
                "diagonal",
                2**n,
                buffers=[(sv.buf, 0), (self._const(diag), 1), (self._const(tpos), 2)],
                scalars=[(k, 3)],
            )
            return sv

        # General path: dense / permutation / controlled, one thread per group.
        tpos_sorted = np.sort(tpos).astype(np.uint32)
        if controls:
            cpos = np.array([n - 1 - c for c in controls], dtype=np.uint32)
        else:
            cpos = np.zeros(1, dtype=np.uint32)  # dummy; nc=0 disables the loop
        nc = len(controls) if controls else 0
        self._dispatch(
            "dense",
            2 ** (n - k),
            buffers=[
                (sv.buf, 0),
                (self._const(mat.reshape(-1)), 1),
                (self._const(tpos), 2),
                (self._const(tpos_sorted), 3),
                (self._const(cpos), 4),
            ],
            scalars=[(k, 5), (nc, 6)],
        )
        return sv

    def measure(
        self,
        sv: MetalState,
        qubits: list[int],
        *,
        collapse: bool = True,
    ) -> list[int]:
        view = self._view(sv)
        npsv = np.array(view, dtype=np.complex64)
        outcomes = _cpu_measure(npsv, qubits, self._rng, collapse=collapse)
        if collapse:
            view[:] = npsv
        return outcomes

    def sample(self, sv: MetalState, qubits: list[int], shots: int) -> Counter:
        npsv = self.to_numpy(sv)
        n = sv.n_qubits
        probs2 = np.abs(npsv.reshape((2,) * n)) ** 2
        sum_axes = tuple(i for i in range(n) if i not in qubits)
        joint = np.sum(probs2, axis=sum_axes)
        order = sorted(range(len(qubits)), key=lambda i: qubits[i])
        joint = np.transpose(joint, order)
        flat = joint.reshape(-1)
        flat = flat / flat.sum()
        indices = self._rng.choice(2 ** len(qubits), size=shots, p=flat)
        result: Counter = Counter()
        for idx in indices:
            result[format(int(idx), f"0{len(qubits)}b")] += 1
        return result

    def abs2sum(self, sv: MetalState, qubits: list[int]) -> np.ndarray:
        n = sv.n_qubits
        probs = np.abs(self.to_numpy(sv).reshape((2,) * n)) ** 2
        sum_axes = tuple(i for i in range(n) if i not in qubits)
        return np.sum(probs, axis=sum_axes).reshape(-1)

    def expectation_pauli(self, sv: MetalState, pauli_strings) -> np.ndarray:
        from macquerel.backends.cpu import CPUBackend

        return CPUBackend().expectation_pauli(self.to_numpy(sv), pauli_strings)


def _cpu_measure(npsv, qubits, rng, *, collapse):
    """Single-amplitude measurement mirroring CPUBackend.measure but using `rng`."""
    n = int(np.log2(len(npsv)))
    state = npsv.reshape((2,) * n)
    probs2 = np.abs(state) ** 2
    outcomes = []
    for q in qubits:
        sum_axes = tuple(i for i in range(n) if i != q)
        marginal = np.sum(probs2, axis=sum_axes)
        p0, p1 = float(np.real(marginal[0])), float(np.real(marginal[1]))
        total = p0 + p1
        outcome = 0 if total < 1e-15 else int(rng.choice([0, 1], p=[p0 / total, p1 / total]))
        outcomes.append(outcome)
        if collapse:
            idx: list = [slice(None)] * n
            idx[q] = 1 - outcome
            state[tuple(idx)] = 0.0
            norm = np.sqrt(np.sum(np.abs(state) ** 2))
            if norm > 1e-15:
                state /= norm
            probs2 = np.abs(state) ** 2
    npsv[:] = state.reshape(-1)
    return outcomes
