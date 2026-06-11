from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

try:
    import mlx.core as mx  # ty: ignore[unresolved-import]

    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False


# Step 24: kick off (asynchronous) evaluation of the state every this many
# gate applications, but only for states of at least _ASYNC_EVAL_MIN_QUBITS.
# P1 (defer evaluation) deliberately stopped eval-ing per gate, but with *no*
# synchronization between observation boundaries a depth-d circuit keeps O(d)
# full-width temporaries in the lazy graph: at 26-28q the observed peak is
# >= ~16x the state size, which drives the machine into swap (the 28q cliff,
# see docs/plan.md baseline). async_eval keeps the pipeline full while letting
# MLX retire and free earlier intermediates, bounding the working set. Small
# states keep the fully-lazy P1 behavior - their graphs are tiny and the win
# there came precisely from not synchronizing.
_ASYNC_EVAL_INTERVAL = 16
_ASYNC_EVAL_MIN_QUBITS = 24


@dataclass
class MLXState:
    """Statevector held as a single complex64 mx.array (P4: native complex storage).

    `perm` tracks the physical axis order of the (2,)*n view (Step 23):
    ``perm[axis] = logical qubit stored on that axis``. Dense gates leave their
    contracted axes in front rather than paying a full transpose+copy to restore
    canonical order after every gate; the permutation is folded in once, at
    readback (`to_numpy` / measure / sample). ``None`` means canonical order.
    """

    data: mx.array  # shape (2**n,), complex64
    n_qubits: int
    perm: tuple[int, ...] | None = None


def _perm_gather_kernel(data, src):
    """Permutation gate: gather amplitudes by source index."""
    return data[src]


# Step 33: custom dense-gate Metal kernel (mx.fast.metal_kernel).
#
# mx.tensordot applies a dense k-qubit gate as matmul over a *permuted copy*
# of the state — its internal transpose of the input is the dominant cost on
# scattered-target circuits (and the reason Step 23 only tracked the output
# axis order). This kernel applies the gate the way the native Metal backend
# does (Step 25): one thread owns the 2**k amplitudes of one group (all basis
# states sharing the same non-target bits), reads them, multiplies by the
# gate matrix, and writes them back to the same positions — no permutation of
# the state in either direction. MLX kernel inputs are const device (issue
# #2547), so unlike the native backend the result lands in a fresh output
# buffer (double-buffered, like every other MLX gate path).
#
# The gate width K and control count NC are baked into the source per
# (k, nc): the m/j loops unroll and amp[]/idx[] live in registers. Threads
# whose control bits are not all set must still copy their group through to
# the output buffer.
_MLX_DENSE_KERNEL_TEMPLATE = r"""
    #define K_FIXED {k}u
    #define KDIM (1u << K_FIXED)
    #define NC {nc}u

    uint3 gid = thread_position_in_grid;
    uint3 gdim = threads_per_grid;
    ulong g = ((ulong)gid.z * gdim.y + gid.y) * gdim.x + gid.x;

    // Expand the group id into a base index with 0 at every target bit
    // (ascending target bit positions so higher bits shift up correctly).
    ulong base = g;
    for (uint j = 0u; j < K_FIXED; ++j) {{
        ulong mask = ((ulong)1 << tpos_sorted[j]) - 1;
        base = ((base & ~mask) << 1) | (base & mask);
    }}

    bool act = true;
    for (uint c = 0u; c < NC; ++c) {{
        if (((base >> cpos[c]) & 1u) == 0u) {{ act = false; break; }}
    }}

    complex64_t amp[KDIM];
    ulong idx[KDIM];
    for (uint m = 0u; m < KDIM; ++m) {{
        ulong id = base;
        for (uint j = 0u; j < K_FIXED; ++j) {{
            uint v = (m >> (K_FIXED - 1u - j)) & 1u;
            id |= (ulong)v << tpos[j];
        }}
        idx[m] = id;
        amp[m] = state[id];
    }}
    if (act) {{
        for (uint m = 0u; m < KDIM; ++m) {{
            complex64_t acc = complex64_t(0.0f, 0.0f);
            for (uint c = 0u; c < KDIM; ++c) {{
                acc += M[m * KDIM + c] * amp[c];
            }}
            out[idx[m]] = acc;
        }}
    }} else {{
        for (uint m = 0u; m < KDIM; ++m) {{
            out[idx[m]] = amp[m];
        }}
    }}
"""

# Beyond this gate width the per-thread amp[]/idx[] arrays (2**k registers)
# start spilling; fall back to the tensordot path. Fusion caps dense gates at
# 4 qubits, so the kernel covers everything the compiler emits.
_MLX_DENSE_KERNEL_MAX_K = 6


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
        # P8: compile the hot permutation gather so MLX fuses the index math +
        # gather into one kernel and caches the trace per input shape.
        self._perm_kernel = mx.compile(_perm_gather_kernel)
        # Step 33: per-(gate width, control count) specialized dense kernels.
        self._dense_kernels: dict[tuple[int, int], object] = {}
        # Step 24: gate applications since the last async_eval kick.
        self._applies_since_eval = 0

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

    @staticmethod
    def _perm_of(sv: MLXState) -> tuple[int, ...]:
        return sv.perm if sv.perm is not None else tuple(range(sv.n_qubits))

    @staticmethod
    def _phys(sv: MLXState, qubits: list[int]) -> list[int]:
        """Map logical qubits to their current physical axes (Step 23)."""
        if sv.perm is None:
            return list(qubits)
        pos = {q: a for a, q in enumerate(sv.perm)}
        return [pos[q] for q in qubits]

    def _canonicalize(self, sv: MLXState) -> None:
        """Fold the deferred axis permutation into the data (one transpose).

        Mutates `sv` so repeated readbacks pay the transpose once. This is the
        only place the Step 23 permutation is materialized.
        """
        perm = self._perm_of(sv)
        if perm == tuple(range(sv.n_qubits)):
            sv.perm = None
            return
        view = sv.data.reshape((2,) * sv.n_qubits)
        # New axis i must hold logical qubit i, currently on axis perm.index(i).
        order = [perm.index(i) for i in range(sv.n_qubits)]
        sv.data = mx.transpose(view, order).reshape(-1)
        sv.perm = None

    def to_numpy(self, sv: MLXState) -> np.ndarray:
        self._canonicalize(sv)
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
            new = self._apply_diagonal(sv, mat, targets)
        elif kind == "permutation" and not controls:
            new = self._apply_permutation(sv, mat, targets)
        else:
            new = self._apply_general(sv, mat, targets, controls)
        self._maybe_async_eval(new)
        return new

    def _maybe_async_eval(self, sv: MLXState) -> None:
        """Step 24: periodically start evaluating the state without blocking."""
        if sv.n_qubits < _ASYNC_EVAL_MIN_QUBITS:
            return
        self._applies_since_eval += 1
        if self._applies_since_eval >= _ASYNC_EVAL_INTERVAL:
            mx.async_eval(sv.data)
            self._applies_since_eval = 0

    def _apply_diagonal(self, sv: MLXState, matrix: np.ndarray, targets: list[int]) -> MLXState:
        """Diagonal gate: broadcast elementwise phase multiply (layout-preserving).

        Step 32: instead of materializing a full 2**n gather table (k shift/or
        passes to build a per-amplitude gate-row index, then a full-width
        gather of the phase), reshape the state so each target qubit gets its
        own length-2 axis — with the gaps between targets collapsed into
        single axes — and broadcast-multiply by the (2,)*k diagonal. One
        elementwise kernel over the state, no index table.
        """
        n = sv.n_qubits
        k = len(targets)
        phys = self._phys(sv, targets)
        diag = np.diag(matrix).astype(np.complex64).reshape((2,) * k)
        # diag axes follow gate target order; sort them to ascending physical
        # axis order so they line up with the state view built below.
        diag = np.ascontiguousarray(np.transpose(diag, tuple(np.argsort(phys))))
        # Compact state view: one axis per target qubit, gaps collapsed, so the
        # broadcast stays <= 2k+1 dimensional regardless of n.
        state_shape: list[int] = []
        diag_shape: list[int] = []
        prev = 0
        for a in sorted(phys):
            if a > prev:
                state_shape.append(1 << (a - prev))
                diag_shape.append(1)
            state_shape.append(2)
            diag_shape.append(2)
            prev = a + 1
        if prev < n:
            state_shape.append(1 << (n - prev))
            diag_shape.append(1)
        diag_mx = mx.array(diag.reshape(diag_shape))
        new_data = (sv.data.reshape(state_shape) * diag_mx).reshape(-1)
        return MLXState(data=new_data, n_qubits=n, perm=sv.perm)

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
        phys = self._phys(sv, targets)

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
        for j, t in enumerate(phys):
            bit = (indices >> (n - 1 - t)) & one
            out_row = out_row | (bit << (k - 1 - j))

        # Map each output row to its source input row.
        in_row = mx.array(gate_perm)[out_row]

        # Source index = output index with its target bits rewritten to in_row.
        src = indices
        for j, t in enumerate(phys):
            shift = n - 1 - t
            clear_mask = mx.array(((1 << n) - 1) ^ (1 << shift), dtype=mx.uint32)
            bit = (in_row >> (k - 1 - j)) & one
            src = (src & clear_mask) | (bit << shift)

        new_data = self._perm_kernel(sv.data, src)
        if has_phase:
            # Multiply each amplitude by its row's nonzero entry (phase gather).
            new_data = mx.array(row_phase)[out_row] * new_data
        return MLXState(data=new_data, n_qubits=n, perm=sv.perm)

    def _tensordot_front(
        self,
        data: mx.array,
        mat: np.ndarray,
        phys_targets: list[int],
        n: int,
    ) -> mx.array:
        """Contract a dense k-qubit matrix against the given physical axes.

        Single complex64 tensordot (vs four real tensordots in the SoA layout).
        The result keeps tensordot's natural output order: contracted axes in
        *front* (gate target order), remaining axes after them in their prior
        relative order. Step 23: the transpose+copy back to canonical order
        that used to follow every dense gate is gone — the caller records the
        new axis permutation on the state instead, and it is materialized once
        at readback (`_canonicalize`).
        """
        k = len(phys_targets)
        mat_mx = mx.array(mat.astype(np.complex64)).reshape((2,) * (2 * k))
        state = data.reshape((2,) * n)
        input_axes = list(range(k, 2 * k))
        return mx.tensordot(mat_mx, state, axes=[input_axes, phys_targets])

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

        Unlike the uncontrolled dense path this *is* layout-preserving: the
        gated tensor is transposed back to the state's current axis order so
        the mx.where select lines up element-by-element with `sv.data`.
        """
        n = sv.n_qubits
        phys = self._phys(sv, targets)
        out = self._tensordot_front(sv.data, matrix, phys, n)

        remaining = [i for i in range(n) if i not in phys]
        dest = phys + remaining
        inv_perm = [0] * n
        for new_pos, old_pos in enumerate(dest):
            inv_perm[old_pos] = new_pos
        gated = mx.transpose(out, inv_perm).reshape(-1)

        indices = self._arange(n)
        one = self._one
        mask = None
        for c in self._phys(sv, controls):
            bit = (indices >> (n - 1 - c)) & one
            cond = bit == one
            mask = cond if mask is None else (mask & cond)  # ty: ignore[unsupported-operator]

        new_data = mx.where(mask, gated, sv.data)  # ty: ignore[invalid-argument-type]
        return MLXState(data=new_data, n_qubits=n, perm=sv.perm)

    def _dense_kernel(self, k: int, nc: int):
        """The Step 33 dense-gate kernel specialized to (gate width, #controls)."""
        key = (k, nc)
        kernel = self._dense_kernels.get(key)
        if kernel is None:
            kernel = mx.fast.metal_kernel(
                name=f"macquerel_dense_k{k}_nc{nc}",
                input_names=["state", "M", "tpos", "tpos_sorted", "cpos"],
                output_names=["out"],
                source=_MLX_DENSE_KERNEL_TEMPLATE.format(k=k, nc=nc),
            )
            self._dense_kernels[key] = kernel
        return kernel

    def _apply_dense_kernel(
        self,
        sv: MLXState,
        mat: np.ndarray,
        targets: list[int],
        controls: list[int],
    ) -> MLXState:
        """Dense/controlled gate via the custom group-per-thread kernel (Step 33).

        Layout-preserving: each thread rewrites its own group in place (well,
        into the same positions of the output buffer), so `perm` carries over
        unchanged — no tensordot input permutation, no Step 23 perm growth.
        """
        n = sv.n_qubits
        k = len(targets)
        phys = self._phys(sv, targets)
        cphys = self._phys(sv, controls) if controls else []
        # Bit position of physical axis a in the linear index (axis 0 = MSB).
        tpos = np.array([n - 1 - a for a in phys], dtype=np.uint32)
        tpos_sorted = np.sort(tpos)
        cpos = (
            np.array([n - 1 - a for a in cphys], dtype=np.uint32)
            if cphys
            else np.zeros(1, dtype=np.uint32)  # dummy; NC=0 disables the loop
        )
        total = 2 ** (n - k)
        kernel = self._dense_kernel(k, len(cphys))
        (out,) = kernel(  # ty: ignore[call-non-callable]
            inputs=[
                sv.data,
                mx.array(mat.reshape(-1)),
                mx.array(tpos),
                mx.array(tpos_sorted),
                mx.array(cpos),
            ],
            grid=(total, 1, 1),
            threadgroup=(min(total, 256), 1, 1),
            output_shapes=[sv.data.shape],
            output_dtypes=[mx.complex64],
        )
        return MLXState(data=out, n_qubits=n, perm=sv.perm)

    def _apply_general(
        self,
        sv: MLXState,
        matrix: np.ndarray,
        targets: list[int],
        controls: list[int] | None,
    ) -> MLXState:
        """General dense gate (Step 33: custom group-per-thread Metal kernel;
        tensordot fallback for gates wider than the kernel's register budget)."""
        n = sv.n_qubits
        mat = matrix.astype(np.complex64)

        if len(targets) <= _MLX_DENSE_KERNEL_MAX_K:
            return self._apply_dense_kernel(sv, mat, targets, controls or [])

        if controls:
            return self._apply_controlled(sv, mat, targets, controls)

        perm = self._perm_of(sv)
        phys = self._phys(sv, targets)
        out = self._tensordot_front(sv.data, mat, phys, n)
        phys_set = set(phys)
        new_perm = tuple(targets) + tuple(q for a, q in enumerate(perm) if a not in phys_set)
        if new_perm == tuple(range(n)):
            new_perm = None  # canonical; keep the fast no-perm path
        return MLXState(data=out.reshape(-1), n_qubits=n, perm=new_perm)

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
        self._canonicalize(sv)
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
        self._canonicalize(sv)
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
