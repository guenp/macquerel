"""Density-matrix simulation with Kraus-operator noise channels (v0.3).

The density matrix of an n-qubit system is carried as its row-major
vectorization: a ``4**n`` complex64 array that the existing statevector
backends treat as an ordinary ``2n``-qubit state. Ket axes occupy qubits
``0..n-1`` of the doubled space, bra axes qubits ``n..2n-1``, so
``vec(rho)[i * 2**n + j] = rho[i, j]``. Everything then reduces to backend
``apply_matrix`` calls the CPU/MLX/Metal backends already implement:

- a unitary ``rho -> U rho U^dagger`` is U on the ket axes followed by
  ``conj(U)`` on the bra axes (controls shift with their targets);
- a Kraus channel ``rho -> sum_k K_k rho K_k^dagger`` is the single dense
  superoperator ``sum_k K_k (x) conj(K_k)`` applied to the channel's ket axes
  + bra axes in one call (`noise.channel_superoperator`);
- measurement probabilities are the diagonal ``rho_ii`` — a stride-``2**n+1``
  slice of the vectorization, read without materializing the full matrix on
  the host where the backend allows it (CPU arrays and Metal's zero-copy
  unified-memory view).

Memory: an n-qubit density matrix costs exactly what a 2n-qubit statevector
costs (``4**n * 8`` bytes complex64), so n qubits of noisy simulation inherit
the 2n-qubit backend ceilings — MLX's int32 ShapeElem caps it at n=15, Metal
reaches n=16 (32 GiB) on a 128 GiB machine. Backend auto-selection reuses the
statevector tiers at the doubled count.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from macquerel.circuit import ChannelOp, Circuit, Gate, MeasureOp
from macquerel.compiler import _resolve_fusion_width, fuse_gates
from macquerel.noise import channel_superoperator
from macquerel.simulator import _make_backend, _select_backend

# MLX rejects arrays of >= 2**31 elements (int32 ShapeElem), and the Metal
# backend is validated to 33 statevector qubits; at the doubled count those
# ceilings land here. CPU has no hard cap, only patience and RAM.
_BACKEND_MAX_DM_QUBITS = {"mlx": 15, "metal": 16}


class DensityMatrixSimulator:
    """Noisy circuit simulation via the vectorized density matrix.

    Accepts the same circuits as `Simulator` plus Kraus channels
    (`Circuit.bit_flip`, `.phase_flip`, `.depolarizing`, `.amplitude_damping`,
    `.phase_damping`, or arbitrary `.kraus`). `backend` is one of ``"auto"``,
    ``"cpu"``, ``"mlx"``, ``"metal"``; ``"auto"`` selects by the *doubled*
    qubit count, since an n-qubit density matrix moves as many bytes as a
    2n-qubit statevector. `seed` makes sampling reproducible.

    >>> qc = Circuit(2)
    >>> qc.h(0).cx(0, 1).depolarizing(0, 0.05).measure_all()
    >>> DensityMatrixSimulator().run(qc, shots=1000)
    """

    def __init__(
        self,
        backend: str = "auto",
        dtype: str = "complex64",
        seed: int | None = None,
    ) -> None:
        if backend not in ("auto", "cpu", "mlx", "metal"):
            raise ValueError(
                f"Unknown backend: {backend!r}. Choose 'cpu', 'mlx', 'metal', or 'auto'."
            )
        self.backend_name = backend
        self.dtype = dtype
        self._np_dtype = np.complex64 if dtype == "complex64" else np.complex128
        self._rng = np.random.default_rng(seed)
        # Like Simulator (Step 34): reuse backend instances across calls so the
        # Metal buffer pool and pipeline caches stay warm. Sampling randomness
        # lives in self._rng, not the backend, so reuse is seed-safe here.
        self._backends: dict[str, object] = {}

    # -- public API ----------------------------------------------------------

    def density_matrix(self, circuit: Circuit) -> np.ndarray:
        """The final density matrix, shape ``(2**n, 2**n)`` complex.

        This materializes the full matrix on the host — ``4**n * 8`` bytes on
        top of the backend's state. Prefer `probabilities` / `run` /
        `expectation_pauli` for large n.
        """
        backend, state = self._evolve(circuit)
        n = circuit.n_qubits
        return backend.to_numpy(state).reshape(2**n, 2**n)

    def probabilities(self, circuit: Circuit) -> np.ndarray:
        """Measurement probabilities ``p(i) = rho_ii``, shape ``(2**n,)``.

        Reads only the ``2**n`` diagonal elements of the vectorized state —
        no full-matrix host copy on the CPU and Metal backends.
        """
        backend, state = self._evolve(circuit)
        return self._diag_probs(backend, state, circuit.n_qubits)

    def run(self, circuit: Circuit, shots: int = 1000) -> Counter:
        """Sample measurement outcomes; mirrors `Simulator.run` semantics.

        At each `MeasureOp` the listed qubits are sampled `shots` times from
        the diagonal of the current density matrix (no collapse); counters
        from multiple MeasureOps are summed. A circuit without any MeasureOp
        samples all qubits at the end.
        """
        n = circuit.n_qubits
        counters: list[Counter] = []

        def on_measure(backend, state, qubits: list[int]) -> None:
            probs = self._diag_probs(backend, state, n)
            counters.append(self._sample(probs, n, qubits, shots))

        backend, state = self._evolve(circuit, on_measure=on_measure)
        if not counters:
            probs = self._diag_probs(backend, state, n)
            return self._sample(probs, n, list(range(n)), shots)
        if len(counters) == 1:
            return counters[0]
        result: Counter = Counter()
        for c in counters:
            result.update(c)
        return result

    def expectation_pauli(self, circuit: Circuit, pauli_strings) -> np.ndarray:
        """``tr(rho P)`` for each ``(coeff, [(pauli_char, qubit), ...])`` term.

        ``tr(rho P) = sum_i vec(rho P)[i*2**n + i]`` and
        ``vec(rho P) = (I (x) P^T) vec(rho)``: each Pauli is applied
        *transposed* to the bra axis of a host copy, then the diagonal is
        summed. One full host readback total, plus one copy per term.
        """
        from macquerel.backends.cpu import CPUBackend
        from macquerel.gates import I as I_gate
        from macquerel.gates import X, Y, Z

        pauli_t = {"X": X().T, "Y": Y().T, "Z": Z().T, "I": I_gate().T}
        backend, state = self._evolve(circuit)
        n = circuit.n_qubits
        vec = backend.to_numpy(state)
        cpu = CPUBackend()
        results = []
        for coeff, terms in pauli_strings:
            work = vec.copy()
            for pauli_char, qubit in terms:
                work = cpu.apply_matrix(work, pauli_t[pauli_char], [n + qubit])
            results.append(coeff * float(np.real(np.sum(work[:: 2**n + 1]))))
        return np.array(results)

    def purity(self, circuit: Circuit) -> float:
        """``tr(rho^2)`` — 1.0 for pure states, ``1/2**n`` when fully mixed.

        For a Hermitian rho, ``tr(rho^2) = sum_ij |rho_ij|^2``, the squared
        norm of the vectorized state: one BLAS dot over a host view, with no
        state-sized temporary.
        """
        backend, state = self._evolve(circuit)
        vec = self._host_view(backend, state)
        return float(np.real(np.vdot(vec, vec)))

    # -- internals -------------------------------------------------------------

    def _backend_name_for(self, n_qubits: int) -> str:
        name = self.backend_name
        if name == "auto":
            name = _select_backend(2 * n_qubits)
        cap = _BACKEND_MAX_DM_QUBITS.get(name)
        if cap is not None and n_qubits > cap:
            raise ValueError(
                f"{name} backend caps density-matrix simulation at {cap} qubits "
                f"(the doubled state exceeds the backend's statevector ceiling); "
                f"got {n_qubits}"
            )
        return name

    def _get_backend(self, name: str):
        backend = self._backends.get(name)
        if backend is None:
            backend = _make_backend(name, self.dtype, None)
            self._backends[name] = backend
        return backend

    def _evolve(self, circuit: Circuit, on_measure=None):
        """Run `circuit` over the vectorized density matrix.

        Returns ``(backend, state)`` with `state` the backend-native doubled
        statevector. `on_measure(backend, state, qubits)` fires at every
        MeasureOp, in circuit order.
        """
        n = circuit.n_qubits
        name = self._backend_name_for(n)
        backend = self._get_backend(name)
        # Fusion sees channels and measurements as barriers. The width default
        # is resolved at the *doubled* qubit count — each fused gate is applied
        # over the 4**n state (twice), so 2n is the count the per-(backend, n)
        # width table was measured against.
        fused = fuse_gates(circuit, max_fused_qubits=_resolve_fusion_width(name, 2 * n))

        state = backend.allocate(2 * n, self._np_dtype)  # vec(|0..0><0..0|)
        for op in fused.ops:
            if isinstance(op, Gate):
                # rho -> U rho U^dagger: U on ket axes, conj(U) on bra axes.
                state = backend.apply_matrix(state, op.matrix, op.targets, op.controls or None)
                state = backend.apply_matrix(
                    state,
                    op.matrix.conj(),
                    [n + t for t in op.targets],
                    [n + c for c in op.controls] if op.controls else None,
                )
            elif isinstance(op, ChannelOp):
                superop = channel_superoperator(op.kraus)
                state = backend.apply_matrix(
                    state, superop, op.qubits + [n + q for q in op.qubits], None
                )
            elif isinstance(op, MeasureOp) and on_measure is not None:
                on_measure(backend, state, op.qubits)
        return backend, state

    @staticmethod
    def _host_view(backend, state) -> np.ndarray:
        """The vectorized state as a host array, zero-copy where possible.

        CPU states *are* NumPy arrays; Metal exposes a zero-copy view over its
        unified-memory buffer (`_view` is also its own sync point); MLX pays
        one full readback (it is the fallback backend, capped at n=15).
        """
        if isinstance(state, np.ndarray):
            return state
        if hasattr(backend, "_view"):
            return backend._view(state)
        return backend.to_numpy(state)

    def _diag_probs(self, backend, state, n: int) -> np.ndarray:
        """``p(i) = Re(rho_ii)``, clipped and normalized, shape ``(2**n,)``.

        The diagonal is the stride-``2**n + 1`` slice of the vectorization;
        slicing the host view copies out only ``2**n`` elements. Tiny negative
        or complex residue from float accumulation is clipped away.
        """
        vec = self._host_view(backend, state)
        diag = np.array(vec[:: 2**n + 1], dtype=np.complex128)
        probs = np.clip(diag.real, 0.0, None)
        total = probs.sum()
        if total < 1e-15:
            raise ValueError("density matrix has vanished trace; cannot sample")
        return probs / total

    def _sample(self, probs: np.ndarray, n: int, qubits: list[int], shots: int) -> Counter:
        """Sample bitstrings over `qubits` from the full diagonal distribution."""
        joint = probs.reshape((2,) * n)
        sum_axes = tuple(i for i in range(n) if i not in qubits)
        if sum_axes:
            joint = joint.sum(axis=sum_axes)
        # Remaining axes follow ascending qubit order; reorder to caller order.
        order = sorted(range(len(qubits)), key=lambda i: qubits[i])
        joint = np.transpose(joint, order).reshape(-1)
        joint = joint / joint.sum()
        indices = self._rng.choice(joint.size, size=shots, p=joint)
        result: Counter = Counter()
        width = len(qubits)
        for idx in indices:
            result[format(int(idx), f"0{width}b")] += 1
        return result
