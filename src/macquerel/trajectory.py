"""Quantum-trajectory (Monte-Carlo wavefunction) simulation (Step 37).

Noisy circuits without the density matrix's ``4**n`` cost: each trajectory is
an ordinary ``2**n`` statevector evolution in which every Kraus channel
``{K_k}`` applies exactly one operator, chosen with the Born probability
``p_k = <psi| K_k^dagger K_k |psi>``, followed by renormalization. Averaging
over trajectories reproduces the channel exactly in expectation —
``E[|psi><psi|] = sum_k K_k rho K_k^dagger`` (Molmer-Castin-Dalibard) — with
statistical error shrinking as ``1/sqrt(trajectories)``.

Memory is one statevector per trajectory, run sequentially, so noisy
simulation inherits the *statevector* ceilings (Metal 33 qubits) instead of
the density matrix's halved ones (n=16). The trade is variance for memory:
`DensityMatrixSimulator` is exact at small n; this simulator is stochastic
but reaches the full statevector range, reusing the same backends and
`ChannelOp`s unchanged.

Jump probabilities cost no state copies: every built-in channel (bit/phase
flip, depolarizing, amplitude/phase damping) has diagonal effect operators
``E_k = K_k^dagger K_k``, so all ``p_k`` come from one `abs2sum` marginal
over the channel's qubits — compatible with Metal's in-place state. Channels
with non-diagonal effects fall back to the channel-qubit reduced density
matrix, built on the host from a (zero-copy where possible) state view.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from macquerel.circuit import ChannelOp, Circuit, Gate, MeasureOp
from macquerel.compiler import fuse_gates
from macquerel.simulator import _make_backend, _select_backend

_EFFECT_ATOL = 1e-7


class TrajectorySimulator:
    """Stochastic noisy simulation over `trajectories` pure-state samples.

    Accepts the same circuits as `DensityMatrixSimulator` (gates, Kraus
    channels, measurements). `backend` / `dtype` / `seed` mirror `Simulator`;
    ``backend="auto"`` selects by the *statevector* qubit count. `seed` makes
    both the Kraus sampling and the measurement sampling reproducible.

    >>> qc = Circuit(2)
    >>> qc.h(0).cx(0, 1).depolarizing(0, 0.05).measure_all()
    >>> TrajectorySimulator(trajectories=200).run(qc, shots=1000)
    """

    def __init__(
        self,
        backend: str = "auto",
        dtype: str = "complex64",
        seed: int | None = None,
        trajectories: int = 100,
        batch_shots: int | str = "auto",
    ) -> None:
        if backend not in ("auto", "cpu", "mlx", "metal"):
            raise ValueError(
                f"Unknown backend: {backend!r}. Choose 'cpu', 'mlx', 'metal', or 'auto'."
            )
        if trajectories < 1:
            raise ValueError(f"trajectories must be >= 1, got {trajectories}")
        self.backend_name = backend
        self.dtype = dtype
        self.trajectories = trajectories
        self.batch_shots = batch_shots
        self._np_dtype = np.complex64 if dtype == "complex64" else np.complex128
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        # Step 34 pattern: unseeded runs reuse backend instances so pipeline
        # caches and buffer pools stay warm across trajectories.
        self._backends: dict[str, object] = {}

    # -- public API ----------------------------------------------------------

    def probabilities(self, circuit: Circuit, trajectories: int | None = None) -> np.ndarray:
        """Trajectory-averaged measurement probabilities, shape ``(2**n,)``."""
        n = circuit.n_qubits
        acc = np.zeros(2**n, dtype=np.float64)
        t_total = self._resolve_trajectories(trajectories)
        for _, backend, state in self._trajectories(circuit, t_total):
            acc += backend.abs2sum(state, list(range(n)))
        acc /= t_total
        return acc / acc.sum()

    def run(self, circuit: Circuit, shots: int = 1000, trajectories: int | None = None) -> Counter:
        """Sample measurement outcomes; mirrors `DensityMatrixSimulator.run`.

        `shots` are split as evenly as possible across the trajectories; at
        each `MeasureOp` a trajectory contributes its share of samples from
        its current pure state (no collapse). A circuit without any MeasureOp
        samples all qubits at the end.
        """
        t_total = self._resolve_trajectories(trajectories)
        shares = self._shot_shares(shots, t_total)
        result: Counter = Counter()

        def on_measure(t: int, backend, state, qubits: list[int]) -> None:
            if shares[t]:
                counts = backend.sample(state, qubits, shares[t], batch_shots=self.batch_shots)
                result.update(counts)

        measured = any(isinstance(op, MeasureOp) for op in circuit.ops)
        for t, backend, state in self._trajectories(circuit, t_total, on_measure=on_measure):
            if not measured and shares[t]:
                qubits = list(range(circuit.n_qubits))
                result.update(
                    backend.sample(state, qubits, shares[t], batch_shots=self.batch_shots)
                )
        return result

    def expectation_pauli(
        self, circuit: Circuit, pauli_strings, trajectories: int | None = None
    ) -> np.ndarray:
        """Trajectory-averaged ``<psi| P |psi>`` for each Pauli-string term."""
        t_total = self._resolve_trajectories(trajectories)
        acc: np.ndarray | None = None
        for _, backend, state in self._trajectories(circuit, t_total):
            vals = np.asarray(backend.expectation_pauli(state, pauli_strings), dtype=np.float64)
            acc = vals if acc is None else acc + vals
        assert acc is not None
        return acc / t_total

    # -- internals -------------------------------------------------------------

    def _resolve_trajectories(self, trajectories: int | None) -> int:
        t = self.trajectories if trajectories is None else trajectories
        if t < 1:
            raise ValueError(f"trajectories must be >= 1, got {t}")
        return t

    @staticmethod
    def _shot_shares(shots: int, t_total: int) -> list[int]:
        base, extra = divmod(shots, t_total)
        return [base + (1 if t < extra else 0) for t in range(t_total)]

    def _backend_for(self, name: str):
        if self._seed is not None:
            # One fresh, derived-seed backend per *call* (not per trajectory —
            # a fresh Metal backend per trajectory pinned a new state buffer
            # each time, ~6x the state at 30q): repeated runs stay
            # bit-identical because the whole RNG stream derives from `seed`,
            # and all trajectories of the call share the buffer pool.
            return _make_backend(name, self.dtype, int(self._rng.integers(2**32)))
        backend = self._backends.get(name)
        if backend is None:
            backend = _make_backend(name, self.dtype, None)
            self._backends[name] = backend
        return backend

    def _trajectories(self, circuit: Circuit, t_total: int, on_measure=None):
        """Yield ``(t, backend, final_state)`` for each of `t_total` trajectories.

        ``on_measure(t, backend, state, qubits)`` fires at every MeasureOp, in
        circuit order, during trajectory ``t``'s evolution.
        """
        n = circuit.n_qubits
        name = self.backend_name if self.backend_name != "auto" else _select_backend(n)
        fused = fuse_gates(circuit, backend=name)  # channels/measures are barriers
        if self._seed is not None:
            # Restart the stream per call so repeated calls on one seeded
            # simulator are bit-identical, mirroring `Simulator`.
            self._rng = np.random.default_rng(self._seed)
        backend = self._backend_for(name)
        state = None
        for t in range(t_total):
            state = self._fresh_state(backend, state, n)
            for op in fused.ops:
                if isinstance(op, Gate):
                    state = backend.apply_matrix(state, op.matrix, op.targets, op.controls or None)
                elif isinstance(op, ChannelOp):
                    state = self._apply_channel(backend, state, op, n)
                elif isinstance(op, MeasureOp) and on_measure is not None:
                    on_measure(t, backend, state, op.qubits)
            yield t, backend, state

    def _fresh_state(self, backend, prev, n: int):
        """A ``|0...0>`` state, reusing the previous trajectory's storage.

        Re-allocating per trajectory accumulated one state-sized footprint
        per trajectory on Metal (measured: released multi-GiB MTLBuffers are
        reclaimed lazily by the driver, and buffers past the pool's 1 GiB cap
        are never recycled), so the state is reset in place instead: through
        the zero-copy unified-memory view on Metal, by `fill` on the CPU's
        ndarray. MLX arrays are immutable; a fresh allocate goes through its
        own buffer pool.
        """
        if prev is not None:
            host = None
            if isinstance(prev, np.ndarray):
                host = prev
            elif hasattr(backend, "_view"):
                host = backend._view(prev)  # flushes pending work, zero-copy
            if host is not None:
                host.fill(0)
                host[0] = 1.0
                return prev
        return backend.allocate(n, self._np_dtype)

    def _apply_channel(self, backend, state, op: ChannelOp, n: int):
        """Apply one sampled Kraus operator and renormalize.

        ``p_k = <psi| E_k |psi>`` with ``E_k = K_k^dagger K_k``. When every
        effect is diagonal (all built-in channels), the probabilities are one
        `abs2sum` marginal over the channel qubits — no state copy, no host
        readback. Non-diagonal effects use the channel-qubit reduced density
        matrix from a host view of the state (zero-copy on CPU/Metal; this
        path costs a host pass over the state and only exotic `kraus(...)`
        channels take it).
        """
        qubits = op.qubits
        c = len(qubits)
        effects = [k.astype(np.complex128).conj().T @ k.astype(np.complex128) for k in op.kraus]

        if all(self._is_diagonal(e) for e in effects):
            marginal = backend.abs2sum(state, sorted(qubits))  # ascending qubit order
            # Effect diagonals follow op.qubits order; permute to ascending.
            order = np.argsort(qubits)
            p = np.array(
                [
                    float(
                        np.dot(
                            np.transpose(np.real(np.diag(e)).reshape((2,) * c), order).reshape(-1),
                            marginal,
                        )
                    )
                    for e in effects
                ]
            )
        else:
            view = self._host_view(backend, state).reshape((2,) * n)
            mat = np.moveaxis(view, qubits, range(c)).reshape(2**c, -1)
            rho_c = mat @ mat.conj().T
            p = np.array([float(np.real(np.trace(e @ rho_c))) for e in effects])

        p = np.clip(p, 0.0, None)
        total = p.sum()
        if total <= 0.0:
            raise ValueError(f"channel {op.name!r} produced a vanishing state norm")
        p /= total
        k = int(self._rng.choice(len(p), p=p))

        state = backend.apply_matrix(state, op.kraus[k].astype(self._np_dtype), qubits, None)
        # Renormalize: a scalar is a diagonal "gate", the cheapest full pass.
        scale = np.eye(2, dtype=self._np_dtype) / np.sqrt(p[k])
        return backend.apply_matrix(state, scale, [qubits[0]], None)

    @staticmethod
    def _is_diagonal(mat: np.ndarray) -> bool:
        off = mat.copy()
        np.fill_diagonal(off, 0)
        return bool(np.allclose(off, 0, atol=_EFFECT_ATOL))

    @staticmethod
    def _host_view(backend, state) -> np.ndarray:
        """Host array over the state, zero-copy where the backend allows."""
        if isinstance(state, np.ndarray):
            return state
        if hasattr(backend, "_view"):
            return backend._view(state)
        return backend.to_numpy(state)
