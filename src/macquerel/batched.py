"""Batched small-circuit simulation (Step 31 / v0.3 `BatchedSimulator`).

The small-n regime is dispatch-bound: a 12-qubit state is a few KB, so the
fixed per-run costs (Python op dispatch, kernel launches, GPU sync) dwarf the
arithmetic, and running a parameter sweep as N independent `Simulator` calls
pays those costs N times. `BatchedSimulator` packs the sweep into *batched*
array ops instead: circuits that share a structure (same gate positions,
targets, and controls — e.g. one VQE/QML ansatz at many parameter settings)
are evolved together as one ``(B, 2**n)`` state tensor, with each gate
position applied across the whole batch in a single batched matmul (or a
single broadcast multiply when every matrix at that position is diagonal).
One launch per gate position instead of one per circuit x gate.

Circuits with differing structures still work: the batch is grouped by
structure signature and each group is evolved batched; a group of one
degenerates to the single-circuit path. All circuits must share `n_qubits`.

Backends: ``cpu`` (NumPy) and ``mlx`` (lazy GPU graph — the whole batched
circuit evaluates in a handful of fused kernels). ``auto`` routes on the
*total* problem size ``log2(B) + n``, since a batch behaves like one state
that big. The native Metal backend is not offered here: its kernels are
specialized for single huge states, which is the opposite regime.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import cast

import numpy as np

from macquerel.circuit import ChannelOp, Circuit, Gate, MeasureOp

try:
    import mlx.core as mx  # ty: ignore[unresolved-import]

    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False


def _signature(circuit: Circuit) -> tuple:
    """Structure signature: everything about the op list except the matrices."""
    sig = []
    for op in circuit.ops:
        if isinstance(op, Gate):
            sig.append(("g", tuple(op.targets), tuple(op.controls)))
        else:
            sig.append(("m", tuple(op.qubits)))
    return tuple(sig)


def _lift_controls(mats: np.ndarray, n_controls: int) -> np.ndarray:
    """Lift batched target matrices to act on (controls + targets) unitarily.

    With the control qubits as the most-significant bits of the lifted gate,
    the all-controls-set subspace is the bottom-right block: the lifted matrix
    is the identity except for that block, which holds the original gate.
    """
    b, d, _ = mats.shape
    dim = d * (2**n_controls)
    out = np.zeros((b, dim, dim), dtype=mats.dtype)
    idx = np.arange(dim - d)
    out[:, idx, idx] = 1.0
    out[:, dim - d :, dim - d :] = mats
    return out


class _Engine:
    """The array ops the batched evolution needs, over NumPy or MLX.

    `xp` is the array module (numpy or mlx.core); both spell transpose /
    reshape / matmul / swapaxes the same way, so the batched gate application
    is written once. States are carried as ``(B, 2**n)`` complex64 arrays in
    the module's native type.
    """

    def __init__(self, xp) -> None:
        self.xp = xp

    def allocate(self, batch: int, n: int):
        sv = np.zeros((batch, 2**n), dtype=np.complex64)
        sv[:, 0] = 1.0
        return self.xp.array(sv) if self.xp is not np else sv

    def apply(self, states, mats: np.ndarray, targets: list[int], n: int):
        """Apply per-circuit matrices at one gate position across the batch.

        `states`: (B, 2**n). `mats`: (B, 2**k, 2**k) NumPy complex64.
        """
        xp = self.xp
        b = states.shape[0]
        k = len(targets)

        diags = self._all_diagonal(mats)
        if diags is not None:
            return self._apply_diagonal(states, diags, targets, n)

        view = states.reshape((b,) + (2,) * n)
        rest = [1 + i for i in range(n) if i not in targets]
        tax = [1 + t for t in targets]
        order = [0, *rest, *tax]
        moved = xp.transpose(view, order).reshape((b, -1, 2**k))
        # out[b, r, i] = sum_j moved[b, r, j] * M[b, i, j]
        out = xp.matmul(moved, xp.swapaxes(xp.array(mats), 1, 2))
        out = out.reshape((b,) + (2,) * n)
        inv = [0] * (n + 1)
        for new_pos, old_pos in enumerate(order):
            inv[old_pos] = new_pos
        return xp.transpose(out, inv).reshape((b, -1))

    @staticmethod
    def _all_diagonal(mats: np.ndarray) -> np.ndarray | None:
        """The (B, 2**k) diagonals if every matrix in the batch is diagonal."""
        d = mats.shape[1]
        off = mats[:, ~np.eye(d, dtype=bool)]
        if not np.allclose(off, 0.0, atol=1e-7):
            return None
        return mats[:, np.eye(d, dtype=bool)]

    def _apply_diagonal(self, states, diags: np.ndarray, targets: list[int], n: int):
        """Broadcast phase multiply across the batch (cf. Step 32)."""
        xp = self.xp
        b = states.shape[0]
        k = len(targets)
        d = diags.reshape((b,) + (2,) * k)
        # Diagonal axes follow gate target order; sort to ascending qubit order.
        d = np.ascontiguousarray(np.transpose(d, [0, *(1 + np.argsort(targets))]))
        target_set = set(targets)
        shape = tuple([b] + [2 if i in target_set else 1 for i in range(n)])
        view = states.reshape((b,) + (2,) * n)
        out = view * xp.array(d.reshape(shape))
        return out.reshape((b, -1))

    def probabilities(self, states, qubits: list[int], n: int) -> np.ndarray:
        """Marginal probabilities (B, 2**len(qubits)) over `qubits`, on host."""
        xp = self.xp
        probs = xp.abs(states.reshape((states.shape[0],) + (2,) * n)) ** 2
        sum_axes = tuple(1 + i for i in range(n) if i not in qubits)
        joint = xp.sum(probs, axis=sum_axes) if sum_axes else probs
        # joint axes follow ascending qubit order; reorder to caller order.
        in_state_order = sorted(range(len(qubits)), key=lambda i: qubits[i])
        order = [0] + [1 + i for i in in_state_order]
        joint = xp.transpose(joint, order)
        joint = joint.reshape((states.shape[0], -1))
        return np.array(joint, dtype=np.float64)

    def to_numpy(self, states) -> np.ndarray:
        if self.xp is np:
            return states
        return np.array(states).astype(np.complex64)


class BatchedSimulator:
    """Simulate many same-width circuits as one batched evolution.

    >>> circuits = [ansatz(theta) for theta in thetas]
    >>> BatchedSimulator().statevectors(circuits)  # (len(thetas), 2**n)
    >>> BatchedSimulator().run(circuits, shots=1000)  # list[Counter]
    """

    def __init__(
        self,
        backend: str = "auto",
        dtype: str = "complex64",
        seed: int | None = None,
    ) -> None:
        if backend not in ("auto", "cpu", "mlx"):
            raise ValueError(
                f"Unknown batched backend: {backend!r}. Choose 'cpu', 'mlx', or 'auto'."
            )
        if backend == "mlx" and not _MLX_AVAILABLE:
            raise ImportError(
                "mlx is not installed. Install it with: pip install mlx\n"
                "Note: mlx requires macOS on Apple Silicon (M1 or later)."
            )
        self.backend_name = backend
        self.dtype = dtype
        self._rng = np.random.default_rng(seed)

    def _engine_for(self, batch: int, n: int) -> _Engine:
        name = self.backend_name
        if name == "auto":
            # A batch of B states of n qubits moves as many bytes as one
            # (log2(B) + n)-qubit state; route it the way auto-select would
            # route that state.
            from macquerel.simulator import _resolve_cpu_max

            total = n + max(0, math.ceil(math.log2(max(batch, 1))))
            name = "mlx" if (_MLX_AVAILABLE and total > _resolve_cpu_max()) else "cpu"
        return _Engine(mx if name == "mlx" else np)

    # -- public API --------------------------------------------------------

    def statevectors(self, circuits: list[Circuit]) -> np.ndarray:
        """Final statevectors, shape ``(len(circuits), 2**n_qubits)``.

        Measurement ops are ignored (like `Simulator.statevector`).
        """
        n = self._common_width(circuits)
        out = np.empty((len(circuits), 2**n), dtype=np.complex64)
        for indices, group in self._groups(circuits):
            engine = self._engine_for(len(group), n)
            states = self._evolve(engine, group, n, on_measure=None)
            out[indices] = engine.to_numpy(states)
        return out

    def run(self, circuits: list[Circuit], shots: int = 1000) -> list[Counter]:
        """Per-circuit measurement counts, one Counter per input circuit.

        Matches `Simulator.run` semantics: at each MeasureOp the listed qubits
        are sampled `shots` times from the current state, and a circuit's
        counters from multiple MeasureOps are summed.
        """
        n = self._common_width(circuits)
        results: list[Counter | None] = [None] * len(circuits)
        for indices, group in self._groups(circuits):
            engine = self._engine_for(len(group), n)
            counters: list[Counter] = [Counter() for _ in group]

            def on_measure(
                states, qubits: list[int], engine=engine, group=group, counters=counters
            ) -> None:
                probs = engine.probabilities(states, qubits, n)
                width = len(qubits)
                num = probs.shape[1]
                for i in range(len(group)):
                    p = probs[i] / probs[i].sum()
                    drawn = self._rng.choice(num, size=shots, p=p)
                    for idx in drawn:
                        counters[i][format(int(idx), f"0{width}b")] += 1

            self._evolve(engine, group, n, on_measure=on_measure)
            for pos, i in enumerate(indices):
                results[i] = counters[pos]
        # Circuits without any MeasureOp get an empty Counter (a Simulator.run
        # would have sampled all qubits; requiring explicit measurement keeps
        # batched semantics unambiguous), so measure_all() before running.
        return [r if r is not None else Counter() for r in results]

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _common_width(circuits: list[Circuit]) -> int:
        if not circuits:
            raise ValueError("BatchedSimulator needs at least one circuit")
        widths = {c.n_qubits for c in circuits}
        if len(widths) != 1:
            raise ValueError(f"All circuits must share n_qubits; got {sorted(widths)}")
        if any(isinstance(op, ChannelOp) for c in circuits for op in c.ops):
            raise ValueError(
                "circuits contain noise channels; a statevector batch cannot represent "
                "mixed states — run them with macquerel.DensityMatrixSimulator"
            )
        return circuits[0].n_qubits

    @staticmethod
    def _groups(circuits: list[Circuit]):
        """Yield (original indices, circuits) per structure signature."""
        by_sig: dict[tuple, list[int]] = {}
        for i, c in enumerate(circuits):
            by_sig.setdefault(_signature(c), []).append(i)
        for indices in by_sig.values():
            yield indices, [circuits[i] for i in indices]

    @staticmethod
    def _evolve(engine: _Engine, group: list[Circuit], n: int, on_measure):
        """Run one aligned group batched; `on_measure(states, qubits)` fires at
        every MeasureOp position. Returns the final (B, 2**n) states."""
        states = engine.allocate(len(group), n)
        ops_per_circuit = [c.ops for c in group]
        for pos in range(len(ops_per_circuit[0])):
            op = ops_per_circuit[0][pos]
            if isinstance(op, MeasureOp):
                if on_measure is not None:
                    on_measure(states, op.qubits)
                continue
            # Signature grouping guarantees ops[pos] is a Gate in every circuit.
            mats = np.stack(
                [cast(Gate, ops[pos]).matrix.astype(np.complex64) for ops in ops_per_circuit]
            )
            targets = op.targets
            if op.controls:
                mats = _lift_controls(mats, len(op.controls))
                targets = op.controls + op.targets
            states = engine.apply(states, mats, targets, n)
        return states
