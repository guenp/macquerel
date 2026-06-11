# Applying gates

This page explains the single operation everything else is built on: applying a small
unitary matrix to a huge amplitude array. It is the inner loop of every backend, and
the indexing picture developed here — bits of the array index ↔ qubits — is the key to
reading any of the kernel code.

## The index is the bitstring

An n-qubit state is a flat array `sv` of 2ⁿ complex numbers. The amplitude of basis
state `|b₀b₁…bₙ₋₁⟩` lives at the index whose binary expansion is that bitstring, with
**qubit 0 as the most significant bit**: qubit `q` occupies bit `n-1-q` of the index.

```text
 n = 3:    index i =  b₀ b₁ b₂   (binary)
                      │  │  └── qubit 2 → bit 0  (LSB)
                      │  └───── qubit 1 → bit 1
                      └──────── qubit 0 → bit 2  (MSB)
```

This convention has a beautiful consequence. Reshape the flat array into an
n-dimensional tensor of shape `(2,)*n` — costless, it's the same memory — and **axis q
of the tensor is exactly qubit q**:

```python
state = sv.reshape((2,) * n)   # state[b0, b1, ..., b_{n-1}] = amplitude of |b0 b1 ... ⟩
```

Now "apply a gate to qubit q" becomes "apply a 2×2 matrix along axis q", which is a
standard array operation (`np.tensordot`). Marginal probabilities are axis sums.
Nearly every line of backend code is a restatement of this picture.

## One gate, one sweep

Take a single-qubit gate U on qubit q. It mixes each amplitude with the amplitude of
the *partner* index that differs only at qubit q's bit:

```text
 U on qubit 1 of a 3-qubit state (partner indices differ at bit 1):

  index:   000   001   010   011   100   101   110   111
            │     │     ▲     ▲     │     │     ▲     ▲
            └─────┼─────┘     │     └─────┼─────┘     │
                  └───────────┘           └───────────┘

  for each pair (i, j = i + 2):
      | sv[i] |      | u00  u01 |   | sv[i] |
      |       |  ←   |          | · |       |
      | sv[j] |      | u10  u11 |   | sv[j] |
```

2ⁿ⁻¹ independent 2×2 multiplications, each touching 2 amplitudes: **one pass over the
state, O(2ⁿ) work**. The general k-qubit case is the same picture scaled up. Group the
2ⁿ indices by their *non-target* bits: each group holds the 2ᵏ amplitudes that share
those bits and differ only at the k target bits, and the gate is an independent
2ᵏ×2ᵏ matrix–vector multiply on every group:

```text
 k-qubit dense gate:   2ⁿ⁻ᵏ groups  ×  (2ᵏ×2ᵏ matrix · 2ᵏ vector)

 ┌─ group 0 ──┐ ┌─ group 1 ──┐ ┌─ group 2 ──┐
 │ a₀ a₁ … a₂ᵏ│ │ …          │ │ …          │   each group: amp[] ← M · amp[]
 └────────────┘ └────────────┘ └────────────┘
   (independent → trivially parallel, and in-place safe)
```

This *group picture* is implemented three ways:

- **CPU** (`backends/cpu.py`): `np.tensordot(gate, state, axes=([k..2k), targets])`
  contracts the gate against the target axes — NumPy's optimized BLAS-backed path —
  then a transpose puts the axes back in place.
- **Metal** (`backends/metal_backend.py`): one GPU thread *owns* one group. It
  reconstructs the group's 2ᵏ indices with bit arithmetic, reads the amplitudes,
  multiplies by the matrix, and writes them back **to the same locations** — groups
  are disjoint, so in-place update is race-free and the state is never copied.
- **MLX** (`backends/mlx_backend.py`): a custom GPU kernel with the same
  one-thread-per-group design (falling back to `mx.tensordot` for very wide gates).

Controlled gates fit the same picture with a twist: a control qubit is a *non-target*
bit, so it is constant across a group, and the whole group either applies the matrix
(all control bits 1) or is left untouched. On the GPU that is a single branch per
thread; on CPU it is a slice selecting the control-set subspace.

## Not all gates cost the same: diagonal / permutation / dense

A dense k-qubit gate does 4ᵏ multiply-adds per group. But most gates that occur in
real circuits have structure that lets the backend skip almost all of that work, so
every backend first *classifies* the matrix (`gates.classify`, memoized by matrix
bytes) into one of three kinds:

```text
 kind          shape of the matrix         example gates             cost per amplitude
 ────────────  ──────────────────────────  ────────────────────────  ──────────────────
 diagonal      ⎡d₀      ⎤                  Z, S, T, RZ, CP, CZ,      1 multiply
               ⎢  d₁    ⎥  off-diagonal    fused phase blocks        (read, scale,
               ⎢    d₂  ⎥  entries ≈ 0                                write in place)
               ⎣      d₃⎦

 permutation   ⎡  φ₀    ⎤  exactly one     X, SWAP, CNOT, and        1 gather + 1
 (monomial)    ⎢      φ₁⎥  unit-magnitude  *phased* products like    multiply
               ⎢φ₂      ⎥  entry per       a fused CX·(RZ⊗RZ)
               ⎣    φ₃  ⎦  row/column

 dense         anything else               H, RX, RY, fused          2ᵏ MACs
                                           multi-qubit blocks
```

- A **diagonal** gate never mixes amplitudes — it just scales each one by a phase
  picked by its own target bits: `sv[i] *= diag[row(i)]`. One read and one write per
  amplitude, no index tables, perfectly in-place. This is why the compiler goes out
  of its way to fuse long runs of phase gates into one wide diagonal
  (see [Optimizations](optimizations.md)).
- A **permutation** (more precisely *monomial*) gate moves each amplitude to a new
  position and multiplies it by one phase: `out[i] = φ · sv[src(i)]`. A gather instead
  of a matmul — 2ᵏ multiplies per group rather than 4ᵏ. One subtlety: `classify`
  accepts any unit-magnitude monomial matrix, so the per-row *phase* must be applied
  too, not just the gather (pure permutations like X and SWAP have all-1 phases).
- Everything else takes the **dense** path described above.

The classification trio and the fast paths it enables are standard in
high-performance simulators — Qulacs, for instance, ships specialized kernels per
gate kind for the same reason [\[5\]](#references). The deeper point: since gate application is
bandwidth-bound, *kind determines cost*. A diagonal gate moves 16 bytes per amplitude
where MLX's dense path moves several times that, and the gap is exactly what the
measured QFT benchmark differences come down to.

## Measurement and sampling

After the gates, `run(shots=...)` needs bitstring samples over the measured qubits.
The probabilities are already in memory, so the backend:

1. squares the amplitudes: `probs = |sv|²` over the `(2,)*n` view;
2. **marginalizes** away unmeasured qubits by summing those axes
   (`joint = probs.sum(axis=unmeasured)`);
3. reorders the surviving axes from ascending qubit order to the caller's
   `measure([q…])` order;
4. flattens and draws `shots` samples from the categorical distribution.

Step 3 hides a classic off-by-an-inverse trap, preserved here as a warning to fellow
implementers. After the sum, the surviving axes hold the measured qubits in
*ascending index order*; result axis `i` must hold `qubits[i]`. The permutation that
does this is each qubit's **rank** among the measured qubits — `argsort(argsort(qubits))`
— not `argsort(qubits)`, which is its inverse. The two coincide on sorted lists and
on any swap of two qubits (involutions are their own inverse), so the bug only shows
up when a measure list permutes three or more qubits cyclically, e.g.
`measure([1, 2, 0])`. Every sampler in the codebase carries a comment pointing at
`CPUBackend.sample` for this reason.

Mid-circuit `measure(qubits, collapse=True)` instead samples one outcome per qubit
sequentially, zeroes the amplitudes inconsistent with it, and renormalizes — the
textbook projective update [\[1\]](#references).

## Expectation values

`expectation_pauli` computes ⟨ψ|P|ψ⟩ for Pauli strings P (sums of terms like
`0.5·X₀Z₂`): apply each term's Paulis to a copy of the state (each is a one-qubit
gate, so this is cheap) and take the inner product with the original. Pauli-string
expectations are the bread and butter of variational algorithms (VQE-style energy
estimation), which is also what `BatchedSimulator` accelerates across parameter
sweeps.

---

Next: [The backends](backends.md) — how the same `apply_matrix` contract is
implemented three different ways, and what each implementation buys.

## References

1. M. A. Nielsen and I. L. Chuang, *Quantum Computation and Quantum Information*,
   Cambridge University Press (2010), §2.2 (measurement), §4.2–4.3 (gates).
2. T. Jones et al., "QuEST and High Performance Simulation of Quantum Computers"
   (2019). [arXiv:1802.08032](https://arxiv.org/abs/1802.08032) — the same
   pairs/groups picture, on CPU clusters.
3. M. Smelyanskiy et al., "qHiPSTER" (2016).
   [arXiv:1601.07195](https://arxiv.org/abs/1601.07195) — bandwidth analysis of
   single- and two-qubit gate sweeps.
4. Google Quantum AI, *qsim* — [https://quantumai.google/qsim](https://quantumai.google/qsim).
5. Y. Suzuki et al., "Qulacs: a fast and versatile quantum circuit simulator"
   (2021). [arXiv:2011.13524](https://arxiv.org/abs/2011.13524) — per-gate-kind
   specialized kernels and fusion.
