# How macquerel works

This section is a guided tour of the library's internals, written for someone who has
seen qubits and circuits before but has never looked inside a simulator. It starts from
first principles — what it even means to simulate a quantum computer on a classical one
— and works down to the Metal kernels, explaining each design decision along the way.

1. **This page** — what a statevector simulator is, why its cost is exponential, and a
   map of how a circuit flows through the library.
2. [Applying gates](applying-gates.md) — how a gate, a small unitary matrix, is applied
   to a state of billions of amplitudes: the indexing convention, the tensor view, and
   the diagonal/permutation/dense classification every backend exploits.
3. [The backends](backends.md) — the three execution engines (NumPy, MLX, Metal): how
   each one stores the state and applies gates, and why each owns a different regime.
4. [Optimizations](optimizations.md) — gate fusion and the other compiler passes, the
   backend-specific optimizations, and the measure-first A/B discipline behind them.
5. [Noise and density matrices](noise.md) — how noisy circuits are simulated by
   reusing the statevector machinery on a *vectorized density matrix*.

If you just want the measured performance numbers and tuning tips, see
[Backends: CPU vs MLX vs Metal](../backends.md); for the API itself, the
[reference](../reference/api.md).

## What a simulator does

A quantum computer prepares qubits, applies gates, and measures. A *statevector
simulator* tracks the complete mathematical description of the qubits — the state
vector — and imitates each of those steps with linear algebra:

```text
 quantum computer                     macquerel
 ─────────────────                    ──────────────────────────────
 n physical qubits          ⟷        an array of 2ⁿ complex numbers
 a gate (physical pulse)    ⟷        a small unitary matrix applied
                                      to that array
 measurement                ⟷        sampling bitstrings from the
                                      squared amplitudes
```

This is the most direct, fully general way to simulate a quantum computer: every
algorithm, every entangled state, every intermediate amplitude is available exactly
(up to floating-point precision). The price is that the array doubles with every
qubit added — the *exponential wall* discussed below.

## Qubits, amplitudes, and the state vector

A classical bit is 0 or 1. A qubit's state is a *superposition*

```text
|ψ⟩ = α·|0⟩ + β·|1⟩        with |α|² + |β|² = 1
```

where α and β are complex numbers called *amplitudes*. One qubit therefore needs two
complex numbers. Two qubits need four — one amplitude per basis state `|00⟩, |01⟩,
|10⟩, |11⟩` — and in general **n qubits need 2ⁿ amplitudes**, because the state can be
a superposition of every n-bit string at once:

```text
|ψ⟩ = Σ over all bitstrings b of  α_b · |b⟩
```

That sum *is* the state vector: macquerel stores it as a single one-dimensional
complex array, indexed by the bitstring read as a binary number. For a 3-qubit GHZ
state `(|000⟩ + |111⟩)/√2`:

```text
 index   bits (q0 q1 q2)   amplitude
 ─────   ───────────────   ─────────
   0          0 0 0          0.7071
   1          0 0 1          0
   2          0 1 0          0
   ...                       0
   7          1 1 1          0.7071
```

Two conventions to remember, because everything downstream depends on them:

- **Qubit 0 is the most significant bit** of the index (so qubit `q` lives at bit
  `n-1-q` of the array index). The bitstrings that `run()` returns read left-to-right
  as qubit 0, 1, 2, …
- Amplitudes are stored as `complex64` (8 bytes each) by default — single precision
  is plenty for circuit simulation and halves the memory and bandwidth of double.

## Gates are matrices

A quantum gate on k qubits is a 2ᵏ×2ᵏ *unitary* matrix (unitary: it preserves the
total probability, U†U = I). Applying it means multiplying the affected amplitudes by
that matrix. The workhorses:

```text
 H = 1/√2 · | 1  1 |        X = | 0 1 |        CX = | 1 0 0 0 |
            | 1 -1 |            | 1 0 |             | 0 1 0 0 |
                                                    | 0 0 0 1 |
 (superposition)             (bit flip)             | 0 0 1 0 |
                                                    (controlled flip)
```

A circuit is just a sequence of these. The Bell-pair circuit from the quickstart,

```python
circuit = Circuit(2)
circuit.h(0).cx(0, 1)
```

evolves the array like this:

```text
  start            after h(0)                after cx(0, 1)
 ┌────────┐       ┌────────────┐            ┌────────────┐
 │ 1      │ |00⟩  │ 0.7071     │ |00⟩       │ 0.7071     │ |00⟩
 │ 0      │ |01⟩  │ 0          │ |01⟩       │ 0          │ |01⟩
 │ 0      │ |10⟩  │ 0.7071     │ |10⟩  ──►  │ 0          │ |10⟩
 │ 0      │ |11⟩  │ 0          │ |11⟩       │ 0.7071     │ |11⟩
 └────────┘       └────────────┘            └────────────┘
```

The key computational fact — the reason simulation is feasible at all — is that a
k-qubit gate is *sparse structure, not a 2ⁿ×2ⁿ matrix*. Applying a 2×2 gate to qubit q
of an n-qubit state never builds a big matrix; it sweeps the array once, combining
amplitudes in pairs whose indices differ only at bit `n-1-q`. The cost of one gate is
**O(2ⁿ)** — one pass over the state — not O(4ⁿ). [Applying gates](applying-gates.md)
makes this precise; it is the inner loop of the whole library.

## Measurement

Measuring all n qubits returns bitstring `b` with probability `|α_b|²` — the *Born
rule*. A simulator has a luxury a real device lacks: the full probability
distribution is sitting right there in memory. So instead of evolving the state once
per shot, macquerel evolves it **once** and then draws all requested shots from the
distribution `p(b) = |α_b|²` (`numpy.random.choice` on CPU, `mx.random.categorical`
on MLX). Mid-circuit `measure()` with collapse is also supported, where the state is
projected and renormalized.

## The exponential wall

Doubling per qubit is brutal in both directions — it is why quantum computers are
interesting and why simulating them is hard:

```text
 qubits   amplitudes        state size (complex64)
 ──────   ───────────       ──────────────────────
   10     1 024              8 KB    (L1 cache)
   20     ~10⁶               8 MB    (L2/L3 cache)
   28     ~2.7·10⁸           2 GiB
   30     ~10⁹               8 GiB
   33     ~8.6·10⁹          64 GiB   (macquerel's ceiling on a 128 GiB Mac)
   45     ~3.5·10¹³          0.5 PB  (a supercomputer's worth of RAM [4])
   50+    —                  beyond any classical machine's memory
```

Past ~33 qubits on a single machine you need different tricks entirely — distributed
memory across nodes [\[3, 4\]](#references), tensor-network contraction for shallow circuits, or
restricted gate sets (Clifford circuits simulate in polynomial time [\[6\]](#references)). Within the
single-machine regime, though, the game is clear: **a gate is one pass over a huge
array, so the simulator's speed is set by memory bandwidth, not arithmetic**. Every
optimization in this library — gate fusion, in-place kernels, lazy evaluation —
is ultimately about moving fewer bytes. That is also why Apple Silicon is an
interesting target: its *unified memory* gives the GPU direct, high-bandwidth access
(hundreds of GB/s) to the same physical RAM the CPU sees, with capacities (up to
128–192 GiB) that discrete GPUs don't reach, and zero-copy handoff between the two.

## The shape of the library

A circuit flows through three layers:

```text
            you
             │  Circuit(3).h(0).cx(0,1).cx(1,2).measure_all()
             ▼
 ┌───────────────────────┐   gate list: H[0], CX[0,1], CX[1,2], M[0,1,2]
 │  circuit.py           │
 │  (the gate list)      │
 └───────────┬───────────┘
             ▼
 ┌───────────────────────┐   fused list: Fused(H,CX,CX)[0,1,2], M[0,1,2]
 │  compiler.py          │   • greedy gate fusion (fewer passes over the state)
 │  (circuit → circuit)  │   • diagonal-run merging, commutation-aware grouping
 └───────────┬───────────┘
             ▼
 ┌───────────────────────┐   simulator.py picks a backend by qubit count:
 │  backends/            │      cpu  ≤15q   (NumPy — dispatch-bound regime)
 │  cpu / mlx / metal    │      metal 16q+  (in-place Metal kernels, to 33q)
 │  (apply, measure,     │      mlx  16–30q (lazy-graph fallback)
 │   sample)             │
 └───────────┬───────────┘
             ▼
        Counter({'000': 503, '111': 497})
```

All three backends implement the same small protocol — `allocate`, `apply_matrix`,
`measure`, `sample`, `expectation_pauli` — and are differential-tested against each
other, so they are interchangeable in correctness and differ only in performance.
Three more simulators reuse the same machinery: `BatchedSimulator` evolves many small
same-shaped circuits as one batched tensor (for parameter sweeps),
`DensityMatrixSimulator` runs *noisy* circuits by treating an n-qubit density matrix
as a 2n-qubit statevector, and `TrajectorySimulator` runs noisy circuits past the
density-matrix ceiling by averaging stochastic pure-state trajectories — see
[Noise and density matrices](noise.md).

## Where this sits in the literature

Statevector (also "full amplitude" or "Schrödinger-style") simulation is the standard
baseline design, and macquerel's structure — flat amplitude array, gate kinds, gate
fusion, bandwidth-oriented kernels — will look familiar from the HPC simulators:
qHiPSTER [\[3\]](#references), QuEST [\[2\]](#references), Qulacs [\[5\]](#references), and Google's qsim. The textbook backing for
everything in these pages is Nielsen & Chuang [\[1\]](#references) (chapters 1–2 and 4 for states,
gates, and measurement; chapter 8 for the noise formalism used in
[Noise and density matrices](noise.md)).

## References

1. M. A. Nielsen and I. L. Chuang, *Quantum Computation and Quantum Information*,
   Cambridge University Press (2010).
2. T. Jones, A. Brown, I. Bush, S. C. Benjamin, "QuEST and High Performance Simulation
   of Quantum Computers", *Scientific Reports* **9**, 10736 (2019).
   [arXiv:1802.08032](https://arxiv.org/abs/1802.08032)
3. M. Smelyanskiy, N. P. D. Sawaya, A. Aspuru-Guzik, "qHiPSTER: The Quantum High
   Performance Software Testing Environment".
   [arXiv:1601.07195](https://arxiv.org/abs/1601.07195)
4. T. Häner and D. S. Steiger, "0.5 Petabyte Simulation of a 45-Qubit Quantum
   Circuit", *SC '17*. [arXiv:1704.01127](https://arxiv.org/abs/1704.01127)
5. Y. Suzuki et al., "Qulacs: a fast and versatile quantum circuit simulator for
   research purpose", *Quantum* **5**, 559 (2021).
   [arXiv:2011.13524](https://arxiv.org/abs/2011.13524)
6. S. Aaronson and D. Gottesman, "Improved simulation of stabilizer circuits",
   *Phys. Rev. A* **70**, 052328 (2004).
   [arXiv:quant-ph/0406196](https://arxiv.org/abs/quant-ph/0406196)
7. J. Preskill, *Lecture Notes on Quantum Computation*,
   [http://theory.caltech.edu/~preskill/ph229/](http://theory.caltech.edu/~preskill/ph229/)
