# Noise and density matrices

Everything so far assumed a *closed* system: gates are perfectly unitary and nothing
disturbs the qubits between them. Real hardware isn't like that — qubits decay, dephase,
and pick up errors from their environment. This page explains how macquerel simulates
that, and how the implementation gets all of it almost for free by reusing the
statevector machinery on a *vectorized* density matrix.

## Why a statevector isn't enough

A statevector describes a system whose state we know exactly — a *pure* state. Noise
introduces classical uncertainty: "with probability 1−p nothing happened, with
probability p the qubit flipped." The result is a *mixed* state — a probabilistic
ensemble of pure states — and no single vector of 2ⁿ amplitudes can represent it.
(It is not the same as superposition: a 50/50 *mixture* of |0⟩ and |1⟩ never
interferes; the superposition (|0⟩+|1⟩)/√2 does.)

The standard fix [\[1, ch. 2 & 8\]](#references) is the **density matrix**

```text
 ρ = Σᵢ pᵢ |ψᵢ⟩⟨ψᵢ|        (2ⁿ × 2ⁿ, Hermitian, trace 1)
```

which carries both quantum amplitudes and classical uncertainty in one object:

```text
                  pure |+⟩ = (|0⟩+|1⟩)/√2        50/50 classical mixture
                  ┌              ┐               ┌              ┐
            ρ  =  │ 0.5    0.5   │               │ 0.5    0     │
                  │ 0.5    0.5   │               │ 0      0.5   │
                  └              ┘               └              ┘
 diagonal:        measurement probabilities      same probabilities!
 off-diagonal:    coherences — the "quantumness" that makes interference work;
                  noise is, in essence, the decay of these entries
```

Useful facts the API exposes directly: measurement probabilities are the diagonal,
`p(i) = ρᵢᵢ` (the Born rule again); expectation values are `tr(ρP)`; and the *purity*
`tr(ρ²)` is 1.0 for a pure state, falling toward `1/2ⁿ` as the state mixes —
`DensityMatrixSimulator.purity` is a one-line noise meter.

## Noise as Kraus operators

How does ρ evolve? Unitaries act as `ρ → UρU†`. Noise is described by a **channel** —
a set of Kraus operators {K₀, K₁, …} with

```text
 ρ  →  Σₖ Kₖ ρ Kₖ†          with the completeness condition  Σₖ Kₖ†Kₖ = I
```

Intuitively, each Kₖ is "one thing the environment might do", and the channel sums
over the possibilities, weighted by how likely each is for the current state. This
operator-sum form can represent any physical (completely positive, trace-preserving)
evolution [\[1, §8.2\]](#references). The built-in channels (`noise.py`) are the textbook set:

```text
 bit_flip(p)             K₀ = √(1−p)·I            K₁ = √p·X
 phase_flip(p)           K₀ = √(1−p)·I            K₁ = √p·Z
 depolarizing(p)         √(1−p)·I, √(p/3)·X, √(p/3)·Y, √(p/3)·Z

 amplitude_damping(γ)    K₀ = | 1    0    |       K₁ = | 0  √γ |
 ("T1 decay": |1⟩             | 0  √(1−γ) |            | 0   0  |
  relaxes to |0⟩)
 phase_damping(λ)        K₀ = | 1    0    |       K₁ = | 0   0  |
 ("T2 dephasing")             | 0  √(1−λ) |            | 0  √λ  |
```

Arbitrary channels — including multi-qubit ones, like correlated two-qubit noise —
go through `circuit.kraus(qubits, operators)`; `validate_kraus` checks completeness at
build time. Channels are circuit ops like any other:

```python
circuit = Circuit(2)
circuit.h(0).cx(0, 1)            # Bell pair...
circuit.depolarizing(0, 0.05)    # ...with a noisy qubit 0
counts = DensityMatrixSimulator().run(circuit, shots=1000)
```

The statevector `Simulator` (and `BatchedSimulator`) reject circuits containing
channels — a statevector cannot represent the mixed output — with an error pointing
here.

## The implementation trick: vectorization

The obvious implementation would be new backends operating on 2ⁿ×2ⁿ matrices. The
actual implementation (`density.py`) is much smaller: **flatten ρ row-major into a
vector of 4ⁿ amplitudes and hand it to the existing statevector backends as if it
were a 2n-qubit state.**

```text
        ρ (2ⁿ × 2ⁿ)                       vec(ρ)  (length 4ⁿ = 2^(2n))
   ┌─────────────────┐
   │ ρ₀₀  ρ₀₁  ρ₀₂ … │    row-major     [ρ₀₀ ρ₀₁ … ρ₀ₘ ρ₁₀ ρ₁₁ … ]
   │ ρ₁₀  ρ₁₁  …     │   ──────────►     └── row index i ──┘└─ col j ─┘
   │ …               │     flatten       vec(ρ)[i·2ⁿ + j] = ρᵢⱼ
   └─────────────────┘
                                          high n bits = i  → "ket qubits" 0…n−1
                                          low  n bits = j  → "bra qubits" n…2n−1
```

The row index becomes qubits 0…n−1 of the doubled state (the *ket* axes) and the
column index becomes qubits n…2n−1 (the *bra* axes). Why this works so neatly comes
down to two identities of the vectorization (for row-major flattening,
`vec(AρB) = (A ⊗ Bᵀ) vec(ρ)`):

- **Unitary gate** `ρ → UρU†`: apply `U` to the gate's *ket* axes, then `conj(U)` to
  the matching *bra* axes — two ordinary `apply_matrix` calls on targets `[t]` and
  `[t+n]`. Controls shift the same way. Narrow control-free gates instead apply the
  combined `U ⊗ conj(U)` superoperator to ket+bra axes at once — one full-state pass
  instead of two, gated by gate kind (diagonal ≤4q, monomial ≤3q, dense ≤2q; wider
  doubled gates lose more in the kernel than the saved pass returns).
- **Kraus channel**: the operator-sum becomes a single matrix on the doubled space —
  the **superoperator** `Σₖ Kₖ ⊗ conj(Kₖ)` — applied to the channel's ket+bra axes
  in *one* dense `apply_matrix` call (`noise.channel_superoperator`). For a 1-qubit
  channel that is just a 4×4 matrix on the doubled state.
- **Probabilities**: the diagonal `ρᵢᵢ` sits at positions `i·(2ⁿ+1)` of the
  vector — a strided slice. On CPU and Metal it is read through a zero-copy host
  view, so sampling never materializes the full matrix on the host.
- **Purity**: for Hermitian ρ, `tr(ρ²) = Σᵢⱼ|ρᵢⱼ|²` — the squared norm of the
  vector, one BLAS dot.
- **Expectation** `tr(ρP)`: a Pauli string is a monomial matrix — one unit-magnitude
  entry per row — so the trace needs exactly one element of ρ per row index:
  `tr(ρP) = Σᵢ phase(i) · ρ[i, i⊕mask]`, a strided gather off the zero-copy host
  view per term. No full readback, no state-sized temporaries (which at n=16 would
  have been ~96 GB).

Nothing in the backends changed: the Metal in-place kernels, the MLX graph, gate
fusion (resolved at the doubled qubit count, with channels acting as fusion
barriers since a channel does not commute with the gates around it), and backend
auto-selection (at the doubled count) all apply verbatim. This superoperator /
"Liouville space" representation is the standard one in open-quantum-systems work
[\[3\]](#references) and is also how Qiskit Aer's density-matrix method is built
[\[4\]](#references); macquerel's
contribution is just noticing the existing backends could run it unmodified.

## What it costs

The doubled state is the whole story of the cost: an n-qubit density matrix **is** a
2n-qubit statevector.

```text
 n (noisy qubits)   doubled state   memory (4ⁿ × 8 B)     backend ceiling
 ────────────────   ─────────────   ──────────────────    ─────────────────
       10                20q             8 MB
       13                26q           0.5 GiB             cpu: patience
       15                30q             8 GiB             mlx ceiling (int32)
       16                32q            32 GiB             metal ceiling
```

So exact noisy simulation tops out at 16 qubits (Metal, a 32 GiB state — measured
6.0 s for a noisy 16-qubit GHZ, sitting exactly on the theoretical memory line; the
dashed series in `benchmarks/data/memory.png`). Quadratic-in-state-size cost is the
fundamental price of exact noise simulation.

## Past the density-matrix wall: `TrajectorySimulator`

The escape hatch for larger noisy circuits is the **quantum-trajectory**
(Monte-Carlo wavefunction) method [\[5\]](#references), shipped as
`TrajectorySimulator`: unravel each channel into stochastic pure-state trajectories
— sample *one* Kraus operator per channel with its Born probability
`pₖ = ⟨ψ|Kₖ†Kₖ|ψ⟩`, apply it, renormalize — and average over trajectories. Each
trajectory is a normal 2ⁿ statevector run on the existing backends, so the 4ⁿ cost
becomes T·2ⁿ time at 2ⁿ memory, exact in expectation with statistical error
~`1/√T`. Two implementation points keep it at true statevector scale: every
built-in channel has *diagonal* effect operators `Kₖ†Kₖ`, so the jump probabilities
come from a single `abs2sum` marginal — no state copies, compatible with Metal's
in-place buffer — and trajectories reset one allocated state in place rather than
reallocating per run, keeping the footprint constant in trajectory count. Measured:
a noisy **30-qubit** GHZ (a 2⁶⁰-amplitude density matrix, forever out of reach)
runs at ~10 s/trajectory on Metal.

```python
sim = TrajectorySimulator(trajectories=200, seed=7)
counts = sim.run(circuit, shots=1000)      # shots spread across trajectories
probs = sim.probabilities(circuit)         # averaged over trajectories
```

## References

1. M. A. Nielsen and I. L. Chuang, *Quantum Computation and Quantum Information*,
   Cambridge University Press (2010) — ch. 2.4 (density operators), ch. 8 (quantum
   noise and channels).
2. J. Preskill, *Lecture Notes on Quantum Computation*, ch. 3 (foundations of the
   density-matrix formalism).
   [http://theory.caltech.edu/~preskill/ph229/](http://theory.caltech.edu/~preskill/ph229/)
3. C. Gardiner and P. Zoller, *Quantum Noise*, Springer (2004) — superoperators and
   Liouville-space methods.
4. Qiskit Aer documentation, *density_matrix simulation method* —
   [https://qiskit.github.io/qiskit-aer/](https://qiskit.github.io/qiskit-aer/)
5. K. Mølmer, Y. Castin, J. Dalibard, "Monte Carlo wave-function method in quantum
   optics", *J. Opt. Soc. Am. B* **10**, 524 (1993) — the trajectory unraveling.
