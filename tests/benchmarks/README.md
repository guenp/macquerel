# macquerel benchmarks

Benchmarks live in `tests/benchmarks/` and are run manually — they are not part of the pytest suite.

## Running the benchmark

```bash
# default: qubit counts 10–24, depth 50, 3 reps
uv run python tests/benchmarks/bench_backends.py

# custom qubit range and depth
uv run python tests/benchmarks/bench_backends.py --qubits 10 14 18 22 26 --depth 30

# more reps for a more stable minimum (slower)
uv run python tests/benchmarks/bench_backends.py --reps 10
```

Each configuration runs `--reps` times; the **minimum** time is reported (best-case throughput, unaffected by OS scheduling noise).

### All options

| Flag | Default | Description |
|---|---|---|
| `--qubits N [N ...]` | `10 14 16 18 20 22 24` | Qubit counts to sweep |
| `--depth N` | `50` | Gates per circuit |
| `--reps N` | `3` | Repetitions; minimum time reported |
| `--seed N` | `42` | RNG seed — same seed produces identical circuits across runs |
| `--json FILE` | — | Write results to a JSON file |
| `--no-chart` | — | Suppress ASCII bar chart |

## Saving and comparing runs

Save results to JSON to compare across backend versions, machine configs, or code changes:

```bash
uv run python tests/benchmarks/bench_backends.py --json results/before.json
# ... make changes ...
uv run python tests/benchmarks/bench_backends.py --json results/after.json
```

Compare two runs directly in Python:

```python
import json

before = json.load(open("results/before.json"))["results"]
after  = json.load(open("results/after.json"))["results"]

print(f"{'qubits':>6}  {'before (ms)':>12}  {'after (ms)':>11}  {'delta':>8}")
for b, a in zip(before, after):
    delta = (a["cpu_ms"] - b["cpu_ms"]) / b["cpu_ms"] * 100
    print(f"{b['n_qubits']:>6}  {b['cpu_ms']:>12.1f}  {a['cpu_ms']:>11.1f}  {delta:>+7.1f}%")
```

## Plotting with matplotlib

Install matplotlib if needed:

```bash
uv add --dev matplotlib
```

Then plot CPU and MLX times vs qubit count:

```python
import json
import matplotlib.pyplot as plt

data = json.load(open("results/run1.json"))["results"]

n      = [r["n_qubits"] for r in data]
cpu_ms = [r["cpu_ms"]   for r in data]
mlx_ms = [r["mlx_ms"]   for r in data if r["mlx_ms"] is not None]
n_mlx  = [r["n_qubits"] for r in data if r["mlx_ms"] is not None]

plt.figure(figsize=(8, 5))
plt.semilogy(n, cpu_ms, "o-", label="CPU (NumPy)")
if mlx_ms:
    plt.semilogy(n_mlx, mlx_ms, "s-", label="MLX")
plt.xlabel("Qubits")
plt.ylabel("Time (ms, log scale)")
plt.title("macquerel backend benchmark — random circuit, depth 50")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("results/benchmark.png", dpi=150)
plt.show()
```

To plot speedup (CPU time / MLX time):

```python
speedups = [r["speedup"] for r in data if r["speedup"] is not None]
n_sp     = [r["n_qubits"] for r in data if r["speedup"] is not None]

plt.figure(figsize=(8, 4))
plt.axhline(1.0, color="gray", linestyle="--", label="breakeven")
plt.plot(n_sp, speedups, "o-")
plt.xlabel("Qubits")
plt.ylabel("Speedup (CPU ms / MLX ms)")
plt.title("MLX speedup over CPU")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/speedup.png", dpi=150)
plt.show()
```

## Interpreting results

**Why MLX is currently slower than CPU:**
The MLX backend today uses NumPy `tensordot` (CPU) for gate application and pays a NumPy↔MLX conversion on every gate call. There is no GPU work of substance. This is the §5.3 "reference gate path" described in the design spec — correct but not fast. Steps 9–10 of the v0.2+ plan (`docs/plan.md`) describe the fixes: persistent SoA MLX state and the `mx.fast.metal_kernel` pairing-loop kernel.

**What to expect after those fixes:**
Once state stays in MLX arrays across calls and the Metal kernel is in place, the crossover point (where MLX beats CPU) should appear around 15–18 qubits, where state-vector size starts to saturate NumPy's cache. Above ~20 qubits the MLX backend should pull ahead as memory bandwidth becomes the bottleneck and the GPU's higher bandwidth advantage kicks in.
