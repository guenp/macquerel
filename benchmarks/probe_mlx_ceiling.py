"""Gate-0 probe: find the qubit count at which released MLX breaks on this machine.

Exercises the operations the MLX backend actually performs on a 2**n complex64
state: allocate, scalar multiply + eval (diagonal path), a uint32 arange index,
a gather (permutation path — the op flagged by MLX issue #3327), and a readback.

Each qubit count is run in its own subprocess so an MLX abort/overflow can't take
down the driver. Usage:  python benchmarks/probe_mlx_ceiling.py 29 30 31 32
"""

from __future__ import annotations

import subprocess
import sys

CHILD = r"""
import sys, traceback
import numpy as np
import mlx.core as mx

n = int(sys.argv[1])
size = 2 ** n
gib = size * 8 / 1024**3
print(f"n={n}: 2**n={size} elements, {gib:.1f} GiB complex64", flush=True)

def step(name, fn):
    try:
        out = fn()
        print(f"  OK   {name}", flush=True)
        return out
    except Exception as e:
        print(f"  FAIL {name}: {type(e).__name__}: {e}", flush=True)
        raise

def _alloc():
    a = mx.zeros(size, dtype=mx.complex64)
    mx.eval(a)
    return a

def _mul(a):
    b = a * mx.array(1.0 + 0j, dtype=mx.complex64)
    mx.eval(b)
    return b

def _arange():
    a = mx.arange(size, dtype=mx.uint32)
    mx.eval(a)
    return a

def _gather(a, i):
    g = a[i]
    mx.eval(g)
    return g

sv = step("allocate mx.zeros(2**n, complex64) + eval", _alloc)
sv = step("scalar multiply (diagonal-like) + eval", lambda: _mul(sv))
idx = step("mx.arange(2**n, uint32) + eval", _arange)
step("gather sv[idx] (permutation path) + eval", lambda: _gather(sv, idx))
step("readback np.array(sv[:4])", lambda: np.array(sv[:4]))

print(f"n={n}: ALL OK", flush=True)
"""


def main() -> None:
    ns = [int(x) for x in sys.argv[1:]] or [29, 30, 31, 32]
    for n in ns:
        print(f"\n===== probing n={n} =====", flush=True)
        proc = subprocess.run(
            [sys.executable, "-c", CHILD, str(n)],
            capture_output=True,
            text=True,
        )
        print(proc.stdout, end="")
        if proc.stderr.strip():
            print("  [stderr]", proc.stderr.strip()[-2000:])
        print(f"  exit code: {proc.returncode}", flush=True)
        if proc.returncode != 0:
            print(f"===== n={n} FAILED — stopping ascent =====", flush=True)
            break


if __name__ == "__main__":
    main()
