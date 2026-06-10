#!/usr/bin/env python3
"""Benchmark released macquerel versions against the current checkout.

This is the PR/regression harness. It runs the same small random circuits for
the CPU, MLX, and Metal backends in isolated Python environments, prints a
comparison table to stdout, and can save JSON for plotting.

Examples:
    uv run python benchmarks/bench_versions.py --versions 0.2.0 --include-current
    uv run python benchmarks/bench_versions.py --versions latest --qubits 8 10 --reps 2
    uv run python benchmarks/bench_versions.py --json benchmarks/data/version_regression.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
import venv
from pathlib import Path

BACKENDS = ["cpu", "mlx", "metal"]
_RESULT_TAG = "__MACQUEREL_VERSION_BENCH__"

_BENCH_CODE = r"""
import argparse, json, time
import numpy as np

import macquerel.gates as g
from macquerel.backends.cpu import CPUBackend

RESULT_TAG = "__MACQUEREL_VERSION_BENCH__"
SINGLE = [g.H, g.X, g.Z, g.S, lambda: g.Rz(0.7), lambda: g.Rx(0.3)]
TWO = [g.CNOT, g.CZ, g.SWAP]

def make_backend(name):
    if name == "cpu":
        return CPUBackend()
    if name == "mlx":
        from macquerel.backends.mlx_backend import MLXBackend
        return MLXBackend()
    if name == "metal":
        from macquerel.backends.metal_backend import MetalBackend
        return MetalBackend()
    raise ValueError(name)

def ops_for(n, depth, seed):
    rng = np.random.default_rng(seed)
    ops = []
    for _ in range(depth):
        if n >= 2 and rng.random() < 0.4:
            pair = rng.choice(n, size=2, replace=False).tolist()
            ops.append((TWO[rng.integers(len(TWO))](), pair))
        else:
            ops.append((SINGLE[rng.integers(len(SINGLE))](), [int(rng.integers(n))]))
    return ops

def flush(backend, sv):
    if isinstance(sv, np.ndarray):
        return
    if type(sv).__name__ == "MetalState":
        _ = backend._view(sv)[0]
        return
    backend.to_numpy(sv)

def run_cell(backend, n, ops):
    sv = backend.allocate(n)
    t0 = time.perf_counter()
    for mat, targets in ops:
        sv = backend.apply_matrix(sv, mat, targets)
    flush(backend, sv)
    return time.perf_counter() - t0

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", required=True)
    p.add_argument("--qubits", nargs="+", type=int, required=True)
    p.add_argument("--depth", type=int, required=True)
    p.add_argument("--reps", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    args = p.parse_args()
    backend = make_backend(args.backend)
    rows = []
    for n in args.qubits:
        ops = ops_for(n, args.depth, args.seed)
        run_cell(backend, n, ops)
        best = min(run_cell(backend, n, ops) for _ in range(args.reps))
        rows.append({"n_qubits": n, "ms": round(best * 1000, 3)})
    print(RESULT_TAG + " " + json.dumps(rows))

if __name__ == "__main__":
    main()
"""


def _latest_pypi_version() -> str:
    with urllib.request.urlopen("https://pypi.org/pypi/macquerel/json", timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return str(data["info"]["version"])


def _venv_python(root: Path) -> Path:
    if sys.platform == "win32":
        return root / "Scripts" / "python.exe"
    return root / "bin" / "python"


def _make_release_env(version: str, extras: list[str]) -> Path:
    root = Path(tempfile.mkdtemp(prefix=f"macquerel-{version}-"))
    venv.EnvBuilder(with_pip=True).create(root)
    python = _venv_python(root)
    spec = f"macquerel=={version}"
    if extras:
        spec = f"macquerel[{','.join(extras)}]=={version}"
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-compile",
            spec,
        ],
        check=True,
        text=True,
    )
    return python


def _run_backend(
    python: Path,
    backend: str,
    qubits: list[int],
    depth: int,
    reps: int,
    seed: int,
    cwd: Path | None,
) -> tuple[list[dict], str | None]:
    cmd = [
        str(python),
        "-c",
        _BENCH_CODE,
        "--backend",
        backend,
        "--qubits",
        *[str(q) for q in qubits],
        "--depth",
        str(depth),
        "--reps",
        str(reps),
        "--seed",
        str(seed),
    ]
    env = os.environ.copy()
    if cwd is None:
        env.pop("PYTHONPATH", None)
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, env=env, timeout=600)
    if proc.returncode != 0:
        lines = proc.stderr.strip().splitlines() or proc.stdout.strip().splitlines()
        return [], (lines[-1] if lines else f"exit {proc.returncode}")[:160]
    for line in proc.stdout.splitlines():
        if line.startswith(_RESULT_TAG):
            return json.loads(line.split(" ", 1)[1]), None
    return [], "no benchmark result emitted"


def _warm_backend_process(
    python: Path,
    backend: str,
    qubits: list[int],
    depth: int,
    seed: int,
    cwd: Path | None,
) -> None:
    # A fresh venv's first Python subprocess can pay one-time import/bytecode/NumPy
    # initialization costs large enough to swamp these sub-millisecond CPU cells.
    _run_backend(
        python=python,
        backend=backend,
        qubits=[min(qubits)],
        depth=max(1, min(depth, 1)),
        reps=1,
        seed=seed,
        cwd=cwd,
    )


def _print_report(payload: dict, threshold: float) -> int:
    runs = payload["runs"]
    baseline_name = payload["baseline"]
    baseline = runs[baseline_name]["results"]
    backends = payload["config"]["backends"]
    exit_code = 0

    print("\nmacquerel released-version regression benchmark")
    print(f"baseline: {baseline_name}")
    print(f"threshold: {threshold:.1f}% slower than baseline flags a regression\n")

    for version, run in runs.items():
        print(f"== {version} ==")
        for backend in backends:
            rows = run["results"].get(backend, [])
            note = run["skipped"].get(backend)
            if note:
                print(f"  {backend:5s} SKIP  {note}")
                if version == baseline_name:
                    exit_code = 1
                continue
            if version != baseline_name and not baseline.get(backend):
                print(f"  {backend:5s} SKIP  baseline has no {backend} results")
                continue
            print(f"  {backend:5s} {'qubits':>6} {'ms':>10} {'vs baseline':>13}")
            base_rows = {r["n_qubits"]: r["ms"] for r in baseline.get(backend, [])}
            for row in rows:
                bms = base_rows.get(row["n_qubits"])
                if bms is None or version == baseline_name:
                    delta = "baseline"
                else:
                    pct = (row["ms"] - bms) / bms * 100
                    delta = f"{pct:+.1f}%"
                    if pct > threshold:
                        exit_code = 1
                        delta += " REGRESSION"
                print(f"  {'':5s} {row['n_qubits']:>6} {row['ms']:>10.3f} {delta:>13}")
        print()
    return exit_code


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--versions", nargs="+", default=["latest"], help="PyPI versions, or 'latest'")
    p.add_argument("--include-current", action="store_true", default=True)
    p.add_argument("--no-current", dest="include_current", action="store_false")
    p.add_argument("--backends", nargs="+", default=BACKENDS, choices=BACKENDS)
    p.add_argument("--qubits", nargs="+", type=int, default=[8, 10, 12])
    p.add_argument("--depth", type=int, default=20)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=25.0)
    p.add_argument(
        "--extras",
        nargs="*",
        default=[],
        help="extras to install for released versions, e.g. --extras mlx metal",
    )
    p.add_argument("--json", metavar="FILE")
    p.add_argument("--plot", metavar="FILE", default=None)
    args = p.parse_args()

    versions = [_latest_pypi_version() if v == "latest" else v for v in args.versions]
    # Preserve order while removing duplicates.
    versions = list(dict.fromkeys(versions))

    python_by_label: dict[str, Path] = {}
    cwd_by_label: dict[str, Path | None] = {}
    for version in versions:
        print(f"Preparing release environment for macquerel {version}...", flush=True)
        python_by_label[f"v{version}"] = _make_release_env(version, args.extras)
        cwd_by_label[f"v{version}"] = None

    if args.include_current:
        python_by_label["current"] = Path(sys.executable)
        cwd_by_label["current"] = Path(__file__).resolve().parents[1]

    payload = {
        "benchmark": "version_regression",
        "config": vars(args),
        "baseline": f"v{versions[-1]}",
        "runs": {},
    }

    for label, python in python_by_label.items():
        payload["runs"][label] = {"results": {}, "skipped": {}}
        for backend in args.backends:
            print(f"Running {label} {backend}...", flush=True)
            _warm_backend_process(
                python=python,
                backend=backend,
                qubits=sorted(args.qubits),
                depth=args.depth,
                seed=args.seed,
                cwd=cwd_by_label[label],
            )
            rows, error = _run_backend(
                python=python,
                backend=backend,
                qubits=sorted(args.qubits),
                depth=args.depth,
                reps=args.reps,
                seed=args.seed,
                cwd=cwd_by_label[label],
            )
            if error:
                payload["runs"][label]["skipped"][backend] = error
            else:
                payload["runs"][label]["results"][backend] = rows

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
        print(f"\nRaw data -> {path}")

    if args.plot:
        from plot_versions import make_plot

        make_plot(payload, args.plot)

    return _print_report(payload, args.threshold)


if __name__ == "__main__":
    raise SystemExit(main())
