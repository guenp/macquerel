#!/bin/bash
# Run the per-step benchmark for the GPU-perf plan (docs/plan.md Steps 21-29).
#
# Usage: run_step_bench.sh <step-label> <commit> <backend> [<backend> ...]
#
# Checks out <commit> in the ../macquerel-bench worktree (so the main working
# tree can keep moving), runs bench_statevector.py there for each backend with
# the step-comparison qubit sets, and writes JSONs tagged with the step label
# and commit id into benchmarks/data/steps/ of the *main* repo.
set -euo pipefail

STEP="$1"; COMMIT="$2"; shift 2
MAIN_REPO="$(cd "$(dirname "$0")/.." && pwd)"
BENCH_TREE="$MAIN_REPO/../macquerel-bench"
OUT_DIR="$MAIN_REPO/benchmarks/data/steps"
mkdir -p "$OUT_DIR"

cd "$BENCH_TREE"
git checkout --quiet "$COMMIT"
uv sync --quiet --extra mlx --extra metal

for BACKEND in "$@"; do
  case "$BACKEND" in
    macquerel-cpu) QUBITS="6 12 16 20 22" ;;
    *)             QUBITS="6 12 16 20 22 24 26 28" ;;
  esac
  SHORT="${BACKEND#macquerel-}"
  OUT="$OUT_DIR/${STEP}-${COMMIT}-${SHORT}.json"
  echo "=== $STEP $BACKEND (commit $COMMIT) qubits: $QUBITS ==="
  uv run python benchmarks/bench_statevector.py \
    --backends "$BACKEND" --qubits $QUBITS --reps 3 \
    --json "$OUT" --plot /tmp/step-bench-discard.png
  uv run python - "$OUT" "$STEP" "$COMMIT" <<'EOF'
import json, sys
path, step, commit = sys.argv[1:4]
doc = json.load(open(path))
doc["step"] = step
doc["commit"] = commit
json.dump(doc, open(path, "w"), indent=2)
EOF
done
echo "=== done: $STEP ==="
