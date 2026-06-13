#!/bin/bash
# Run the per-step benchmark for the GPU-perf plan (docs/plan.md Steps 21-39)
# through the ASV harness (benchmarks/asv_benchmarks/).
#
# Usage: run_step_bench.sh <step-label> <commit> <backend> [<backend> ...]
#
# `asv run` builds <commit> in its own clone under benchmarks/.asv/ (so the
# main working tree can keep moving) and times the statevector cells for the
# requested backends, one process per cell. The results are then exported to
# benchmarks/data/steps/<step>-<commit>-<backend>.json in the same format the
# previous bench_statevector.py-based runner wrote, so plot_steps.py renders
# identical charts from them. Qubit sets match the old runner: cpu 6-22,
# mlx/metal 6-28 (encoded in asv_benchmarks/statevector.py).
set -euo pipefail

STEP="$1"; COMMIT="$2"; shift 2
MAIN_REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$MAIN_REPO"
FULL="$(git rev-parse "$COMMIT")"

BENCH_RE="time_statevector\\(.*'($(IFS='|'; echo "$*"))'"
echo "=== $STEP (commit ${FULL:0:7}) backends: $* ==="
(cd benchmarks && uv run --extra bench asv run "${FULL}^!" \
  --bench "$BENCH_RE" --record-samples --show-stderr)

uv run python benchmarks/plot_steps.py --export-asv "$STEP" "$FULL" "$@"
echo "=== done: $STEP ==="
