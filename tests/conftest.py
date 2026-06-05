import os

# Pin the fusion width during the test session so fuse_gates() is fast and
# deterministic by default (no per-run autotuning measurement, no dependence on
# a machine-specific on-disk cache). Tests that exercise the autotuner itself
# delete this override explicitly. `setdefault` respects a value the developer
# has already exported.
os.environ.setdefault("MACQUEREL_FUSION_WIDTH", "4")
