import os

# fuse_gates() already defaults to a fixed width of 4 with no measurement, so the
# suite is fast and deterministic out of the box. Pinning the env var here just
# guards against a developer who has exported MACQUEREL_FUSION_WIDTH=auto (which
# would otherwise trigger the per-chip measurement on every fuse). Tests that
# exercise the autotuner/resolver set or delete this override explicitly.
if os.environ.get("MACQUEREL_FUSION_WIDTH", "").strip().lower() == "auto":
    os.environ["MACQUEREL_FUSION_WIDTH"] = "4"
