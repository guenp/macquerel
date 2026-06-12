"""Shared helpers for ASV benchmarks.

The existing benchmark scripts live in ``benchmarks/`` and some of them import
their siblings as top-level modules. ASV imports from ``asv_benchmarks/``, so we
add the legacy benchmark directory to ``sys.path`` before reusing those helpers.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Any

LEGACY_BENCHMARKS = Path(__file__).resolve().parents[1]
if str(LEGACY_BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(LEGACY_BENCHMARKS))


def optional_backend(factory, *args: Any, **kwargs: Any) -> Any:
    """Build an optional backend or mark the ASV cell as unsupported."""
    try:
        return factory(*args, **kwargs)
    except Exception as exc:
        raise NotImplementedError(str(exc)[:160]) from exc


def force_gc() -> None:
    """Keep repeated ASV cells from retaining large statevector temporaries."""
    gc.collect()
