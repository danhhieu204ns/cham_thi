"""Shared bootstrap for baseline script wrappers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import sys


BASELINE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = BASELINE_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def run(main: Callable[[], int]) -> None:
    raise SystemExit(main())
