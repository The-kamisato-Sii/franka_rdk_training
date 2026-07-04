#!/usr/bin/env python
"""WMAM HTTP deployment server for Franka-dual checkpoints."""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

RLINF_ROOT = Path(__file__).resolve().parents[2]
if str(RLINF_ROOT) not in sys.path:
    sys.path.insert(0, str(RLINF_ROOT))

from deploy.dreamzero.serve_franka_dual_dreamzero import run_server


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        run_server("wmam")
