#!/usr/bin/env python3
"""Run UEFB-G v1 public benchmark evaluator on a per-image metrics submission."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.uefbg_eval import run_cli


if __name__ == '__main__':
    run_cli()
