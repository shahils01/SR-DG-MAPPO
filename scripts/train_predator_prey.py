#!/usr/bin/env python
"""Source-tree entry point for predator-prey MARL training."""

from pathlib import Path
import sys


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sr_dg_mappo.train import cli


if __name__ == "__main__":
    cli()
