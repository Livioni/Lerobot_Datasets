#!/usr/bin/env python3
"""Solve camera-frame closed TCP predictions for ARX-X5."""
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robotwin_ik._common import main

if __name__ == "__main__":
    main('ARX-X5')
