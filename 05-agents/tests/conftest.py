from __future__ import annotations

import pathlib
import sys

# `05-agents` starts with a digit, so it is not an importable package name.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
