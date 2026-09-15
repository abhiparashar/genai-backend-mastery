from __future__ import annotations

import pathlib
import sys

# `04-rag` starts with a digit, so it is not an importable package name.
# Putting the module root on sys.path makes `import ragkit` work from the repo
# root without packaging ceremony.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
