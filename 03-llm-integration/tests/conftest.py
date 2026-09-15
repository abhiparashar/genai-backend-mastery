from __future__ import annotations

import pathlib
import sys

# Directory names in this repo start with digits (`03-llm-integration`), which
# are not valid Python identifiers, so the module directory cannot be imported
# as a package. Putting the module root on sys.path makes `import llmkit` work
# from the repo root without any packaging ceremony.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
