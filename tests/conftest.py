import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (
    ROOT,
    *(os.path.join(ROOT, d) for d in ("source", "pipeline", "simulator")),
):
    if path not in sys.path:
        sys.path.insert(0, path)
