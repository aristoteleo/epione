"""scPair parity: runs ``tests/parity/translate_scbutterfly/parity_check.py`` with ``--method scpair``.

    python parity_check.py train --out RUNS
    python parity_check.py gates --out RUNS --json results.json
"""
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--method", "scpair"] + sys.argv[1:]
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "translate_scbutterfly" / "parity_check.py"), run_name="__main__")
