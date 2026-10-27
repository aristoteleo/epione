"""Recorded parity of the spatial-method ports against the upstream code.

The gates are pre-registered in tests/parity/spatial_ports/<method>/MANIFEST.md
(sha256-frozen). The runs need the upstream clones, GPU and the public fixtures,
so this test only checks the stored gate results; it is skipped unless
EPIONE_SPATIAL_PARITY_DIR points at a directory holding <method>/gates.json
(e.g. tests/parity/spatial_ports itself).
"""
import hashlib
import json
import os
from pathlib import Path

import pytest

PARITY = os.environ.get("EPIONE_SPATIAL_PARITY_DIR")
pytestmark = pytest.mark.skipif(not PARITY, reason="EPIONE_SPATIAL_PARITY_DIR not set")
HERE = Path(__file__).parent / "parity" / "spatial_ports"


@pytest.mark.parametrize("method", ["cosmos", "present", "instinct", "candies", "praga", "miso", "spamosaic"])
def test_manifest_frozen_and_required_gates_pass(method):
    base = Path(PARITY) / method
    gates = base / "gates.json"
    if not gates.exists():
        pytest.skip(f"no recorded gates for {method}")
    man = HERE / method / "MANIFEST.md"
    # first line: original freeze; later "A<n> <sha>" lines: documented amendments
    lines = [l.split() for l in (HERE / method / "MANIFEST.sha256").read_text().splitlines() if l.strip()]
    hashes = [l[1] if l[0].startswith("A") else l[0] for l in lines if len(l[0]) == 64 or l[0].startswith("A")]
    want = hashes[-1]
    assert hashlib.sha256(man.read_bytes()).hexdigest() == want, "MANIFEST changed after it was frozen"
    res = json.loads(gates.read_text())
    assert res["PASS_required"], json.dumps(res.get("gates", {}), indent=1)
