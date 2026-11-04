"""Doublet parity checks against references outside the repository.

``EPIONE_DOUBLETS_PARITY_DIR`` points at the reference directory, which holds:

- ``archr_fixture/out/seed{1,2,3}_scores.tsv``: ArchR 1.0.3 ``addDoubletScores``
  (Sample, nFrags, DoubletScore, DoubletEnrichment) on the heme fixture;
- ``archr_fixture/out/seed1_filterDoublets_removed.txt``: cells ArchR's
  ``filterDoublets`` removed;
- ``archr_fixture/out/tile_<sample>.h5``: the binarised TileMatrix given to both
  sides;
- ``analysis/*.json``: gate results written by the drivers.

Gate F1 is re-verified here from the raw references. Gate A results are read
from ``analysis/gate_a.json`` and ``gate_a_amendment.json``. Recomputing them
takes about 25 minutes, so the drivers do that (see MANIFEST.md).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def parity_dir() -> Path | None:
    d = os.environ.get("EPIONE_DOUBLETS_PARITY_DIR")
    return Path(d) if d else None


def gate_f1(ref: Path) -> dict:
    """epi.pp.filter_doublets(method='archr') vs ArchR filterDoublets on ArchR's scores."""
    import anndata as ad
    import epione as epi
    out = ref / "archr_fixture/out"
    df = pd.read_csv(out / "seed1_scores.tsv", sep="\t", index_col=0)
    a = ad.AnnData(obs=df.rename(columns={"DoubletScore": "archr_doublet_score",
                                          "DoubletEnrichment": "archr_doublet_enrichment"}))
    epi.pp.filter_doublets(a, method="archr", sample_key="Sample")
    ours = set(df.index) - set(a.obs_names)
    archr = set((out / "seed1_filterDoublets_removed.txt").read_text().split())
    return dict(identical=ours == archr, n_epione=len(ours), n_archr=len(archr))


def recorded(ref: Path, name: str) -> dict:
    return json.loads((ref / "analysis" / name).read_text())
