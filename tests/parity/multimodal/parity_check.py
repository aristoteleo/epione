"""Parity checks for epione's multimodal functions (see MANIFEST.md).

Usage::

    EPIONE_MULTIMODAL_PARITY_DIR=/path/to/multimodal_parity \
        python tests/parity/multimodal/parity_check.py combined_dims
    python tests/parity/multimodal/parity_check.py wnn

References live outside the repository: ``combined_dims/`` holds ArchR's
inputs and ``addCombinedDims`` outputs, ``wnn/`` holds the shared embeddings
and Seurat's / muon's WNN results.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PARITY = Path(os.environ.get("EPIONE_MULTIMODAL_PARITY_DIR", ""))

C_CONFIGS = {
    # id: (dim_weights, dims_to_use, scaleDims as given to ArchR, cor_cutoff)
    "C1": (None, None, None, 0.75),
    "C2": ([1.0, 0.5], None, None, 0.75),
    "C3": (None, list(range(1, 21)), True, 0.75),
    "C4": (None, None, False, 0.5),
}


def _read_tsv(path):
    df = pd.read_csv(path, sep="\t", index_col=0)
    return df.to_numpy(dtype=np.float64), list(df.index)


def check_combined_dims() -> dict:
    import anndata as ad
    import epione as epi

    d = PARITY / "combined_dims"
    reps, info = {}, {}
    for rd in ("IterativeLSI", "Harmony"):
        reps[rd], cells = _read_tsv(d / f"in_{rd}.tsv")
        info[rd] = json.loads((d / f"in_{rd}.json").read_text())
    adata = ad.AnnData(np.zeros((len(cells), 1)), obs=pd.DataFrame(index=cells))
    for rd, m in reps.items():
        adata.obsm[rd] = m

    results = {}
    for cid, (w, dims, scale_arg, cut) in C_CONFIGS.items():
        # ArchR getReducedDims: a stored scaleDims of NA forces FALSE; NULL uses the stored value.
        scales, ctd = [], {}
        for rd in ("IterativeLSI", "Harmony"):
            stored = info[rd]["scaleDims"]
            if stored is None:
                s = False
            elif scale_arg is None:
                s = bool(stored)
            else:
                s = bool(scale_arg)
            scales.append(s)
            key = "corToDepth_scaled" if s else "corToDepth_none"
            ctd[rd] = info[rd][key] if info[rd][key] is not None else None
        out = epi.tl.combined_dims(adata, ["IterativeLSI", "Harmony"], dim_weights=w, dims_to_use=dims,
                                   scale_dims=scales, cor_cutoff=cut, cor_to_depth=ctd, key_added=cid)
        ref, ref_cells = _read_tsv(d / f"out_{cid}.tsv")
        assert ref_cells == cells
        same_shape = out.shape == ref.shape
        maxdiff = float(np.max(np.abs(out - ref))) if same_shape else float("inf")
        results[cid] = {"shape_epione": list(out.shape), "shape_archr": list(ref.shape),
                        "max_abs_diff": maxdiff, "pass": bool(same_shape and maxdiff <= 1e-10)}
    return results


def main(argv):
    what = argv[1] if len(argv) > 1 else "combined_dims"
    res = {"combined_dims": check_combined_dims}[what]() if what == "combined_dims" else check_wnn()
    print(json.dumps(res, indent=2))
    return 0 if all(v.get("pass", True) for v in res.values() if isinstance(v, dict)) else 1


def check_wnn() -> dict:  # filled in below once the references exist
    raise NotImplementedError


if __name__ == "__main__":
    sys.exit(main(sys.argv))
