"""Named within-type RNA axes for the v2 corpus, as explicit stage-2 conditioning inputs.

Why this exists
---------------
Measured on the patient cohort, a 0.73 SD group shift along the chamber-identity axis arrives at z as a
0.0013 cosine wiggle inside a 512-dimensional vector whose principal structure is cell type, and the
decoder then faithfully passes on something buried. The answer is to give the within-type variation an
axis of its own, next to the embedding, rather than hoping the embedding carries it.

What an axis is here
--------------------
A signed contrast between two curated marker sets, scored per cell on the log-normalised RNA and
standardised within the tissue where it is defined::

    score = mean(log1p CPM of the + markers) - mean(log1p CPM of the - markers)

Not a principal component. v1's within-type arm used unnamed components of a 480-gene panel, and that
panel was already at its information ceiling; an axis that cannot be named cannot be checked against
biology, and a component of a small panel mostly re-encodes the panel.

Tissue scope is part of the definition
--------------------------------------
Chamber identity means something in heart and nothing in liver. Each axis therefore ships a mask, and a
cell outside the axis's tissue gets value 0 with mask 0 -- never a fabricated score. :mod:`._v2_decoder`
multiplies by the mask, so an undefined axis contributes nothing rather than a confident zero.

    from epione.models.aceso import build_axes
    build_axes("data/train_v2")           # -> data/train_v2/axes.npz (score, mask, names) + axes.json
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

# The chamber axis is the one the within-type figures are about, and its marker sets are those used to
# define the identity axis on the patient cohort, so the training-time axis and the figure's axis are
# the same quantity.
AXES: Dict[str, dict] = {
    "chamber": dict(
        tissue=["heart"],
        pos=["MYH7", "MYL2", "IRX4", "HEY2", "FHL2", "MYL3", "PLN", "NPPB", "HOPX", "IRX5"],
        neg=["NPPA", "MYL7", "MYL4", "SLN", "GJA5", "KCNJ3", "NR2F2", "KCNA5", "PITX2", "BMP10"],
        note="ventricular (+) vs atrial (-)"),
    # A second, tissue-general axis so the mechanism is not demonstrated on one contrast alone. Cell
    # cycle is defined in every tissue and is the classic within-type axis an atlas averages away.
    "cycle": dict(
        tissue=None,
        pos=["MKI67", "TOP2A", "CCNB1", "CDK1", "BUB1", "AURKA", "PLK1", "CENPF", "UBE2C", "TPX2"],
        neg=["CDKN1A", "CDKN1B", "CDKN2A", "RB1", "GAS1", "GADD45A"],
        note="proliferative (+) vs arrested (-); defined in every tissue"),
}


def build_axes(store, axes: Optional[Dict[str, dict]] = None, *, chunk: int = 100_000, log=print) -> str:
    """Score every cell of a v2 store on each named axis and write ``axes.npz`` / ``axes.json``.

    ``store`` is the directory holding ``cells.tsv`` and ``rna.h5ad`` (see :mod:`._store`). Returns the
    path of the written ``axes.npz``, whose arrays are ``score`` and ``mask`` (cells x axes) and ``names``.
    """
    import h5py
    import pandas as pd

    axes = AXES if axes is None else axes
    store = Path(store)
    O = pd.read_csv(store / "cells.tsv", sep="\t")
    with h5py.File(store / "rna.h5ad") as f:
        genes = np.array([g.decode() if isinstance(g, bytes) else str(g) for g in f["var/_index"][:]])
        indptr = f["X/indptr"][:]
        n = len(indptr) - 1
        assert n == len(O), (n, len(O))
        log(f"store {n:,} cells x {len(genes):,} genes")
        gi = {g: i for i, g in enumerate(genes)}

        want = sorted({g for A in axes.values() for g in A["pos"] + A["neg"] if g in gi})
        cols = np.array([gi[g] for g in want])
        log(f"  {len(want)}/{sum(len(A['pos']) + len(A['neg']) for A in axes.values())} marker genes present")

        # one pass, reading only the marker columns; also the per-cell total for CPM
        sub = np.zeros((n, len(cols)), np.float32)
        tot = np.zeros(n, np.float64)
        csort = np.argsort(cols)
        cs = cols[csort]
        t0 = time.time()
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            a, b = int(indptr[s]), int(indptr[e])
            ind = f["X/indices"][a:b]
            dat = f["X/data"][a:b]
            rowlen = np.diff(indptr[s:e + 1])
            rows = np.repeat(np.arange(s, e) - s, rowlen)
            np.add.at(tot, np.arange(s, e), np.bincount(rows, weights=dat, minlength=e - s))
            pos_ = np.searchsorted(cs, ind)
            hit = (pos_ < len(cs)) & (cs[np.clip(pos_, 0, len(cs) - 1)] == ind)
            if hit.any():
                sub[rows[hit] + s, csort[pos_[hit]]] = dat[hit]
            if (s // chunk) % 5 == 0:
                log(f"  {e:,}/{n:,}  [{time.time() - t0:.0f}s]")

    cpm = sub / np.clip(tot, 1, None)[:, None] * 1e4
    L = np.log1p(cpm)
    col_of = {g: i for i, g in enumerate(want)}
    organ = O["organ"].astype(str).str.lower().to_numpy()

    names, S, M = [], [], []
    for nm, A in axes.items():
        p = [col_of[g] for g in A["pos"] if g in col_of]
        q = [col_of[g] for g in A["neg"] if g in col_of]
        if not p or not q:
            log(f"  !! {nm}: markers missing, skipped")
            continue
        sc = L[:, p].mean(1) - L[:, q].mean(1)
        mask = (np.ones(len(sc), bool) if A["tissue"] is None
                else np.isin(organ, [t.lower() for t in A["tissue"]]))
        # standardised inside the tissue where it is defined, so the scale is comparable between axes and
        # a cell outside that tissue is zeroed rather than given an out-of-distribution value
        if mask.sum() > 10:
            mu, sd = sc[mask].mean(), sc[mask].std()
            sc = np.where(mask, (sc - mu) / max(sd, 1e-6), 0.0).astype(np.float32)
        names.append(nm)
        S.append(sc)
        M.append(mask.astype(np.float32))
        log(f"  {nm:9s} {len(p)}+/{len(q)}- markers, defined in {int(mask.sum()):,} cells "
            f"({mask.mean() * 100:.1f}%), sd {sc[mask].std():.3f}")

    S = np.vstack(S).T.astype(np.float32)
    M = np.vstack(M).T.astype(np.float32)
    tmp = store / "axes.tmp.npz"
    np.savez_compressed(tmp, score=S, mask=M, names=np.array(names))
    os.replace(tmp, store / "axes.npz")
    json.dump({k: dict(tissue=v["tissue"], pos=v["pos"], neg=v["neg"], note=v["note"])
               for k, v in axes.items() if k in names},
              open(store / "axes.json", "w"), indent=1)
    log(f"\n-> {store}/axes.npz   {S.shape} (cells x axes): {names}")
    for i, nm in enumerate(names):
        d = M[:, i] > 0
        log(f"  {nm:9s} defined {int(d.sum()):>9,}  range {S[d, i].min():+.2f} .. {S[d, i].max():+.2f}")
    return str(store / "axes.npz")
