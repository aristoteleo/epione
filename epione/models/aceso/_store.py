"""The consolidated ACESO v2 training store: one directory, one row order, ready to ship to Marlowe.

One directory holds everything the v2 trainers need, with ONE row order shared by every array, because
the commonest silent failure in a multi-modal store is a mis-joined row order::

    data/train_v2/
      cells.tsv        one row per cell: cell_id, corpus, sample, organ, cell_type, stage_or_age,
                       n_umi, n_genes          <- THE row order for everything else
      rna.h5ad         cells x scGPT-vocab genes, int32 counts, symbol-keyed
      z_epiagent.npy   cells x 512, row i == cells.tsv row i      (written by the EpiAgent step)
      fragments/       per-sample tabix'd fragments in real-cell barcode space (symlinked, not copied)
      splits.json      train / val / test BY SAMPLE, never by cell
      manifest.json    sizes, md5s, and the provenance of every piece

:func:`build_training_store` writes it, :class:`TrainingStore` reads it, and :mod:`._v2_encoder` and
:mod:`._v2_decoder` consume it. :mod:`._axes` adds ``axes.npz`` next to ``cells.tsv`` in the same row
order, and :mod:`._exon` keys its pooling masks on the gene symbols of ``rna.h5ad``.

Why the RNA store is symbol-keyed, and why that needs care
----------------------------------------------------------
scGPT's vocabulary is 60,697 **symbols**, while the ingested h5ads are indexed on **Ensembl** -- chosen
there because HDMA's 38,705 symbols are not unique, so a symbol index would make a gene lookup resolve
silently to whichever row came first. Going back to symbols here therefore has to **sum the duplicate
rows**, not pick one. Measured on one HDMA sample: 24,540 of 38,705 genes are in scGPT's vocabulary and
they carry **96.9% of UMIs** -- the genes dropped are overwhelmingly low-expression, and the ones kept
include the non-coding regulators (XIST, KCNQ1OT1, MALAT1, HAND2-AS1) that an ESM-2 gene space could
not hold. That is why the backbone stayed scGPT.

Why splits are by sample
------------------------
Cells from one 10x channel share ambient RNA, barcode collisions and batch; a cell-level split leaks all
three and inflates every metric. v1 held out whole samples (``E7.5_rep2 / E8.5_rep2 / E8.75_rep2``) and
this keeps that discipline across 76 + 97 samples.

Usage
-----
::

    from epione.models.aceso._store import StoreConfig, build_training_store, TrainingStore
    build_training_store(StoreConfig(corpus="hdma", organ="Heart"))                 # one organ
    build_training_store(StoreConfig(corpus="hdma,encode", all_samples=True))       # all ingested
    store = TrainingStore("data/train_v2")

or, equivalently, from the command line::

    python -m epione.models.aceso._store --corpus hdma --organ Heart
    python -m epione.models.aceso._store --corpus hdma,encode --all
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ._sites import register_site, require_sites, site_path

# The ingest directories and HDMA's two metadata tables are resources like any other, so they are
# registered rather than hardcoded; a machine that keeps them elsewhere sets the variable or the
# sites file instead of editing this module.
for _name, _env, _cand in (
    ("hdma_ingested", "HDMA_INGESTED", "{R}/data/hdma/ingested"),
    ("hdma_cell_meta", "HDMA_CELL_META", "{R}/data/hdma/meta/per_cell_meta.csv"),
    ("hdma_ts2", "HDMA_TS2", "{R}/data/hdma/meta/ts2.tsv"),
    ("encode_ingested", "ENCODE_INGESTED", "{R}/data/encode_adult/ingested"),
):
    register_site(_name, _cand, env=_env)

_UNSET = object()


@dataclass
class StoreConfig:
    """What to consolidate and where from; every path field is resolved through :mod:`._sites` if ``None``."""

    corpus: str = "hdma,encode"
    organ: Optional[str] = None
    all_samples: bool = False
    # an explicit sample list, so a store can be built from exactly the samples that already have z
    # rather than waiting for the whole corpus to finish embedding
    samples: Optional[str] = None
    out: Optional[str] = None
    val_frac: float = 0.08
    test_frac: float = 0.08
    seed: int = 0
    # resolved at run time: project root, the scGPT model directory holding vocab.json, and the
    # per-corpus ingest directories
    root: Optional[str] = None
    scgpt_model: Optional[str] = None
    hdma_ingested: Optional[str] = None
    hdma_cell_meta: Optional[str] = None
    hdma_ts2: Optional[str] = None
    encode_ingested: Optional[str] = None

    def path(self, name: str) -> Path:
        """This config's value for a path field, else the :mod:`._sites` resolution of the same name."""
        return Path(getattr(self, name) or site_path(name))


def hdma_samples(c: StoreConfig) -> List[dict]:
    """The ingested HDMA samples selected by ``c``, each with its per-cell metadata and label table."""
    import pandas as pd

    meta = pd.read_csv(c.path("hdma_cell_meta"), low_memory=False)
    meta["sample"] = meta.cb.str.split("#").str[0]
    meta["barcode"] = meta.cb.str.split("#").str[1]
    ts2 = pd.read_csv(c.path("hdma_ts2"), sep="\t").set_index("Cluster")
    ing = c.path("hdma_ingested")
    out = []
    for rna in sorted(ing.glob("*.rna.h5ad")):
        tag = rna.name.replace(".rna.h5ad", "")
        organ = tag.split("_")[2] if len(tag.split("_")) > 2 else ""
        if c.organ and not c.samples and organ != c.organ:
            continue
        pcw = tag.split("PCW")[-1] if "PCW" in tag else ""
        out.append(dict(corpus="hdma", tag=tag, rna=rna, organ=organ, stage=f"PCW{pcw}",
                        meta=meta[meta["sample"] == tag], labels=ts2))
    return out


def encode_samples(c: StoreConfig) -> List[dict]:
    """The ingested ENCODE adult samples selected by ``c``; they carry no label table."""
    ing = c.path("encode_ingested")
    out = []
    for rna in sorted(ing.glob("*.rna.h5ad")):
        tag = rna.name.replace(".rna.h5ad", "")
        organ = tag.split("_")[0]
        if c.organ and not c.samples and organ != c.organ:
            continue
        out.append(dict(corpus="encode", tag=tag, rna=rna, organ=organ, stage="", meta=None,
                        labels=None))
    return out


def build_training_store(config: StoreConfig = StoreConfig(), *, log=print) -> str:
    """Consolidate the ingested corpora into one v2 training store; returns the store directory."""
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp

    c = config
    need = [n for n in ("root", "scgpt_model") if getattr(c, n) is None]
    if need:
        require_sites(*need)
    R = c.path("root")
    SCGPT = c.path("scgpt_model")
    OUT = R / "data/train_v2"
    if c.out:
        OUT = Path(c.out) if Path(c.out).is_absolute() else R / c.out
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fragments").mkdir(exist_ok=True)

    VOCAB = set(json.load(open(SCGPT / "vocab.json")).keys())
    log(f"scGPT vocabulary: {len(VOCAB):,} tokens")

    S = []
    for name in c.corpus.split(","):
        S += {"hdma": hdma_samples, "encode": encode_samples}[name](c)
    if c.samples:
        want = set(c.samples.split(","))
        S = [s for s in S if s["tag"] in want]
    log(f"{len(S)} sample(s) to consolidate")
    if not S:
        sys.exit("nothing ingested for this corpus/organ")

    blocks, obs_rows = [], []
    var_sym = None
    unreadable = []
    for i, s in enumerate(S):
        t0 = time.time()
        # One unreadable input must not stop the other 146. At this scale a half-written h5ad from an
        # interrupted ingest step is a normal event, and the atomic-rename writes added below make it
        # rarer -- but a store build that dies on file 5 of 147 wastes more than it protects.
        try:
            A = ad.read_h5ad(s["rna"])
            _ = A.X[:1]                      # force the filter pipeline, which is where corruption shows
        except Exception as e:
            unreadable.append((s["tag"], f"{type(e).__name__}"))
            log(f"  [{i+1}/{len(S)}] {s['tag']}: UNREADABLE ({type(e).__name__}) -- skipped")
            continue
        # Ensembl -> symbol, then restrict to scGPT's vocabulary and SUM duplicate symbols.
        sym = (A.var.symbol.astype(str).to_numpy() if "symbol" in A.var.columns
               else A.var_names.astype(str).to_numpy())
        keep = np.isin(sym, list(VOCAB))
        X = A.X.tocsc()[:, keep]
        k_sym = sym[keep]
        # group identical symbols by a sparse right-multiply: a (kept genes x unique symbols) 0/1 matrix.
        # This sums duplicates exactly, where `A[:, first_index]` would silently drop the others.
        uniq, inv = np.unique(k_sym, return_inverse=True)
        G = sp.csr_matrix((np.ones(len(inv), np.float32), (np.arange(len(inv)), inv)),
                          shape=(len(inv), len(uniq)))
        X = (X @ G).tocsr()
        if var_sym is None:
            var_sym = uniq
        elif not np.array_equal(var_sym, uniq):
            # different samples keep different gene sets; align onto the union seen so far
            allsym = np.union1d(var_sym, uniq)

            def expand(M, cols, target):
                idx = pd.Index(target).get_indexer(cols)
                P = sp.csr_matrix((np.ones(len(idx), np.float32), (np.arange(len(idx)), idx)),
                                  shape=(len(idx), len(target)))
                return (M.tocsr() @ P).tocsr()
            blocks = [expand(B, var_sym, allsym) for B in blocks]
            X = expand(X, uniq, allsym)
            var_sym = allsym
        X.data = np.rint(X.data).astype(np.float32)
        blocks.append(X)

        cells = A.obs_names.astype(str).to_numpy()
        if s["corpus"] == "hdma":
            m = s["meta"].set_index("barcode")
            cl = m.reindex(cells).Cluster
            ct = s["labels"].reindex(cl).L3_annot.to_numpy()
        else:
            ct = np.array(["unlabelled"] * len(cells))
        nu = np.asarray(X.sum(1)).ravel()
        ng = np.diff(X.indptr)
        obs_rows.append(pd.DataFrame(dict(
            cell_id=[f"{s['tag']}#{b}" for b in cells], corpus=s["corpus"], sample=s["tag"],
            organ=s["organ"], cell_type=ct,
            stage_or_age=(s["stage"] if s["corpus"] == "hdma"
                          else A.obs.age.iloc[0] if "age" in A.obs.columns else ""),
            n_umi=nu.astype(np.int64), n_genes=ng.astype(np.int32))))
        # fragments: symlink rather than copy, so the store adds no bytes on this filesystem
        for suf in (".fragments.tsv.gz", ".fragments.tsv.gz.tbi"):
            src = s["rna"].parent / (s["tag"] + suf)
            dst = OUT / "fragments" / (s["tag"] + suf)
            if src.exists() and not dst.exists():
                dst.symlink_to(src)
        log(f"  [{i+1}/{len(S)}] {s['tag']}: {X.shape[0]:,} cells x {X.shape[1]:,} genes "
            f"({ng.mean():.0f} genes/cell) [{time.time()-t0:.0f}s]")

    if unreadable:
        log(f"\n{len(unreadable)} sample(s) skipped as unreadable:")
        for tag, why in unreadable:
            log(f"   {tag}: {why}")
        log("  delete those .rna.h5ad files and re-run the ingest to regenerate them")
    X = sp.vstack(blocks).tocsr()
    del blocks
    O = pd.concat(obs_rows, ignore_index=True)
    assert X.shape[0] == len(O), (X.shape, len(O))
    X.indices = X.indices.astype(np.int32)        # column ids: 39,028 genes, int32 is ample
    # INDPTR MUST STAY int64 ONCE THE CORPUS PASSES 2.1e9 NONZEROS. The full v2 store holds 3,409,034,954,
    # so int32 wraps at row 1,241,158 and every later row pointer goes negative. Nothing raises: h5py
    # reads a negative start as an offset from the END of the array, so rows past that point return a
    # real-looking count vector belonging to a DIFFERENT cell. Measured 2026-11-12: row sums matched
    # cells.tsv's n_umi for 40/40 rows before the wrap and 0/40 after, and a stage-1 run had already begun
    # training with 37.1% of its cells paired to the wrong RNA. Cast only when it is safe to.
    if X.indptr[-1] <= np.iinfo(np.int32).max:
        X.indptr = X.indptr.astype(np.int32)
    else:
        X.indptr = X.indptr.astype(np.int64)
        log(f"  indptr kept int64: {X.indptr[-1]:,} nonzeros exceeds the int32 limit "
            f"{np.iinfo(np.int32).max:,}")

    # ---------------------------------------------------------------- splits, by SAMPLE
    rng = np.random.default_rng(c.seed)
    samples = O["sample"].unique()
    rng.shuffle(samples)
    n_te = max(1, int(round(len(samples) * c.test_frac)))
    n_va = max(1, int(round(len(samples) * c.val_frac)))
    split = {"test": sorted(samples[:n_te].tolist()),
             "val": sorted(samples[n_te:n_te + n_va].tolist()),
             "train": sorted(samples[n_te + n_va:].tolist())}
    O["split"] = O["sample"].map({s: k for k, v in split.items() for s in v})

    O.to_csv(OUT / "cells.tsv", sep="\t", index=False)
    AD = ad.AnnData(X=X, obs=pd.DataFrame(index=pd.Index(O.cell_id, name=None)),
                    var=pd.DataFrame(index=pd.Index(var_sym, name=None)))
    AD.uns["scgpt_vocab"] = str(SCGPT / "vocab.json")
    AD.write_h5ad(OUT / "rna.h5ad", compression="gzip", compression_opts=4)
    json.dump(split, open(OUT / "splits.json", "w"), indent=1)

    man = dict(built=time.strftime("%Y-%m-%dT%H:%M:%S"), n_cells=int(len(O)),
               n_genes=int(X.shape[1]), nnz=int(X.nnz),
               corpora={k: int(v) for k, v in O.corpus.value_counts().items()},
               organs={k: int(v) for k, v in O.organ.value_counts().items()},
               samples=int(O["sample"].nunique()), split_sizes={k: len(v) for k, v in split.items()},
               scgpt_vocab_size=len(VOCAB),
               rna_md5=hashlib.md5((OUT / "rna.h5ad").read_bytes()).hexdigest())
    json.dump(man, open(OUT / "manifest.json", "w"), indent=1)

    log(f"\n*** {len(O):,} cells x {X.shape[1]:,} scGPT-vocab genes, {X.nnz/1e6:.1f}M nnz ***")
    log(f"  corpora: {man['corpora']}")
    log(f"  organs:  {man['organs']}")
    log(f"  splits (by sample): " + ", ".join(f"{k} {len(v)}" for k, v in split.items()))
    log(f"  median genes/cell {np.median(np.diff(X.indptr)):.0f}, median UMI {np.median(O.n_umi):.0f}")
    log(f"\n-> {OUT}")
    log("  STILL MISSING: z_epiagent.npy -- written by the EpiAgent step, which must use THIS "
        "cells.tsv row order")
    return str(OUT)


class TrainingStore:
    """Read-only view of a v2 training store directory; every piece is read on first use.

    ``cells`` is the row order; ``genes``, ``splits``, ``manifest``, ``z``, ``axes`` and ``fragments``
    are the pieces written alongside it (see the module docstring for the contract).
    """

    def __init__(self, path):
        self.path = Path(path)
        if not self.path.is_dir():
            raise FileNotFoundError(f"{self.path} is not a directory; a v2 training store is a directory "
                                    f"holding cells.tsv, rna.h5ad, splits.json and fragments/")
        self._cells = _UNSET
        self._genes = _UNSET
        self._splits = _UNSET
        self._manifest = _UNSET
        self._z = _UNSET
        self._axes = _UNSET
        self._fragments = _UNSET

    def _need(self, name: str, why: str) -> Path:
        """The store's path for ``name``, or a ``FileNotFoundError`` saying what it should have been."""
        q = self.path / name
        if not q.exists():
            raise FileNotFoundError(f"{q} is missing; a v2 training store must hold {name}: {why}")
        return q

    @property
    def cells(self):
        """``cells.tsv`` as a DataFrame -- THE row order for every other array in the store."""
        if self._cells is _UNSET:
            import pandas as pd
            q = self._need("cells.tsv", "one row per cell, the row order of rna.h5ad and z_epiagent.npy")
            self._cells = pd.read_csv(q, sep="\t", low_memory=False)
        return self._cells

    @property
    def genes(self) -> List[str]:
        """The gene symbols of ``rna.h5ad`` (its ``var`` index), read without touching ``X``."""
        if self._genes is _UNSET:
            import h5py
            q = self._need("rna.h5ad", "cells x scGPT-vocab genes, int32 counts, symbol-keyed")
            with h5py.File(q, "r") as f:
                self._genes = [g.decode() if isinstance(g, bytes) else str(g) for g in f["var/_index"][:]]
        return self._genes

    @property
    def splits(self) -> Dict[str, List[str]]:
        """``splits.json``: train / val / test as lists of sample tags, never cells."""
        if self._splits is _UNSET:
            q = self._need("splits.json", "train / val / test BY SAMPLE, never by cell")
            self._splits = json.loads(q.read_text())
        return self._splits

    @property
    def manifest(self) -> Optional[dict]:
        """``manifest.json`` -- sizes, md5s and provenance; ``None`` if the store was not finished."""
        if self._manifest is _UNSET:
            q = self.path / "manifest.json"
            self._manifest = json.loads(q.read_text()) if q.exists() else None
        return self._manifest

    @property
    def z(self) -> np.ndarray:
        """``z_epiagent.npy`` memory-mapped read-only: row i is ``cells`` row i."""
        if self._z is _UNSET:
            q = self._need("z_epiagent.npy", "cells x 512 EpiAgent states in cells.tsv row order, "
                                             "written by the EpiAgent step")
            self._z = np.load(q, mmap_mode="r")
        return self._z

    @property
    def axes(self) -> Optional[dict]:
        """``axes.npz`` as ``{score, mask, names}`` (see :mod:`._axes`), or ``None`` if not built."""
        if self._axes is _UNSET:
            q = self.path / "axes.npz"
            if not q.exists():
                self._axes = None
            else:
                z = np.load(q, allow_pickle=False)
                self._axes = dict(score=z["score"], mask=z["mask"], names=[str(n) for n in z["names"]])
        return self._axes

    @property
    def fragments(self) -> Dict[str, str]:
        """Sample tag -> path of its tabix'd ``fragments/*.fragments.tsv.gz``."""
        if self._fragments is _UNSET:
            d = self.path / "fragments"
            self._fragments = {q.name[:-len(".fragments.tsv.gz")]: str(q)
                               for q in sorted(d.glob("*.fragments.tsv.gz"))} if d.is_dir() else {}
        return self._fragments

    def summary(self):
        """Per corpus: cells, samples, organs, and the cells whose sample has a fragment file."""
        import pandas as pd

        O = self.cells
        g = O.groupby("corpus", sort=True)
        has = O["sample"].isin(set(self.fragments)).groupby(O["corpus"], sort=True).sum()
        return pd.DataFrame(dict(cells=g.size(), samples=g["sample"].nunique(),
                                 organs=g["organ"].nunique(), cells_with_fragments=has))


def main(argv=None):
    """CLI: build the v2 training store from the ingested corpora."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="hdma,encode")
    ap.add_argument("--organ")
    ap.add_argument("--all", action="store_true", dest="all_samples")
    # an explicit sample list, so a store can be built from exactly the samples that already have z
    # rather than waiting for the whole corpus to finish embedding
    ap.add_argument("--samples", help="comma-separated sample tags, overrides --organ")
    ap.add_argument("--out", default=None)
    ap.add_argument("--val-frac", type=float, default=0.08)
    ap.add_argument("--test-frac", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    return build_training_store(StoreConfig(corpus=a.corpus, organ=a.organ, all_samples=a.all_samples,
                                            samples=a.samples, out=a.out, val_frac=a.val_frac,
                                            test_frac=a.test_frac, seed=a.seed))


if __name__ == "__main__":  # pragma: no cover
    main()
