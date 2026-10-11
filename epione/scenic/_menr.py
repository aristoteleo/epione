"""Motif-enrichment results → SCENIC+ cistromes, plus pycistarget-compatible
HDF5 I/O.

:func:`merge_cistromes` reproduces scenicplus
``data_wrangling.cistarget_wrangling.get_and_merge_cistromes`` (the
``prepare_menr`` step). It merges, per TF, the region-set coordinates of the
motif hits of every cisTarget and DEM result. Each TF is annotated directly
or with extended evidence, and the result is restricted to the regions of
the multiome object. Output is two region × TF boolean :class:`AnnData`
objects.

:func:`write_hdf5` / :func:`read_hdf5` use pycistarget's on-disk layout, so
results can be exchanged with upstream scenicplus.
"""
from __future__ import annotations

import os
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ._cistarget import MotifEnrichmentResult
from ._motif_annotation import motifs_per_tf, tf_list


def _iter_results(results) -> Iterable[MotifEnrichmentResult]:
    if isinstance(results, MotifEnrichmentResult):
        yield results
        return
    if isinstance(results, Mapping):
        results = results.values()
    for r in results:
        if isinstance(r, (str, os.PathLike)):
            yield from read_hdf5(r).values()
        elif isinstance(r, Mapping):
            yield from r.values()
        else:
            yield r


def merge_cistromes(
    results,
    scplus_regions: Optional[Iterable[str]] = None,
    direct_annotation: Sequence[str] = ("Direct_annot",),
    extended_annotation: Sequence[str] = ("Orthology_annot",),
):
    """Merge cisTarget/DEM results into direct and extended cistromes.

    Parameters
    ----------
    results
        :class:`MotifEnrichmentResult` objects, ``{name: result}`` dicts
        (as returned by :func:`run_cistarget` / :func:`run_dem`), or paths to
        pycistarget-format ``.hdf5`` files, or a list mixing any of these.
    scplus_regions
        Regions kept in the cistromes (the scATAC ``var_names``). ``None``
        keeps all.

    Returns
    -------
    ``(direct, extended)`` :class:`anndata.AnnData`. ``X`` is region × TF
    boolean (CSC), ``var["motifs"]`` holds the comma-joined sorted motifs,
    and ``var["is_extended"]`` marks the type. TFs are sorted. Region rows
    are sorted. Upstream uses set order, so only the order differs.
    """
    import anndata as ad

    keep = None if scplus_regions is None else set(scplus_regions)
    merged = {False: {}, True: {}}
    for res in _iter_results(results):
        me = res.motif_enrichment
        if len(me) == 0:
            continue
        hits = res.motif_hits.get("Region_set", res.motif_hits.get("region_set", {}))
        for extended, annotation in ((False, direct_annotation), (True, extended_annotation)):
            for tf in tf_list(me, annotation):
                motifs = motifs_per_tf(me, tf, annotation)
                regs: Set[str] = set()
                for m in motifs:
                    if m not in hits:
                        raise ValueError(f"motif enrichment table and motif hits don't match for {tf}")
                    regs.update(hits[m])
                if keep is not None:
                    regs &= keep
                cur = merged[extended].setdefault(tf, [set(), set()])
                cur[0].update(motifs)
                cur[1].update(regs)

    def to_adata(d, extended):
        tfs = sorted(d)
        regions = sorted(set().union(*[d[t][1] for t in tfs])) if tfs else []
        pos = {r: i for i, r in enumerate(regions)}
        rows, cols = [], []
        for j, t in enumerate(tfs):
            idx = [pos[r] for r in d[t][1]]
            rows.extend(idx)
            cols.extend([j] * len(idx))
        X = sp.csc_matrix((np.ones(len(rows), dtype=bool), (rows, cols)),
                          shape=(len(regions), len(tfs)), dtype=bool)
        a = ad.AnnData(X=X, obs=pd.DataFrame(index=regions), var=pd.DataFrame(index=tfs))
        a.var["motifs"] = [",".join(sorted(d[t][0])) for t in tfs]
        a.var["is_extended"] = extended
        return a

    return to_adata(merged[False], False), to_adata(merged[True], True)


# ---------------------------------------------------------------------------
# pycistarget-compatible hdf5
# ---------------------------------------------------------------------------

_META = {
    "cisTarget": ["species", "auc_threshold", "nes_threshold", "rank_threshold",
                  "annotation_version", "annotation_to_use", "motif_similarity_fdr",
                  "orthologous_identity_threshold"],
    "DEM": ["species", "adjpval_thr", "log2fc_thr", "mean_fg_thr", "motif_hit_thr",
            "annotation_version", "annotation_to_use", "motif_similarity_fdr",
            "orthologous_identity_threshold"],
}

_DTYPES = {"Logo": "S", "Region_set": "S", "Direct_annot": "S", "Motif_similarity_annot": "S",
           "Orthology_annot": "S", "Motif_similarity_and_Orthology_annot": "S",
           "NES": np.float64, "AUC": np.float64, "Rank_at_max": np.float64,
           "Motif_hits": np.int64, "Target": "S", "Query": "S", "Log2FC": np.float64,
           "Adjusted_pval": np.float64, "Mean_fg": np.float64, "Mean_bg": np.float64}


def _require_tables():
    import importlib.util
    if importlib.util.find_spec("tables") is None:
        raise ImportError("pycistarget-format hdf5 files store tables with pandas.HDFStore; "
                          "install PyTables (`pip install tables` or `epione[scenic]`).")


def write_hdf5(results, path: Union[str, os.PathLike], mode: str = "w") -> str:
    """Write results in pycistarget's ``write_hdf5`` layout (readable by
    ``pycistarget.input_output.read_hdf5`` and scenicplus ``prepare_menr``).
    Results without enriched motifs are skipped, as in the scenicplus CLI.
    Needs PyTables."""
    import h5py

    _require_tables()
    path = os.fspath(path)
    first = True
    for res in _iter_results(results):
        if len(res.motif_enrichment) == 0:
            continue
        with h5py.File(path, mode if first else "a") as h5:
            first = False
            root = h5.create_group(res.name)
            meta = root.create_group("metadata")
            for k in _META[res.kind]:
                v = res.params.get(k)
                if v is not None:
                    meta.create_dataset(k, data=v)
            meta.create_dataset("version", data="epione")
            meta.create_dataset("type", data=res.kind)
            for grp, key in (("cistromes", res.cistromes), ("motif_hits", res.motif_hits)):
                g = root.create_group(grp)
                for sub, h5name in (("Database", "database"), ("Region_set", "region_set")):
                    gg = g.create_group(h5name)
                    for k, v in key.get(sub, {}).items():
                        gg.create_dataset(k, data=np.array(v, dtype="S"))
        me = res.motif_enrichment.copy()
        me = me.astype({k: v for k, v in _DTYPES.items() if k in me.columns})
        if "Motif_hit_thr" in me:
            me["Motif_hit_thr"] = me["Motif_hit_thr"].astype(np.float64)
        me.to_hdf(path, key=f"{res.name}/motif_enrichment", append=True, mode="r+")
        r2d = res.regions_to_db.astype({k: "S" for k in ("Target", "Query")})
        r2d.to_hdf(path, key=f"{res.name}/regions_to_db", append=True, mode="r+")
    return path


def read_hdf5(path: Union[str, os.PathLike]) -> Dict[str, MotifEnrichmentResult]:
    """Read a pycistarget-format hdf5 (written by pycistarget or
    :func:`write_hdf5`). Needs PyTables."""
    import h5py

    _require_tables()
    path = os.fspath(path)
    out = {}
    with h5py.File(path, "r") as h5:
        names = list(h5.keys())
        parsed = {}
        for name in names:
            g = h5[name]
            kind = g["metadata"]["type"][()].decode()
            params = {}
            for k in g["metadata"].keys():
                v = g["metadata"][k][()]
                if isinstance(v, bytes):
                    v = v.decode()
                elif isinstance(v, np.ndarray) and v.dtype.kind == "S":
                    v = [x.decode() for x in v]
                params[k] = v

            def grab(grp):
                d = {}
                for sub, key in (("database", "Database"), ("region_set", "Region_set")):
                    d[key] = {k: [x.decode() for x in grp[sub][k][:]]
                              for k in grp[sub].keys()} if sub in grp else {}
                return d
            parsed[name] = (kind, params, grab(g["cistromes"]), grab(g["motif_hits"]))
    for name, (kind, params, cis, hits) in parsed.items():
        me = pd.read_hdf(path, key=f"{name}/motif_enrichment")
        r2d = pd.read_hdf(path, key=f"{name}/regions_to_db")
        for c in me.columns:
            if me[c].dtype == object and len(me) and isinstance(me[c].iloc[0], bytes):
                me[c] = me[c].str.decode("utf-8")
        for c in ("Target", "Query"):
            if c in r2d and len(r2d) and isinstance(r2d[c].iloc[0], bytes):
                r2d[c] = r2d[c].str.decode("utf-8")
        out[name] = MotifEnrichmentResult(name, kind, me, hits, cis, r2d, params)
    return out
