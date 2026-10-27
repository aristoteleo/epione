"""Dispatcher behind ``epi.tl.spatial_domains(method=<spatial method>)``.

Each entry runs one ported spatial-domain method end to end (embedding +
domains) and returns the per-spot labels. The ported methods live in their own
modules (``_cosmos``, ``_present``, ``_instinct``, ``_candies``) and are also
public as ``epi.tl.cosmos`` etc.

Inputs: ``adata`` is the ATAC AnnData (raw peak counts in ``X``,
``obsm["spatial"]``) or a MuData with ``mod["atac"]`` (and ``mod["rna"]``);
multimodal methods take the RNA AnnData as ``rna=``.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

__all__ = ["SPATIAL_METHODS", "run_spatial_method"]


def _split(adata, rna):
    mods = getattr(adata, "mod", None)
    if mods is not None:
        atac = mods.get("atac", mods.get("peaks"))
        rna = rna if rna is not None else mods.get("rna")
        return atac, rna, adata
    return adata, rna, None


def _mclust(emb, n_domains, random_seed=2020, n_comps=20):
    import anndata as ad

    from ._spatial import spatial_domains

    t = ad.AnnData(np.zeros((emb.shape[0], 1), dtype=np.float32))
    t.obsm["E"] = np.asarray(emb, dtype=float)
    return spatial_domains(t, int(n_domains), use_rep="E", method="mclust", use_pca=True,
                           n_comps=min(n_comps, emb.shape[1]), random_seed=random_seed).to_numpy()


def _cosmos(atac, rna, n_domains, *, use_rep: Optional[str] = "X_lsi", rna_rep: Optional[str] = None,
            random_seed: int = 42, **kw):
    from ._cosmos import cosmos

    if rna is None:
        raise ValueError("method='cosmos' integrates two modalities: pass rna=<RNA AnnData>")
    rep = use_rep if (use_rep is not None and use_rep in atac.obsm) else None
    emb = cosmos(rna, atac, use_rep1=rna_rep, use_rep2=rep, random_seed=random_seed, **kw)
    return _mclust(emb, n_domains), "X_cosmos"


def _present(atac, rna, n_domains, *, random_seed: int = 2024, **kw):
    from ._present import present

    out = present(adata_rna=rna, adata_atac=atac, nclusters=int(n_domains), random_seed=random_seed, **kw)
    lab = pd.Series(out.obs["LeidenClusters"].astype(str).to_numpy(), index=out.obs_names)
    return lab.reindex(atac.obs_names).to_numpy(), kw.get("key_added", "X_present")


def _instinct(atac, rna, n_domains, *, batch_key: str = "slice", seed: int = 1236, **kw):
    from ._instinct import instinct

    if isinstance(atac, (list, tuple)):
        slices, names = list(atac), kw.pop("slice_names", None)
        concat = instinct(slices, names, n_domains=n_domains, seed=seed, **kw)
        return concat.obs["instinct_domain"].to_numpy(), "X_instinct"
    if batch_key not in atac.obs:
        raise ValueError("method='instinct' integrates several slices: pass a list of AnnData or set "
                         f"obs[{batch_key!r}] on the concatenated AnnData")
    names = [str(x) for x in pd.unique(atac.obs[batch_key])]
    slices = [atac[atac.obs[batch_key].astype(str) == n].copy() for n in names]
    concat = instinct(slices, names, n_domains=n_domains, seed=seed, **kw)
    lab = pd.Series(pd.NA, index=atac.obs_names, dtype="object")
    for n, s in zip(names, slices):
        lab.loc[s.obs_names] = s.obs["instinct_domain"].astype(object).to_numpy()
    emb = np.full((atac.n_obs, concat.obsm["X_instinct"].shape[1]), np.nan, dtype=np.float32)
    for s in slices:
        emb[atac.obs_names.get_indexer(s.obs_names)] = s.obsm["X_instinct"]
    atac.obsm["X_instinct"] = emb
    return lab.to_numpy(), "X_instinct"


def _candies(atac, rna, n_domains, *, seed_offset: int = 0, **kw):
    from ._candies import candies

    if rna is None:
        raise ValueError("method='candies' denoises/integrates RNA + ATAC: pass rna=<RNA AnnData>")
    candies(rna, atac, n_domains=n_domains, seed_offset=seed_offset, **kw)
    return atac.obs["candies_domain"].to_numpy(), "X_candies"


def _praga(atac, rna, n_domains, *, random_seed: int = 2024, **kw):
    from ._praga import praga

    if rna is None:
        raise ValueError("method='praga' integrates RNA + ATAC: pass rna=<RNA AnnData>")
    praga(rna, atac, n_clusters=n_domains, random_seed=random_seed, **kw)
    return atac.obs["praga_domain"].to_numpy(), "X_praga"


def _miso(atac, rna, n_domains, *, seed: int = 100, **kw):
    from ._miso import miso, miso_features

    feats = [miso_features(atac, "atac")] + ([miso_features(rna, "rna")] if rna is not None else [])
    emb, labels = miso(feats, n_clusters=n_domains, seed=seed, **kw)
    atac.obsm["X_miso"] = emb
    return labels, "X_miso"


SPATIAL_METHODS: Dict[str, Callable] = {
    "cosmos": _cosmos,
    "present": _present,
    "instinct": _instinct,
    "candies": _candies,
    "praga": _praga,
    "miso": _miso,
}


def run_spatial_method(adata, n_domains, *, method: str, rna=None, key_added: str = "domain", **kwargs):
    """Run a registered spatial method and store its domains in ``obs[key_added]``."""
    if method not in SPATIAL_METHODS:
        raise ValueError(f"unknown spatial method {method!r}; available: {sorted(SPATIAL_METHODS)}")
    atac, rna, container = _split(adata, rna)
    labels, emb_key = SPATIAL_METHODS[method](atac, rna, n_domains, **kwargs)
    target = container if container is not None else atac
    if isinstance(target, (list, tuple)):
        return labels
    lab = pd.Series(labels, index=target.obs_names).astype("object")
    cats = sorted({str(x) for x in lab.dropna()}, key=lambda s: (len(s), s))
    target.obs[key_added] = pd.Categorical(lab.where(lab.isna(), lab.astype(str)), categories=cats)
    if container is not None and emb_key in atac.obsm:
        container.obsm[emb_key] = atac.obsm[emb_key]
    target.uns.setdefault("spatial_domains", {})[key_added] = {"method": method, "embedding": emb_key,
                                                                "n_domains": int(n_domains)}
    return target.obs[key_added]
