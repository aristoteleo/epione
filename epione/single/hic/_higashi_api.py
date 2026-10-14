"""Glue between the AnnData-level sc-Hi-C API and the Higashi port.

:func:`impute_cells` (``method="higashi"``) and :func:`embedding`
(``method="higashi"``) share one trained model per AnnData. The model is
kept in an in-process cache keyed by ``adata.uns['hic']['higashi']['key']``
and, when an output directory is known, saved as ``higashi_model.pt`` next to
the imputed matrices so a later session can reuse it.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

_CACHE: Dict[str, object] = {}

HIGASHI_DEFAULTS = dict(
    dimensions=64,
    loss_mode="zinb",
    neighbor_num=5,
    local_transfer_range=1,
    minimum_distance=None,     # one bin
    maximum_distance=-1,
    embedding_epoch=60,
    no_nbr_epoch=45,
    with_nbr_epoch=30,
    optimizer="sequential",    # or "torch20_foreach" (see _higashi._model.ListwiseAdam)
)


def _config(chromosomes, random_state, higashi_params):
    cfg = dict(HIGASHI_DEFAULTS)
    unknown = set(higashi_params or {}) - set(cfg)
    if unknown:
        raise TypeError(f"unknown higashi_params: {sorted(unknown)}; allowed: {sorted(cfg)}")
    cfg.update(higashi_params or {})
    cfg["chromosomes"] = list(chromosomes)
    cfg["random_state"] = int(random_state)
    return cfg


def _chromosomes(adata, chromosomes):
    info = adata.uns.get("hic", {})
    chroms = list(chromosomes) if chromosomes is not None else list(info.get("chromosomes", []))
    if not chroms:
        raise ValueError("no chromosomes: adata.uns['hic']['chromosomes'] is empty; pass chromosomes=...")
    return chroms


def _train_stage1(adata, cfg, device, verbose):
    import cooler
    from ._higashi import Higashi, contacts_from_cools, process_contacts

    cool_paths = [str(p) for p in adata.obs["cool_path"].astype(str)]
    chroms = cfg["chromosomes"]
    clr = cooler.Cooler(cool_paths[0])
    res = int(clr.binsize)
    sizes = {c: int(clr.chromsizes[c]) for c in chroms}
    contacts = contacts_from_cools(cool_paths, chroms)
    present = np.unique(contacts["cell_id"].to_numpy())
    if len(present) != len(cool_paths):
        missing = sorted(set(range(len(cool_paths))) - set(present.tolist()))
        names = [adata.obs_names[i] for i in missing[:5]]
        raise ValueError(f"{len(missing)} cell(s) have no intra-chromosomal contacts on the chosen "
                         f"chromosomes (e.g. {names}); remove them before running Higashi")
    data = process_contacts(contacts, sizes, chroms, res, random_state=cfg["random_state"])
    del contacts
    kw = {k: cfg[k] for k in HIGASHI_DEFAULTS}
    model = Higashi(data, impute_list=chroms, device=device, seed=cfg["random_state"], verbose=verbose, **kw)
    model.prep_model()
    model.train_for_embeddings()
    return model


def _save(model, cfg, path):
    import torch
    torch.save({"config": cfg, "data": model.data, "stage_models": model._stage_models,
                "cell_embeddings": model.cell_embeddings, "history": model.history, "timing": model.timing,
                "cell_neighbor_list": getattr(model, "cell_neighbor_list", None),
                "cell_neighbor_weight_list": getattr(model, "cell_neighbor_weight_list", None)}, path)


def _load(path, device, verbose):
    import torch
    from ._higashi import Higashi
    dev = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    # map_location also moves the plain tensor attributes inside the pickled modules
    ck = torch.load(path, map_location=dev, weights_only=False)
    cfg = ck["config"]
    kw = {k: cfg[k] for k in HIGASHI_DEFAULTS}
    model = Higashi(ck["data"], impute_list=cfg["chromosomes"], device=dev, seed=cfg["random_state"],
                    verbose=verbose, **kw)
    model.prep_model()  # rebuilds samplers and generators; network weights come from the snapshots
    model._stage_models = dict(ck["stage_models"])
    model.cell_embeddings = ck["cell_embeddings"]
    model.history, model.timing = ck["history"], ck["timing"]
    if ck.get("cell_neighbor_list") is not None:
        model.cell_neighbor_list = ck["cell_neighbor_list"]
        model.cell_neighbor_weight_list = ck["cell_neighbor_weight_list"]
    return model, cfg


def get_model(adata, cfg, *, device=None, verbose=True, checkpoint: Optional[Path] = None):
    """Return a Higashi model trained at least through stage 1 for ``adata``,
    reusing the cached / saved one when its configuration matches."""
    info = adata.uns.setdefault("hic", {}).setdefault("higashi", {})
    key = info.get("key")
    if key in _CACHE and _CACHE[key][1] == cfg:
        return _CACHE[key][0]
    ck = checkpoint or (Path(info["checkpoint"]) if info.get("checkpoint") else None)
    if ck is not None and Path(ck).exists():
        model, saved_cfg = _load(ck, device, verbose)
        if saved_cfg == cfg:
            key = info.get("key") or uuid.uuid4().hex
            info["key"] = key
            _CACHE[key] = (model, cfg)
            return model
    model = _train_stage1(adata, cfg, device, verbose)
    key = uuid.uuid4().hex
    info.clear()
    info.update({"key": key, "config": {k: v for k, v in cfg.items() if v is not None}})
    _CACHE[key] = (model, cfg)
    return model


def higashi_impute(adata, out_dir, *, chromosomes=None, random_state=0, device=None, higashi_params=None,
                   neighbors=True, overwrite=False, progress=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chroms = _chromosomes(adata, chromosomes)
    cfg = _config(chroms, random_state, higashi_params)
    ck = out_dir / "higashi_model.pt"
    model = get_model(adata, cfg, device=device, verbose=progress, checkpoint=ck)
    stage = "stage3" if neighbors else "stage2"
    if "stage2" not in model._stage_models:
        model.train_for_imputation_nbr_0()
    if neighbors and "stage3" not in model._stage_models:
        model.train_for_imputation_with_nbr()
    _save(model, cfg, ck)
    info = adata.uns["hic"]["higashi"]
    info["checkpoint"] = str(ck)

    cell_ids = list(adata.obs_names)
    todo = [i for i, cid in enumerate(cell_ids) if overwrite or not (out_dir / f"{cid}.npz").exists()]
    pairs = model.impute_pairs()
    n_bins = {c: int(model.data.chrom_start_end[model.chrom_list.index(c), 1]
                     - model.data.chrom_start_end[model.chrom_list.index(c), 0]) for c in chroms}
    pbar = None
    if progress:
        try:
            from tqdm.auto import tqdm
            pbar = tqdm(total=len(todo), desc="impute_cells (higashi)")
        except ImportError:
            pass

    def on_cell(i, vals):
        mats = {}
        for c, v in vals.items():
            n = n_bins[c]
            m = np.zeros((n, n), dtype=np.float32)
            xy = pairs[c][1]
            m[xy[:, 0], xy[:, 1]] = v
            m[xy[:, 1], xy[:, 0]] = v
            mats[c] = m
        np.savez(out_dir / f"{cell_ids[i]}.npz", **mats)
        if pbar:
            pbar.update(1)

    if todo:
        (model.impute_with_nbr if neighbors else model.impute_no_nbr)(on_cell, cells=todo)
    if pbar:
        pbar.close()
    adata.obsm["X_higashi"] = np.asarray(model.cell_embeddings, dtype=np.float32)
    adata.uns["hic"]["imputed_dir"] = str(out_dir)
    adata.uns["hic"]["impute_params"] = {
        "method": "higashi", "neighbors": bool(neighbors), "chromosomes": chroms,
        **{k: v for k, v in cfg.items() if k not in ("chromosomes",) and v is not None}}
    return out_dir


def higashi_embedding(adata, *, chromosomes=None, random_state=0, device=None, higashi_params=None,
                      work_dir=None, verbose=True):
    import anndata as ad
    import pandas as pd
    chroms = _chromosomes(adata, chromosomes)
    cfg = _config(chroms, random_state, higashi_params)
    ck = Path(work_dir) / "higashi_model.pt" if work_dir is not None else None
    model = get_model(adata, cfg, device=device, verbose=verbose, checkpoint=ck)
    if ck is not None:
        Path(work_dir).mkdir(parents=True, exist_ok=True)
        _save(model, cfg, ck)
        adata.uns["hic"]["higashi"]["checkpoint"] = str(ck)
    emb = np.asarray(model.cell_embeddings, dtype=np.float32)
    new = ad.AnnData(X=emb.copy(), obs=adata.obs.copy(), uns=dict(adata.uns),
                     var=pd.DataFrame(index=[f"higashi_{i}" for i in range(emb.shape[1])]))
    new.obsm["X_higashi"] = emb
    new.uns["hic"] = dict(adata.uns.get("hic", {}))
    new.uns["hic"]["embedding_method"] = "higashi"
    new.uns["hic"]["embedding_chromosomes"] = chroms
    new.uns["higashi"] = {"history": pd.DataFrame([h for h in model.history if h["stage"] == 1])}
    return new
