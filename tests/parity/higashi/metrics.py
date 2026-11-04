"""Parity metrics for Higashi (definitions fixed in MANIFEST.md).

Run directories (reference or Epione) contain:
  cell_embeddings_final.npy
  temp/<chrom>_exp_nbr_<k>_impute.hdf5  ('coordinates' (P, 2) local bins, 'cell_<i>' (P,))
"""
from __future__ import annotations

import os

import h5py
import numpy as np


def embedding_metrics(emb, labels, repli=None):
    from sklearn.cluster import KMeans
    from sklearn.linear_model import Ridge
    from sklearn.metrics import accuracy_score, adjusted_mutual_info_score, adjusted_rand_score
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    emb = np.asarray(emb, dtype=np.float64)
    labels = np.asarray(labels)
    km = KMeans(n_clusters=len(np.unique(labels)), n_init=10, random_state=0).fit_predict(emb)
    out = {
        'ARI': float(adjusted_rand_score(labels, km)),
        'AMI': float(adjusted_mutual_info_score(labels, km)),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    pred = cross_val_predict(KNeighborsClassifier(n_neighbors=10), emb, labels, cv=cv)
    out['kNN_acc'] = float(accuracy_score(labels, pred))
    if repli is not None:
        out['repli_R2'] = float(np.mean(cross_val_score(Ridge(alpha=1.0), emb, repli,
                                                        cv=KFold(5, shuffle=True, random_state=0), scoring='r2')))
    out['_kmeans'] = km
    out['_knn_pred'] = pred
    return out


def _pair_pos(coords, n):
    lut = np.full(n * n, -1, dtype=np.int64)
    lut[coords[:, 0].astype(np.int64) * n + coords[:, 1]] = np.arange(len(coords))
    return lut


def load_imputation_eval(run_dir, k, chroms, n_bins, n_cells, heldout, groups, full):
    """Collect, per chromosome, the imputed values at the held-out evaluation pairs
    and the per-group mean imputed maps; and the matching raw pseudo-bulk.

    Returns (eval_values aligned with ``heldout`` rows, list of (z_imp, z_raw) per group).
    """
    ev_vals = np.full(len(heldout['label']), np.nan)
    group_ids = np.unique(groups)
    per_group = {g: {'imp': [], 'raw': [], 'stratum': []} for g in group_ids}
    for ci, c in enumerate(chroms):
        n = int(n_bins[c])
        with h5py.File(os.path.join(run_dir, 'temp', f'{c}_exp_nbr_{k}_impute.hdf5'), 'r') as f:
            coords = np.asarray(f['coordinates']).astype(np.int64)
            mat = np.stack([np.asarray(f[f'cell_{i}'], dtype=np.float64) for i in range(n_cells)])
        lut = _pair_pos(coords, n)
        sel = np.where(heldout['chrom'] == ci)[0]
        pos = lut[heldout['b1'][sel] * n + heldout['b2'][sel]]
        assert (pos >= 0).all(), 'evaluation pair outside the imputed set'
        ev_vals[sel] = mat[heldout['cell'][sel], pos]
        dist = coords[:, 1] - coords[:, 0]
        fs = np.where((full['chrom'] == ci) & (full['b2'] - full['b1'] >= 1))[0]
        fpos = lut[full['b1'][fs].astype(np.int64) * n + full['b2'][fs]]
        for g in group_ids:
            cells = np.where(groups == g)[0]
            imp = mat[cells].mean(axis=0)
            in_g = np.isin(full['cell'][fs], cells)
            raw = np.bincount(fpos[in_g], weights=full['count'][fs][in_g].astype(np.float64), minlength=len(coords))
            per_group[g]['imp'].append(imp)
            per_group[g]['raw'].append(raw)
            per_group[g]['stratum'].append(ci * 100000 + dist)
    pb = {g: tuple(np.concatenate(per_group[g][key]) for key in ('imp', 'raw', 'stratum')) for g in group_ids}
    return ev_vals, pb


def heldout_auroc(ev_vals, heldout, min_pos=10):
    from sklearn.metrics import roc_auc_score
    cells = heldout['cell']
    lab = heldout['label']
    order = np.argsort(cells, kind='stable')
    bounds = np.searchsorted(cells[order], np.arange(cells.max() + 2))
    aucs = []
    for i in range(cells.max() + 1):
        idx = order[bounds[i]:bounds[i + 1]]
        if lab[idx].sum() < min_pos or (lab[idx] == 0).sum() == 0:
            continue
        aucs.append(roc_auc_score(lab[idx], ev_vals[idx]))
    return float(np.mean(aucs)), np.asarray(aucs)


def _zscore_by_stratum(x, stratum):
    out = np.full(len(x), np.nan)
    uniq, inv = np.unique(stratum, return_inverse=True)
    cnt = np.bincount(inv)
    mu = np.bincount(inv, weights=x) / cnt
    var = np.bincount(inv, weights=(x - mu[inv]) ** 2) / cnt
    sd = np.sqrt(var)
    ok = sd[inv] > 0
    out[ok] = (x[ok] - mu[inv][ok]) / sd[inv][ok]
    return out


def pseudobulk_r(pb):
    rs = {}
    for g, (imp, raw, stratum) in pb.items():
        zi = _zscore_by_stratum(imp, stratum)
        zr = _zscore_by_stratum(raw, stratum)
        ok = np.isfinite(zi) & np.isfinite(zr)
        rs[g] = float(np.corrcoef(zi[ok], zr[ok])[0, 1])
    return float(np.mean(list(rs.values()))), rs
