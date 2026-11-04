"""G1: the parity metrics vs independent from-scratch implementations, on a run's outputs.

usage: python g1_metrics.py <fixture_dir> <run_dir> <out.json>
"""
import json
import math
import os
import pickle
import sys

import numpy as np
from scipy.special import gammaln
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import embedding_metrics, heldout_auroc, load_imputation_eval, pseudobulk_r  # noqa: E402

fixture, run_dir, out = sys.argv[1:4]
li = pickle.load(open(os.path.join(fixture, 'label_info.pickle'), 'rb'))
labels = np.asarray(li['cell type'])
emb = np.load(os.path.join(run_dir, 'cell_embeddings_final.npy')).astype(np.float64)
em = embedding_metrics(emb, labels)
km = em['_kmeans']


def contingency(a, b):
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    t = np.zeros((len(ua), len(ub)), dtype=np.int64)
    for x, y in zip(ia, ib):
        t[x, y] += 1
    return t


def ari_scratch(a, b):
    t = contingency(a, b)
    comb = lambda x: x * (x - 1) / 2.0
    s_ij = sum(comb(v) for v in t.ravel())
    s_a = sum(comb(v) for v in t.sum(1))
    s_b = sum(comb(v) for v in t.sum(0))
    n = t.sum()
    exp = s_a * s_b / comb(n)
    return (s_ij - exp) / (0.5 * (s_a + s_b) - exp)


def ami_scratch(a, b):
    t = contingency(a, b).astype(float)
    n = t.sum()
    ai, bj = t.sum(1), t.sum(0)
    mi = 0.0
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            if t[i, j] > 0:
                mi += t[i, j] / n * math.log(n * t[i, j] / (ai[i] * bj[j]))
    h = lambda p: -sum(x / n * math.log(x / n) for x in p if x > 0)
    emi = 0.0
    for i in range(len(ai)):
        for j in range(len(bj)):
            lo, hi = int(max(1, ai[i] + bj[j] - n)), int(min(ai[i], bj[j]))
            for nij in range(lo, hi + 1):
                logp = (gammaln(ai[i] + 1) + gammaln(bj[j] + 1) + gammaln(n - ai[i] + 1) + gammaln(n - bj[j] + 1)
                        - gammaln(n + 1) - gammaln(nij + 1) - gammaln(ai[i] - nij + 1) - gammaln(bj[j] - nij + 1)
                        - gammaln(n - ai[i] - bj[j] + nij + 1))
                emi += nij / n * math.log(n * nij / (ai[i] * bj[j])) * math.exp(logp)
    mean_h = 0.5 * (h(ai) + h(bj))
    return (mi - emi) / (mean_h - emi)


def knn_scratch(x, y, k=10):
    from sklearn.model_selection import StratifiedKFold
    classes = np.unique(y)
    pred = np.empty(len(y), dtype=object)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(x, y):
        d = ((x[te, None, :] - x[None, tr, :]) ** 2).sum(-1)
        nn = np.argsort(d, axis=1, kind='stable')[:, :k]
        for row, idx in zip(te, nn):
            counts = np.array([(y[tr][idx] == c).sum() for c in classes])
            pred[row] = classes[np.argmax(counts)]
    return float(np.mean(pred == y))


def auroc_scratch(score, lab):
    r = rankdata(score)
    n1, n0 = lab.sum(), (lab == 0).sum()
    return (r[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


res = {'ARI': [em['ARI'], ari_scratch(labels, km)], 'AMI': [em['AMI'], ami_scratch(labels, km)],
       'kNN_acc': [em['kNN_acc'], knn_scratch(emb, labels)]}

sizes = {l.split('\t')[0]: int(l.split('\t')[1]) for l in open(os.path.join(fixture, 'chrom_sizes.txt'))}
chroms = list(sizes)
n_bins = {c: int(np.ceil(s / 1_000_000)) for c, s in sizes.items()}
heldout = dict(np.load(os.path.join(fixture, 'heldout.npz')))
full = dict(np.load(os.path.join(fixture, 'full_pixels.npz')))
ev, pb = load_imputation_eval(run_dir, 0, chroms, n_bins, len(labels), heldout, labels, full)
m_auc, per_cell = heldout_auroc(ev, heldout)
aucs = []
for i in range(heldout['cell'].max() + 1):
    sel = heldout['cell'] == i
    lab = heldout['label'][sel]
    if lab.sum() >= 10 and (lab == 0).sum() > 0:
        aucs.append(auroc_scratch(ev[sel], lab))
res['AUROC_nbr0'] = [m_auc, float(np.mean(aucs))]

m_pb, _ = pseudobulk_r(pb)
rs = []
for g, (imp, raw, stratum) in pb.items():
    zi, zr = np.full(len(imp), np.nan), np.full(len(imp), np.nan)
    for s in np.unique(stratum):
        idx = np.where(stratum == s)[0]
        for vec, z in ((imp, zi), (raw, zr)):
            v = vec[idx]
            mu = sum(v) / len(v)
            sd = math.sqrt(sum((t - mu) ** 2 for t in v) / len(v))
            if sd > 0:
                z[idx] = (v - mu) / sd
    ok = np.isfinite(zi) & np.isfinite(zr)
    a, b = zi[ok], zr[ok]
    ma, mb = a.mean(), b.mean()
    rs.append(float(((a - ma) * (b - mb)).sum() / math.sqrt(((a - ma) ** 2).sum() * ((b - mb) ** 2).sum())))
res['PB_r_nbr0'] = [m_pb, float(np.mean(rs))]
summary = {k: {'metrics_py': v[0], 'scratch': v[1], 'abs_diff': abs(v[0] - v[1]), 'pass': abs(v[0] - v[1]) <= 1e-9}
           for k, v in res.items()}
summary['G1_pass'] = all(v['pass'] for v in summary.values())
json.dump(summary, open(out, 'w'), indent=1, default=float)
print(json.dumps(summary, indent=1, default=float))
