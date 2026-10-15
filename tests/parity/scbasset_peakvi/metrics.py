"""Parity metrics (MANIFEST G2 / P2), shared by every side."""
import numpy as np


def knn_accuracy(emb, labels, k=15):
    """Leave-one-out k-NN majority vote (Euclidean, self excluded); ties -> smallest code."""
    from sklearn.neighbors import NearestNeighbors

    codes, y = np.unique(np.asarray(labels).astype(str), return_inverse=True)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(emb)
    _, idx = nn.kneighbors(emb)
    nb = []
    for i in range(len(y)):
        row = idx[i][idx[i] != i][:k]
        nb.append(row)
    nb = np.asarray(nb)
    votes = np.zeros((len(y), len(codes)), dtype=np.int64)
    for j in range(k):
        np.add.at(votes, (np.arange(len(y)), y[nb[:, j]]), 1)
    return float((votes.argmax(1) == y).mean())


def cluster_scores(emb, labels, seed=0):
    import anndata as ad
    import scanpy as sc
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    a = ad.AnnData(np.zeros((emb.shape[0], 1), dtype=np.float32))
    a.obsm["E"] = np.asarray(emb, dtype=np.float32)
    sc.pp.neighbors(a, n_neighbors=15, use_rep="E", random_state=seed)
    sc.tl.leiden(a, resolution=1.0, flavor="igraph", n_iterations=2, directed=False, random_state=seed)
    lab = np.asarray(labels).astype(str)
    cl = a.obs["leiden"].astype(str).to_numpy()
    return float(adjusted_rand_score(lab, cl)), float(adjusted_mutual_info_score(lab, cl))


def embedding_metrics(emb, labels):
    from sklearn.metrics import silhouette_score

    ari, ami = cluster_scores(emb, labels)
    return {"ARI": ari, "AMI": ami, "kNN_acc": knn_accuracy(emb, labels),
            "silhouette": float(silhouette_score(emb, np.asarray(labels).astype(str)))}


def per_cell_auroc(y, p):
    """Mean of per-cell sklearn ROC AUC over test peaks (rows = peaks)."""
    from sklearn.metrics import roc_auc_score

    out = []
    for j in range(y.shape[1]):
        yj = y[:, j]
        if 0 < yj.sum() < len(yj):
            out.append(roc_auc_score(yj, p[:, j]))
    return float(np.mean(out))


def gate(a, b, floor):
    """|mean a - mean b| <= max(2 * max(sd_a, sd_b), floor)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    sd = max(a.std(ddof=1) if len(a) > 1 else 0.0, b.std(ddof=1) if len(b) > 1 else 0.0)
    d = abs(a.mean() - b.mean())
    tol = max(2 * sd, floor)
    return {"mean_a": a.mean(), "sd_a": a.std(ddof=1) if len(a) > 1 else 0.0, "mean_b": b.mean(),
            "sd_b": b.std(ddof=1) if len(b) > 1 else 0.0, "delta": d, "tol": tol, "pass": bool(d <= tol)}


FLOORS = {"ARI": 0.05, "AMI": 0.03, "kNN_acc": 0.02, "silhouette": 0.03, "test_AUROC": 0.01}
