"""Gene panels: cross-species gene mapping, the panel space, panel-agnostic sampling, depth thinning.

ACESO's state encoder must accept *any* targeted panel. During training each batch draws one
panel (all cells of a dataset share a panel):

* with probability ``p_target`` the target platform's panel, counts thinned to that platform's
  per-cell depth distribution;
* otherwise a random panel of log-uniform size in ``[min_genes, max_genes]`` drawn uniformly from
  the top ``pool`` highly variable genes, counts thinned to ``target depth x (size / |target
  panel|) x log-uniform[0.5, 2]`` (depth scales with panel size, platform efficiency varies 4x).

Modes: ``mixed`` (ACESO) | ``fixed`` (target panel only) | ``random`` (never the target panel;
zero-shot panel generalisation).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp


def mgi_orthologs(rpt: str, source: str = "mouse", target: str = "human") -> Dict[str, str]:
    """1:1 orthologs from MGI ``HOM_MouseHumanSequence.rpt`` (symbol -> symbol)."""
    h = pd.read_csv(rpt, sep="\t")
    key = h.columns[0]
    org = h["Common Organism Name"].str.split(",").str[0].str.strip()
    a, b = h[org == source], h[org == target]
    na, nb = a.groupby(key).size(), b.groupby(key).size()
    one = sorted(set(na[na == 1].index) & set(nb[nb == 1].index))
    a = a[a[key].isin(one)].set_index(key)["Symbol"]; b = b[b[key].isin(one)].set_index(key)["Symbol"]
    return dict(zip(a.loc[one], b.loc[one]))


def hvg_rank(X: sp.spmatrix, rows: Optional[np.ndarray] = None, n_bins: int = 20) -> np.ndarray:
    """Rank genes by normalised dispersion of log1p(CP10k) within mean bins (1 = most variable)."""
    X = sp.csr_matrix(X if rows is None else X[rows])
    tot = np.asarray(X.sum(1)).ravel()
    L = sp.diags(1e4 / np.maximum(tot, 1)) @ X; L.data = np.log1p(L.data)
    mu = np.asarray(L.mean(0)).ravel(); var = np.asarray(L.multiply(L).mean(0)).ravel() - mu ** 2
    disp = np.where(mu > 0, var / np.maximum(mu, 1e-12), 0)
    bins = pd.qcut(mu, n_bins, labels=False, duplicates="drop")
    z = pd.Series(disp).groupby(bins).transform(lambda v: (v - v.mean()) / (v.std() + 1e-12)).to_numpy().copy()
    z[mu <= 0] = -np.inf
    return pd.Series(-z).rank(method="first").astype(int).to_numpy()


@dataclass
class PanelSpace:
    """Genes the encoder can read for a reference dataset.

    ``genes``: reference gene symbols (columns of the count matrix); ``vocab_ids``: encoder token
    id of each gene; ``target``: column indices of the target platform panel; ``pool``: column
    indices random panels are drawn from.
    """
    genes: np.ndarray
    vocab_ids: np.ndarray
    target: np.ndarray
    pool: np.ndarray

    @classmethod
    def build(cls, genes: Sequence[str], vocab: Dict[str, int], target_panel: Sequence[str], X: sp.spmatrix,
              rows: Optional[np.ndarray] = None, orthologs: Optional[Dict[str, str]] = None, pool: int = 5000):
        """Keep genes that map (via ``orthologs`` if given) into ``vocab``; rank HVGs on ``rows`` of ``X``."""
        genes = np.asarray(genes, dtype=object)
        mapped = np.array([orthologs.get(g) if orthologs else g for g in genes], dtype=object)
        keep = np.array([m is not None and m in vocab for m in mapped])
        cols = np.flatnonzero(keep)
        rank = hvg_rank(sp.csr_matrix(X)[:, cols], rows)
        tset = set(target_panel)
        return cls(genes=genes[cols], vocab_ids=np.array([vocab[m] for m in mapped[cols]], dtype=np.int64),
                   target=np.flatnonzero([g in tset for g in genes[cols]]), pool=np.flatnonzero(rank <= pool)), cols


def thin_counts(M: sp.spmatrix, depth: np.ndarray, rng: np.random.Generator) -> sp.csr_matrix:
    """Binomially thin each row of ``M`` to (at most) ``depth[i]`` expected counts."""
    M = sp.csr_matrix(M, dtype=np.float32, copy=True); tot = np.asarray(M.sum(1)).ravel()
    p = np.minimum(1.0, depth / np.maximum(tot, 1))
    M.data = rng.binomial(M.data.astype(np.int64), np.repeat(p, np.diff(M.indptr))).astype(np.float32)
    M.eliminate_zeros()
    return M


@dataclass
class PanelSampler:
    """Draws (panel columns, depth scale) per batch; see module docstring."""
    space: PanelSpace
    target_depths: np.ndarray
    mode: str = "mixed"
    p_target: float = 0.5
    min_genes: int = 100
    max_genes: int = 1000
    efficiency_range: tuple = (0.5, 2.0)

    def __post_init__(self):
        if self.mode not in ("mixed", "fixed", "random"):
            raise ValueError(self.mode)

    def draw(self, rng: np.random.Generator):
        if self.mode == "fixed" or (self.mode == "mixed" and rng.random() < self.p_target):
            return self.space.target, 1.0
        n = int(np.exp(rng.uniform(np.log(self.min_genes), np.log(self.max_genes))))
        cols = np.sort(rng.choice(self.space.pool, n, replace=False))
        eff = float(np.exp(rng.uniform(*np.log(self.efficiency_range))))
        return cols, n / len(self.space.target) * eff

    def counts(self, X: sp.spmatrix, rows: np.ndarray, cols: np.ndarray, scale: float, rng: np.random.Generator):
        """Panel counts of ``rows`` thinned to a depth drawn from the target distribution x ``scale``."""
        return thin_counts(sp.csr_matrix(X)[rows][:, cols], rng.choice(self.target_depths, len(rows)) * scale, rng)
