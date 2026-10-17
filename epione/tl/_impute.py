"""MAGIC imputation weights (port of ArchR ``addImputeWeights`` / ``imputeMatrix``).

ArchR (MIT licence, Granja et al. 2021) adapts MAGIC (van Dijk et al. 2018):
cells are split into random blocks of ~``sample_cells``; within each block a
kNN graph on the reduced dimensions is turned into an adaptive Gaussian
kernel, Markov-normalised and raised to the power ``td + 1``. Imputing a
matrix multiplies each block's diffusion operator into the block's cells and
averages over ``n_rep`` random blockings.
"""
from __future__ import annotations

import math
from typing import Literal, Optional, Sequence, Union

import numpy as np
import scipy.sparse as sp
from anndata import AnnData
from sklearn.neighbors import NearestNeighbors

from ..utils import console


class _RRandom:
    """R's default RNG (Mersenne-Twister, ``set.seed`` scrambling, rejection
    ``sample()``) so that a seed gives the same cell blocks as ArchR."""

    _I2_32M1 = 2.328306437080797e-10

    def __init__(self, seed: int):
        s = int(seed) & 0xFFFFFFFF
        for _ in range(50):
            s = (69069 * s + 1) & 0xFFFFFFFF
        key = np.empty(625, dtype=np.uint32)
        for j in range(625):
            s = (69069 * s + 1) & 0xFFFFFFFF
            key[j] = s
        self._bg = np.random.MT19937(0)
        # key[0] is R's `mti` slot; FixupSeeds sets it to N=624 -> regenerate.
        self._bg.state = {"bit_generator": "MT19937",
                          "state": {"key": key[1:].copy(), "pos": 624}}

    def unif_rand(self) -> float:
        u = float(self._bg.random_raw()) * 2.3283064365386963e-10
        if u <= 0.0:
            return 0.5 * self._I2_32M1
        if 1.0 - u <= 0.0:
            return 1.0 - 0.5 * self._I2_32M1
        return u

    def _rbits(self, bits: int) -> int:
        v = 0
        for _ in range(0, bits + 1, 16):
            v = 65536 * v + int(math.floor(self.unif_rand() * 65536))
        return v & ((1 << bits) - 1)

    def _unif_index(self, dn: int) -> int:
        if dn <= 0:
            return 0
        bits = int(math.ceil(math.log2(dn)))
        while True:
            dv = self._rbits(bits)
            if dn > dv:
                return dv

    def permutation(self, n: int) -> np.ndarray:
        """0-based equivalent of R's ``sample(seq_len(n), n)``."""
        x = list(range(n))
        out = np.empty(n, dtype=np.int64)
        m = n
        for i in range(n):
            j = self._unif_index(m)
            out[i] = x[j]
            m -= 1
            x[j] = x[m]
        return out


def _row_zscores(m: np.ndarray) -> np.ndarray:
    mu = m.mean(axis=1, keepdims=True)
    sd = m.std(axis=1, ddof=1, keepdims=True)
    return (m - mu) / sd


def _bin_size(n: int, sample_cells: int) -> float:
    cutoffs = n / np.arange(1, 1001, dtype=np.float64)
    best = cutoffs[np.argmin(np.abs(cutoffs - sample_cells))]
    return min(best + 1.0, float(n))


def _block_diffusion(
    X: np.ndarray, k: int, td: int, ka: int, epsilon: float,
) -> np.ndarray:
    n = X.shape[0]
    nn = NearestNeighbors(n_neighbors=k, algorithm="kd_tree").fit(X)
    dist, idx = nn.kneighbors(X)
    if ka > 0:
        dist = dist / dist[:, [ka - 1]]
    rows = np.repeat(np.arange(n), k)
    cols = idx.ravel()
    vals = dist.ravel() if epsilon > 0 else np.ones(n * k)
    # W + t(W) built from COO triplets: R's Matrix keeps the stored zero
    # self-distances, so after exp() every cell carries a self-loop of 1.
    W = sp.coo_matrix(
        (np.concatenate([vals, vals]),
         (np.concatenate([rows, cols]), np.concatenate([cols, rows]))),
        shape=(n, n),
    ).tocsr()
    if epsilon > 0:
        W.data = np.exp(-(W.data / epsilon ** 2))
    rs = np.asarray(W.sum(axis=1)).ravel()
    W = sp.diags(1.0 / rs) @ W
    Wt = W
    for _ in range(td):
        Wt = Wt @ W
    return Wt.toarray() if sp.issparse(Wt) else np.asarray(Wt)


def add_impute_weights(
    adata: AnnData,
    use_rep: str = "X_birmt",
    *,
    dims_to_use: Optional[Sequence[int]] = None,
    scale_dims: bool = False,
    depth_key: Optional[str] = None,
    cor_cut_off: float = 0.75,
    td: int = 3,
    ka: int = 4,
    sample_cells: Optional[int] = 5000,
    n_rep: int = 2,
    k: int = 15,
    epsilon: float = 1.0,
    seed: int = 1,
    rng: Literal["R", "numpy"] = "R",
    blocks: Optional[Sequence[Sequence[np.ndarray]]] = None,
    key_added: str = "impute_weights",
    verbose: bool = True,
) -> None:
    """Compute MAGIC imputation weights (ArchR ``addImputeWeights``).

    Arguments:
        adata: AnnData with a reduced-dimension embedding in ``adata.obsm``.
        use_rep: key of the embedding in ``adata.obsm`` (ArchR ``reducedDims``).
        dims_to_use: 0-based columns of the embedding to keep. Default all.
        scale_dims: z-score each cell across dimensions (ArchR ``scaleDims``).
            Leave ``False`` when the embedding is already scaled, e.g. an
            ArchR ``getReducedDims`` export.
        depth_key: ``adata.obs`` column with per-cell fragment counts. When
            given, drop dimensions with ``|cor(dim, log10(depth + 1))| >
            cor_cut_off`` (ArchR ``corCutOff``).
        cor_cut_off: depth-correlation cut-off used with ``depth_key``.
        td: diffusion time; the block operator is ``W ** (td + 1)``.
        ka: kNN autotune — distances are divided by the distance to the
            ``ka``-th neighbour (self counts as neighbour 1, as in ArchR).
        sample_cells: target block size. ``None`` (or more than ``n_obs``)
            uses one block with every cell and forces ``n_rep = 1``.
        n_rep: number of random blockings averaged by :func:`impute_matrix`.
        k: neighbours per cell, including the cell itself.
        epsilon: Gaussian kernel width; ``0`` gives an unweighted kNN graph.
        seed: RNG seed for the random blockings.
        rng: ``"R"`` reproduces R's ``set.seed(seed); sample()`` so the blocks
            equal ArchR's (``threads = 1``); ``"numpy"`` uses
            ``numpy.random.default_rng(seed)``.
        blocks: explicit blockings, one list per replicate of 0-based cell
            index arrays; overrides ``sample_cells`` / ``n_rep`` / ``rng``.
        key_added: key in ``adata.uns`` for the weights.
        verbose: print progress.

    Returns:
        None. Stores ``adata.uns[key_added]`` with ``"weights"`` (a list per
        replicate of ``{"cells": int array, "weights": dense array}`` blocks)
        and ``"params"``.
    """
    if use_rep not in adata.obsm:
        raise KeyError(f"add_impute_weights: adata.obsm[{use_rep!r}] not found")
    X = np.asarray(adata.obsm[use_rep], dtype=np.float64)
    if dims_to_use is not None:
        X = X[:, list(dims_to_use)]
    if scale_dims:
        X = _row_zscores(X)
    if depth_key is not None:
        depth = np.log10(np.asarray(adata.obs[depth_key], dtype=np.float64) + 1.0)
        cors = np.array([np.corrcoef(X[:, j], depth)[0, 1] for j in range(X.shape[1])])
        keep = np.abs(cors) <= cor_cut_off
        if verbose and (~keep).any():
            console.info(f"[add_impute_weights] filtering {int((~keep).sum())} dims "
                         f"correlated > {cor_cut_off} to log10(depth + 1)")
        X = X[:, keep]
    n = X.shape[0]

    if blocks is not None:
        rep_blocks = [[np.asarray(b, dtype=np.int64) for b in rep] for rep in blocks]
    else:
        if sample_cells is not None and sample_cells > n:
            sample_cells = None
        if sample_cells is None:
            rep_blocks = [[np.arange(n)]]
        else:
            bin_size = _bin_size(n, sample_cells)
            gen = _RRandom(seed) if rng == "R" else np.random.default_rng(seed)
            rep_blocks = []
            for _ in range(n_rep):
                perm = gen.permutation(n)
                bid = np.ceil(np.arange(1, n + 1) / bin_size).astype(np.int64)
                rep_blocks.append([perm[bid == b] for b in np.unique(bid)])

    weights = []
    for r, rep in enumerate(rep_blocks):
        if verbose:
            console.info(f"[add_impute_weights] replicate {r + 1}/{len(rep_blocks)}: "
                         f"{len(rep)} block(s) of ~{len(rep[0]):,} cells")
        weights.append([
            {"cells": b, "weights": _block_diffusion(X[b], k, td, ka, epsilon)}
            for b in rep
        ])

    adata.uns[key_added] = {
        "weights": weights,
        "params": dict(use_rep=use_rep, td=td, k=k, ka=ka, epsilon=epsilon,
                       sample_cells=-1 if sample_cells is None else sample_cells,
                       n_rep=len(rep_blocks), seed=seed, rng=rng),
    }


def impute_matrix(
    adata: AnnData,
    genes: Optional[Sequence[str]] = None,
    *,
    X: Optional[Union[np.ndarray, sp.spmatrix]] = None,
    layer: Optional[str] = None,
    weights_key: str = "impute_weights",
    layer_added: Optional[str] = None,
) -> np.ndarray:
    """Impute a cells × features matrix with MAGIC weights (ArchR ``imputeMatrix``).

    Arguments:
        adata: AnnData holding ``adata.uns[weights_key]`` from
            :func:`add_impute_weights`.
        genes: subset of ``adata.var_names`` to impute. Default all.
        X: impute this ``(n_obs, n_features)`` matrix instead of
            ``adata.X`` / ``adata.layers[layer]``.
        layer: layer to impute instead of ``adata.X``.
        weights_key: key in ``adata.uns`` written by :func:`add_impute_weights`.
        layer_added: if set (and all features are imputed), also store the
            result in ``adata.layers[layer_added]`` for plotting with
            ``epi.pl.embedding(..., layer=layer_added)`` — the analogue of
            ArchR ``plotEmbedding(imputeWeights = ...)``.

    Returns:
        Dense ``(n_obs, n_features)`` float64 array of imputed values.
    """
    if weights_key not in adata.uns:
        raise KeyError(f"impute_matrix: run add_impute_weights first "
                       f"(adata.uns[{weights_key!r}] missing)")
    if X is None:
        sub = adata if genes is None else adata[:, list(genes)]
        X = sub.layers[layer] if layer is not None else sub.X
    if X.shape[0] != adata.n_obs:
        raise ValueError("impute_matrix: X must have one row per cell")
    weights = adata.uns[weights_key]["weights"]
    out = np.zeros(X.shape, dtype=np.float64)
    for rep in weights:
        for blk in rep:
            cells = np.asarray(blk["cells"])
            sub_x = X[cells]
            sub_x = sub_x.toarray() if sp.issparse(sub_x) else np.asarray(sub_x, dtype=np.float64)
            out[cells] += np.asarray(blk["weights"]) @ sub_x
    out /= len(weights)
    if layer_added is not None:
        if out.shape[1] != adata.n_vars:
            raise ValueError("impute_matrix: layer_added needs every feature imputed "
                             "(genes=None)")
        adata.layers[layer_added] = out
    return out
