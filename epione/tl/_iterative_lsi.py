"""
ArchR-style Iterative LSI for scATAC-seq data.

Reference:
    Granja, J.M., Corces, M.R., Pierce, S.E. et al. ArchR is a scalable software
    package for integrative single-cell chromatin accessibility analysis.
    Nat Genet 53, 403-411 (2021).

Algorithm (LSIMethod = 2, "Seurat" style, default in ArchR):
    1. Initial feature selection: keep the top `total_features` most accessible
       features (binarized), then clip the top `filter_quantile` tail.
    2. Optionally sub-sample `sample_cells_pre` cells for the first round to
       keep runtime bounded on very large datasets.
    3. TF-logIDF normalise, randomized SVD, drop components that correlate with
       sequencing depth (|r| > cor_cut_off).
    4. Cluster the embedding with Leiden (resolution can vary per iteration).
    5. Pick the top `var_features` features whose log-normalised pseudobulk
       accessibility varies most across clusters.
    6. Repeat steps 3-5 `iterations - 1` more times; the final LSI is run on
       all cells using the last feature set.

Results stored on the AnnData:
    adata.obsm[key_added]              : (n_obs, n_dims) final embedding
    adata.varm[key_added + "_loadings"]: (n_vars, n_dims) feature loadings
                                         (zero for features not selected in the
                                         final round)
    adata.uns[key_added]               : dict with keys
        "stdev"          - singular values of the final SVD
        "features_final" - boolean mask (n_vars,) of features used last round
        "depth_cor"      - per-component correlation with log10(depth)
        "kept_dims"      - indices of components kept after cor_cut_off filter
        "feature_sums"   - per-feature totals over the fitted cells (IDF)
        "n_cells_fit"    - number of cells in the final SVD
        "scale_mean", "scale_sd" - dim scaling of the kept components
        "iterations"     - list of per-iteration info (features, clusters)
        "params"         - the parameters the call was made with
"""
from __future__ import annotations

from typing import Literal, Optional, Sequence, Union
import numpy as np
import scipy.sparse as sp
from anndata import AnnData
import sklearn.utils.extmath

from ..utils import console


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------

def _binarize(X):
    """Return a binary (0/1) copy of X with dtype float32."""
    if sp.issparse(X):
        if X.format == "csr":
            # Same result as copy().astype(float32) + (data > 0) +
            # eliminate_zeros(), without the intermediate float32 copy.
            data = (X.data > 0).astype(np.float32)
            if data.all():
                # Structure unchanged: share indices / indptr with X (read-only
                # use downstream) instead of copying them.
                out = sp.csr_matrix((data, X.indices, X.indptr), shape=X.shape, copy=False)
            else:
                out = sp.csr_matrix((data, X.indices.copy(), X.indptr.copy()), shape=X.shape)
                out.eliminate_zeros()
            out.has_sorted_indices = bool(X.has_sorted_indices)
            return out
        out = X.copy().astype(np.float32)
        out.data = (out.data > 0).astype(np.float32)
        out.eliminate_zeros()
        return out
    return (X > 0).astype(np.float32)


def _all_positive(a: np.ndarray, chunk: int = 1 << 24) -> bool:
    """``(a > 0).all()`` without a full-size boolean temporary."""
    for i in range(0, len(a), chunk):
        if not (a[i:i + chunk] > 0).all():
            return False
    return True


def _row_sums(X):
    from ._lsi_kernels import BinaryCSR
    if isinstance(X, BinaryCSR):
        return X.row_sums()
    if sp.issparse(X):
        return np.asarray(X.sum(axis=1)).ravel()
    return X.sum(axis=1)


def _col_sums(X):
    from ._lsi_kernels import BinaryCSR
    if isinstance(X, BinaryCSR):
        return X.col_sums()
    if sp.issparse(X):
        return np.asarray(X.sum(axis=0)).ravel()
    return X.sum(axis=0)


def _tf_logidf(X, scale_to: float = 10000.0, n_jobs=None):
    """Seurat v3 / ArchR LSIMethod=2 normalisation.

    Computes log1p( (X / rowSums(X)) * idf * scale_to )
    where idf = log(1 + ncells / colSums(X)).

    Returns sparse CSR if X is sparse, else dense float32.
    """
    n_cells = X.shape[0]
    col = _col_sums(X)
    col_safe = np.where(col > 0, col, 1.0)
    idf = np.log(1.0 + n_cells / col_safe)

    row = _row_sums(X)
    row_safe = np.where(row > 0, row, 1.0)

    if sp.issparse(X) and X.format == "csr" and X.dtype == np.float32:
        # Threaded, bit-identical form of the generic branch below.
        from ._lsi_kernels import tf_logidf_data
        return tf_logidf_data(X, (1.0 / row_safe).astype(np.float32),
                              idf.astype(np.float32), scale_to, n_jobs=n_jobs)
    if sp.issparse(X):
        X = X.tocsr()
        # tf = X / row
        inv_row = (1.0 / row_safe).astype(np.float32)
        tf = X.multiply(inv_row[:, None])
        # * idf
        tf = tf.multiply(idf.astype(np.float32))
        # * scale_to
        tf = tf.tocsr()
        tf.data = np.log1p(tf.data * scale_to).astype(np.float32)
        return tf
    else:
        tf = (X / row_safe[:, None]) * idf[None, :] * scale_to
        return np.log1p(tf).astype(np.float32)


class _ThreadedCSR(sp.csr_matrix):
    """``csr_matrix`` whose sparse x dense products run on ``n_jobs``
    threads over row blocks.

    Each block is multiplied by SciPy's own ``csr_matvecs`` kernel (which
    releases the GIL) and every output row is computed exactly as in the
    single call, so results are bit-identical. ``.T`` is a lazy
    :class:`_CSRTranspose` (no copy), bit-identical as well.
    """

    _n_jobs = 1
    _transposed = None

    @classmethod
    def wrap(cls, X, n_jobs: int):
        out = cls(X, copy=False)
        out._n_jobs = max(1, int(n_jobs))
        return out

    def _blocks(self):
        blocks = getattr(self, "_row_blocks", None)
        if blocks is None:
            n_rows = self.shape[0]
            nb = max(1, min(n_rows, 4 * self._n_jobs))
            edges = np.linspace(0, n_rows, nb + 1).astype(np.int64)
            blocks = []
            for r0, r1 in zip(edges[:-1], edges[1:]):
                a, b = self.indptr[r0], self.indptr[r1]
                blocks.append(sp.csr_matrix(
                    (self.data[a:b], self.indices[a:b], self.indptr[r0:r1 + 1] - a),
                    shape=(int(r1 - r0), self.shape[1]), copy=False))
            self._row_blocks = blocks
        return blocks

    def _matmul_multivector(self, other):
        if self._n_jobs <= 1 or self.ndim != 2:
            return super()._matmul_multivector(other)
        from concurrent.futures import ThreadPoolExecutor
        other = np.ascontiguousarray(other)
        with ThreadPoolExecutor(max_workers=self._n_jobs) as pool:
            parts = list(pool.map(lambda blk: blk @ other, self._blocks()))
        return np.vstack(parts)

    def _rmatmul_dispatch(self, other):
        # other @ A == (A.T @ other.T).T, as SciPy evaluates it (csc_matvecs
        # on the transpose), via the lazy transpose.
        if self._n_jobs <= 1 or np.isscalar(other) or sp.issparse(other):
            return super()._rmatmul_dispatch(other)
        other = np.asarray(other)
        if other.ndim != 2:
            return super()._rmatmul_dispatch(other)
        return (self.transpose() @ other.T).T

    def transpose(self, axes=None, copy=False):
        if axes is not None or copy or self._n_jobs <= 1:
            return super().transpose(axes=axes, copy=copy)
        if self._transposed is None:
            self._transposed = _CSRTranspose(self)
        return self._transposed


class _CSRTranspose:
    """Lazy ``A.T`` of a :class:`_ThreadedCSR` for ``randomized_svd``.

    ``A.T @ Q`` runs a numba kernel over column (= output row) ranges of
    ``A`` on ``n_jobs`` threads; every output row accumulates its entries in
    increasing row-of-``A`` order with the same float operations as SciPy's
    ``csc_matvecs`` (what ``A.T @ Q`` calls), so it is bit-identical without
    materialising the transpose. ``Q.T @ A.T`` is evaluated as
    ``(A @ Q).T`` exactly like SciPy's ``__rmatmul__``.
    """

    __array_ufunc__ = None          # ndarray @ self -> self.__rmatmul__

    def __init__(self, base):
        self._base = base
        self.shape = (base.shape[1], base.shape[0])
        self.dtype = base.dtype
        self.ndim = 2

    @property
    def T(self):
        return self._base

    def transpose(self, *args, **kwargs):
        return self._base

    def __matmul__(self, other):
        from ._lsi_kernels import csr_t_matmul
        other = np.asarray(other)
        if other.ndim == 1:
            return csr_t_matmul(self._base, other[:, None], self._base._n_jobs)[:, 0]
        return csr_t_matmul(self._base, other, self._base._n_jobs)

    def __rmatmul__(self, other):
        return (self._base @ np.asarray(other).T).T


def _randomized_svd(X, n_components: int, random_state: int = 0, device: str = "cpu",
                    n_jobs: Optional[int] = None):
    """Randomized SVD returning U, s, Vt with largest singular values first."""
    n_components = int(min(n_components, min(X.shape) - 1))
    if device != "cpu":
        # torch port of sklearn's randomized_svd: same Gaussian test matrix
        # (same RandomState), same LU / QR power iterations, same sign flip.
        from ..core._gpu import torch_device
        from ..core._gpu_ops import randomized_svd
        return randomized_svd(X, n_components, dev=torch_device(device),
                              random_state=random_state, n_iter="auto")
    if sp.issparse(X) and X.format == "csr" and X.has_sorted_indices:
        from ..pp._fragment_io import resolve_n_jobs
        X = _ThreadedCSR.wrap(X, resolve_n_jobs(n_jobs))
    U, s, Vt = sklearn.utils.extmath.randomized_svd(
        X, n_components=n_components,
        random_state=random_state, n_iter="auto",
    )
    return U, s, Vt


def _project(X_new, V, s):
    """Project new cells into an existing SVD space: X_new @ V @ diag(1/s)."""
    # V is (n_vars, k); s is (k,)
    if sp.issparse(X_new):
        proj = X_new.dot(V)
    else:
        proj = X_new @ V
    s_safe = np.where(s > 0, s, 1.0)
    return proj / s_safe[None, :]


def _select_top_features(X_bin, n_keep: int, filter_quantile: float):
    """Rank features by total accessibility and keep the top n_keep, after
    trimming features above the upper `filter_quantile`."""
    totals = _col_sums(X_bin)
    n_feat = totals.shape[0]

    # Trim the top quantile (ArchR: filterQuantile = 0.995 -> drop top 0.5%)
    upper = np.quantile(totals[totals > 0], filter_quantile) if (totals > 0).any() else np.inf
    mask = (totals > 0) & (totals <= upper)

    idx = np.where(mask)[0]
    ranked = idx[np.argsort(-totals[idx])]
    keep = ranked[:n_keep]

    out = np.zeros(n_feat, dtype=bool)
    out[keep] = True
    return out


def _scale_dims(U: np.ndarray) -> np.ndarray:
    """Z-score each SVD component (column-wise)."""
    out = U - U.mean(axis=0, keepdims=True)
    sd = out.std(axis=0, ddof=1, keepdims=True)
    sd[sd == 0] = 1.0
    return out / sd


def _correlate_with_depth(U: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Pearson correlation between each SVD component and log10(depth)."""
    x = np.log10(np.asarray(depth, dtype=np.float64) + 1.0)
    x = x - x.mean()
    x_sd = x.std()
    if x_sd == 0:
        return np.zeros(U.shape[1])
    cors = np.zeros(U.shape[1])
    for i in range(U.shape[1]):
        y = U[:, i] - U[:, i].mean()
        y_sd = y.std()
        if y_sd == 0:
            cors[i] = 0.0
        else:
            cors[i] = float(np.dot(x, y) / (x_sd * y_sd * len(x)))
    return cors


def _variable_features_from_clusters(
    X_bin, clusters: np.ndarray, var_features: int, scale_to: float = 10000.0,
    binary: bool = False,
) -> np.ndarray:
    """Top-N variable features across cluster pseudobulks.

    Mirrors ArchR's ``.getTopFeatures`` when ``selection == "var"``: take the
    **mean** binary accessibility per cluster (= fraction of cells with the
    feature open), log2-transform with a pseudocount of 1, then pick the
    ``var_features`` rows with the highest across-cluster variance.
    ``scale_to`` is kept only for API parity – it is not applied to the mean.

    Falls back to "most accessible features" if every column is constant
    across clusters (degenerate case on tiny data).
    """
    labels = np.asarray(clusters)
    uniq, idx, counts = np.unique(labels, return_inverse=True, return_counts=True)
    n_feat = X_bin.shape[1]
    pb = np.zeros((len(uniq), n_feat), dtype=np.float32)

    from ._lsi_kernels import BinaryCSR
    X_csr = X_bin.tocsr() if sp.issparse(X_bin) else X_bin
    group_sums = None
    if isinstance(X_csr, BinaryCSR) or (
            binary and sp.issparse(X_csr) and X_csr.dtype == np.float32):
        # One pass instead of a row subset per cluster; sums of 0/1 float32
        # values are exact integers, so identical to sub.sum(axis=0).
        from ._lsi_kernels import group_col_sums
        group_sums = group_col_sums(X_csr, idx.ravel(), len(uniq))

    for i, c in enumerate(uniq):
        sel = np.where(labels == c)[0]
        if group_sums is not None:
            sums = group_sums[i]
        elif sp.issparse(X_csr):
            sub = X_csr[sel]
            sums = np.asarray(sub.sum(axis=0)).ravel()
        else:
            sub = X_csr[sel]
            sums = sub.sum(axis=0)
        # ArchR-style: per-cluster MEAN (fraction of cells with peak open).
        n_i = max(int(counts[i]), 1)
        pb[i] = np.log2(sums / n_i + 1.0)

    var = pb.var(axis=0, ddof=1)
    n_keep = int(min(var_features, (var > 0).sum()))
    if n_keep == 0:
        totals = _col_sums(X_bin)
        keep = np.argsort(-totals)[: min(var_features, n_feat)]
    else:
        keep = np.argsort(-var)[:n_keep]
    out = np.zeros(n_feat, dtype=bool)
    out[keep] = True
    return out


# ---------------------------------------------------------------------------
# Internal clustering helper
# ---------------------------------------------------------------------------

class _TorchKNNTransformer:
    """Minimal sklearn-style kNN transformer for :func:`scanpy.pp.neighbors`:
    exact cosine / euclidean search on the GPU. ``fit_transform`` returns the
    ``(n, n)`` CSR distance graph with ``n_neighbors`` entries per row, self
    first at distance 0, in increasing distance -- the layout scanpy reads
    with its constant-nnz shortcut."""

    def __init__(self, n_neighbors: int, metric: str, device: str):
        self.n_neighbors = int(n_neighbors)
        self.metric = metric
        self.device = device

    def get_params(self, deep: bool = True):
        return {"n_neighbors": self.n_neighbors, "metric": self.metric}

    def fit_transform(self, X):
        from ..core._gpu import torch_device
        from ..core._gpu_ops import knn
        X = np.asarray(X)
        n = X.shape[0]
        k = min(self.n_neighbors, n)
        idx, dist = knn(X, k, dev=torch_device(self.device), metric=self.metric)
        return sp.csr_matrix(
            (dist.ravel(), idx.ravel(), np.arange(0, n * k + 1, k)),
            shape=(n, n),
        )


def _pynndescent_no_compress(n_obs: int, n_neighbors: int, metric: str, random_state):
    """``PyNNDescentTransformer`` configured exactly as
    ``scanpy.pp.neighbors(transformer=None)`` configures it, whose
    ``fit_transform`` skips the final ``index_.compress_index()``."""
    import scanpy as sc
    from pynndescent import PyNNDescentTransformer

    class _NoCompress(PyNNDescentTransformer):
        def fit_transform(self, X, y=None, **fit_params):
            self.fit(X, compress_index=False)
            return self.transform(X=None)

    return _NoCompress(
        n_neighbors=n_neighbors, metric=metric, metric_kwds={},
        random_state=random_state, n_jobs=sc.settings.n_jobs,
        n_trees=min(64, 5 + round(n_obs ** 0.5 / 20.0)),
        n_iters=max(5, round(np.log2(n_obs))),
    )


def _knn_leiden(
    emb: np.ndarray, n_neighbors: int, resolution: float, seed: int,
    device: str = "cpu",
) -> np.ndarray:
    """Build a cosine kNN graph on `emb` and run Leiden.

    On a GPU device the kNN search is exact (torch brute force); on the CPU
    scanpy chooses (pynndescent above 4,096 cells). Connectivities and Leiden
    are scanpy's in both cases."""
    import scanpy as sc
    from anndata import AnnData as _A

    # Tiny adata with emb as .obsm["X_lsi"]
    tmp = _A(np.zeros((emb.shape[0], 1), dtype=np.float32))
    tmp.obsm["X_emb"] = emb.astype(np.float32)
    if device != "cpu":
        sc.pp.neighbors(
            tmp, use_rep="X_emb",
            n_neighbors=int(n_neighbors), metric="cosine", random_state=seed,
            transformer=_TorchKNNTransformer(int(n_neighbors), "cosine", device),
        )
    elif emb.shape[0] >= 4096:
        # The transformer scanpy picks itself for >= 4096 cells (same
        # PyNNDescentTransformer arguments, so the same graph), minus the
        # post-fit ``compress_index`` that only prepares (and JIT-compiles)
        # the query path -- this throw-away graph is never queried.
        sc.pp.neighbors(
            tmp, use_rep="X_emb",
            n_neighbors=int(n_neighbors), metric="cosine", random_state=seed,
            transformer=_pynndescent_no_compress(
                n_obs=emb.shape[0], n_neighbors=int(n_neighbors),
                metric="cosine", random_state=seed),
        )
    else:
        sc.pp.neighbors(
            tmp, use_rep="X_emb",
            n_neighbors=int(n_neighbors), metric="cosine", random_state=seed,
        )
    sc.tl.leiden(
        tmp, resolution=float(resolution),
        random_state=seed, key_added="leiden",
        flavor="igraph", directed=False, n_iterations=2,
    )
    return tmp.obs["leiden"].to_numpy()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def iterative_lsi(
    adata: AnnData,
    n_components: int = 30,
    iterations: int = 2,
    var_features: int = 25000,
    total_features: int = 500000,
    filter_quantile: float = 0.995,
    resolution: Union[float, Sequence[float]] = 0.2,
    n_neighbors: int = 20,
    sample_cells_pre: Optional[int] = 10000,
    sample_cells_final: Optional[int] = None,
    binarize: bool = True,
    scale_to: float = 10000.0,
    scale_dims: bool = True,
    cor_cut_off: float = 0.75,
    depth_col: Optional[str] = None,
    seed: int = 0,
    key_added: str = "X_iterative_lsi",
    layer: Optional[str] = None,
    copy: bool = False,
    verbose: bool = True,
    device: str = "cpu",
    n_jobs: Optional[int] = None,
) -> Optional[AnnData]:
    """Perform ArchR-style iterative LSI on a cell-by-feature matrix.

    Parameters
    ----------
    adata
        Cell-by-peak or cell-by-tile AnnData.
    n_components
        Number of singular vectors to retain *before* depth filtering.
    iterations
        Number of LSI rounds. Must be >= 1. Typical value is 2 or 3.
    var_features
        Number of top-variable features to keep after the first round.
    total_features
        Number of features retained by the initial top-accessibility filter.
    filter_quantile
        Upper accessibility quantile to clip (mirrors ArchR filterQuantile).
    resolution
        Leiden resolution. Scalar, or a sequence of length ``iterations - 1``
        giving the resolution for each intermediate clustering step.
    n_neighbors
        k for the kNN graph used during intermediate clustering.
    sample_cells_pre
        If set and the dataset has more cells than this, run the first LSI on
        a random subsample; remaining cells are projected into that SVD space
        for clustering purposes only. Pass ``None`` to use all cells.
    sample_cells_final
        Same idea for the final SVD (rarely needed).
    binarize
        Binarise the matrix before LSI (ATAC default).
    scale_to
        Multiplier inside log1p (ArchR uses 10000).
    scale_dims
        Z-score each SVD component after fitting.
    cor_cut_off
        Drop components whose absolute correlation with log10(depth) exceeds
        this value. Set to 1.0 to keep all components.
    depth_col
        Column in ``adata.obs`` giving per-cell depth. If ``None``, row sums
        of the input matrix are used.
    seed
        Seed for SVD and Leiden.
    key_added
        Base key for ``.obsm``, ``.varm`` (``key_added + '_loadings'``) and
        ``.uns`` outputs.
    layer
        Optional ``adata.layers[layer]`` to use instead of ``adata.X``.
    copy
        If True, work on and return a copy.
    verbose
        Print progress.
    device
        ``"cpu"`` (default) runs the NumPy / SciPy / sklearn implementation.
        ``"cuda"`` (``"cuda:N"``, or ``"auto"``) runs each round's randomized
        SVD with a torch port of sklearn's ``randomized_svd`` (same random
        test matrix for the same ``seed``, so a single round agrees with the
        CPU to floating-point rounding) and the kNN search of the
        intermediate clustering as an exact GPU search. The CPU path lets
        scanpy use approximate pynndescent search there, so with
        ``iterations >= 2`` the intermediate clusters -- and hence the
        variable features of later rounds -- can differ slightly from the
        CPU run, as they do between CPU seeds. TF-logIDF, feature selection
        and Leiden stay on the CPU. Needs ``pip install epione[gpu]``.
    n_jobs
        Threads for the sparse x dense products of the CPU randomized SVD
        (row-blocked SciPy kernels; results are bit-identical to a single
        thread). ``None`` (default) uses every CPU available to the process.

    Examples
    --------
    >>> import epione as ep
    >>> # adata: cells x tiles (or peaks); binary or counts
    >>> ep.tl.iterative_lsi(
    ...     adata,
    ...     iterations=2,
    ...     var_features=25000,
    ...     depth_col="n_fragment",
    ...     seed=1,
    ... )
    >>> adata.obsm["X_iterative_lsi"].shape
    (n_obs, n_components_kept)

    See also
    --------
    epione.tl.lsi : a single-round LSI (no iterative feature refinement).
    """
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    from ..core._gpu import resolve_device
    device = resolve_device(device)

    if copy:
        adata = adata.copy()

    # Resolve source matrix
    X_full = adata.layers[layer] if layer is not None else adata.X
    # Materialise anndataoom BackedArray → scipy.sparse. Iterative LSI
    # scans X multiple times (binarisation, feature selection, TF-IDF)
    # and the lazy BackedArray doesn't implement `X > 0` / arithmetic.
    if type(X_full).__name__ in ("BackedArray", "_SubsetBackedArray",
                                  "TransformedBackedArray", "ScaledBackedArray"):
        parts = []
        for _, _, chunk in X_full.chunked():
            parts.append(chunk)
        if sp.issparse(parts[0]):
            X_full = sp.vstack(parts).tocsr()
        else:
            X_full = np.vstack(parts)
    n_obs, n_vars = X_full.shape

    # 1. Binarise if requested. For a CSR matrix with only positive stored
    # values the binary matrix is its structure with all-ones data: keep it
    # as a data-free view (every derived quantity is identical) instead of
    # allocating a float32 copy of the values.
    from ._lsi_kernels import BinaryCSR, csr_select
    if (binarize and sp.issparse(X_full) and X_full.format == "csr"
            and X_full.has_sorted_indices and X_full.nnz and _all_positive(X_full.data)):
        X_bin = BinaryCSR(X_full.indptr, X_full.indices, X_full.shape)
    else:
        X_bin = _binarize(X_full) if binarize else X_full
        if sp.issparse(X_bin):
            X_bin = X_bin.tocsr()

    # 2. Initial top-accessibility feature set
    features = _select_top_features(X_bin, n_keep=total_features, filter_quantile=filter_quantile)
    if verbose:
        console.info(f"[iterative_lsi] Initial feature set: {int(features.sum()):,} / {n_vars:,}")

    # Per-cell depth for correlation filtering
    if depth_col is not None and depth_col in adata.obs.columns:
        depth_all = adata.obs[depth_col].to_numpy()
    else:
        depth_all = _row_sums(X_bin)

    # Normalise resolution into a list of length (iterations - 1)
    if iterations == 1:
        resolutions = []
    elif isinstance(resolution, (int, float, np.integer, np.floating)):
        resolutions = [float(resolution)] * (iterations - 1)
    else:
        resolutions = [float(r) for r in resolution]
        if len(resolutions) < iterations - 1:
            resolutions = resolutions + [resolutions[-1]] * (iterations - 1 - len(resolutions))
        resolutions = resolutions[: iterations - 1]

    rng = np.random.default_rng(seed)
    iter_info = []

    # Iterative rounds -----------------------------------------------------
    for it in range(iterations):
        is_final = it == iterations - 1

        # Choose cells for this round's SVD
        if is_final and sample_cells_final and n_obs > sample_cells_final:
            fit_idx = rng.choice(n_obs, size=sample_cells_final, replace=False)
        elif (not is_final) and sample_cells_pre and n_obs > sample_cells_pre:
            fit_idx = rng.choice(n_obs, size=sample_cells_pre, replace=False)
        else:
            fit_idx = np.arange(n_obs)

        # Subset to current feature mask
        feat_idx = np.where(features)[0]
        if isinstance(X_bin, BinaryCSR) or (sp.issparse(X_bin) and X_bin.format == "csr"):
            # == X_bin[fit_idx][:, feat_idx], threaded
            X_sub = csr_select(
                X_bin, feat_idx, n_jobs=n_jobs,
                rows=None if np.array_equal(fit_idx, np.arange(n_obs)) else fit_idx)
        elif sp.issparse(X_bin):
            X_sub = X_bin[fit_idx][:, feat_idx]
        else:
            X_sub = X_bin[np.ix_(fit_idx, feat_idx)]

        if verbose:
            console.info(
                f"[iterative_lsi] Iter {it+1}/{iterations} | "
                f"fit on {len(fit_idx):,} cells x {len(feat_idx):,} features"
            )

        # TF-logIDF + SVD
        X_norm = _tf_logidf(X_sub, scale_to=scale_to, n_jobs=n_jobs)
        # Only its column sums are needed later; free the values now.
        sub_col_sums = _col_sums(X_sub)
        del X_sub
        U, s, Vt = _randomized_svd(X_norm, n_components=n_components, random_state=seed + it,
                                   device=device, n_jobs=n_jobs)

        # Depth-correlation filter
        depth_fit = depth_all[fit_idx]
        depth_cor = _correlate_with_depth(U, depth_fit)
        kept_dims = np.where(np.abs(depth_cor) <= cor_cut_off)[0]
        if kept_dims.size == 0:
            # Keep everything as a safety net
            kept_dims = np.arange(U.shape[1])

        # Project any cells not used for fitting into the same SVD space for
        # clustering / final embedding purposes.
        if len(fit_idx) == n_obs:
            U_full = U
        else:
            if isinstance(X_bin, BinaryCSR) or (sp.issparse(X_bin) and X_bin.format == "csr"):
                X_rest = csr_select(X_bin, feat_idx, n_jobs=n_jobs)
            elif sp.issparse(X_bin):
                X_rest = X_bin[:, feat_idx]
            else:
                X_rest = X_bin[:, feat_idx]
            X_rest_norm = _tf_logidf(X_rest, scale_to=scale_to, n_jobs=n_jobs)
            U_full = _project(X_rest_norm, Vt.T, s)
            # Overwrite the rows we actually fitted with the exact values
            U_full[fit_idx] = U

        emb = U_full[:, kept_dims]
        emb_mean = emb.mean(axis=0)
        emb_sd = emb.std(axis=0, ddof=1)
        if scale_dims:
            emb = _scale_dims(emb)

        iter_record = {
            "iteration": it + 1,
            "n_cells_fit": int(len(fit_idx)),
            "n_features": int(features.sum()),
            "kept_dims": kept_dims.tolist(),
            "depth_cor": depth_cor.tolist(),
        }

        if not is_final:
            res = resolutions[it]
            clusters = _knn_leiden(emb, n_neighbors=n_neighbors, resolution=res, seed=seed + it,
                                   device=device)
            iter_record["resolution"] = res
            iter_record["n_clusters"] = int(len(np.unique(clusters)))

            # Variable-feature selection for the next round
            features = _variable_features_from_clusters(
                X_bin, clusters, var_features=var_features, scale_to=scale_to,
                binary=binarize,
            )
            if verbose:
                console.info(
                    f"[iterative_lsi]   -> {iter_record['n_clusters']} clusters; "
                    f"selected {int(features.sum()):,} variable features for next round"
                )

        iter_info.append(iter_record)

        if is_final:
            # Store the final embedding and loadings
            adata.obsm[key_added] = emb.astype(np.float32)

            loadings = np.zeros((n_vars, emb.shape[1]), dtype=np.float32)
            loadings[feat_idx] = Vt[kept_dims].T.astype(np.float32)
            adata.varm[key_added + "_loadings"] = loadings

            # Flatten per-iteration history into array-of-arrays for h5ad
            # friendliness (nested dicts with variable-length lists don't
            # serialise cleanly).
            iter_summary = {
                "iteration": np.asarray(
                    [r["iteration"] for r in iter_info], dtype=np.int32),
                "n_cells_fit": np.asarray(
                    [r["n_cells_fit"] for r in iter_info], dtype=np.int64),
                "n_features": np.asarray(
                    [r["n_features"] for r in iter_info], dtype=np.int64),
                "n_clusters": np.asarray(
                    [r.get("n_clusters", 0) for r in iter_info], dtype=np.int32),
                "resolution": np.asarray(
                    [r.get("resolution", np.nan) for r in iter_info],
                    dtype=np.float32),
            }

            # LSI model for projecting new profiles (epione.tl.LSIModel):
            # IDF denominators over the fitted cells and the dim scaling.
            fit_sums = sub_col_sums.astype(np.float64)
            adata.uns[key_added] = {
                "stdev": s.astype(np.float32),
                "feature_sums": fit_sums,
                "n_cells_fit": int(len(fit_idx)),
                "scale_mean": emb_mean.astype(np.float64),
                "scale_sd": emb_sd.astype(np.float64),
                "features_final": features.astype(np.int8),
                "depth_cor": depth_cor.astype(np.float32),
                "kept_dims": kept_dims.astype(np.int32),
                "iterations": iter_summary,
                "params": {
                    k: (list(v) if isinstance(v, (list, tuple))
                        else ("" if v is None else v))
                    for k, v in dict(
                        n_components=n_components,
                        iterations=iterations,
                        var_features=var_features,
                        total_features=total_features,
                        filter_quantile=filter_quantile,
                        resolution=list(resolutions),
                        n_neighbors=n_neighbors,
                        sample_cells_pre=sample_cells_pre,
                        sample_cells_final=sample_cells_final,
                        binarize=binarize,
                        scale_to=scale_to,
                        scale_dims=scale_dims,
                        cor_cut_off=cor_cut_off,
                        depth_col=depth_col,
                        seed=seed,
                    ).items()
                },
            }
            if verbose:
                console.info(
                    f"[iterative_lsi] Done. Stored embedding "
                    f"({emb.shape[0]:,} x {emb.shape[1]}) in adata.obsm['{key_added}']"
                )

    return adata if copy else None
