"""torch kernels shared by Epione's accelerated functions.

* :func:`randomized_svd` -- a line-by-line torch port of
  :func:`sklearn.utils.extmath.randomized_svd` (Halko et al. 2011). The
  Gaussian test matrix is drawn from the same NumPy ``RandomState`` as
  sklearn, the power iterations use the same LU / QR normalisers and the
  same ``svd_flip`` sign convention, so for a given ``random_state`` the
  result matches sklearn up to floating-point rounding.
* :func:`knn` -- exact brute-force k-nearest neighbours, query-blocked so the
  distance tile stays within the memory budget.

All functions take and return NumPy arrays; device tensors stay internal.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp

from ._gpu import (
    SparseOperand, capped_budget, get_torch, memory_budget, row_blocks, torch_dtype,
)


# ---------------------------------------------------------------------------
# Randomized SVD
# ---------------------------------------------------------------------------

def _svd_flip(u, v, u_based_decision: bool = True):
    """torch port of :func:`sklearn.utils.extmath.svd_flip`."""
    torch = get_torch()
    if u_based_decision:
        idx = torch.argmax(torch.abs(u), dim=0)
        signs = torch.sign(u[idx, torch.arange(u.shape[1], device=u.device)])
    else:
        idx = torch.argmax(torch.abs(v), dim=1)
        signs = torch.sign(v[torch.arange(v.shape[0], device=v.device), idx])
    u = u * signs[None, :]
    v = v * signs[:, None]
    return u, v


class _DenseOperand:
    def __init__(self, A):
        self.A = A
        self.shape = tuple(A.shape)

    def matmul(self, D):
        return self.A @ D

    def rmatmul(self, D):
        return self.A.T @ D


class _TOp:
    """Transposed view of an operand exposing matmul / rmatmul."""

    def __init__(self, op):
        self.op = op
        self.shape = (op.shape[1], op.shape[0])

    def matmul(self, D):
        return self.op.rmatmul(D)

    def rmatmul(self, D):
        return self.op.matmul(D)


def randomized_svd(
    M,
    n_components: int,
    *,
    dev,
    dtype=None,
    n_oversamples: int = 10,
    n_iter: Union[int, str] = "auto",
    power_iteration_normalizer: str = "auto",
    transpose: Union[bool, str] = "auto",
    flip_sign: bool = True,
    random_state=None,
    budget: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """torch port of :func:`sklearn.utils.extmath.randomized_svd`.

    Parameters
    ----------
    M
        SciPy sparse matrix, NumPy array, or an already-built operator: a
        :class:`~epione.core._gpu.SparseOperand` or any object with
        ``shape``, a torch ``dtype``, ``matmul(D)`` (``M @ D``) and
        ``rmatmul(D)`` (``M.T @ D``) on device tensors.
    dev
        torch device.
    dtype
        NumPy float dtype of the computation; defaults to ``M``'s float dtype
        (float64 for integer input), matching sklearn.
    budget
        Device bytes the sparse operand may occupy before it is streamed.

    Returns ``(U, s, Vt)`` as NumPy arrays, exactly like sklearn.
    """
    from sklearn.utils import check_random_state

    torch = get_torch()
    is_op = isinstance(M, SparseOperand) or (
        hasattr(M, "matmul") and hasattr(M, "rmatmul") and not sp.issparse(M)
        and not isinstance(M, np.ndarray))
    if dtype is None:
        if is_op:
            dtype = np.float32 if M.dtype == torch.float32 else np.float64
        else:
            src = np.dtype(getattr(M, "dtype", np.float64))
            dtype = src if src.kind == "f" else np.float64
    dtype = np.dtype(dtype)
    tdt = torch_dtype(dtype)

    if is_op:
        op = M
    elif sp.issparse(M):
        op = SparseOperand(M, dev, tdt, budget=budget)
    else:
        op = _DenseOperand(torch.as_tensor(np.asarray(M, dtype=dtype), device=dev))

    random_state = check_random_state(random_state)
    n_random = n_components + n_oversamples
    n_samples, n_features = op.shape
    if n_iter == "auto":
        n_iter = 7 if n_components < 0.1 * min(op.shape) else 4
    if transpose == "auto":
        transpose = n_samples < n_features
    A = _TOp(op) if transpose else op

    # ---- range finder (sklearn _randomized_range_finder) ----
    Q = random_state.normal(size=(A.shape[1], n_random))
    if dtype == np.float32:
        Q = Q.astype(np.float32, copy=False)
    Q = torch.as_tensor(Q, device=dev, dtype=tdt)

    if power_iteration_normalizer == "auto":
        power_iteration_normalizer = "none" if n_iter <= 2 else "LU"

    def _qr(X):
        return torch.linalg.qr(X, mode="reduced")[0]

    def _lu(X):
        # sklearn uses scipy.linalg.lu(X, permute_l=True) on this tall-skinny
        # (n x (k + p)) block. Doing exactly that on the host keeps the
        # pivoting identical and costs one small transfer per iteration
        # (torch's CUDA LU of such shapes is MAGMA's batched kernel, which is
        # slower here and prints a performance banner).
        from scipy import linalg
        PL = linalg.lu(X.cpu().numpy(), permute_l=True, check_finite=False)[0]
        return torch.as_tensor(PL, device=X.device)

    if power_iteration_normalizer == "QR":
        normalizer = _qr
    elif power_iteration_normalizer == "LU":
        normalizer = _lu
    else:
        normalizer = lambda X: X  # noqa: E731

    for _ in range(n_iter):
        Q = normalizer(A.matmul(Q))
        Q = normalizer(A.rmatmul(Q))
    Q = _qr(A.matmul(Q))

    # ---- project + small SVD ----
    B = A.rmatmul(Q).T            # Q.T @ A
    Uhat, s, Vt = torch.linalg.svd(B, full_matrices=False)
    del B
    U = Q @ Uhat
    if flip_sign:
        if not transpose:
            U, Vt = _svd_flip(U, Vt)
        else:
            U, Vt = _svd_flip(U, Vt, u_based_decision=False)

    if transpose:
        out = (Vt[:n_components, :].T, s[:n_components], U[:, :n_components].T)
    else:
        out = (U[:, :n_components], s[:n_components], Vt[:n_components, :])
    return tuple(np.ascontiguousarray(t.detach().cpu().numpy()) for t in out)


# ---------------------------------------------------------------------------
# Exact kNN
# ---------------------------------------------------------------------------

_KNN_METRICS = {
    "euclidean", "l2", "sqeuclidean", "cosine", "correlation",
    "manhattan", "cityblock", "l1", "chebyshev", "minkowski",
}


def knn(
    X: np.ndarray,
    n_neighbors: int,
    *,
    dev,
    metric: str = "euclidean",
    metric_kwds: Optional[dict] = None,
    dtype=np.float64,
    budget: Optional[int] = None,
    Y: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact brute-force kNN of the rows of ``Y`` (default ``X``) among ``X``.

    Mirrors ``sklearn.neighbors.NearestNeighbors(algorithm="brute")
    .fit(X).kneighbors(Y)``: when ``Y is X`` each point is its own first
    neighbour. Distances are computed in ``dtype`` (float64 by default, as
    sklearn upcasts float32 euclidean distances to float64).

    Returns ``(indices int64 (n_query, k), distances float32 (n_query, k))``,
    sorted by increasing distance.
    """
    torch = get_torch()
    metric_kwds = dict(metric_kwds or {})
    metric = str(metric).lower()
    if metric not in _KNN_METRICS:
        raise NotImplementedError(
            f"metric={metric!r} has no GPU implementation; supported: "
            f"{sorted(_KNN_METRICS)}. Use device='cpu'."
        )
    tdt = torch_dtype(dtype)
    Xd = torch.as_tensor(np.ascontiguousarray(X), device=dev).to(tdt)
    Yd = Xd if Y is None else torch.as_tensor(np.ascontiguousarray(Y), device=dev).to(tdt)
    n_ref = Xd.shape[0]
    k = int(min(n_neighbors, n_ref))

    if metric in ("cosine", "correlation"):
        if metric == "correlation":
            Xd = Xd - Xd.mean(dim=1, keepdim=True)
            Yd = Yd - Yd.mean(dim=1, keepdim=True) if Y is not None else Xd
        xn = torch.linalg.vector_norm(Xd, dim=1, keepdim=True)
        Xn = Xd / torch.where(xn > 0, xn, torch.ones_like(xn))
        if Y is None:
            Yn = Xn
        else:
            yn = torch.linalg.vector_norm(Yd, dim=1, keepdim=True)
            Yn = Yd / torch.where(yn > 0, yn, torch.ones_like(yn))
    else:
        x_sq = (Xd * Xd).sum(dim=1)

    if budget is None:
        # the distance tile only sets the query batch size: cap it so an
        # exact search over many cells does not fill a shared GPU
        budget = capped_budget(dev, 4.0)
    itemsize = torch.tensor([], dtype=tdt).element_size()
    # distance tile + topk scratch per query row
    bytes_per_row = 3 * n_ref * itemsize
    idx_out = np.empty((Yd.shape[0], k), dtype=np.int64)
    dist_out = np.empty((Yd.shape[0], k), dtype=np.float32)
    for blk in row_blocks(Yd.shape[0], bytes_per_row, budget):
        if metric in ("cosine", "correlation"):
            D = 1.0 - Yn[blk] @ Xn.T
            D.clamp_(min=0.0, max=2.0)
        elif metric in ("euclidean", "l2", "sqeuclidean"):
            q = Yd[blk]
            D = (q * q).sum(dim=1, keepdim=True) - 2.0 * (q @ Xd.T) + x_sq[None, :]
            D.clamp_(min=0.0)
        else:
            p = {"manhattan": 1.0, "cityblock": 1.0, "l1": 1.0,
                 "chebyshev": float("inf")}.get(metric, float(metric_kwds.get("p", 2)))
            D = torch.cdist(Yd[blk], Xd, p=p)
        if Y is None:
            # A point's distance to itself is exactly 0. (The expansion
            # |x|^2 + |y|^2 - 2 x.y leaves a ~1e-15 residue, which UMAP's
            # connectivities would otherwise mistake for the nearest
            # neighbour distance rho.)
            rows = torch.arange(blk.start, blk.stop, device=dev)
            D[rows - blk.start, rows] = 0.0
        vals, inds = torch.topk(D, k, dim=1, largest=False, sorted=True)
        if metric in ("euclidean", "l2"):
            vals = torch.sqrt(vals)
        idx_out[blk] = inds.cpu().numpy()
        dist_out[blk] = vals.to(torch.float32).cpu().numpy()
        del D, vals, inds
    return idx_out, dist_out
