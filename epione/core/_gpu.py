"""Shared plumbing for Epione's optional torch (GPU) backend.

Every accelerated function in Epione takes a ``device=`` argument:

``"cpu"`` (the default)
    The original NumPy / SciPy implementation. Nothing in this module is
    touched and torch is never imported, so CPU results are unchanged.
``"cuda"``, ``"cuda:N"``
    The torch implementation on that GPU. Raises if torch or CUDA is
    missing, so that a typo never silently falls back to the CPU.
``"auto"``
    ``"cuda"`` when torch is importable and a CUDA device is visible,
    otherwise ``"cpu"``.

torch is imported lazily (only when a non-CPU device is resolved), so the
base install does not need it. Install it with ``pip install epione[gpu]``.

The helpers here are deliberately small:

* :func:`resolve_device` / :func:`torch_device` -- device resolution;
* :func:`to_torch_csr` / :func:`csr_from_torch` -- SciPy CSR <-> torch CSR;
* :func:`row_blocks` / :func:`csr_row_blocks` -- batching plans that bound
  device memory (pure NumPy, no torch needed);
* :class:`SparseOperand` -- a CSR matrix that lives on the device when it
  fits the memory budget and is streamed in row blocks otherwise, with
  ``X @ D`` and ``X.T @ D`` products;
* :func:`memory_budget` -- the number of device bytes a single call may
  use (``EPIONE_GPU_MEMORY_GB`` overrides it).
"""
from __future__ import annotations

import os
from typing import Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp

__all__ = [
    "resolve_device",
    "is_torch_device",
    "torch_device",
    "get_torch",
    "memory_budget",
    "row_blocks",
    "csr_row_blocks",
    "to_torch_csr",
    "csr_from_torch",
    "to_numpy",
    "SparseOperand",
    "torch_dtype",
    "empty_cache",
]

_CPU = "cpu"
_ENV_BUDGET = "EPIONE_GPU_MEMORY_GB"


# ---------------------------------------------------------------------------
# torch import + device resolution
# ---------------------------------------------------------------------------

def get_torch():
    """Import and return :mod:`torch`, with an install hint if missing."""
    try:
        import torch
    except ImportError as e:  # pragma: no cover - exercised without torch
        raise ImportError(
            "This code path needs PyTorch. Install the GPU extra with "
            "`pip install epione[gpu]` (or a CUDA build of torch from "
            "https://pytorch.org), or call with device='cpu'."
        ) from e
    return torch


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_device(device: Union[str, "object", None] = "cpu") -> str:
    """Normalise a ``device=`` argument to ``"cpu"`` or a torch device string.

    Parameters
    ----------
    device
        ``"cpu"`` (or ``None``) selects the NumPy / SciPy implementation;
        ``"cuda"`` / ``"cuda:N"`` / a :class:`torch.device` selects the torch
        implementation; ``"auto"`` picks ``"cuda"`` when available.

    Returns
    -------
    str
        ``"cpu"`` or a canonical torch device string such as ``"cuda:0"``.

    Raises
    ------
    ImportError
        A GPU device was requested but torch is not installed.
    RuntimeError
        A CUDA device was requested but none is visible.
    ValueError
        Unrecognised device string.
    """
    if device is None:
        return _CPU
    name = str(device).strip().lower()
    if name == _CPU:
        return _CPU
    if name == "auto":
        if not _torch_available():
            return _CPU
        import torch
        return "cuda:0" if torch.cuda.is_available() else _CPU
    if name.startswith("cuda"):
        torch = get_torch()
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"device={device!r} requested but torch sees no CUDA device "
                f"(torch {torch.__version__}, built for CUDA "
                f"{torch.version.cuda}). Use device='cpu' or device='auto'."
            )
        dev = torch.device(name)
        idx = 0 if dev.index is None else dev.index
        if idx >= torch.cuda.device_count():
            raise RuntimeError(
                f"device={device!r}: only {torch.cuda.device_count()} CUDA "
                "device(s) visible."
            )
        return f"cuda:{idx}"
    if name == "torch-cpu":
        # Internal / testing hook: run the torch implementation on the CPU.
        # Lets the torch code paths be exercised on machines without a GPU.
        get_torch()
        return "torch-cpu"
    raise ValueError(
        f"unknown device {device!r}; expected 'cpu', 'cuda', 'cuda:N' or 'auto'."
    )


def is_torch_device(resolved: str) -> bool:
    """True when a :func:`resolve_device` result selects the torch path."""
    return resolved != _CPU


def torch_device(resolved: str):
    """:class:`torch.device` for a resolved, non-CPU device string."""
    torch = get_torch()
    if resolved == "torch-cpu":
        return torch.device("cpu")
    return torch.device(resolved)


def torch_dtype(np_dtype):
    """torch floating dtype matching a NumPy floating dtype."""
    torch = get_torch()
    np_dtype = np.dtype(np_dtype)
    if np_dtype == np.float64:
        return torch.float64
    if np_dtype == np.float32:
        return torch.float32
    raise TypeError(f"unsupported dtype {np_dtype}")


def empty_cache(dev) -> None:
    """Release cached CUDA blocks (no-op elsewhere)."""
    torch = get_torch()
    if getattr(dev, "type", None) == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Memory budget + batching plans (pure NumPy)
# ---------------------------------------------------------------------------

def memory_budget(dev=None, fraction: float = 0.5) -> int:
    """Device bytes a single accelerated call may allocate.

    ``EPIONE_GPU_MEMORY_GB`` (float, GiB) wins when set. Otherwise the budget
    is ``fraction`` of the memory currently free on ``dev``, so a call on a
    GPU shared with other jobs stays within what is left. On a CPU torch
    device the budget is 4 GiB.
    """
    env = os.environ.get(_ENV_BUDGET)
    if env:
        return int(float(env) * 2**30)
    if dev is not None and getattr(dev, "type", None) == "cuda":
        torch = get_torch()
        free, _total = torch.cuda.mem_get_info(dev)
        return int(free * fraction)
    return 4 * 2**30


def row_blocks(n_rows: int, bytes_per_row: float, max_bytes: float,
               min_rows: int = 1) -> List[slice]:
    """Split ``range(n_rows)`` into contiguous slices of bounded size.

    Each slice holds at most ``max(min_rows, max_bytes // bytes_per_row)``
    rows. ``n_rows == 0`` returns an empty list.
    """
    n_rows = int(n_rows)
    if n_rows <= 0:
        return []
    per = int(max_bytes // max(float(bytes_per_row), 1.0))
    per = max(int(min_rows), per, 1)
    return [slice(s, min(s + per, n_rows)) for s in range(0, n_rows, per)]


def csr_row_blocks(indptr: np.ndarray, max_nnz: int,
                   max_rows: Optional[int] = None) -> List[slice]:
    """Split CSR rows into contiguous slices with at most ``max_nnz``
    non-zeros (a single row that is larger than ``max_nnz`` gets its own
    slice) and, optionally, at most ``max_rows`` rows."""
    indptr = np.asarray(indptr, dtype=np.int64)
    n_rows = indptr.shape[0] - 1
    if n_rows <= 0:
        return []
    max_nnz = max(int(max_nnz), 1)
    out = []
    start = 0
    while start < n_rows:
        # last row index `end` such that indptr[end] - indptr[start] <= max_nnz
        end = int(np.searchsorted(indptr, indptr[start] + max_nnz, side="right")) - 1
        end = max(end, start + 1)
        if max_rows is not None:
            end = min(end, start + int(max_rows))
        end = min(end, n_rows)
        out.append(slice(start, end))
        start = end
    return out


# ---------------------------------------------------------------------------
# SciPy <-> torch conversion
# ---------------------------------------------------------------------------

def _index_dtype(n: int):
    torch = get_torch()
    return torch.int32 if n < 2**31 - 1 else torch.int64


def to_torch_csr(X, dev, dtype=None):
    """SciPy sparse (or dense) matrix -> torch sparse CSR tensor on ``dev``.

    Indices are stored as int32 when they fit (cuSPARSE accepts both), which
    halves index memory. ``dtype`` defaults to the matrix's float dtype
    (integers become float32).
    """
    torch = get_torch()
    if not isinstance(X, _CSRView):
        X = sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()
    if dtype is None:
        dtype = torch_dtype(X.dtype) if X.dtype.kind == "f" else torch.float32
    idx_dt = _index_dtype(max(X.nnz, max(X.shape)))
    crow = torch.from_numpy(np.ascontiguousarray(X.indptr)).to(dev, idx_dt)
    col = torch.from_numpy(np.ascontiguousarray(X.indices)).to(dev, idx_dt)
    val = torch.from_numpy(np.ascontiguousarray(X.data)).to(dev, dtype)
    return _csr_tensor(crow, col, val, X.shape, dev)


def _csr_tensor(crow, col, val, shape, dev):
    import warnings
    torch = get_torch()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"Sparse CSR tensor support is in beta")
        return torch.sparse_csr_tensor(crow, col, val, size=tuple(shape), device=dev)


def csr_from_torch(T) -> sp.csr_matrix:
    """torch sparse CSR tensor -> SciPy CSR matrix on the host."""
    crow = T.crow_indices().cpu().numpy()
    col = T.col_indices().cpu().numpy()
    val = T.values().cpu().numpy()
    return sp.csr_matrix((val, col, crow), shape=tuple(T.shape))


def to_numpy(t) -> np.ndarray:
    """Detach a tensor to a host NumPy array."""
    return t.detach().cpu().numpy()


def csr_row_ids(T):
    """Row index of every stored value of a torch CSR tensor."""
    torch = get_torch()
    crow = T.crow_indices()
    counts = (crow[1:] - crow[:-1]).to(torch.int64)
    return torch.repeat_interleave(
        torch.arange(T.shape[0], device=crow.device), counts)


# ---------------------------------------------------------------------------
# Sparse operand: resident or streamed
# ---------------------------------------------------------------------------

class SparseOperand:
    """A CSR matrix ``X`` used in repeated ``X @ D`` / ``X.T @ D`` products.

    ``X`` is uploaded once when it fits in ``budget`` device bytes, and its
    transpose is built on the device (CSR -> CSC is a device-side sort) when
    that fits too. Otherwise ``X`` stays on the host and is streamed to the
    device in row blocks for every product (``X.T @ D`` is then accumulated
    block by block as ``X_b.T @ D_b``), so device memory stays bounded by
    ``budget`` regardless of matrix size. The host never builds ``X.T``.

    Parameters
    ----------
    X
        SciPy sparse matrix (any format; converted to CSR).
    dev
        torch device.
    dtype
        torch dtype of the values.
    budget
        Device bytes the operand itself may occupy.
    value_fn
        Optional ``f(values, row, col) -> values`` applied on the device to
        the stored values when they are uploaded, where ``row`` / ``col`` are
        global int64 indices into ``X``. Lets an element-wise normalisation
        such as TF-IDF run on the device without materialising the
        normalised matrix on the host.
    """

    def __init__(self, X, dev, dtype, budget: Optional[int] = None,
                 value_fn=None):
        torch = get_torch()
        self.torch = torch
        self.dev = dev
        self.dtype = dtype
        self.value_fn = value_fn
        X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
        self.shape = X.shape
        self._X = X
        itemsize = torch.tensor([], dtype=dtype).element_size()
        idx = 4 if max(X.nnz, max(X.shape)) < 2**31 - 1 else 8
        self._bytes_one = X.nnz * (itemsize + idx) + (X.shape[0] + X.shape[1] + 2) * idx
        if budget is None:
            budget = memory_budget(dev)
        self.budget = int(budget)
        per_nnz = itemsize + idx + 16          # values + index + int64 row/col ids
        self._max_nnz = max(int(self.budget // (4 * per_nnz)), 1)
        self._Xd = self._XTd = None
        # resident X needs its own bytes plus transient row ids / the
        # transposition scratch (about one more copy)
        if 2 * self._bytes_one + 8 * X.nnz <= self.budget:
            self._Xd = self._upload(X, 0)
            if 3 * self._bytes_one <= self.budget:
                self._XTd = _transpose_csr(self._Xd)
        self.resident = self._Xd is not None and self._XTd is not None

    def _upload(self, Xh, row_offset: int):
        torch = self.torch
        T = to_torch_csr(Xh, self.dev, self.dtype if self.value_fn is None else None)
        if self.value_fn is None:
            if T.dtype != self.dtype:
                T = T.to(self.dtype)
            return T
        rows = csr_row_ids(T) + int(row_offset)
        cols = T.col_indices().to(torch.int64)
        vals = self.value_fn(T.values(), rows, cols).to(self.dtype)
        del rows, cols
        return _csr_tensor(T.crow_indices(), T.col_indices(), vals, T.shape, self.dev)

    @property
    def T(self) -> "_Transposed":
        return _Transposed(self)

    def _blocks(self):
        for blk in csr_row_blocks(self._X.indptr, self._max_nnz):
            yield blk, self._upload(_csr_rows_view(self._X, blk), blk.start)

    def matmul(self, D):
        """``X @ D`` for a dense device tensor ``D`` (n_cols x k)."""
        if self._Xd is not None:
            return self._Xd @ D
        out = self.torch.empty((self.shape[0], D.shape[1]), dtype=D.dtype, device=self.dev)
        for blk, sub in self._blocks():
            out[blk] = sub @ D
            del sub
        return out

    def rmatmul(self, D):
        """``X.T @ D`` for a dense device tensor ``D`` (n_rows x k)."""
        if self._XTd is not None:
            return self._XTd @ D
        out = self.torch.zeros((self.shape[1], D.shape[1]), dtype=D.dtype, device=self.dev)
        if self._Xd is not None:
            # X resident, its transpose too large: transpose row blocks of X
            for blk in csr_row_blocks(self._X.indptr, self._max_nnz):
                sub = _csr_slice_device(self._Xd, blk)
                out += _transpose_csr(sub) @ D[blk]
                del sub
            return out
        for blk, sub in self._blocks():
            out += _transpose_csr(sub) @ D[blk]
            del sub
        return out

    def __matmul__(self, D):
        return self.matmul(D)


def _transpose_csr(T):
    """CSR tensor of ``T.T``, built on ``T``'s device."""
    C = T.to_sparse_csc()
    return _csr_tensor(C.ccol_indices(), C.row_indices(), C.values(),
                       (T.shape[1], T.shape[0]), T.device)


def _csr_slice_device(T, blk: slice):
    """Rows ``blk`` of a device CSR tensor (shares values / indices)."""
    crow = T.crow_indices()
    s0, s1 = crow[blk.start], crow[blk.stop]
    return _csr_tensor(crow[blk.start:blk.stop + 1] - s0, T.col_indices()[s0:s1],
                       T.values()[s0:s1], (blk.stop - blk.start, T.shape[1]), T.device)


class _CSRView:
    """Rows of a SciPy CSR matrix as zero-copy array views, accepted by
    :func:`to_torch_csr` (``X[blk]`` would copy ``data`` and ``indices``)."""

    def __init__(self, X: sp.csr_matrix, blk: slice):
        s0, s1 = int(X.indptr[blk.start]), int(X.indptr[blk.stop])
        self.indptr = X.indptr[blk.start:blk.stop + 1] - s0
        self.indices = X.indices[s0:s1]
        self.data = X.data[s0:s1]
        self.shape = (blk.stop - blk.start, X.shape[1])
        self.nnz = s1 - s0
        self.dtype = X.dtype


def _csr_rows_view(X: sp.csr_matrix, blk: slice) -> "_CSRView":
    return _CSRView(X, blk)


class _Transposed:
    def __init__(self, op: SparseOperand):
        self.op = op
        self.shape = (op.shape[1], op.shape[0])

    @property
    def T(self):
        return self.op

    def matmul(self, D):
        return self.op.rmatmul(D)

    def rmatmul(self, D):
        return self.op.matmul(D)

    def __matmul__(self, D):
        return self.matmul(D)


def _np_dtype(tdtype):
    torch = get_torch()
    return {torch.float32: np.float32, torch.float64: np.float64}[tdtype]
