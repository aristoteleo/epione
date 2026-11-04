"""Building blocks shared by the torch translation methods.

Per-chromosome layers fused into batched matmuls, sparse-to-dense batching
with prefetch, and early stopping on a validation loss.
"""
from __future__ import annotations

import copy

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn


class ChromLinear(nn.Module):
    """One Linear layer per chromosome block, run as a single batched matmul.

    ``mode="in"`` maps each chromosome's peaks to ``per_chrom`` units (a first
    ATAC encoder layer); ``mode="out"`` maps each chromosome's ``per_chrom``
    units back to its peaks (a last ATAC decoder layer). Each block is
    initialised like its own ``nn.Linear`` (``init="xavier"`` additionally draws
    Xavier-uniform weights); blocks are zero-padded to the largest chromosome
    and the padding never receives a gradient, so this is the same function as
    a list of per-chromosome Linear layers.
    """

    def __init__(self, chroms, per_chrom, mode, init="xavier"):
        super().__init__()
        self.chroms = list(chroms)
        self.mode = mode
        n, cmax = len(self.chroms), max(self.chroms)
        din, dout = (cmax, per_chrom) if mode == "in" else (per_chrom, cmax)
        W = torch.zeros(n, din, dout)
        b = torch.zeros(n, dout)
        for j, c in enumerate(self.chroms):
            fi, fo = (c, per_chrom) if mode == "in" else (per_chrom, c)
            lin = nn.Linear(fi, fo)
            if init == "xavier":
                nn.init.xavier_uniform_(lin.weight)
            if mode == "in":
                W[j, :c, :] = lin.weight.data.T
                b[j] = lin.bias.data
            else:
                W[j, :, :c] = lin.weight.data.T
                b[j, :c] = lin.bias.data
        self.weight = nn.Parameter(W)
        self.bias = nn.Parameter(b)
        starts = np.concatenate([[0], np.cumsum(self.chroms)[:-1]])
        # flat peak index -> (block, offset) gather/scatter map
        blk = np.repeat(np.arange(n), self.chroms)
        off = np.arange(sum(self.chroms)) - np.repeat(starts, self.chroms)
        self.register_buffer("flat_index", torch.from_numpy(blk * cmax + off), persistent=False)
        self.cmax, self.per_chrom = cmax, per_chrom

    def forward(self, x):
        B, n = x.shape[0], len(self.chroms)
        if self.mode == "in":
            xp = x.new_zeros(B, n * self.cmax).index_copy(1, self.flat_index, x)
            y = torch.bmm(xp.view(B, n, self.cmax).transpose(0, 1), self.weight) + self.bias[:, None, :]
            return y.transpose(0, 1).reshape(B, n * self.per_chrom)
        xb = x.view(B, n, self.per_chrom).transpose(0, 1)
        y = torch.bmm(xb, self.weight) + self.bias[:, None, :]
        # index_select (backward: index_add_) rather than advanced indexing, whose
        # backward is a slow sort-based scatter; the indices are unique
        return torch.index_select(y.transpose(0, 1).reshape(B, n * self.cmax), 1, self.flat_index)


class BlockLinear(nn.Module):
    """``n`` independent Linear layers of equal size on consecutive input blocks.

    Input ``(B, n * din)`` -> output ``(B, n * dout)``; block ``j`` of the output
    depends only on block ``j`` of the input. Each block is initialised like
    its own ``nn.Linear(din, dout)``.
    """

    def __init__(self, n, din, dout):
        super().__init__()
        W = torch.empty(n, din, dout)
        b = torch.empty(n, dout)
        for j in range(n):
            lin = nn.Linear(din, dout)
            W[j] = lin.weight.data.T
            b[j] = lin.bias.data
        self.weight = nn.Parameter(W)
        self.bias = nn.Parameter(b)
        self.n, self.din, self.dout = n, din, dout

    def forward(self, x):
        B = x.shape[0]
        y = torch.bmm(x.view(B, self.n, self.din).transpose(0, 1), self.weight) + self.bias[:, None, :]
        return y.transpose(0, 1).reshape(B, self.n * self.dout)


class _Batches:
    """Row batches of several matrices (CSR or dense numpy) as float32 tensors on ``device``.

    The next batch is prepared in a background thread. On CUDA, large sparse
    batches are sent as CSR and densified on the GPU (less to copy); small ones
    are densified on the CPU (fewer kernel launches). Both give identical tensors.
    """

    def __init__(self, *mats, device, sparse_transfer_min_rows: int = 128):
        from concurrent.futures import ThreadPoolExecutor

        self.mats, self.device = mats, device
        self.min_rows = sparse_transfer_min_rows if device.type == "cuda" else None
        self.pool = ThreadPoolExecutor(max_workers=1)

    def _one(self, X, idx):
        if not sp.issparse(X):
            return torch.from_numpy(np.ascontiguousarray(X[idx], dtype=np.float32))
        m = X[idx]
        if self.min_rows is not None and len(idx) >= self.min_rows:
            import warnings

            with warnings.catch_warnings():  # "sparse CSR support is in beta"
                warnings.simplefilter("ignore", UserWarning)
                return torch.sparse_csr_tensor(torch.from_numpy(m.indptr.astype(np.int64)),
                                               torch.from_numpy(m.indices.astype(np.int64)),
                                               torch.from_numpy(m.data), size=m.shape)
        return torch.from_numpy(m.toarray())

    def _load(self, idx):
        return tuple(self._one(X, idx) for X in self.mats)

    def _to_device(self, t):
        t = t.to(self.device)
        return t.to_dense() if t.layout == torch.sparse_csr else t

    def get(self, idx):
        return tuple(self._to_device(t) for t in self._load(idx))

    def iterate(self, batches):
        """Yield a tuple of device tensors for each index batch, prefetching the next."""
        if not batches:
            return
        fut = self.pool.submit(self._load, batches[0])
        for k in range(len(batches)):
            ts = fut.result()
            if k + 1 < len(batches):
                fut = self.pool.submit(self._load, batches[k + 1])
            yield tuple(self._to_device(t) for t in ts)


def _epoch_batches(idx: np.ndarray, batch_size: int, gen: torch.Generator, shuffle: bool):
    """DataLoader-like batches; drop the last batch only if it would hold one cell."""
    n = len(idx)
    order = idx[torch.randperm(n, generator=gen).numpy()] if shuffle else idx
    drop_last = shuffle and n % batch_size == 1
    stop = n - (n % batch_size) if drop_last else n
    return [order[i:i + batch_size] for i in range(0, stop, batch_size)]


class _EarlyStop:
    """Stop after ``patience`` epochs without a lower validation loss; keep the best weights.

    Ties count as an improvement, as in scButterfly's ``EarlyStopping``.
    """

    def __init__(self, patience):
        self.patience, self.best, self.counter, self.stop, self.state = patience, None, 0, False, None

    def __call__(self, val_loss, net):
        score = -val_loss
        if self.best is None or score >= self.best:
            # the reference counts ``score < best`` as no improvement, so ties reset
            self.best, self.counter = score, 0
            self.state = copy.deepcopy(net.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


def _mean(xs) -> float:
    """Mean of per-batch loss tensors, synchronising once."""
    return float(torch.stack([x.float() for x in xs]).mean().item()) if xs else float("nan")


def _load_parts(net, state, modules):
    """Restore the given submodules of ``net`` from a full ``state_dict`` snapshot."""
    names = {id(m): n for n, m in net.named_children()}
    for m in modules:
        prefix = names[id(m)] + "."
        m.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)})


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
