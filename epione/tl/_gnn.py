"""Plain-PyTorch graph layers shared by the spatial-method ports.

The spatial ports (COSMOS, PRESENT, CANDIES, ...) were written upstream on
PyTorch Geometric (PyG). Epione does not depend on PyG, so the few pieces they
use are re-implemented here with the *same* arithmetic and the same
random-number consumption as PyG 2.6:

* :func:`scatter_sum` / :func:`scatter_max` -- ``Tensor.scatter_add_`` and
  ``Tensor.scatter_reduce_(..., "amax", include_self=False)``, which is what
  ``torch_geometric.utils.scatter`` calls for these reductions;
* :func:`segment_softmax` -- ``torch_geometric.utils.softmax`` (index path);
* :func:`glorot_` / :func:`uniform_` -- ``torch_geometric.nn.inits``;
* :class:`GCNConv` -- ``torch_geometric.nn.GCNConv`` with the default
  ``add_self_loops=True, normalize=True`` (``gcn_norm`` with
  ``add_remaining_self_loops``), flow ``source_to_target``;
* :class:`GATConv` -- ``torch_geometric.nn.GATConv`` (no edge features),
  ``remove_self_loops`` + ``add_self_loops``, LeakyReLU 0.2, attention dropout.

Edges are given as a ``(2, E)`` LongTensor ``edge_index`` whose first row is
the source and second row the target node (PyG convention); messages flow
source -> target and are summed at the target.

PyG is MIT-licensed (c) 2023 PyG Team <team@pyg.org>; the code below is a
re-implementation of the parts listed above, see ``NOTICE``.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "scatter_sum", "scatter_max", "segment_softmax", "glorot_", "uniform_",
    "gcn_norm", "gcn_norm_sparse", "GCNConv", "GATConv", "reset_module",
]


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Sum ``src`` rows into ``dim_size`` buckets given by ``index`` (dim 0)."""
    size = (dim_size,) + tuple(src.shape[1:])
    idx = index.view((-1,) + (1,) * (src.dim() - 1)).expand_as(src)
    return src.new_zeros(size).scatter_add_(0, idx, src)


def scatter_max(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Row-wise max into buckets (empty buckets are 0), as PyG on CPU / no grad."""
    size = (dim_size,) + tuple(src.shape[1:])
    idx = index.view((-1,) + (1,) * (src.dim() - 1)).expand_as(src)
    return src.new_zeros(size).scatter_reduce_(0, idx, src, reduce="amax", include_self=False)


def segment_softmax(src: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Softmax of edge scores grouped by target node (``torch_geometric.utils.softmax``)."""
    src_max = scatter_max(src.detach(), index, num_nodes)
    out = src - src_max.index_select(0, index)
    out = out.exp()
    out_sum = scatter_sum(out, index, num_nodes) + 1e-16
    out_sum = out_sum.index_select(0, index)
    return out / out_sum


def glorot_(t: Optional[torch.Tensor]) -> None:
    """``torch_geometric.nn.inits.glorot`` (uniform, a = sqrt(6 / (fan_in + fan_out)))."""
    if t is not None:
        a = math.sqrt(6.0 / (t.size(-2) + t.size(-1)))
        t.data.uniform_(-a, a)


def uniform_(size: int, t: Optional[torch.Tensor]) -> None:
    """``torch_geometric.nn.inits.uniform``: U(-1/sqrt(size), 1/sqrt(size))."""
    if t is not None:
        bound = 1.0 / math.sqrt(size)
        t.data.uniform_(-bound, bound)


def reset_module(value) -> None:
    """``torch_geometric.nn.inits.reset``: call ``reset_parameters`` recursively."""
    if hasattr(value, "reset_parameters"):
        value.reset_parameters()
    else:
        for child in value.children() if hasattr(value, "children") else []:
            reset_module(child)


def _add_remaining_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    mask = edge_index[0] != edge_index[1]
    loop = torch.arange(0, num_nodes, device=edge_index.device).view(1, -1).repeat(2, 1)
    return torch.cat([edge_index[:, mask], loop], dim=1)


def gcn_norm(edge_index: torch.Tensor, num_nodes: int, dtype=torch.float32):
    """Symmetric GCN normalisation with self loops (PyG ``gcn_norm``)."""
    edge_index = _add_remaining_self_loops(edge_index, num_nodes)
    edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_sum(edge_weight, col, num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    return edge_index, edge_weight


def gcn_norm_sparse(adj: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """PyG ``gcn_norm`` for a torch sparse COO ``adj_t``: add self loops (fill 1), D^-1/2 A D^-1/2.

    Self loops are appended to the existing entries and the tensor is coalesced,
    so an existing diagonal value v becomes v + 1 (``torch_geometric.utils.add_self_loops``).
    """
    adj = adj.coalesce()
    index, value = adj.indices(), adj.values()
    loop = torch.arange(0, num_nodes, device=index.device).view(1, -1).repeat(2, 1)
    index = torch.cat([index, loop], dim=1)
    value = torch.cat([value, torch.full((num_nodes,), 1.0, dtype=value.dtype, device=value.device)], dim=0)
    adj = torch.sparse_coo_tensor(index, value, (num_nodes, num_nodes)).coalesce()
    index, value = adj.indices(), adj.values()
    col, row = index[0], index[1]
    deg = scatter_sum(value, col, num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    value = deg_inv_sqrt[row] * value * deg_inv_sqrt[col]
    return torch.sparse_coo_tensor(index, value, (num_nodes, num_nodes)).coalesce()


class _Linear(nn.Module):
    """``torch_geometric.nn.dense.linear.Linear`` with glorot weights, optional zero bias."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        glorot_(self.weight)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class GCNConv(nn.Module):
    """Graph convolution of Kipf & Welling (PyG ``GCNConv`` defaults, ``cached=False``)."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True, **_):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.lin = _Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """``edge_index``: ``(2, E)`` LongTensor, or a torch sparse COO adjacency (``adj_t``)."""
        n = x.size(0)
        if edge_index.is_sparse:
            adj = gcn_norm_sparse(edge_index, n)
            x = self.lin(x)
            out = torch.sparse.mm(adj.to_sparse_csr(), x)
        else:
            ei, ew = gcn_norm(edge_index, n, x.dtype)
            x = self.lin(x)
            out = scatter_sum(ew.view(-1, 1) * x.index_select(0, ei[0]), ei[1], n)
        if self.bias is not None:
            out = out + self.bias
        return out


class GATConv(nn.Module):
    """Graph attention (Velickovic et al.) with PyG ``GATConv`` semantics.

    ``heads`` attention heads, ``concat`` or average them; self loops are
    removed and re-added; LeakyReLU(``negative_slope``) on the summed source /
    target scores, softmax over each target's incoming edges, dropout on the
    attention coefficients during training, optional bias.
    """

    def __init__(self, in_channels: int, out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0, add_self_loops: bool = True,
                 bias: bool = True, **_):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.heads, self.concat = heads, concat
        self.negative_slope, self.dropout = negative_slope, dropout
        self.add_self_loops = add_self_loops
        self.lin_src = _Linear(in_channels, heads * out_channels, bias=False)
        self.lin_dst = self.lin_src
        self.att_src = nn.Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_channels))
        if bias and concat:
            self.bias = nn.Parameter(torch.empty(heads * out_channels))
        elif bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        glorot_(self.att_src)
        glorot_(self.att_dst)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, return_attention_weights=None):
        H, C = self.heads, self.out_channels
        n = x.size(0)
        xs = self.lin_src(x).view(-1, H, C)
        a_src = (xs * self.att_src).sum(dim=-1)
        a_dst = (xs * self.att_dst).sum(-1)
        if self.add_self_loops:
            mask = edge_index[0] != edge_index[1]
            loop = torch.arange(0, n, device=edge_index.device).view(1, -1).repeat(2, 1)
            edge_index = torch.cat([edge_index[:, mask], loop], dim=1)
        src, dst = edge_index[0], edge_index[1]
        alpha = a_src.index_select(0, src) + a_dst.index_select(0, dst)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = segment_softmax(alpha, dst, n)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        msg = alpha.unsqueeze(-1) * xs.index_select(0, src)
        out = scatter_sum(msg, dst, n)
        out = out.view(-1, H * C) if self.concat else out.mean(dim=1)
        if self.bias is not None:
            out = out + self.bias
        if isinstance(return_attention_weights, bool):
            return out, (edge_index, alpha)
        return out
