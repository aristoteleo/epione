"""Higashi neural-network modules (hypergraph Hyper-SAGNN + cell-dependent GNN).

Adapted from Higashi (Zhang, Zhou & Ma, *Nat. Biotechnol.* 2022),
https://github.com/ma-compbio/Higashi @ 42702a8, ``higashi/Higashi_backend/
Modules.py`` and ``Functions.py``. MIT License, Copyright (c) 2020 Ma Lab at
CMU; the full notice is in ``LICENSE_HIGASHI`` next to this file.

Changes relative to upstream, none of which alter the computation:

* the module-global ``device`` is replaced by an explicit ``device`` argument
  (or the device of the input tensor);
* classes and code paths Higashi never reaches in its default configuration
  (``SparseEmbedding`` sparse/CPU modes, ``MeanAggregator`` with
  ``transfer_range > 0`` via list-of-sets input, ``forward_GCN`` moving
  averages) are kept only to the extent the default path needs them;
* attribute names are kept identical so a reference ``state_dict`` loads into
  these modules unchanged (used by the parity gate G1b).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def swish(x):
    return x * torch.sigmoid(x)


activation_func = swish


# Code adapted by upstream from scVI.
def log_zinb_positive(x, mu, theta, pi, eps=1e-8):
    """Log-likelihood of a zero-inflated negative binomial (logit ``pi``)."""
    softplus_pi = F.softplus(-pi)
    log_theta_eps = torch.log(theta + eps)
    log_theta_mu_eps = torch.log(theta + mu + eps)
    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

    case_zero = F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

    case_non_zero = (
        -softplus_pi
        + pi_theta_log
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )
    mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)
    return mul_case_zero + mul_case_non_zero


class SparseEmbedding(nn.Module):
    """Fixed (non-trainable) feature lookup table.

    Upstream's dense path: ``embedding[x, :]``. The table is a plain tensor
    attribute (not a parameter or buffer), as upstream.
    """

    def __init__(self, embedding_weight, device="cpu"):
        super().__init__()
        if isinstance(embedding_weight, np.ndarray):
            embedding_weight = torch.from_numpy(np.asarray(embedding_weight))
        self.embedding = embedding_weight.to(device)

    def forward(self, x):
        return self.embedding[x.to(self.embedding.device), :]


class TiedAutoEncoder(nn.Module):
    """Auto-encoder with (optionally) tied weights; upstream ``TiedAutoEncoder``."""

    def __init__(self, shape_list, use_bias=True, tied_list=None, add_activation=False,
                 dropout=None, layer_norm=False, activation=None, device="cpu"):
        super().__init__()
        if tied_list is None:
            tied_list = []
        self.add_activation = add_activation
        self.weight_list = []
        self.reverse_weight_list = []
        self.bias_list = []
        self.use_bias = use_bias
        self.recon_bias_list = []
        self.shape_list = shape_list
        self.activation = activation if activation is not None else activation_func
        self._device = device
        for i in range(len(shape_list) - 1):
            p = nn.parameter.Parameter(torch.empty(shape_list[i + 1], shape_list[i], device=device))
            self.weight_list.append(p)
            if i not in tied_list:
                self.reverse_weight_list.append(
                    nn.parameter.Parameter(torch.empty(shape_list[i + 1], shape_list[i], device=device)))
            else:
                self.reverse_weight_list.append(p)
            self.bias_list.append(nn.parameter.Parameter(torch.empty(shape_list[i + 1], device=device)))
            self.recon_bias_list.append(nn.parameter.Parameter(torch.empty(shape_list[i], device=device)))

        self.recon_bias_list = self.recon_bias_list[::-1]
        self.reverse_weight_list = self.reverse_weight_list[::-1]
        self.weight_list = nn.ParameterList(self.weight_list)
        self.reverse_weight_list = nn.ParameterList(self.reverse_weight_list)
        self.bias_list = nn.ParameterList(self.bias_list)
        self.recon_bias_list = nn.ParameterList(self.recon_bias_list)
        self.reset_parameters()
        self.dropout = nn.Dropout(dropout) if dropout is not None else None
        self.layer_norm = nn.LayerNorm(shape_list[-1]) if layer_norm else None
        self.tied_list = tied_list
        self.input_dropout = nn.Dropout(0.1)

    def reset_parameters(self):
        for i, _ in enumerate(self.weight_list):
            nn.init.kaiming_uniform_(self.weight_list[i], a=0.0, mode='fan_in', nonlinearity='leaky_relu')
            nn.init.kaiming_uniform_(self.reverse_weight_list[i], a=0.0, mode='fan_out',
                                     nonlinearity='leaky_relu')
        for i, _ in enumerate(self.bias_list):
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_list[i])
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_list[i], -bound, bound)
        temp_weight_list = self.weight_list[::-1]
        for i, _ in enumerate(self.recon_bias_list):
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(temp_weight_list[i])
            bound = 1 / math.sqrt(fan_out)
            nn.init.uniform_(self.recon_bias_list[i], -bound, bound)

    def encoder(self, input):
        encoded_feats = input
        for i in range(len(self.weight_list)):
            if self.use_bias:
                encoded_feats = F.linear(encoded_feats, self.weight_list[i], self.bias_list[i])
            else:
                encoded_feats = F.linear(encoded_feats, self.weight_list[i])
            if i < len(self.weight_list) - 1:
                encoded_feats = self.activation(encoded_feats)
                if self.dropout is not None:
                    encoded_feats = self.dropout(encoded_feats)
        if self.layer_norm is not None:
            encoded_feats = self.layer_norm(encoded_feats)
        if self.add_activation:
            encoded_feats = self.activation(encoded_feats)
        return encoded_feats

    def decoder(self, encoded_feats):
        if self.add_activation:
            reconstructed_output = encoded_feats
        else:
            reconstructed_output = self.activation(encoded_feats)
        for i in range(len(self.reverse_weight_list)):
            reconstructed_output = F.linear(reconstructed_output, self.reverse_weight_list[i].t(),
                                            self.recon_bias_list[i])
            if i < len(self.recon_bias_list) - 1:
                reconstructed_output = self.activation(reconstructed_output)
        return reconstructed_output

    def forward(self, input, return_recon=False):
        encoded_feats = self.encoder(input)
        if return_recon:
            # Upstream applies the activation here *and* again at the top of
            # ``decoder`` when ``add_activation`` is False; kept as is.
            if not self.add_activation:
                reconstructed_output = self.activation(encoded_feats)
            else:
                reconstructed_output = encoded_feats
            if self.dropout is not None:
                reconstructed_output = self.dropout(reconstructed_output)
            reconstructed_output = self.decoder(reconstructed_output)
            return encoded_feats, reconstructed_output
        return encoded_feats

    def fit(self, data, epochs=10, batch_size=-1, targets=None, early_stop=True,
            generator=None, random_state=None, verbose=False):
        """Pre-train with MSE (upstream ``fit(..., sparse=False)``).

        If the code dimension is below the input dimension the first encoder
        weight and the last decoder weight are initialised with PCA loadings,
        as upstream.
        """
        from sklearn.decomposition import PCA
        device = self.weight_list[0].device
        if self.shape_list[1] < data.shape[1]:
            pca = PCA(n_components=self.shape_list[1], random_state=random_state).fit(data)
            comp = torch.from_numpy(pca.components_).float().to(device)
            self.weight_list[0].data = comp.clone()
            self.reverse_weight_list[-1].data = comp.clone()
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        data = torch.from_numpy(data).to(device)
        if batch_size < 0:
            batch_size = int(len(data))
        no_improve_count = 0
        loss_best = None
        loss = None
        i = 0
        for i in range(epochs):
            batch_index = torch.randint(0, int(len(data)), (batch_size,), generator=generator).to(device)
            encode, recon = self.forward(data[batch_index], return_recon=True)
            optimizer.zero_grad()
            loss = F.mse_loss(recon, targets[batch_index])
            if i == 0:
                loss_best = float(loss.item())
            loss.backward()
            optimizer.step()
            if early_stop and i >= 50:
                if loss.item() < loss_best * 0.99:
                    loss_best = loss.item()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                if no_improve_count >= 30:
                    break
        if verbose and loss is not None:
            print("cell-feature AE pre-train: loss %.4f best %.4f epochs %d" % (loss.item(), loss_best, i))


class AutoEncoder(nn.Module):
    """Untied auto-encoder; upstream uses it only when the cell targets differ
    in width from the cell inputs (co-assay mode)."""

    def __init__(self, encoder_shape_list, decoder_shape_list, use_bias=True, add_activation=False,
                 dropout=None, layer_norm=False, device="cpu"):
        super().__init__()
        self.add_activation = add_activation
        self.use_bias = use_bias
        self.weight_list = nn.ModuleList([nn.Linear(encoder_shape_list[i], encoder_shape_list[i + 1]).to(device)
                                          for i in range(len(encoder_shape_list) - 1)])
        self.reverse_weight_list = nn.ModuleList([nn.Linear(decoder_shape_list[i], decoder_shape_list[i + 1]).to(device)
                                                  for i in range(len(decoder_shape_list) - 1)])
        self.dropout = nn.Dropout(dropout) if dropout is not None else None
        self.layer_norm_stack = None
        if layer_norm:
            self.layer_norm_stack = [nn.LayerNorm(encoder_shape_list[i + 1]).to(device)
                                     for i in range(len(encoder_shape_list) - 1)]

    def encoder(self, input):
        encoded_feats = input
        for i in range(len(self.weight_list)):
            encoded_feats = self.weight_list[i](encoded_feats)
            if i < len(self.weight_list) - 1:
                encoded_feats = activation_func(encoded_feats)
                if self.dropout is not None:
                    encoded_feats = self.dropout(encoded_feats)
            if self.layer_norm_stack is not None:
                encoded_feats = self.layer_norm_stack[i](encoded_feats)
        if self.add_activation:
            encoded_feats = activation_func(encoded_feats)
        return encoded_feats

    def decoder(self, encoded_feats):
        reconstructed_output = encoded_feats if self.add_activation else activation_func(encoded_feats)
        for i in range(len(self.reverse_weight_list)):
            reconstructed_output = self.reverse_weight_list[i](reconstructed_output)
            if i < len(self.reverse_weight_list) - 1:
                reconstructed_output = activation_func(reconstructed_output)
        return reconstructed_output

    def forward(self, input, return_recon=False):
        encoded_feats = self.encoder(input)
        if return_recon:
            reconstructed_output = encoded_feats
            if self.dropout is not None:
                reconstructed_output = self.dropout(reconstructed_output)
            return encoded_feats, self.decoder(reconstructed_output)
        return encoded_feats


class MultipleEmbedding(nn.Module):
    """Routes each node id to the auto-encoder of its node type.

    Node ids are 1-based; ``num_list = cumsum([n_cells, n_bins_chr1, ...])``.
    Ids ``1..n_cells`` use ``wstack[0]`` (cells), the next chromosome's bins
    ``wstack[1]`` and so on.
    """

    def __init__(self, embedding_weights, dim, num_list, target_weights=None, device="cpu"):
        super().__init__()
        if target_weights is None:
            target_weights = embedding_weights
        self.dim = dim
        self._device = device
        self.num_list = torch.tensor([0] + list(num_list)).to(device)
        table = torch.zeros(int(num_list[-1]) + 1).long()
        for i in range(len(self.num_list) - 1):
            table[int(self.num_list[i]) + 1:int(self.num_list[i + 1]) + 1] = i
        one_hot = torch.zeros([len(table), int(table.max()) + 1])
        one_hot[torch.arange(len(table)), table] = 1
        one_hot[0] = 0
        self.searchsort_table = one_hot.bool().to(device)

        self.embeddings = [SparseEmbedding(w, device) for w in embedding_weights]
        self.targets = [SparseEmbedding(w, device) for w in target_weights]
        test = torch.zeros(1, device=device).long()
        self.input_size = [w(test).shape[-1] for w in self.embeddings]
        self.layer_norm = nn.LayerNorm(self.dim).to(device)

        wstack = []
        if self.input_size[0] == target_weights[0].shape[-1]:
            wstack.append(TiedAutoEncoder([self.input_size[0], self.dim], add_activation=False, tied_list=[],
                                          device=device))
        else:
            wstack.append(AutoEncoder([self.input_size[0], self.dim], [self.dim, target_weights[0].shape[-1]],
                                      add_activation=True, device=device))
        for i in range(1, len(self.embeddings)):
            wstack.append(TiedAutoEncoder([self.input_size[i], self.dim], add_activation=True, tied_list=[],
                                          device=device))
        self.wstack = nn.ModuleList(wstack)
        self.on_hook_embedding = nn.ModuleList([nn.Sequential(w, self.wstack[i])
                                                for i, w in enumerate(self.embeddings)])
        self.on_hook_set = set(range(len(self.embeddings)))
        self.off_hook_embedding = [i for i in range(len(self.embeddings))]
        self.features = self.forward

    def forward(self, x, *args, route_nn=None):
        if len(x.shape) > 1:
            sz_b, len_seq = x.shape
            x = x.view(-1)
            reshape_flag = True
        else:
            reshape_flag = False
        if route_nn is None:
            final = torch.zeros((len(x), self.dim), device=x.device).float()
            ind = self.searchsort_table[x]
            node_type = torch.nonzero(torch.any(ind, dim=0)).view(-1)
            for i in node_type:
                mask = ind[:, i]
                if int(i) in self.on_hook_set:
                    final[mask] = self.on_hook_embedding[i](x[mask] - self.num_list[i] - 1)
                else:
                    final[mask] = self.off_hook_embedding[i](x[mask] - self.num_list[i] - 1)
        else:
            i = route_nn
            if int(i) in self.on_hook_set:
                final = self.on_hook_embedding[i](x - self.num_list[i] - 1)
            else:
                final = self.off_hook_embedding[i](x - self.num_list[i] - 1)
        if reshape_flag:
            final = final.view(sz_b, len_seq, -1)
        return final

    def off_hook(self, off_hook_list=()):
        """Freeze the listed node-type auto-encoders and cache their outputs."""
        off_hook_list = list(off_hook_list) or list(range(len(self.wstack)))
        for index in off_hook_list:
            ae = self.wstack[index]
            for p in ae.parameters():
                p.requires_grad = False
            ids = torch.arange(start=0, end=int(self.num_list[index + 1] - self.num_list[index]),
                               device=self.num_list.device)
            with torch.no_grad():
                embed = self.on_hook_embedding[index](ids).detach()
            self.off_hook_embedding[index] = SparseEmbedding(embed, embed.device)
            self.on_hook_set.discard(index)

    def on_hook(self, on_hook_list=()):
        on_hook_list = list(on_hook_list) or list(range(len(self.wstack)))
        for index in on_hook_list:
            for p in self.wstack[index].parameters():
                p.requires_grad = True
            self.on_hook_set.add(index)


class PositionwiseFeedForward(nn.Module):
    """Position-wise MLP built from 1x1 ``Conv1d`` layers (upstream)."""

    def __init__(self, dims, dropout=None, reshape=False, use_bias=True, residual=False, layer_norm=False):
        super().__init__()
        self.dims = dims
        self.w_stack = nn.ModuleList([nn.Conv1d(dims[i], dims[i + 1], 1, bias=use_bias)
                                      for i in range(len(dims) - 1)])
        self.reshape = reshape
        self.layer_norm = nn.LayerNorm(dims[0])
        self.dropout = nn.Dropout(dropout) if dropout is not None else None
        self.residual = residual
        self.layer_norm_flag = layer_norm
        self.alpha = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        output = self.layer_norm(x) if self.layer_norm_flag else x
        output = output.transpose(1, 2)
        for i in range(len(self.w_stack) - 1):
            output = self.w_stack[i](output)
            output = activation_func(output)
            if self.dropout is not None:
                output = self.dropout(output)
        output = self.w_stack[-1](output)
        output = output.transpose(1, 2)
        if self.reshape:
            output = output.view(output.shape[0], -1, 1)
        if self.dims[0] == self.dims[-1] and self.residual:
            output = output + x
        return output


class FeedForward(nn.Module):
    """Plain MLP (upstream ``FeedForward``)."""

    def __init__(self, dims, dropout=None, reshape=False, use_bias=True):
        super().__init__()
        self.w_stack = nn.ModuleList([nn.Linear(dims[i], dims[i + 1], use_bias) for i in range(len(dims) - 1)])
        self.dropout = nn.Dropout(dropout) if dropout is not None else None
        self.reshape = reshape

    def forward(self, x):
        output = x
        for i in range(len(self.w_stack) - 1):
            output = self.w_stack[i](output)
            output = activation_func(output)
            if self.dropout is not None:
                output = self.dropout(output)
        output = self.w_stack[-1](output)
        if self.reshape:
            output = output.view(output.shape[0], -1, 1)
        return output


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    @staticmethod
    def masked_softmax(vector, mask, dim=-1):
        # upstream ``memory_efficient=False`` branch
        if mask is None:
            return torch.nn.functional.softmax(vector, dim=dim)
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        result = torch.nn.functional.softmax(vector * mask, dim=dim)
        result = result * mask
        return result / (result.sum(dim=dim, keepdim=True) + 1e-13)

    def forward(self, q, k, v, diag_mask, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask, -float('inf'))
        attn = self.masked_softmax(attn, diag_mask, dim=-1)
        return torch.bmm(attn, v), attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout, diag_mask, input_dim):
        super().__init__()
        self.d_model = d_model
        self.input_dim = input_dim
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(input_dim, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(input_dim, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(input_dim, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.fc1 = FeedForward([n_head * d_v, d_model], use_bias=False)
        self.fc2 = FeedForward([n_head * d_v, d_model], use_bias=False)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.layer_norm3 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout) if dropout is not None else None
        self.diag_mask_flag = diag_mask
        self.alpha_static = torch.nn.Parameter(torch.zeros(1))
        self.alpha_dynamic = torch.nn.Parameter(torch.zeros(1))

    def forward(self, q, k, v, diag_mask=None, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        q = self.layer_norm1(q)
        k = self.layer_norm2(k)
        v = self.layer_norm3(v)
        sz_b, len_q, _ = q.shape
        sz_b, len_k, _ = k.shape
        sz_b, len_v, _ = v.shape
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)
        # upstream caches an (n*b, L, L) bool mask = 1 - I; broadcasting is equivalent
        dm = torch.ones((len_v, len_v), device=q.device)
        if self.diag_mask_flag:
            dm = dm - torch.eye(len_v, len_v, device=q.device)
        dm = dm.bool()[None]
        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)
        dynamic, attn = self.attention(q, k, v, dm, mask=mask)
        dynamic = dynamic.view(n_head, sz_b, len_q, d_v).permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        static = v.view(n_head, sz_b, len_q, d_v).permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        dynamic = self.dropout(self.fc1(dynamic)) if self.dropout is not None else self.fc1(dynamic)
        static = self.dropout(self.fc2(static)) if self.dropout is not None else self.fc2(static)
        return dynamic, static, attn


class MeanAggregator_with_weights(nn.Module):
    """Weighted mean over a node's contact-graph neighbours.

    ``to_neighs`` is ``(indices, values, unique_nodes)`` as built by the
    sampler: a sparse ``(n_nodes, n_unique)`` row-normalised weight matrix.
    """

    def __init__(self, features):
        super().__init__()
        self.features = features

    def forward(self, nodes_real, to_neighs, num_sample=10, route_nn=None):
        indices, v, unique_nodes_list = to_neighs
        device = nodes_real.device
        unique_nodes_list = unique_nodes_list.to(device, non_blocking=True)
        indices = indices.to(device, non_blocking=True)
        v = v.to(device, non_blocking=True)
        embed_matrix = self.features(unique_nodes_list, route_nn=route_nn)
        # sparse (n_nodes x n_unique) @ embed, written as a scatter-add: the same
        # sum as upstream's ``torch.sparse_coo_tensor(...).mm`` without the
        # coalescing sort
        out = torch.zeros((len(nodes_real), embed_matrix.shape[-1]), dtype=embed_matrix.dtype, device=device)
        return out.index_add(0, indices[0], embed_matrix[indices[1]] * v[:, None].to(embed_matrix.dtype))

    def forward_GCN(self, nodes, adj, route_nn=None):
        embed_matrix = self.features(nodes, route_nn=route_nn)
        indices, data, shape = adj
        device = nodes.device
        mask = torch.sparse_coo_tensor(indices.to(device), data.to(device), torch.Size([shape[0], shape[1]]),
                                       device=device)
        return mask.mm(embed_matrix)


class MeanAggregator(nn.Module):
    """1-D genomic-neighbour aggregator; only used when ``transfer_range > 0``
    (never in Higashi's default configuration). Holds ``features`` so the
    module tree (and state_dict keys) match upstream."""

    def __init__(self, features):
        super().__init__()
        self.features = features


class GraphSageEncoder_with_weights(nn.Module):
    """Cell-dependent GNN (stage 2/3): a bin node's embedding is
    ``swish(W [mean_w(neighbour emb); self emb])`` over that cell's (or its
    k nearest cells') contact graph."""

    def __init__(self, features, linear_features=None, feature_dim=64, embed_dim=64, num_sample=10,
                 gcn=False, num_list=None, transfer_range=0, device="cpu"):
        super().__init__()
        self.features = features
        self.linear_features = linear_features
        self.feat_dim = feature_dim
        self.aggregator = MeanAggregator_with_weights(self.features)
        self.linear_aggregator = MeanAggregator(self.linear_features)
        self.num_sample = num_sample
        self.transfer_range = transfer_range
        if transfer_range != 0:
            raise NotImplementedError("transfer_range > 0 is not used by Higashi and not ported")
        self.gcn = gcn
        self.embed_dim = embed_dim
        input_size = 1 + (0 if gcn else 1)
        self.nn = nn.Linear(input_size * self.feat_dim, embed_dim)
        self.num_list = torch.as_tensor(num_list)
        self._device = device
        self.bin_feats = torch.zeros([int(self.num_list[-1]) + 1, self.feat_dim * input_size],
                                     dtype=torch.float, device=device)
        self.fix = False
        self.off_hook_mode = False

    def start_fix(self):
        self.fix = True
        ids = (torch.arange(int(self.num_list[0])) + 1).long().to(self.bin_feats.device).view(-1)
        self.cell_feats = self.features(ids)

    def fix_cell2(self, cell, bin_ids=None, sparse_matrix=None, route_nn_list=None):
        """Pre-compute every bin's GNN input for one cell (imputation)."""
        self.fix = True
        self.eval()
        input_size = 1 + (0 if self.gcn else 1)
        self.bin_feats = torch.zeros([int(self.num_list[-1]) + 1, self.feat_dim * input_size],
                                     dtype=torch.float, device=self.bin_feats.device)
        with torch.no_grad():
            for chrom, bin_id in enumerate(bin_ids):
                nodes_flatten = torch.from_numpy(bin_id).long().to(self.bin_feats.device)
                neigh_feats = self.aggregator.forward_GCN(nodes_flatten, sparse_matrix[chrom],
                                                          route_nn=route_nn_list[chrom])
                list1 = [neigh_feats]
                if not self.gcn:
                    list1.append(self.features(nodes_flatten))
                self.bin_feats[nodes_flatten] = torch.cat(list1, dim=-1).detach().clone()

    def forward(self, nodes, to_neighs, *args, route_nn=None):
        if self.off_hook_mode:
            return self.forward_off_hook(nodes, to_neighs, route_nn=route_nn)
        return self.forward_on_hook(nodes, to_neighs, route_nn=route_nn)

    def forward_off_hook(self, nodes, to_neighs, *args, route_nn=None):
        sz_b, len_seq = nodes.shape
        nodes_flatten = nodes[:, 1:].contiguous().view(-1)
        cell_feats = self.cell_feats[nodes[:, 0] - 1, :]
        neigh_feats = activation_func(self.nn(self.bin_feats[nodes_flatten, :])).view(sz_b, len_seq - 1, -1)
        return (torch.cat([cell_feats[:, None, :], neigh_feats], dim=1).view(sz_b, len_seq, -1),
                torch.cat([cell_feats[:, None, :], self.features(nodes_flatten).view(sz_b, len_seq - 1, -1)],
                          dim=1).view(sz_b, len_seq, -1))

    def forward_on_hook(self, nodes, to_neighs, *args, route_nn=None):
        sz_b, len_seq = nodes.shape
        nodes_flatten = nodes[:, 1:].contiguous().view(-1)
        cell_feats = self.features(nodes[:, 0], route_nn=0)
        # Upstream does not forward ``route_nn`` to the aggregator and routes the
        # neighbour ids through the per-node-type lookup instead. All neighbours
        # of a single-chromosome batch are bins of that chromosome, so routing
        # them straight to its auto-encoder gives the same values without the
        # host-device synchronisations of the lookup.
        neigh_feats = self.aggregator.forward(nodes_flatten, to_neighs, self.num_sample,
                                              route_nn=route_nn).view(sz_b, len_seq - 1, -1)
        list1 = [neigh_feats]
        self_feats = None
        if not self.gcn:
            self_feats = self.features(nodes_flatten, route_nn=route_nn).view(sz_b, len_seq - 1, -1)
            list1.append(self_feats)
        combined = torch.cat(list1, dim=-1) if len(list1) > 1 else list1[0]
        combined = activation_func(self.nn(combined))
        combined = torch.cat([cell_feats[:, None, :], combined], dim=1).view(sz_b, len_seq, -1)
        return combined, torch.cat([cell_feats[:, None, :], self_feats], dim=1).view(sz_b, len_seq, -1)


class EncoderLayer(nn.Module):
    """Self-attention over the (cell, bin_i, bin_j) hyperedge + position-wise FFN."""

    def __init__(self, n_head, d_model, d_k, d_v, dropout_mul, dropout_pff, diag_mask, bottle_neck,
                 dynamic_nn=None, static_nn=None):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.mul_head_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout_mul,
                                                diag_mask=diag_mask, input_dim=bottle_neck)
        self.pff_n1 = PositionwiseFeedForward([d_model, d_model, d_model], dropout=dropout_pff,
                                              residual=True, layer_norm=True)
        residual = bottle_neck == d_model
        self.pff_n2 = PositionwiseFeedForward([bottle_neck, d_model, d_model], dropout=dropout_pff,
                                              residual=residual, layer_norm=True)
        self.dynamic_nn = dynamic_nn
        self.static_nn = static_nn
        self.dropout = nn.Dropout(0.2)

    def forward(self, dynamic, static, chrom_info, slf_attn_mask, non_pad_mask, chroms_in_batch):
        if type(chrom_info) is tuple:
            chrom_info, to_neighs = chrom_info
        else:
            to_neighs = chrom_info
        final_chroms_in_batch = None
        if chroms_in_batch is not None and len(chroms_in_batch) == 1:
            final_chroms_in_batch = int(chroms_in_batch[0])
        if isinstance(self.dynamic_nn, GraphSageEncoder_with_weights):
            dynamic, static = self.dynamic_nn(dynamic, to_neighs, route_nn=final_chroms_in_batch)
        else:
            static_cell = self.static_nn(static[:, 0], to_neighs, route_nn=0)
            static_bin = self.static_nn(static[:, 1:].contiguous(), to_neighs, route_nn=final_chroms_in_batch)
            static = torch.cat([static_cell[:, None, :], static_bin], dim=1)
            dynamic = static
        dynamic, static, attn = self.mul_head_attn(dynamic, dynamic, static)
        dynamic = self.pff_n1(dynamic)
        return dynamic, static


class Hyper_SAGNN(nn.Module):
    """Higashi's hyperedge scorer. Returns (mean, dispersion, dropout-logit)
    heads for the ZINB likelihood (or a single logit in classification mode)."""

    def __init__(self, n_head, d_model, d_k, d_v, diag_mask, bottle_neck, attribute_dict=None,
                 cell_feats=None, encoder_dynamic_nn=None, encoder_static_nn=None, chrom_num=1, device="cpu"):
        super().__init__()
        self.pff_classifier = PositionwiseFeedForward([d_model, int(d_model / 2), 1])
        self.pff_classifier_var = PositionwiseFeedForward([d_model, int(d_model / 2), 1])
        self.pff_classifier_proba = PositionwiseFeedForward([d_model, int(d_model / 2), 1])
        self.encode_list = []
        self.encode1 = EncoderLayer(n_head, d_model, d_k, d_v, dropout_mul=0.3, dropout_pff=0.4,
                                    diag_mask=diag_mask, bottle_neck=bottle_neck,
                                    dynamic_nn=encoder_dynamic_nn, static_nn=encoder_static_nn)
        self.diag_mask_flag = diag_mask
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.3)
        self.attribute_dict = None
        if attribute_dict is not None:
            self.attribute_dict = torch.from_numpy(attribute_dict).to(device)
            input_size = self.attribute_dict.shape[-1] * 2 + cell_feats.shape[-1]
            self.extra_proba = FeedForward([input_size, 4, 1])
            self.extra_proba2 = FeedForward([input_size, 4, 1])
            self.extra_proba3 = FeedForward([input_size, 4, 1])
            self.attribute_dict_embedding = nn.Embedding(len(self.attribute_dict), 1, padding_idx=0)
            self.attribute_dict_embedding.weight = nn.Parameter(self.attribute_dict)
            self.attribute_dict_embedding.weight.requires_grad = False
            self.cell_feats = torch.from_numpy(cell_feats).to(device)
        self.only_distance = False
        self.only_model = False
        self.chrom_num = chrom_num
        self.d_model = d_model

    def get_embedding(self, x, x_chrom, slf_attn_mask=None, non_pad_mask=None, chroms_in_batch=None):
        dynamic, static = self.encode1(x, x, x_chrom, slf_attn_mask, non_pad_mask, chroms_in_batch=chroms_in_batch)
        if torch.isnan(dynamic).any():
            raise FloatingPointError("NaN in Higashi hyperedge embedding")
        return dynamic, static

    def forward(self, x, x_chrom, mask=None, chroms_in_batch=None):
        x = x.long()
        if self.attribute_dict is not None:
            if not self.only_model:
                cell_part = self.cell_feats[x[:, 0]]
            else:
                cell_part = torch.zeros((len(x), self.cell_feats.shape[-1]), dtype=torch.float, device=x.device)
            distance = torch.cat([self.attribute_dict_embedding(x[:, 1]), self.attribute_dict_embedding(x[:, 2]),
                                  cell_part], dim=-1)
            distance_proba = self.extra_proba(distance)
            distance_proba2 = self.extra_proba2(distance)
            distance_proba3 = self.extra_proba3(distance)
        else:
            distance_proba = torch.zeros((len(x), 1), dtype=torch.float, device=x.device)
            distance_proba2 = torch.zeros((len(x), 1), dtype=torch.float, device=x.device)
            distance_proba3 = torch.zeros((len(x), 1), dtype=torch.float, device=x.device)
        if self.only_distance:
            return distance_proba2, distance_proba3, distance_proba
        dynamic, static = self.get_embedding(x, x_chrom, chroms_in_batch=chroms_in_batch)
        dynamic = self.layer_norm1(dynamic)
        static = self.layer_norm2(static)
        output = (dynamic - static) ** 2 if self.diag_mask_flag else dynamic
        output_proba = torch.mean(self.pff_classifier_proba(static), dim=-2, keepdim=False) + distance_proba
        output_mean = torch.mean(self.pff_classifier(output), dim=-2, keepdim=False) + distance_proba2
        output_var = torch.mean(self.pff_classifier_var(static), dim=-2, keepdim=False) + distance_proba3
        return output_mean, output_var, output_proba

    def predict(self, input, input_chrom, batch_size=96, activation=None):
        """Score hyperedges (numpy ``(n, 3)`` of 1-based node ids)."""
        self.eval()
        device = self.layer_norm1.weight.device
        output = []
        with torch.no_grad():
            for j in range(math.ceil(len(input) / batch_size)):
                x = torch.as_tensor(input[j * batch_size:min((j + 1) * batch_size, len(input))],
                                    dtype=torch.long).to(device)
                x_chrom = torch.from_numpy(input_chrom[j * batch_size:min((j + 1) * batch_size, len(input))]
                                           ).long().to(device)
                o, _, _ = self(x, x_chrom)
                if activation is not None:
                    o = activation(o)
                output.append(o.detach().cpu())
        self.train()
        return torch.cat(output, dim=0).numpy()
