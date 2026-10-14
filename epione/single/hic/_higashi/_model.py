"""Higashi training and imputation driver.

Adapted from Higashi (https://github.com/ma-compbio/Higashi @ 42702a8,
``higashi/Higashi_wrapper.py`` class ``Higashi`` and ``higashi/Impute.py``).
MIT License, Copyright (c) 2020 Ma Lab at CMU; see ``LICENSE_HIGASHI``.

The three training stages, their losses, optimisers, epoch lengths, early
stopping and the imputation procedure follow upstream. Differences:

* in-memory state instead of ``temp_dir`` files; checkpoints are kept as
  deep copies (upstream ``torch.save`` / ``torch.load`` round-trips them);
* batches are built in the main process by the vectorised sampler
  (``_sampler.py``) instead of a process pool;
* upstream's stage-1 "dynamic pair ratio" is updated in the parent process
  after the sampler workers have been forked, so the workers keep the value
  they were forked with (0.0) for the whole stage. We reproduce that
  *effective* behaviour: the sampler's ``pair_ratio`` is fixed when a stage
  starts (stage 1: 0.0, stages 2/3: 0.5). Verified on the reference run
  (``tests/parity/higashi/MANIFEST.md``);
* seeded random streams (upstream draws from fresh OS entropy).
"""
from __future__ import annotations

import copy
import math
import time
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix, vstack

from ._modules import (GraphSageEncoder_with_weights, Hyper_SAGNN, MultipleEmbedding, log_zinb_positive)
from ._process import HigashiInput
from ._sampler import BatchMaker, ContactIndex, DataGenerator, NegativeSampler, NeighborMaskBuilder


def _standardize_scalar(a):
    """``StandardScaler().fit_transform(a.reshape(-1, 1)).reshape(a.shape)``."""
    from sklearn.preprocessing import StandardScaler
    return StandardScaler().fit_transform(a.reshape((-1, 1))).reshape((len(a), -1))


def moving_avg(adj, moving_range):
    from scipy.stats import norm
    adj_origin = adj.copy()
    adj = adj.copy()
    adj = adj * norm.pdf(0)
    for i in range(moving_range * 2):
        before_list = [adj_origin[0, :]] * (i + 1) + [adj_origin[:-(i + 1), :]]
        adj_before = vstack(before_list)
        after_list = [adj_origin[i + 1:, :]] + [adj_origin[-1, :]] * (i + 1)
        adj_after = vstack(after_list)
        adj = adj + (adj_after + adj_before) * norm.pdf((i + 1) / moving_range)
    return adj


def generate_binpair(start, end, min_bin_, max_bin_):
    samples = []
    for bin1 in range(start, end):
        for bin2 in range(bin1 + min_bin_, min(bin1 + max_bin_, end)):
            samples.append([bin1, bin2])
    return np.array(samples).reshape((-1, 2)) + 1


class Higashi:
    """Higashi (Zhang, Zhou & Ma 2022) on a preprocessed :class:`HigashiInput`.

    Typical use mirrors upstream's API::

        model = Higashi(data, seed=0)
        model.prep_model()
        model.train_for_embeddings()          # stage 1 -> cell embeddings
        emb = model.fetch_cell_embeddings()
        model.train_for_imputation_nbr_0()    # stage 2
        model.impute_no_nbr(on_cell)          # imputation without neighbours
        model.train_for_imputation_with_nbr() # stage 3
        model.impute_with_nbr(on_cell)        # imputation with k neighbours

    Arguments mirror upstream's config keys (``dimensions``, ``loss_mode``,
    ``rank_thres``, ``neighbor_num``, ``local_transfer_range``,
    ``minimum_distance``, ``maximum_distance``, ``embedding_epoch``,
    ``no_nbr_epoch``, ``with_nbr_epoch``, ``impute_list``), with upstream's
    defaults.
    """

    def __init__(self, data: HigashiInput, *, dimensions: int = 64, loss_mode: str = "zinb", rank_thres: float = 1,
                 neighbor_num: int = 5, local_transfer_range: int = 1, minimum_distance: Optional[int] = None,
                 maximum_distance: int = -1, embedding_epoch: int = 60, no_nbr_epoch: int = 45,
                 with_nbr_epoch: int = 30, impute_list: Optional[Sequence[str]] = None, device=None, seed: int = 0,
                 update_num_per_training_epoch: int = 1000, update_num_per_eval_epoch: int = 10,
                 verbose: bool = True):
        if loss_mode not in ("zinb", "classification", "rank", "regression"):
            raise ValueError(f"unknown loss_mode {loss_mode!r}")
        self.data = data
        self.dimensions = int(dimensions)
        self.mode = loss_mode
        self.rank_thres = rank_thres
        self.neighbor_num = int(neighbor_num) + 1
        self.local_transfer_range = int(local_transfer_range)
        self.res = int(data.resolution)
        self.minimum_distance = self.res if minimum_distance is None else int(minimum_distance)
        self.maximum_distance = int(maximum_distance)
        self.embedding_epoch = int(embedding_epoch)
        self.no_nbr_epoch = int(no_nbr_epoch)
        self.with_nbr_epoch = int(with_nbr_epoch)
        self.chrom_list = list(data.chrom_list)
        self.impute_list = list(impute_list) if impute_list is not None else list(self.chrom_list)
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.seed = int(seed)
        self.update_num_per_training_epoch = int(update_num_per_training_epoch)
        self.update_num_per_eval_epoch = int(update_num_per_eval_epoch)
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.cell_embeddings = None
        self.history = []
        self.timing = {}
        self._stage_models = {}

    # ------------------------------------------------------------------ setup
    def _log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    def generate_attributes(self):
        d = self.data
        embeddings, targets = [], []
        cell_node_feats = [np.asarray(a) for a in d.cell_feats]
        from sklearn.preprocessing import StandardScaler
        if self.num[0] >= 10:
            cell_node_feats1 = np.concatenate(cell_node_feats, axis=-1)
            cell_node_feats1 = _standardize_scalar(cell_node_feats1)
            cell_node_feats2 = [StandardScaler().fit_transform(x) for x in cell_node_feats]
            cell_node_feats2 = np.concatenate(cell_node_feats2, axis=-1)
            cell_node_feats2 = _standardize_scalar(cell_node_feats2)
        else:
            cell_node_feats1 = np.concatenate(cell_node_feats, axis=-1)
            cell_node_feats2 = cell_node_feats1
        targets.append(cell_node_feats2.astype('float32'))
        embeddings.append(cell_node_feats1.astype('float32'))
        for i, c in enumerate(self.chrom_list):
            temp = np.asarray(d.bin_feats[i]).astype('float32')
            temp = _standardize_scalar(temp)
            chrom = np.zeros((len(temp), len(self.chrom_list))).astype('float32')
            chrom[:, i] = 1
            temp = np.concatenate([temp, chrom], axis=-1)
            embeddings.append(temp)
            targets.append(temp)
        attribute_all = []
        for i in range(len(self.num) - 1):
            chrom = np.zeros((self.num[i + 1], len(self.chrom_list)))
            chrom[:, i] = 1
            coor = np.arange(self.num[i + 1]).reshape((-1, 1)).astype('float32')
            coor /= self.num[1]
            attribute_all.append(np.concatenate([chrom, coor], axis=-1))
        attribute_all = np.concatenate(attribute_all, axis=0)
        attribute_dict = np.concatenate([np.zeros((self.num[0] + 1, attribute_all.shape[-1])), attribute_all],
                                        axis=0).astype('float32')
        return embeddings, attribute_dict, targets

    def prep_model(self):
        d = self.data
        dev = self.device
        self.num = np.asarray(d.num)
        num = self.num
        self.median_total_sparsity_cell = float(np.median(d.sparsity))
        if self.median_total_sparsity_cell >= 0.05:
            self.contractive_flag, self.contractive_loss_weight = True, 1e-3
        else:
            self.contractive_flag, self.contractive_loss_weight = False, 0.0
        self.cell_num = int(num[0])
        n_cell_feats = int(sum(np.asarray(a).shape[1] for a in d.cell_feats))
        if n_cell_feats > self.dimensions > self.cell_num:
            # upstream initialises the cell auto-encoder with a PCA of this size
            raise ValueError(f"dimensions={self.dimensions} exceeds the number of cells ({self.cell_num}); "
                             f"lower higashi_params['dimensions']")
        self.cell_ids = torch.arange(self.cell_num).long().to(dev)
        self.batch_size = min(int(256 * max((1000000 / self.res), 1) * max(num[0] / 6000, 1)), 1280)
        self.num_list = np.cumsum(num)
        self.max_bin = int(np.max(num[1:]))
        total_possible = 0
        for c in d.chrom_start_end:
            n = int(c[1] - c[0])
            total_possible += n * (n + 1) // 2
        total_possible *= self.cell_num
        sparsity = np.sum([len(a) + len(b) for a, b in zip(d.train_data, d.test_data)]) / total_possible
        if sparsity > 0.3:
            neg_num = 1
        elif sparsity > 0.2:
            neg_num = 2
        elif sparsity > 0.15:
            neg_num = 3
        elif sparsity > 0.1:
            neg_num = 4
        else:
            neg_num = 5
        self.neg_num = neg_num
        self.batch_size *= (1 + neg_num)
        self._log("Higashi: cells %d, node type num %s, neg_num %d, batch %d, contractive %s"
                  % (self.cell_num, list(map(int, num)), neg_num, self.batch_size, self.contractive_flag))

        cell_feats = np.sum(d.extra_cell_feats, axis=-1, keepdims=True)
        self.cell_feats1 = torch.from_numpy(np.asarray(d.cell2weight)).float().to(dev)
        cell_feats = np.log1p(cell_feats)
        self.cell_feats = np.concatenate([np.zeros((1, cell_feats.shape[-1])), cell_feats], axis=0).astype('float32')

        embeddings_initial, attribute_dict, targets_initial = self.generate_attributes()
        self.attribute_dict = attribute_dict
        self.sparse_chrom_list = [[m.astype('float32') for m in mats] for mats in d.sparse_gcn]

        train_weight = [np.asarray(w, dtype='float32') for w in d.train_weight]
        test_weight = [np.asarray(w, dtype='float32') for w in d.test_weight]
        if self.mode == 'classification':
            m = np.mean(train_weight)

            def tw(weight):
                weight = np.log2(weight + 1)
                weight[weight >= np.quantile(weight, 0.99)] = np.quantile(weight, 0.99)
                return weight / m * neg_num
            train_weight = [tw(x) for x in train_weight]
            test_weight = [tw(x) for x in test_weight]
        elif self.mode == 'rank':
            train_weight = [x + 1 for x in train_weight]
            test_weight = [x + 1 for x in test_weight]

        self.node_embedding_init = MultipleEmbedding(embeddings_initial, self.dimensions, self.num_list,
                                                     targets_initial, device=dev)
        gen = torch.Generator().manual_seed(self.seed)
        self.node_embedding_init.wstack[0].fit(
            embeddings_initial[0], 300, batch_size=1024,
            targets=torch.from_numpy(targets_initial[0]).float().to(dev), generator=gen,
            random_state=self.seed, verbose=self.verbose)
        self.higashi_model = Hyper_SAGNN(
            n_head=8, d_model=self.dimensions, d_k=16, d_v=16, diag_mask=True, bottle_neck=self.dimensions,
            attribute_dict=attribute_dict, cell_feats=self.cell_feats, encoder_dynamic_nn=self.node_embedding_init,
            encoder_static_nn=self.node_embedding_init, chrom_num=len(self.chrom_list), device=dev).to(dev)

        self.contact_index = ContactIndex(self.sparse_chrom_list, self.num_list)
        self.sampler = NegativeSampler(self.contact_index, d.start_end_dict, self.max_bin, neg_num, self.rng)
        self.nbr_builder = NeighborMaskBuilder(self.sparse_chrom_list, self.num_list, self.rng)
        self.batch_maker = BatchMaker(self.sampler, self.nbr_builder, self.mode, self.rng)
        pos_bs = int(self.batch_size / (neg_num + 1))
        self.training_data_generator = DataGenerator(d.train_data, d.train_chrom, train_weight, pos_bs,
                                                     self.num_list, self.rng)
        self.validation_data_generator = DataGenerator(d.test_data, d.test_chrom, test_weight, pos_bs,
                                                       self.num_list, self.rng)

    # --------------------------------------------------------------- training
    def forward_batch_hyperedge(self, batch_data, batch_weight, batch_chrom, batch_to_neighs, y, chroms_in_batch):
        model = self.higashi_model
        x, w = batch_data, batch_weight
        pred, pred_var, pred_proba = model(x, (batch_chrom, batch_to_neighs), chroms_in_batch=chroms_in_batch + 1)
        if self.use_recon:
            adj = self.node_embedding_init.embeddings[0](self.cell_ids).float()
            targets = self.node_embedding_init.targets[0](self.cell_ids).float()
            embed, recon = self.node_embedding_init.wstack[0](adj, return_recon=True)
            mse_loss = F.mse_loss(recon, targets)
        else:
            mse_loss = torch.as_tensor([0], dtype=torch.float).to(self.device)
        if self.mode == 'classification':
            main_loss = F.binary_cross_entropy_with_logits(pred, y, weight=w)
        elif self.mode == 'rank':
            pred = F.softplus(pred)
            diff = (pred.view(-1, 1) - pred.view(1, -1)).view(-1)
            diff_w = (w.view(-1, 1) - w.view(1, -1)).view(-1)
            mask_rank = torch.abs(diff_w) > self.rank_thres
            diff = diff[mask_rank].float()
            diff_w = diff_w[mask_rank]
            main_loss = F.binary_cross_entropy_with_logits(diff, (diff_w > 0).float())
            if not self.use_recon and self.neg_num > 0:
                mz = w == 0
                mse_loss = mse_loss + F.mse_loss(w[mz].float(), pred[mz].float())
        elif self.mode == 'zinb':
            pred = F.softplus(pred)
            extra = self.cell_feats1[batch_data[:, 0] - 1].view(-1, 1)
            pred = pred * extra
            pred_var = F.softplus(pred_var)
            pred = torch.clamp(pred, min=1e-8, max=1e8)
            pred_var = torch.clamp(pred_var, min=1e-8, max=1e8)
            main_loss = -log_zinb_positive(w.float(), pred.float(), pred_var.float(), pred_proba.float())
            main_loss = main_loss.mean()
        else:  # regression
            main_loss = F.mse_loss(pred.float().view(-1), w.float().view(-1))
        return pred, main_loss, mse_loss

    def _to_device(self, batch):
        x, y, w, x_chrom, to_neighs = batch
        dev = self.device
        x = torch.as_tensor(x, dtype=torch.long).to(dev, non_blocking=True)
        y = torch.from_numpy(y).float().to(dev, non_blocking=True)
        w = torch.from_numpy(w).float().to(dev, non_blocking=True)
        return x, y, w, x_chrom, to_neighs

    def _batch_stream(self, n):
        """Build ``n`` training batches in a background thread (upstream uses
        a process pool). One producer consumes the random stream in order, so
        the batches are the same as building them inline."""
        import queue
        import threading
        q = queue.Queue(maxsize=16)

        def producer():
            try:
                for _ in range(n):
                    e, c, wt, chroms = self.training_data_generator.next_iter()
                    q.put((self.batch_maker.make(e, c, wt, training=True), chroms))
            except BaseException as ex:  # surface sampler errors in the consumer
                q.put(ex)

        th = threading.Thread(target=producer, daemon=True)
        th.start()
        for _ in range(n):
            item = q.get()
            if isinstance(item, BaseException):
                raise item
            yield item
        th.join()

    def train_epoch(self, optimizer_list):
        model = self.higashi_model
        model.train()
        bce_total, mse_total, n_batches = 0.0, 0.0, 0
        for batch, chroms in self._batch_stream(self.update_num_per_training_epoch):
            x, y, w, x_chrom, to_neighs = self._to_device(batch)
            pred, loss_bce, loss_mse = self.forward_batch_hyperedge(x, w, x_chrom, to_neighs, y, chroms)
            if self.use_recon:
                wl = self.node_embedding_init.wstack[0].weight_list[0]
                for opt in optimizer_list:
                    opt.zero_grad(set_to_none=True)
                loss_bce.backward(retain_graph=True)
                main_norm = wl.grad.data.norm(2) if wl.grad is not None else 0.0
                for opt in optimizer_list:
                    opt.zero_grad(set_to_none=True)
                loss_mse.backward(retain_graph=True)
                recon_norm = wl.grad.data.norm(2)
                ratio = self.beta * main_norm / recon_norm
                ratio1 = max(ratio, 100 * self.median_total_sparsity_cell - 3)
                if self.contractive_flag:
                    contractive_loss = 0.0
                    ae = self.node_embedding_init.wstack[0]
                    for i in range(len(ae.weight_list)):
                        contractive_loss += torch.sum(ae.weight_list[i] ** 2)
                        contractive_loss += torch.sum(ae.reverse_weight_list[i] ** 2)
                else:
                    contractive_loss = 0.0
            else:
                contractive_loss, ratio1 = 0.0, 0.0
            train_loss = self.alpha * loss_bce + ratio1 * loss_mse + self.contractive_loss_weight * contractive_loss
            for opt in optimizer_list:
                opt.zero_grad(set_to_none=True)
            train_loss.backward()
            for opt in optimizer_list:
                opt.step()
            bce_total += loss_bce.item()
            mse_total += loss_mse.item()
            n_batches += 1
        return bce_total / n_batches, mse_total / n_batches

    def eval_epoch(self):
        model = self.higashi_model
        model.eval()
        total = 0.0
        with torch.no_grad():
            for _ in range(self.update_num_per_eval_epoch):
                e, c, wt, chroms = self.validation_data_generator.next_iter()
                batch = self.batch_maker.make(e, c, wt, training=False)
                x, y, w, x_chrom, to_neighs = self._to_device(batch)
                _, loss, _ = self.forward_batch_hyperedge(x, w, x_chrom, to_neighs, y, chroms)
                total += loss.item()
        return total / self.update_num_per_eval_epoch

    def train(self, optimizer, epochs, stage, save_embed=False):
        self.batch_maker.graphsage = isinstance(self.higashi_model.encode1.dynamic_nn, GraphSageEncoder_with_weights)
        # the sampler sees the pair ratio in force when the stage starts (see module docstring)
        self.sampler.pair_ratio = self.pair_ratio
        self.sampler.steps = self.steps
        no_improve = 0
        best_train_loss = 1000
        for epoch_i in range(epochs):
            if save_embed:
                self.save_embeddings()
            t = time.time()
            bce_loss, mse_loss = self.train_epoch(optimizer)
            valid = self.eval_epoch()
            rec = dict(stage=stage, epoch=epoch_i, train_loss=bce_loss, recon_mse=mse_loss, valid_loss=valid,
                       seconds=time.time() - t)
            self.history.append(rec)
            self._log("[stage %d] epoch %d/%d  train %.4f  mse %.4f  valid %.4f  (%.1fs)"
                      % (stage, epoch_i + 1, epochs, bce_loss, mse_loss, valid, rec['seconds']))
            if self.dynamic_pair_ratio:
                # upstream updates the parent-process value only (never reaches the sampler)
                self.pair_ratio = self.pair_ratio + 0.1 if self.pair_ratio < 0.5 else 0.5
            else:
                if bce_loss < best_train_loss - 1e-3:
                    best_train_loss = bce_loss
                    no_improve = 0
                else:
                    no_improve += 1
            if no_improve >= 4:
                self._log("[stage %d] no improvement, early stopping" % stage)
                break

    def save_embeddings(self):
        model = self.higashi_model
        model.eval()
        with torch.no_grad():
            ids = torch.arange(1, self.num_list[-1] + 1).long().to(self.device).view(-1)
            embeddings = []
            for j in range(math.ceil(len(ids) / self.batch_size)):
                x = ids[j * self.batch_size:min((j + 1) * self.batch_size, len(ids))]
                embeddings.append(self.node_embedding_init(x).detach().cpu().numpy())
            embeddings = np.concatenate(embeddings, axis=0)
        self.cell_embeddings = embeddings[:int(self.num_list[0])]
        self.all_embeddings = embeddings
        return embeddings

    def _snapshot(self, name):
        self._stage_models[name] = copy.deepcopy(self.higashi_model)

    def _restore(self, name):
        self.higashi_model = copy.deepcopy(self._stage_models[name])
        self.node_embedding_init = self.higashi_model.encode1.static_nn

    def train_for_embeddings(self, max_epochs=None):
        t0 = time.time()
        params = list(self.higashi_model.parameters()) + list(self.node_embedding_init.parameters())
        # upstream passes the embedding parameters twice (they are also inside
        # the model); the duplicates get two Adam updates per step. Kept as is.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            optimizer = torch.optim.Adam(params, lr=1e-3)
        self.alpha, self.beta = 1.0, 1e-2
        self.steps = 1
        self.pair_ratio = 0.0
        self.dynamic_pair_ratio = True
        self.higashi_model.only_distance = False
        self.use_recon = True
        self.train([optimizer], self.embedding_epoch if max_epochs is None else max_epochs, stage=1,
                   save_embed=True)
        self._snapshot("stage1")
        self.save_embeddings()
        self.timing['stage1'] = time.time() - t0

    def _distance_bins(self):
        max_bin = int(1e5) if self.maximum_distance < 0 else int(self.maximum_distance / self.res)
        min_bin = 0 if self.minimum_distance < 0 else int(self.minimum_distance / self.res)
        return min_bin, max_bin

    def train_for_imputation_nbr_0(self):
        t0 = time.time()
        self._restore("stage1")
        self.save_embeddings()
        self.node_embedding_init.off_hook([0])
        min_bin, max_bin = self._distance_bins()
        self.training_data_generator.filter_edges(min_bin, max_bin)
        self.validation_data_generator.filter_edges(min_bin, max_bin)
        self.alpha, self.beta = 1.0, 1e-3
        self.dynamic_pair_ratio = False
        self.use_recon = False
        self.contractive_flag, self.contractive_loss_weight = False, 0.0
        self.pair_ratio = 0.5
        node_embedding2 = GraphSageEncoder_with_weights(
            features=self.node_embedding_init, linear_features=self.node_embedding_init,
            feature_dim=self.dimensions, embed_dim=self.dimensions, num_sample=8, gcn=False,
            num_list=self.num_list, transfer_range=0, device=self.device).to(self.device)
        self.higashi_model.encode1.dynamic_nn = node_embedding2
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            optimizer = torch.optim.Adam(list(self.higashi_model.parameters())
                                         + list(self.node_embedding_init.parameters()), lr=1e-3)
        self.steps = 2
        self.nbr_builder.set_matrices(self.sparse_chrom_list)
        self.train([optimizer], self.no_nbr_epoch, stage=2)
        self._snapshot("stage2")
        self.timing['stage2'] = time.time() - t0

    def get_cell_neighbor(self, start=1):
        from sklearn.metrics import pairwise_distances
        v = self.cell_embeddings
        distance = pairwise_distances(v, metric='euclidean')
        distance_sorted = np.sort(distance, axis=-1)
        distance /= np.quantile(distance_sorted[:, 1:self.neighbor_num].reshape((-1)), q=0.25)
        nbr_list = [[] for _ in range(self.num[0] + 1)]
        w_list = [[] for _ in range(self.num[0] + 1)]
        for i, d in enumerate(distance):
            neighbor = np.argsort(d)[:self.neighbor_num]
            weight = np.sort(d)[:self.neighbor_num]
            new_w = np.exp(-weight[start:])
            new_w /= np.sum(new_w)
            nbr_list[i + 1] = (neighbor + 1)[start:]
            w_list[i + 1] = new_w
        return nbr_list, w_list

    def train_for_imputation_with_nbr(self):
        t0 = time.time()
        self._restore("stage2")
        self.save_embeddings()
        min_bin, max_bin = self._distance_bins()
        self.training_data_generator.filter_edges(min_bin, max_bin)
        self.validation_data_generator.filter_edges(min_bin, max_bin)
        self.alpha, self.beta = 1.0, 1e-3
        self.dynamic_pair_ratio = False
        self.use_recon = False
        self.contractive_flag, self.contractive_loss_weight = False, 0.0
        self.pair_ratio = 0.5
        self.cell_neighbor_list, self.cell_neighbor_weight_list = self.get_cell_neighbor(0)
        weighted = []
        for c in range(len(self.chrom_list)):
            mats = []
            for cell in np.arange(self.num_list[0]) + 1:
                mtx = 0
                for nbr, w in zip(self.cell_neighbor_list[cell], self.cell_neighbor_weight_list[cell]):
                    mtx = mtx + w * self.sparse_chrom_list[c][nbr - 1]
                mats.append(csr_matrix(mtx))
            weighted.append(mats)
        self.nbr_builder.set_matrices(weighted)
        optimizer = torch.optim.Adam(self.higashi_model.parameters(), lr=1e-3)
        self.steps = 3
        self.train([optimizer], self.with_nbr_epoch, stage=3)
        self._snapshot("stage3")
        self.timing['stage3'] = time.time() - t0

    # ------------------------------------------------------------- imputation
    def impute_pairs(self):
        """Per chromosome: local ``(i, j)`` bin pairs that get imputed."""
        min_bin, max_bin = self._distance_bins()
        out = {}
        for chrom in self.impute_list:
            j = self.chrom_list.index(chrom)
            sb = generate_binpair(int(self.num_list[j]), int(self.num_list[j + 1]), min_bin, max_bin)
            out[chrom] = (sb, sb - int(self.num_list[j]) - 1)
        return out

    def _impute(self, stage, with_nbr, on_cell: Callable[[int, Dict[str, np.ndarray]], None], cells=None):
        t0 = time.time()
        model = copy.deepcopy(self._stage_models[stage])
        model.eval()
        model.only_model = True
        emb = model.encode1.static_nn
        emb.off_hook()
        gs = model.encode1.dynamic_nn
        with torch.no_grad():
            gs.start_fix()
        gs.off_hook_mode = True
        pairs = self.impute_pairs()
        bin_ids, route_nn_list, big, big_chrom, slices = [], [], [], [], {}
        start = 0
        for chrom in self.impute_list:
            j = self.chrom_list.index(chrom)
            route_nn_list.append(j + 1)
            bin_ids.append(np.arange(self.num_list[j], self.num_list[j + 1]) + 1)
            sb = pairs[chrom][0]
            big.append(np.concatenate([np.ones((len(sb), 1), dtype='int'), sb], axis=-1))
            big_chrom.append(np.ones(len(sb), dtype='int') * j)
            slices[chrom] = (start, start + len(sb))
            start += len(sb)
        big = np.concatenate(big, axis=0)
        big_chrom = np.concatenate(big_chrom, axis=0)
        activation = torch.sigmoid if self.mode == 'classification' else F.softplus
        from sklearn.preprocessing import normalize
        cells = range(self.cell_num) if cells is None else cells
        with torch.no_grad():
            for i in cells:
                cell = i + 1
                cell_chrom_list = []
                for chrom in self.impute_list:
                    c = self.chrom_list.index(chrom)
                    if with_nbr:
                        mtx = 0
                        for nbr, w in zip(self.cell_neighbor_list[cell], self.cell_neighbor_weight_list[cell]):
                            mtx = mtx + w * self.sparse_chrom_list[c][nbr - 1]
                    else:
                        mtx = self.sparse_chrom_list[c][cell - 1]
                    adj = moving_avg(csr_matrix(mtx), self.local_transfer_range)
                    adj = csr_matrix(adj)
                    adj.data = np.log1p(adj.data)
                    adj = normalize(adj, norm='l1', axis=1).astype('float32')
                    a = adj.tocoo()
                    cell_chrom_list.append([torch.from_numpy(np.asarray([a.row, a.col]).astype(np.int64)),
                                            torch.from_numpy(a.data), adj.shape])
                gs.fix_cell2(cell, bin_ids, cell_chrom_list, route_nn_list=route_nn_list)
                big[:, 0] = cell
                proba = model.predict(big, big_chrom, batch_size=int(1e5), activation=activation).reshape((-1))
                on_cell(i, {chrom: proba[s:e] for chrom, (s, e) in slices.items()})
        self.timing['impute_' + stage] = time.time() - t0

    def impute_no_nbr(self, on_cell, cells=None):
        """Imputation with the stage-2 model (each cell's own contact graph)."""
        self._impute("stage2", False, on_cell, cells)

    def impute_with_nbr(self, on_cell, cells=None):
        """Imputation with the stage-3 model (k-nearest-cell weighted graph)."""
        self._impute("stage3", True, on_cell, cells)

    def fetch_cell_embeddings(self):
        return self.cell_embeddings
