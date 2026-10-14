"""Tests for the Higashi port (:mod:`epione.single.hic._higashi`) and the
``method="higashi"`` paths of :func:`impute_cells` / :func:`embedding`.

Numerical parity against the official implementation is established by the
metric-level gates in ``tests/parity/higashi`` (GPU, real data). These are
fast CPU checks of the plumbing and of the invariants of the port.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

import epione  # noqa: E402,F401
import epione.utils  # noqa: E402,F401  (import order)
from epione.single.hic._higashi import Higashi, process_contacts  # noqa: E402
from epione.single.hic._higashi._sampler import ContactIndex, NegativeSampler  # noqa: E402

HERE = Path(__file__).parent


@pytest.fixture(autouse=True, scope="module")
def _one_torch_thread():
    # tiny CPU models: intra-op threading only adds contention
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def _contacts(n_cells=24, n_chrom=2, n_bins=30, seed=0):
    """Synthetic binned contacts with a distance decay; two groups of cells
    differ in their decay so the embedding has something to find."""
    rng = np.random.default_rng(seed)
    rows = []
    res = 100_000
    for cell in range(n_cells):
        scale = 2.0 if cell < n_cells // 2 else 8.0
        for c in range(n_chrom):
            n = 300
            b1 = rng.integers(0, n_bins, n)
            d = np.minimum(rng.exponential(scale, n).astype(int), n_bins - 1)
            b2 = np.minimum(b1 + d, n_bins - 1)
            for x, y in zip(b1, b2):
                rows.append((cell, f"c{c}", int(x) * res, f"c{c}", int(y) * res, 1))
    df = pd.DataFrame(rows, columns=["cell_id", "chrom1", "pos1", "chrom2", "pos2", "count"])
    sizes = {f"c{c}": n_bins * res for c in range(n_chrom)}
    return df, sizes, res


def test_process_contacts_shapes_and_determinism():
    df, sizes, res = _contacts()
    a = process_contacts(df, sizes, list(sizes), res, random_state=3)
    b = process_contacts(df, sizes, list(sizes), res, random_state=3)
    assert list(a.num) == [24, 30, 30]
    assert a.chrom_start_end.tolist() == [[0, 30], [30, 60]]
    for i in range(2):
        np.testing.assert_array_equal(a.train_data[i], b.train_data[i])
        np.testing.assert_allclose(a.cell_feats[i], b.cell_feats[i])
        tr, te = len(a.train_data[i]), len(a.test_data[i])
        assert tr == int(0.85 * (tr + te))
        # hyperedges are (cell, bin, bin), 1-based, bins offset by n_cells, distance >= 2
        e = a.train_data[i]
        assert e[:, 0].min() >= 1 and e[:, 0].max() <= 24
        assert (e[:, 2] - e[:, 1] >= 2).all()
        assert e[:, 1].min() > 24
    # the depth-normalised graphs are symmetric, off-diagonal except empty rows
    g = a.sparse_gcn[0][0].toarray()
    np.testing.assert_allclose(g, g.T, rtol=1e-6)
    assert a.bin_feats[0].shape == (30, 30)


def test_negative_sampler_respects_upstream_rules():
    df, sizes, res = _contacts()
    d = process_contacts(df, sizes, list(sizes), res, random_state=0)
    num_list = np.cumsum(d.num)
    index = ContactIndex(d.sparse_gcn, num_list)
    rng = np.random.default_rng(0)
    max_bin = int(np.max(d.num[1:]))
    sampler = NegativeSampler(index, d.start_end_dict, max_bin, 5, rng)
    x = d.train_data[0][:200]
    chrom = d.train_chrom[0][:200]
    for steps, ratio in ((1, 0.0), (2, 0.5)):
        sampler.steps, sampler.pair_ratio = steps, ratio
        neg, neg_chrom = sampler.generate(x, chrom)
        assert len(neg) > 0
        dist = neg[:, 2] - neg[:, 1]
        assert ((dist > 1) & (dist < max_bin)).all()
        assert (neg[:, 0] >= 1).all() and (neg[:, 0] <= d.num[0]).all()
        # every accepted negative has an empty (clipped) 3x3 window in its cell
        assert not index.check_nonzero(neg, neg_chrom).any()
        # the positives themselves are occupied
        assert index.check_nonzero(x.astype(np.int64), chrom).mean() > 0.95


def test_state_dict_keys_match_upstream():
    """Module attribute names mirror upstream, so reference checkpoints load."""
    df, sizes, res = _contacts(n_cells=40)
    d = process_contacts(df, sizes, list(sizes), res, random_state=0)
    m = Higashi(d, dimensions=16, seed=0, embedding_epoch=0, no_nbr_epoch=0, with_nbr_epoch=0, device="cpu",
                verbose=False)
    m.prep_model()
    keys = set(m.higashi_model.state_dict())
    expected = set(json.loads((HERE / "parity" / "higashi" / "upstream_state_dict_keys.json").read_text())["stage1"])
    # the reference has 23 bin node types (20 chromosomes + cells); here 2 chromosomes
    norm = lambda ks: {k for k in ks if not any(f".{i}." in k for i in range(3, 30))}
    assert norm(keys) == norm(expected)


def test_three_stage_pipeline_cpu():
    df, sizes, res = _contacts(n_cells=40)
    d = process_contacts(df, sizes, list(sizes), res, random_state=0)
    m = Higashi(d, dimensions=16, seed=0, embedding_epoch=2, no_nbr_epoch=1, with_nbr_epoch=1, device="cpu",
                update_num_per_training_epoch=3, update_num_per_eval_epoch=1, neighbor_num=3, verbose=False)
    m.prep_model()
    m.train_for_embeddings()
    emb = m.fetch_cell_embeddings()
    assert emb.shape == (40, 16) and np.isfinite(emb).all()
    m.train_for_imputation_nbr_0()
    got = {}
    m.impute_no_nbr(lambda i, v: got.__setitem__(i, v), cells=[0, 5])
    pairs = m.impute_pairs()
    assert set(got) == {0, 5}
    assert got[0]["c0"].shape == (len(pairs["c0"][1]),)
    assert (got[0]["c0"] > 0).all()  # softplus output
    m.train_for_imputation_with_nbr()
    assert len(m.cell_neighbor_list[1]) == 4  # self + 3 neighbours
    got5 = {}
    m.impute_with_nbr(lambda i, v: got5.__setitem__(i, v), cells=[0])
    assert np.isfinite(got5[0]["c1"]).all()
    assert [h["stage"] for h in m.history] == [1, 1, 2, 3]


def test_impute_cells_and_embedding_higashi(tmp_path):
    pytest.importorskip("cooler")
    from test_single_hic import _build_collection
    cool_paths, cell_ids, obs = _build_collection(tmp_path, n_per_group=4)
    adata = epione.single.hic.load_cool_collection(cool_paths, cell_ids=cell_ids, obs=obs)
    params = dict(dimensions=4, embedding_epoch=1, no_nbr_epoch=1, with_nbr_epoch=1, neighbor_num=2)
    import epione.single.hic._higashi._model as hm
    orig = hm.Higashi.__init__

    def small(self, *a, **k):  # keep the CPU test short: 2 batches per epoch
        orig(self, *a, **k)
        self.update_num_per_training_epoch, self.update_num_per_eval_epoch = 2, 1
    hm.Higashi.__init__ = small
    try:
        out = epione.single.hic.impute_cells(adata, tmp_path / "hig", method="higashi", device="cpu",
                                             higashi_params=params, progress=False)
        z = np.load(out / f"{cell_ids[0]}.npz")
        assert set(z.files) == {"chr_a", "chr_b"}
        m = z["chr_a"]
        assert m.shape == (30, 30) and np.allclose(m, m.T) and (np.diag(m) == 0).all()
        assert adata.obsm["X_higashi"].shape == (8, 4)
        assert (out / "higashi_model.pt").exists()
        emb = epione.single.hic.embedding(adata, method="higashi", device="cpu", higashi_params=params)
        np.testing.assert_allclose(emb.obsm["X_higashi"], adata.obsm["X_higashi"])  # reused, not retrained
        # the scHiCluster-style PCA embedding also runs on Higashi's imputation
        pca = epione.single.hic.embedding(adata, n_components=3)
        assert pca.obsm["X_pca"].shape == (8, 3)
    finally:
        hm.Higashi.__init__ = orig


def test_method_validation():
    import anndata as ad
    a = ad.AnnData(np.zeros((2, 0)))
    with pytest.raises(ValueError):
        epione.single.hic.embedding(a, method="nope")
