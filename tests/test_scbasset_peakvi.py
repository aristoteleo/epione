"""epione.models.scbasset, epi.tl.scbasset and epi.tl.peakvi (small, CPU)."""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

torch = pytest.importorskip("torch")
pytest.importorskip("pyfaidx")

import anndata as ad  # noqa: E402

import epione as epi  # noqa: E402
from epione.models import scbasset as sb  # noqa: E402


def test_tower_filters_match_basenji():
    assert sb.tower_filters() == [288, 323, 363, 407, 456, 512]


def test_split_matches_official_algorithm():
    tr, te, va = sb.split_train_test_val(1000)
    np.random.seed(10)  # calico/scBasset utils.split_train_test_val, verbatim
    ids = np.arange(1000)
    tv = np.random.choice(ids, int(len(ids) * (1 - 0.9)), replace=False)
    v = np.random.choice(tv, int(len(tv) / 2), replace=False)
    np.testing.assert_array_equal(tr, np.setdiff1d(ids, tv))
    np.testing.assert_array_equal(va, v)
    np.testing.assert_array_equal(te, np.setdiff1d(tv, v))


def test_peak_sequences_window_and_edges(tmp_path):
    rng = np.random.default_rng(0)
    seq = "".join(rng.choice(list("ACGT"), 5000))
    fa = tmp_path / "g.fa"
    fa.write_text(">c1\n" + seq + "\n")
    peaks = pd.DataFrame({"chr": ["c1", "c1", "c1"], "start": [2000, 10, 4900], "end": [2600, 300, 4990]})
    codes, nmask = sb.peak_sequences(peaks, str(fa), return_n_mask=True)
    assert codes.shape == (3, 1344) and codes.dtype == np.int8
    s0 = (2000 + 2600) // 2 - 672
    exp = np.array(["ACGT".index(c) for c in seq[s0:s0 + 1344]])
    np.testing.assert_array_equal(codes[0], exp)
    assert nmask[1, : 672 - 155].all() and not nmask[1, 672 - 155:].any()  # left N padding
    assert nmask[2, -(4945 + 672 - 5000):].all()  # right N padding
    assert codes.min() >= 0 and codes.max() <= 3


def test_peak_table_parses_names():
    var = pd.DataFrame(index=["chr1:100-600", "chr2-5-50"])
    t = sb.peak_table(var)
    assert t.values.tolist() == [["chr1", 100, 600], ["chr2", 5, 50]]


def test_binarize_drops_stored_zeros():
    X = sp.csr_matrix(np.array([[0, 3, 1], [2, 0, 0]], dtype=float))
    X.data[X.data == 1] = 0  # a stored zero
    Y = sb.binarize(X)
    assert Y.shape == (3, 2)
    np.testing.assert_array_equal(Y.toarray(), [[0, 1], [1, 0], [0, 0]])


def test_keras_auc_close_to_exact_auc():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    y = (rng.random((2000, 5)) < 0.3).astype(np.float32)
    p = np.clip(rng.random((2000, 5)) * 0.7 + 0.3 * y, 0, 1).astype(np.float32)
    m = sb.KerasMultiLabelAUC(5)
    m.update(torch.from_numpy(p), torch.from_numpy(y))
    exact = np.mean([roc_auc_score(y[:, j], p[:, j]) for j in range(5)])
    assert abs(m.result() - exact) < 5e-3


def test_keras_h5_roundtrip(tmp_path):
    h5py = pytest.importorskip("h5py")
    net = sb.ScBassetNet(7)
    path = tmp_path / "w.h5"
    convs = [net.stem.conv] + [b.conv for b in net.tower] + [net.pre.conv]
    bns = [net.stem.bn] + [b.bn for b in net.tower] + [net.pre.bn, net.bn_d]
    for b in bns:  # non-trivial running stats
        b.running_mean.normal_(); b.running_var.uniform_(0.5, 2)
    names = []
    with h5py.File(path, "w") as f:
        def put(name, ws):
            g = f.create_group(name)
            g.attrs["weight_names"] = [f"{name}/{k}:0".encode() for k in ws]
            for k, v in ws.items():
                g[f"{name}/{k}:0"] = v
            names.append(name.encode())
        for i, c in enumerate(convs):
            put("conv1d" + (f"_{i}" if i else ""), {"kernel": c.weight.detach().permute(2, 1, 0).numpy()})
        for i, b in enumerate(bns):
            put("batch_normalization" + (f"_{i}" if i else ""),
                {"gamma": b.weight.detach().numpy(), "beta": b.bias.detach().numpy(),
                 "moving_mean": b.running_mean.numpy(), "moving_variance": b.running_var.numpy()})
        put("dense", {"kernel": net.dense.weight.detach().T.numpy()})
        put("dense_1", {"kernel": net.final.weight.detach().T.numpy(), "bias": net.final.bias.detach().numpy() + 1})
        f.attrs["layer_names"] = names
    other = sb.load_keras_h5(sb.ScBassetNet(7), path)
    x = torch.randint(0, 4, (3, 1344))
    net.eval(); other.eval()
    with torch.no_grad():
        np.testing.assert_allclose(other(x).numpy(), net(x).numpy() + 1, atol=1e-5)


def _toy(n_cells=40, n_peaks=160, seed=0):
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 4, (n_peaks, 1344)).astype(np.int8)
    X = sp.csr_matrix((rng.random((n_cells, n_peaks)) < 0.2).astype(np.float32))
    a = ad.AnnData(X)
    a.var_names = [f"chr1:{i * 2000}-{i * 2000 + 500}" for i in range(n_peaks)]
    return a, codes


def test_tl_scbasset_small_cpu(tmp_path):
    a, codes = _toy()
    m = epi.tl.scbasset(a, codes=codes, max_epochs=2, device="cpu", verbose=False)
    assert a.obsm["X_scbasset"].shape == (40, 32)
    assert a.obs["scbasset_intercept"].shape == (40,)
    assert a.uns["scbasset"]["epochs_run"] == 2
    imp = m.impute(device="cpu")
    assert imp.shape == (40, 160) and (imp >= 0).all() and (imp <= 1).all()
    m.save(tmp_path / "m.pt")
    m2 = sb.ScBasset(codes, a.X).load(tmp_path / "m.pt")
    np.testing.assert_array_equal(m2.cell_embedding(), m.cell_embedding())


def test_motif_score_from_local_library(tmp_path):
    a, codes = _toy()
    m = sb.ScBasset(codes, a.X)
    rng = np.random.default_rng(1)
    lib = tmp_path / "lib"
    (lib / "shuffled_peaks_motifs").mkdir(parents=True)
    bg = ["".join(rng.choice(list("ACGT"), 1344)) for _ in range(6)]
    (lib / "shuffled_peaks.fasta").write_text("".join(f">b{i}\n{s}\n" for i, s in enumerate(bg)))
    (lib / "shuffled_peaks_motifs" / "TF1.fasta").write_text(
        "".join(f">m{i}\n{s[:600]}GGGACTTTCC{s[610:]}\n" for i, s in enumerate(bg)))
    s = m.motif_score("TF1", motif_dir=lib, device="cpu")
    assert s.shape == (40,) and abs(s.mean()) < 1e-6 and abs(s.std() - 1) < 1e-6
    assert m.available_motifs(lib) == ["TF1"]


def test_tl_peakvi_small_cpu():
    pytest.importorskip("scvi")
    a, _ = _toy(n_cells=60, n_peaks=80)
    a.X = a.X * 2
    epi.tl.peakvi(a, n_latent=4, max_epochs=2, accelerator="cpu")
    assert a.obsm["X_peakvi"].shape == (60, 4)
