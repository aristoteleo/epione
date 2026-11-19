"""epione.models.grelu. The adapter/pass-through tests need gReLU and are
skipped without it; the dataset builder needs only anndata."""
import numpy as np
import pandas as pd
import pytest


def test_pseudobulk_anndata_sums_groups():
    ad = pytest.importorskip("anndata")
    import scipy.sparse as sp

    from epione.models.grelu import pseudobulk_anndata
    rng = np.random.RandomState(0)
    X = sp.random(40, 6, density=0.4, random_state=0, format="csr",
                  data_rvs=lambda k: rng.randint(1, 4, k))
    obs = pd.DataFrame({"ct": rng.choice(["b", "t"], 40)}, index=[f"c{i}" for i in range(40)])
    var = pd.DataFrame(index=[f"chr2_{10*i}_{10*i+5}" for i in range(6)])
    pb = pseudobulk_anndata(ad.AnnData(X=X, obs=obs, var=var), "ct")
    ref = pd.DataFrame(X.toarray()).groupby(obs["ct"].to_numpy()).sum()
    assert np.array_equal(pb.X, ref.loc[list(pb.obs_names)].to_numpy().astype(np.float32))
    assert list(pb.var.columns) == ["chrom", "start", "end"]
    assert pb.var["chrom"].iloc[0] == "chr2" and pb.var["end"].iloc[1] == 15
    assert pb.obs["n_cells"].sum() == 40


def test_wrap_chrombpnet_heads_exact():
    torch = pytest.importorskip("torch")
    pytest.importorskip("grelu")
    from epione.models.chrombpnet import BPNet, ChromBPNet, glorot_init_
    from epione.models.grelu import HEADS, wrap
    torch.manual_seed(0)
    kw = dict(input_len=256, output_len=100)
    m = ChromBPNet(glorot_init_(BPNet(8, 2, **kw)), glorot_init_(BPNet(8, 2, **kw))).eval()
    x = torch.randint(0, 2, (3, 4, 256)).float()
    with torch.no_grad():
        prof, cnt = m(x)
        assert torch.equal(wrap(m, head="counts")(x), cnt[:, :, None])
        assert torch.equal(wrap(m, head="profile_logits")(x), prof[:, None])
    lm = wrap(m)
    assert lm.data_params["train"]["seq_len"] == 256
    assert lm.data_params["tasks"]["name"] == ["counts"]
    assert set(HEADS["chrombpnet"]) >= {"counts", "profile"}


def test_wrap_rejects_unknown_head():
    torch = pytest.importorskip("torch")
    pytest.importorskip("grelu")
    from epione.models.chrombpnet import BPNet
    from epione.models.grelu import wrap
    with pytest.raises(ValueError, match="head"):
        wrap(BPNet(4, 2, input_len=256, output_len=100), head="coverage")


def test_genome_arg_writes_sizes(tmp_path):
    pytest.importorskip("pysam")
    from epione.models.grelu._bridge import _genome_arg
    fa = tmp_path / "g.fa"
    fa.write_text(">chrA\nACGTACGTAC\n>chrB\nACGT\n")
    out = _genome_arg(str(fa))
    assert out == str(fa)
    assert (tmp_path / "g.fa.sizes").read_text() == "chrA\t10\nchrB\t4\n"


def test_modisco_recovers_planted_motif():
    pytest.importorskip("modiscolite")
    pytest.importorskip("memelite")
    from epione.models import grelu as eg

    rng = np.random.default_rng(0)
    n, L, motif = 400, 200, "TGACTCATCC"
    idx = rng.integers(0, 4, (n, L))
    for i in range(n):
        p = int(rng.integers(40, L - 50))
        idx[i, p:p + len(motif)] = ["ACGT".index(b) for b in motif]
    X = np.zeros((n, 4, L), np.float32)
    X[np.arange(n)[:, None], idx, np.arange(L)[None]] = 1
    hyp = rng.normal(0, 0.01, (n, 4, L)).astype(np.float32)
    for i in range(n):                      # attribution only at the motif, on its bases
        s = "".join("ACGT"[j] for j in idx[i])
        p = s.find(motif)
        hyp[i, idx[i, p:p + 10], np.arange(p, p + 10)] += 1.0
    ref = {"AP1": np.eye(4)[["ACGT".index(b) for b in "TGACTCA"]].T * 0.97 + 0.0075,
           "GATA": np.eye(4)[["ACGT".index(b) for b in "AGATAA"]].T * 0.97 + 0.0075}
    res = eg.modisco(hyp, X, window=None, motifs=ref, n_matches=1)
    assert len(res.table) >= 1
    top = res.table.iloc[0]
    assert top["sign"] == "pos"
    assert "TGACTCA" in top["consensus"] or "TGAGTCA" in top["consensus"]   # either strand
    assert top["match"] == "AP1"
    assert res.contrib[top["pattern"]].shape[1] == 4


def test_modisco_import_keeps_matplotlib_backend():
    pytest.importorskip("modiscolite")
    import matplotlib

    from epione.models.grelu._modisco import _import_tfmodisco

    before = matplotlib.get_backend()
    _import_tfmodisco()
    assert matplotlib.get_backend() == before
