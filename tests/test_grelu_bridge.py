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
