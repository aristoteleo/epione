"""PRINT / scPrinter integration: exactness against upstream, plus the
parts that need no torch.

The parity tests execute scPrinter's own functions -- lifted verbatim
from its source by AST, written to a temporary module (``torch.jit.script``
needs real source) and imported -- on the same inputs as Epione's, and
require **bit-identical** output: ``np.array_equal``, not a tolerance.

They skip unless ``SCPRINTER_SRC`` points at the ``scprinter`` package
directory of a clone of https://github.com/buenrostrolab/scPrinter and
torch is installed. The remaining tests run in CI.
"""
from __future__ import annotations

import ast
import importlib.util
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

pysam = pytest.importorskip("pysam")

from epione._insertions import InsertionReader, groups_to_matrix  # noqa: E402
from epione.tl import _print as P  # noqa: E402
from epione.tl._tn5_bias import KmerBias, PrintBias  # noqa: E402

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Synthetic genome and fragments
# ---------------------------------------------------------------------------

CHROMS = {"chrA": 12_345, "chrB": 8_002}   # odd sizes: several CNN chunks


@pytest.fixture(scope="module")
def genome(tmp_path_factory):
    d = tmp_path_factory.mktemp("genome")
    fa = d / "g.fa"
    with open(fa, "w") as fh:
        for c, n in CHROMS.items():
            s = np.array(list(RNG.choice(list("ACGT"), size=n)))
            s[3000:3040] = "N"                        # an N run
            s[5000:5100] = np.char.lower(s[5000:5100])  # soft-masked
            seq = "".join(s)
            fh.write(f">{c}\n" + "\n".join(seq[i:i + 60] for i in range(0, n, 60)) + "\n")
    pysam.faidx(str(fa))
    return fa


N_CELLS = 30
BARCODES = [f"BC{i:02d}" for i in range(N_CELLS)]
LABELS = np.array(["T"] * 10 + ["B"] * 10 + ["M"] * 10)


@pytest.fixture(scope="module")
def fragments(tmp_path_factory):
    """Fragments with a cut-depleted 'footprint' at chrA:6000 in group T."""
    recs = []
    for c, n in CHROMS.items():
        for _ in range(9_000):
            s = int(RNG.integers(0, n - 400))
            e = s + int(RNG.integers(30, 350))
            i = int(RNG.integers(0, N_CELLS))
            if c == "chrA" and LABELS[i] == "T" and (5990 <= s <= 6010 or 5990 <= e <= 6010):
                continue
            recs.append((c, s, e, BARCODES[i]))
    recs.sort()
    d = tmp_path_factory.mktemp("frags")
    raw, gz = d / "f.tsv", d / "f.tsv.gz"
    raw.write_text("".join(f"{c}\t{s}\t{e}\t{b}\t1\n" for c, s, e, b in recs))
    pysam.tabix_compress(str(raw), str(gz), force=True)
    pysam.tabix_index(str(gz), preset="bed", force=True, keep_original=True)
    return gz


@pytest.fixture(scope="module")
def adata(fragments):
    ad = pytest.importorskip("anndata")
    a = ad.AnnData(X=np.zeros((N_CELLS, 1), dtype=np.float32),
                   obs=pd.DataFrame({"celltype": LABELS}, index=BARCODES))
    a.uns["files"] = {"fragments": str(fragments)}
    a.uns["reference_sequences"] = dict(CHROMS)
    return a


# ---------------------------------------------------------------------------
# Torch-free tests (run in CI)
# ---------------------------------------------------------------------------

def test_window_sums_geometry():
    """Centre = 2R bases, each flank = 2*(R//2) bases, adjacent."""
    x = RNG.random((2, 300))
    for R in (2, 5, 10, 33):
        left, centre, right = P._window_sums(x, R, R)
        h = int(R / 2)
        for i in (60, 150, 222):
            pad = lambda a, b: x[:, max(a, 0):max(b, 0)].sum(-1)
            np.testing.assert_allclose(centre[:, i], pad(i - R + 1, i + R + 1))
            np.testing.assert_allclose(left[:, i], pad(i - R - 2 * h + 1, i - R + 1))
            np.testing.assert_allclose(right[:, i], pad(i + R + 1, i + R + 2 * h + 1))


def test_window_sums_are_float64():
    """scPrinter's rz_conv pads with float64 zeros; matching that dtype
    promotion is part of matching its numbers."""
    for part in P._window_sums(np.ones((1, 50), np.float32), 4, 4):
        assert part.dtype == np.float64


def test_tiling_matches_scprinter():
    """pyranges.window(10) plus scPrinter's ``End.iloc[-1] += 1``."""
    t = P._tile_sites(1000, 1995, 10)
    assert len(t) == 100
    assert (t["Start"].iloc[:3] == [1000, 1010, 1020]).all()
    assert t["End"].iloc[-2] == 1990 and t["Start"].iloc[-1] == 1990
    assert t["End"].iloc[-1] == 1996          # 1995, widened by one
    try:
        import pyranges as pr
    except ImportError:
        return
    try:
        ref = pr.PyRanges(pd.DataFrame({"Chromosome": ["c"], "Start": [1000],
                                        "End": [1995]})).window(10).df
    except Exception:  # some sorted_nearest builds are broken (py2 `long`)
        return
    np.testing.assert_array_equal(ref["Start"], t["Start"])
    np.testing.assert_array_equal(ref["End"].to_numpy()[:-1], t["End"].to_numpy()[:-1])


def test_group_insertions(fragments):
    r = InsertionReader(fragments, BARCODES, chrom_sizes=CHROMS)
    groups = [np.flatnonzero(LABELS == g) for g in ("T", "B")]
    gi = r.group_insertions("chrA", 1000, 1600, groups)
    full = r.window_insertions("chrA", 1000, 1600).toarray()
    np.testing.assert_array_equal(gi[0], full[groups[0]].sum(0))
    np.testing.assert_array_equal(
        r.group_insertions("chrA", 1000, 1600, groups_to_matrix(groups, N_CELLS)), gi)
    np.testing.assert_array_equal(r.group_insertions("chrA", 1000, 1600)[0], full.sum(0))


def test_cut_convention_follows_the_bias_model(fragments):
    """PRINT bias assumes cuts at (s, e); the hexamer table (s, e-1).
    Reading with the wrong one shifts every end-cut by a base."""
    assert PrintBias.shift == (0, 0)
    assert KmerBias.shift == (0, -1)
    a = InsertionReader(fragments, BARCODES, shift=PrintBias.shift)
    b = InsertionReader(fragments, BARCODES, shift=KmerBias.shift)
    x = a.group_insertions("chrB", 2000, 2600)[0]
    y = b.group_insertions("chrB", 2000, 2600)[0]
    assert x.sum() == pytest.approx(y.sum(), abs=4)   # only edge cuts differ
    assert not np.array_equal(x, y)


def test_kmer_bias_matches_the_inline_lookup(genome):
    """KmerBias is a lift of the lookup get_footprints inlines; it must
    give the same numbers for the same table."""
    from epione.tl._footprint import _codes_to_kmers, _seq_to_codes
    table = RNG.random(4 ** 6) + 0.5
    kb = KmerBias(genome, table)
    fa = pysam.FastaFile(str(genome))
    lo, hi, k = 4000, 4400, 6
    seq = fa.fetch("chrA", lo - 3, hi + 2).upper()
    km = _codes_to_kmers(_seq_to_codes(seq), k)
    want = np.where(km >= 0, table[np.clip(km, 0, None)], 1.0)
    np.testing.assert_allclose(kb.track("chrA", lo, hi), want, rtol=1e-6)
    # near the contig start the sequence is N-padded, not an error
    assert kb.track("chrA", 0, 50).shape == (50,)


def test_binomial_null_runs_without_torch(adata, genome):
    from epione.tl import multi_scale_footprint_region
    table = np.ones(4 ** 6)
    fp = multi_scale_footprint_region(
        adata, [{"chrom": "chrA", "center": 6000}], groupby="celltype",
        genome=genome, bias_table=table, flank=150, scales=[4, 8, 16],
        min_cells_per_group=5, verbose=False)
    assert fp.null == "binomial" and fp.bias_model == "kmer"
    assert fp.score.shape == (3, 3, 301)
    t = fp.groups.index("T")
    # the depleted site scores higher in T than elsewhere in T's window
    assert fp.score[t, 1, 150] > np.median(fp.score[t, 1])


def test_dispersion_null_refuses_a_non_print_bias(adata, genome):
    from epione.tl import multi_scale_footprint_region
    with pytest.raises(ValueError, match="uncalibrated"):
        multi_scale_footprint_region(
            adata, [{"chrom": "chrA", "center": 6000}], groupby="celltype",
            genome=genome, bias_table=np.ones(4 ** 6), null="dispersion",
            min_cells_per_group=5, verbose=False)


def test_smooth_radius_zero_disables_smoothing():
    """Was a division by zero before the scPrinter-aligned post-processing."""
    p = RNG.random((2, 100))
    np.testing.assert_allclose(P._pvalue_to_score(p, 0), -np.log10(p))


# ---------------------------------------------------------------------------
# Exactness against upstream scPrinter
# ---------------------------------------------------------------------------

_SRC = os.environ.get("SCPRINTER_SRC")


def _have_upstream():
    return bool(_SRC) and (pathlib.Path(_SRC) / "footprint.py").exists()


needs_upstream = pytest.mark.skipif(
    not _have_upstream(),
    reason="set SCPRINTER_SRC to the scprinter/ package dir of a scPrinter clone",
)


@pytest.fixture(scope="module")
def upstream(tmp_path_factory):
    """scPrinter's own numeric functions, verbatim, as an importable module."""
    pytest.importorskip("torch")
    pytest.importorskip("pyfaidx")
    src = pathlib.Path(_SRC)
    want = {
        "utils.py": {"rz_conv", "DNA_one_hot", "load_entire_hdf5",
                     "loadDispModel", "loadBindingScoreModel_pt", "strided_lastaxis"},
        "footprint.py": {"predictDispersion_jit", "footprintWindowSum",
                         "footprintScoring", "regionFootprintScore",
                         "multiscale_footprints"},
        "genome.py": {"predict_genome_tn5_bias", "valid_batch_size"},
        "TFBS.py": {"getRegionBindingScore"},
    }
    parts = [
        "import os, warnings, time",
        "import numpy as np, torch, h5py, scipy, scipy.stats",
        "import torch.nn.functional as F",
        "from scipy.ndimage import maximum_filter",
        "from pyfaidx import Fasta",
        "from tqdm.auto import trange",
    ]
    u = (src / "utils.py").read_text().splitlines()
    i = next(k for k, l in enumerate(u) if l.startswith("DNAmapping ="))
    parts.append("\n".join(u[i:i + 6]))
    for fname, names in want.items():
        text = (src / fname).read_text()
        lines = text.splitlines(keepends=True)
        found = set()
        for node in ast.parse(text).body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                first = (node.decorator_list[0].lineno if node.decorator_list
                         else node.lineno) - 1
                parts.append("".join(lines[first:node.end_lineno]))
                found.add(node.name)
        assert found == names, f"{fname}: missing {names - found}"
    mod_path = tmp_path_factory.mktemp("upstream") / "scprinter_ref.py"
    mod_path.write_text("\n\n".join(parts) + "\n")
    spec = importlib.util.spec_from_file_location("scprinter_ref", mod_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["scprinter_ref"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def dispersion():
    try:
        return P.DispersionModel.load()
    except Exception as exc:  # no network
        pytest.skip(f"cannot fetch dispersion model: {exc}")


@needs_upstream
def test_cnn_bias_bit_identical(upstream, genome, tmp_path):
    """Region-wise PrintBias == scPrinter's genome-wide chunked inference,
    including both contig ends and regions spanning chunk boundaries."""
    import h5py
    ref_h5 = tmp_path / "bias.h5"
    model = P.fetch_print_model("tn5_bias")
    upstream.predict_genome_tn5_bias(str(genome), str(ref_h5), tn5_model=model)
    pb = PrintBias(genome)
    with h5py.File(ref_h5) as f:
        for c, n in CHROMS.items():
            ref = np.array(f[c])
            assert ref.shape == (n,)
            for s, e in [(0, 400), (n - 400, n), (950, 2100), (2990, 3060),
                         (4980, 5120), (0, n)]:
                got = pb.track(c, s, e)
                assert np.array_equal(got, ref[s:e]), (c, s, e, np.abs(got - ref[s:e]).max())


@needs_upstream
def test_dispersion_model_loads_identically(upstream, dispersion):
    ref = upstream.loadDispModel(dispersion.path)
    for r in (2, 10, 57, 100):
        for k in ("featureMean", "featureSD", "targetMean", "targetSD"):
            assert np.array_equal(ref[str(r)][k], dispersion[r][k])
        for a, b in zip(ref[str(r)]["modelWeights"], dispersion[r]["weights"]):
            assert np.array_equal(a.numpy(), b)


@pytest.fixture(scope="module")
def tracks(genome):
    """A realistic bias (the CNN) and Poisson cuts with a depleted centre."""
    pytest.importorskip("torch")
    bias = PrintBias(genome).track("chrA", 5500, 6500)
    lam = bias / bias.mean() * 3.0
    ins = RNG.poisson(np.tile(lam, (4, 1))).astype(np.float32)
    ins[0, 480:520] = 0                       # a footprint in group 0
    ins[2] = 0                                # an empty group
    return ins, bias


@needs_upstream
def test_pvalues_bit_identical(upstream, dispersion, tracks):
    ins, bias = tracks
    ref_models = upstream.loadDispModel(dispersion.path)
    for r in P.PRINT_SCALES:
        want = upstream.footprintScoring(ins, bias, ref_models[str(r)], r, r)
        got = P.print_footprint_pvalues(ins, bias, r, dispersion)
        assert np.array_equal(got, want), (r, np.nanmax(np.abs(got - want)))


@needs_upstream
@pytest.mark.parametrize("smooth", [None, 5, 0])
def test_multiscale_score_bit_identical(upstream, dispersion, tracks, smooth):
    ins, bias = tracks
    ref_models = upstream.loadDispModel(dispersion.path)
    want = upstream.multiscale_footprints(ins, bias, ref_models,
                                          modes=np.arange(2, 101),
                                          smoothRadius=smooth)
    got = P.print_footprint_score(ins, bias, dispersion,
                                  smooth_radius="auto" if smooth is None else smooth)
    assert got.shape == want.shape
    assert np.array_equal(got, want), np.abs(got - want).max()


@needs_upstream
def test_tf_binding_bit_identical(upstream, dispersion, tracks):
    """Motif sites on both strands, including ones too close to an edge."""
    ins, bias = tracks
    region_start = 5500
    sites = pd.DataFrame({
        "Start": [5550, 5650, 5800, 5995, 6100, 6380, 6450],
        "End":   [5560, 5662, 5812, 6005, 6110, 6392, 6460],
        "Strand": ["+", "-", "+", "-", "+", "-", "+"],
        "Score": 1.0,
    })
    ref_model = upstream.loadBindingScoreModel_pt(P.fetch_print_model("tf_binding"))
    ref_disp = upstream.loadDispModel(dispersion.path)
    disp_list = [ref_disp[str(s)] for s in ref_model["scales"]]

    class _Region:  # stands in for the pyranges object; only .df is read
        df = pd.DataFrame({"Start": [region_start], "End": [region_start + len(bias)]})

    import torch
    # scPrinter's get_binding_score runs this under torch.no_grad()
    # (tools.py, `with warnings.catch_warnings(), torch.no_grad():`).
    with torch.no_grad():
        want = upstream.getRegionBindingScore(
            "r", ins, bias, _Region(), disp_list, ref_model, sites=sites.copy())
    net, scales, _ = P._load_tfbs_model()
    rel, keep, got = P._region_binding_score(
        ins, bias, region_start, sites.copy(), scales, dispersion, net)
    assert np.array_equal(rel[keep], want["position"])
    assert np.array_equal(keep, want["siteFilter"])
    assert np.array_equal(got, want["BindingScore"]), np.abs(got - want["BindingScore"]).max()


@needs_upstream
def test_end_to_end_region_matches_upstream(upstream, dispersion, adata, genome):
    """The public function, from fragments: same insertions + bias in,
    scPrinter's multiscale_footprints on those must equal ours."""
    from epione.tl import multi_scale_footprint_region
    fp = multi_scale_footprint_region(
        adata, [{"chrom": "chrA", "center": 6000}], groupby="celltype",
        bias=PrintBias(genome), flank=300, min_cells_per_group=5,
        dispersion=dispersion, verbose=False)
    assert fp.null == "dispersion" and fp.bias_model == "print"
    ref = upstream.multiscale_footprints(
        fp.insertions.astype(np.float64), fp.bias.astype(np.float64),
        upstream.loadDispModel(dispersion.path), modes=fp.scales,
        smoothRadius=None)
    assert np.array_equal(fp.score, ref.astype(np.float32))
    t = fp.groups.index("T")
    assert fp.score[t, :10, 280:320].max() > fp.score[fp.groups.index("B"), :10, 280:320].max()


def test_numpy_fallback_agrees_to_float32(dispersion, tracks, monkeypatch):
    """Without torch the MLP runs in numpy: not bit-identical, but within
    float32 rounding."""
    ins, bias = tracks
    exact = P.print_footprint_pvalues(ins, bias, 20, dispersion)
    import builtins
    real_import = builtins.__import__

    def no_torch(name, *a, **k):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("hidden for this test")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    approx = P.print_footprint_pvalues(ins, bias, 20, dispersion)
    np.testing.assert_allclose(approx, exact, rtol=1e-4, atol=1e-6)


def test_tf_binding_score_public_api(adata, genome, dispersion):
    """From fragments: tiled scores, consistent with the region kernel."""
    pytest.importorskip("torch")
    from epione.tl import tf_binding_score
    res = tf_binding_score(
        adata, pd.DataFrame({"chrom": ["chrA"], "start": [5500], "end": [6500]}),
        groupby="celltype", bias=PrintBias(genome), dispersion=dispersion,
        min_cells_per_group=5, verbose=False)
    assert res.score.shape == (3, len(res.sites))
    assert ((res.score >= 0) & (res.score <= 1)).all()
    # tiles every 10 bp, minus the context_radius edges
    assert res.sites["position"].min() > 100 and res.sites["position"].max() <= 900
    long = res.to_frame()
    assert len(long) == res.score.size and set(long["group"]) == set(res.groups)
    # Same numbers as the kernel on the same tracks
    tr = res.tracks["chrA:5500-6500"]
    net, scales, _ = P._load_tfbs_model()
    _, keep, sc = P._region_binding_score(
        tr["insertions"], tr["bias"], 5500, P._tile_sites(5500, 6500, 10),
        scales, dispersion, net)
    assert np.array_equal(sc, res.score)


def test_plots_render(adata, genome, dispersion):
    pytest.importorskip("torch")
    import matplotlib
    matplotlib.use("Agg")
    from epione.tl import multi_scale_footprint_region, tf_binding_score
    from epione.pl import plot_multi_scale_footprint_region, plot_tf_binding_score
    fp = multi_scale_footprint_region(
        adata, [{"chrom": "chrA", "center": 6000}], groupby="celltype",
        bias=PrintBias(genome), dispersion=dispersion, flank=200,
        min_cells_per_group=5, verbose=False)
    fig, axes = plot_multi_scale_footprint_region(fp, show=False)
    assert len(axes) == 3 and len(fig.track_axes) == 3
    fig, axes = plot_multi_scale_footprint_region(fp, tracks=False, zoom=100, show=False)
    assert len(fig.track_axes) == 0
    res = tf_binding_score(
        adata, pd.DataFrame({"chrom": ["chrA"], "start": [5500], "end": [6500]}),
        groupby="celltype", bias=PrintBias(genome), dispersion=dispersion,
        min_cells_per_group=5, verbose=False)
    fig, axes = plot_tf_binding_score(res, groups=["T", "B"], show=False)
    assert len(axes) == 3
