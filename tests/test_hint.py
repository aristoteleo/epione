"""HINT-ATAC port (``epione.tl._hint``).

The synthetic tests run anywhere: they check the fast paths against literal
transcriptions of the RGT 1.0.2 code they replace, and that the public
functions run end to end on a small BAM.

The parity tests compare with official ``rgt-hint`` / ``rgt-motifanalysis``
output on the TOBIAS chr4 fixture (gates in
``tests/parity/hint_atac/MANIFEST.md``). They skip unless
``EPIONE_HINT_PARITY_DIR`` (reference outputs) and ``EPIONE_HINT_DATA``
(fixture) are set.
"""
from __future__ import annotations

import os
import sys
from math import ceil, floor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pysam = pytest.importorskip("pysam")

from epione.tl import _hint as H  # noqa: E402

RNG = np.random.default_rng(7)
CHROM_LEN = 6_000


# ---------------------------------------------------------------------------
# Fixtures: genome, paired-end BAM, bias tables
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def genome(tmp_path_factory):
    d = tmp_path_factory.mktemp("hint_genome")
    fa = d / "g.fa"
    s = np.array(list(RNG.choice(list("ACGT"), size=CHROM_LEN)))
    s[2000:2010] = "N"
    s[3000:3050] = np.char.lower(s[3000:3050])
    seq = "".join(s)
    with open(fa, "w") as fh:
        fh.write(">chrS\n" + "\n".join(seq[i:i + 60] for i in range(0, len(seq), 60)) + "\n")
    pysam.faidx(str(fa))
    return fa


@pytest.fixture(scope="module")
def bam(tmp_path_factory):
    d = tmp_path_factory.mktemp("hint_bam")
    raw, out = d / "raw.bam", d / "reads.bam"
    header = {"HD": {"VN": "1.6"}, "SQ": [{"SN": "chrS", "LN": CHROM_LEN}]}
    with pysam.AlignmentFile(str(raw), "wb", header=header) as fh:
        n = 0
        for _ in range(4000):
            # enrich fragments around two "peaks", with a protected core
            centre = RNG.choice([1500, 4000])
            s = int(RNG.normal(centre, 150))
            L = int(RNG.choice([60, 90, 120, 200, 250, 320]))
            if abs(s + L // 2 - centre) < 8 and RNG.random() < 0.8:
                continue
            e = s + L
            if s < 1 or e >= CHROM_LEN - 1:
                continue
            for mate in (1, 2):
                a = pysam.AlignedSegment()
                a.query_name = f"f{n}"
                rl = min(40, L)
                a.query_sequence = "A" * rl
                a.query_qualities = pysam.qualitystring_to_array("I" * rl)
                a.reference_id = 0
                a.cigarstring = f"{rl}M"
                a.mapping_quality = 60
                if mate == 1:
                    a.flag = 99
                    a.reference_start = s
                    a.next_reference_start = e - rl
                    a.template_length = L
                else:
                    a.flag = 147
                    a.reference_start = e - rl
                    a.next_reference_start = s
                    a.template_length = -L
                a.next_reference_id = 0
                fh.write(a)
            n += 1
    pysam.sort("-o", str(out), str(raw))
    pysam.index(str(out))
    return out


def _random_table(k, seed):
    rng = np.random.default_rng(seed)
    import itertools
    keys = ["".join(p) for p in itertools.product("ACGT", repeat=k)]
    rng.shuffle(keys)
    return {key: float(np.round(rng.lognormal(0, 0.7), 6)) for key in keys}


@pytest.fixture(scope="module")
def bias(genome):
    return H.HintBias(genome, _random_table(8, 1), _random_table(8, 2))


PEAKS = [("chrS", 1300, 1700), ("chrS", 3800, 4200), ("chrS", 2950, 3110)]


# ---------------------------------------------------------------------------
# Literal transcriptions of RGT 1.0.2 (signalProcessing.py), for comparison
# ---------------------------------------------------------------------------

def _seqsum(xs):
    # Python 3.7's sum() of floats: plain left-to-right addition
    s = 0
    for x in xs:
        s = s + x
    return s


def _revcomp(s):
    return s.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def rgt_bias_correction_atac(bam, fasta, tables, chrName, start, end, forward_shift, reverse_shift,
                             paired=False, min_length=None, max_length=None):
    window = 50
    fBiasDict, rBiasDict = tables
    k_nb = len(list(fBiasDict.keys())[0])
    p1, p2 = start, end
    p1_w, p2_w = p1 - (window // 2), p2 + (window // 2)
    if paired:
        lo = hi = int(k_nb / 2.)
    else:
        lo, hi = int(floor(k_nb / 2.)), int(ceil(k_nb / 2.))
    p1_wk, p2_wk = p1_w - lo, p2_w + hi
    nf = [0.0] * (p2_w - p1_w)
    nr = [0.0] * (p2_w - p1_w)
    for read in bam.fetch(chrName, p1_w, p2_w):
        if read.is_unmapped:
            continue
        tl = abs(read.template_length)
        if min_length is not None and not tl > min_length:
            continue
        if max_length is not None and not tl <= max_length:
            continue
        if not read.is_reverse:
            cut_site = read.pos + forward_shift
            if p1_w <= cut_site < p2_w:
                nf[cut_site - p1_w] += 1.0
        else:
            cut_site = read.aend + reverse_shift - 1
            if p1_w <= cut_site < p2_w:
                nr[cut_site - p1_w] += 1.0
    Nf, Nr = [], []
    fSum, rSum = _seqsum(nf[:window]), _seqsum(nr[:window])
    fLast, rLast = nf[0], nr[0]
    for i in range((window // 2), len(nf) - (window // 2)):
        Nf.append(fSum)
        Nr.append(rSum)
        fSum -= fLast
        fSum += nf[i + (window // 2)]
        fLast = nf[i - (window // 2) + 1]
        rSum -= rLast
        rSum += nr[i + (window // 2)]
        rLast = nr[i - (window // 2) + 1]
    currStr = str(fasta.fetch(chrName, p1_wk, p2_wk - 1)).upper()
    currRevComp = _revcomp(str(fasta.fetch(chrName, p1_wk + 1, p2_wk)).upper())
    af, ar = [], []
    for i in range(hi, len(currStr) - lo + 1):
        fseq = currStr[i - lo:i + hi]
        rseq = currRevComp[len(currStr) - hi - i:len(currStr) + lo - i]
        af.append(fBiasDict.get(fseq, 1.0))
        ar.append(rBiasDict.get(rseq, 1.0))
    fSum, rSum = _seqsum(af[:window]), _seqsum(ar[:window])
    fLast, rLast = af[0], ar[0]
    bf, br = [], []
    for i in range((window // 2), len(af) - (window // 2)):
        bf.append(Nf[i - (window // 2)] * (af[i] / fSum))
        br.append(Nr[i - (window // 2)] * (ar[i] / rSum))
        fSum -= fLast
        fSum += af[i + (window // 2)]
        fLast = af[i - (window // 2) + 1]
        rSum -= rLast
        rSum += ar[i + (window // 2)]
        rLast = ar[i - (window // 2) + 1]
    return (nf, nr), (af, ar), (bf, br)


# ---------------------------------------------------------------------------
# Synthetic tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("paired,lo,hi", [(False, None, None), (True, None, 145), (True, 145, None)])
def test_bias_correction_matches_rgt_transcription(bam, genome, bias, paired, lo, hi):
    b = pysam.AlignmentFile(str(bam))
    fa = pysam.FastaFile(str(genome))
    tables = (bias.forward.table, bias.reverse.table)
    for chrom, s, e in PEAKS:
        (nf, nr), (af, ar), (bf, br) = rgt_bias_correction_atac(
            b, fa, tables, chrom, s, e, 5, -5, paired=paired, min_length=lo, max_length=hi)
        mf, mr = H._cut_counts(b, chrom, s - 25, e + 25, 5, -5, lo, hi)
        assert np.array_equal(mf, nf) and np.array_equal(mr, nr)
        kf, kr = bias.kmer_bias(chrom, s - 25, e + 25, paired=paired, fasta=fa)
        assert np.array_equal(kf, af) and np.array_equal(kr, ar)
        assert np.array_equal(H._correct(mf, kf), bf)            # bit-identical
        assert np.array_equal(H._correct(mr, kr), br)


def test_numba_and_python_kernels_agree(bias, genome):
    n = RNG.poisson(2, 500).astype(float)
    a = RNG.lognormal(0, 1, 500)
    assert np.array_equal(H._correct_py(n, a, 50), H._correct(n, a, 50))


def test_kmer_slow_path_odd_k(genome):
    t = _random_table(5, 3)
    hb = H.HintBias(genome, t, _random_table(5, 4))
    fa = pysam.FastaFile(str(genome))
    for paired in (False, True):
        af, ar = hb.kmer_bias("chrS", 1000, 1100, paired=paired)
        lo, hi = hb._halves(paired)
        cur = fa.fetch("chrS", 1000 - lo, 1100 + hi - 1).upper()
        rc = _revcomp(fa.fetch("chrS", 1000 - lo + 1, 1100 + hi).upper())
        L = len(cur)
        exp_f = [t.get(cur[i - lo:i + hi], 1.0) for i in range(hi, L - lo + 1)]
        assert np.array_equal(af, exp_f)
        assert len(ar) == len(af)


def test_bias_track_and_factory(genome, bias):
    from epione.tl import tn5_bias
    hb = tn5_bias(genome, method="hint", table_f=bias.forward.table, table_r=bias.reverse.table)
    assert isinstance(hb, H.HintBias) and hb.shift == (1, -1)
    tr = hb.track("chrS", 1000, 1200)
    f, r = hb.track_stranded("chrS", 1000, 1200)
    assert tr.shape == (200,) and np.allclose(tr, f + r)


def test_bias_table_roundtrip(tmp_path, bias):
    pf, pr = tmp_path / "F.txt", tmp_path / "R.txt"
    for p, t in ((pf, bias.forward.table), (pr, bias.reverse.table)):
        p.write_text("".join(f"{k}\t{v}\n" for k, v in t.items()))
    f, r = H.load_hint_bias_table(pf, pr)
    assert f == bias.forward.table and r == bias.reverse.table


def test_percentile_matches_scipy():
    from scipy.stats import scoreatpercentile
    for n in (1, 2, 7, 50, 401):
        x = RNG.normal(size=n)
        assert H._percentile(x, 98) == scoreatpercentile(x, 98)


def test_viterbi_matches_bruteforce():
    hmm = H.load_hint_hmm("single")
    X = RNG.normal(0.3, 0.3, size=(6, 4))
    st = hmm.decode(X)
    flp = hmm.log_likelihood(X)
    import itertools
    with np.errstate(divide="ignore"):
        ls, lt = np.log(hmm.startprob), np.log(hmm.transmat)
    best, arg = -np.inf, None
    for path in itertools.product(range(hmm.n_states), repeat=X.shape[0]):
        v = ls[path[0]] + flp[0, path[0]] + sum(lt[path[t - 1], path[t]] + flp[t, path[t]]
                                                for t in range(1, len(path)))
        if v > best:
            best, arg = v, path
    assert tuple(st) == arg
    assert np.array_equal(H._viterbi_py(hmm._log_start, hmm._log_trans, flp), st)


def test_hmm_models_load():
    s, p = H.load_hint_hmm("single"), H.load_hint_hmm("paired")
    assert (s.n_states, s.n_features, s.fp_state) == (7, 4, 6)
    assert (p.n_states, p.n_features, p.fp_state) == (9, 8, 8)
    with pytest.raises(ValueError):
        s.decode(np.full((3, 4), np.nan))


def test_calls_from_states():
    st = np.array([0, 6, 6, 1, 6, 6, 6])
    assert H._calls_from_states("c", 100, 107, st, 6, 50) == [("c", 101, 103), ("c", 104, 107)]
    assert H._calls_from_states("c", 100, 107, st, 6, 3) == [("c", 101, 103)]


def test_merge_and_intersect():
    regs = [["c", 0, 5], ["c", 5, 9], ["c", 8, 12], ["c", 20, 22]]
    assert H._merge([list(r) for r in regs]) == [["c", 0, 5], ["c", 5, 12], ["c", 20, 22]]
    assert H._intersect_original([["c", 1, 3], ["c", 30, 31]], [("c", 0, 10)]) == [["c", 1, 3]]


@pytest.mark.parametrize("paired", [False, True])
def test_footprints_end_to_end(tmp_path, bam, bias, paired):
    peaks = pd.DataFrame(PEAKS, columns=["chrom", "start", "end"])
    fp = H.hint_footprints(bam, peaks, bias=bias, paired_end=paired, output=tmp_path / "fp.bed",
                           return_states=True, verbose=False)
    assert list(fp.columns) == ["chrom", "start", "end", "name", "tag_count"]
    assert fp.attrs["n_peaks"] == 3 and fp.attrs["paired_end"] is paired
    assert (tmp_path / "fp.info").exists()
    lines = (tmp_path / "fp.bed").read_text().splitlines()
    assert len(lines) == len(fp)
    if len(fp):
        assert lines[0].split("\t")[5] == "."
    # parallel run gives the same calls
    fp2 = H.hint_footprints(bam, peaks, bias=bias, paired_end=paired, n_jobs=2, verbose=False)
    pd.testing.assert_frame_equal(fp.reset_index(drop=True), fp2.reset_index(drop=True))
    sig = H.hint_signal(bam, "chrS", 1300, 1700, bias=bias, paired_end=paired)
    assert sig["X"].shape == (400, 8 if paired else 4)


def test_paired_autodetect(bam):
    assert H._detect_paired(bam) is True


def test_match_and_differential(tmp_path, bam, genome, bias):
    motifs = {"M1.AAA": np.array([[8, 0, 8, 0], [0, 8, 0, 0], [0, 0, 0, 8], [0, 0, 0, 0]]),
              "M2.CG": np.array([[0, 0, 1, 0, 0], [5, 0, 1, 5, 0], [0, 5, 1, 0, 5], [0, 0, 1, 0, 0]])}
    regions = pd.DataFrame(PEAKS, columns=["chrom", "start", "end"])
    mp = H.hint_match_motifs(regions, motifs, genome, fpr=0.05, output=tmp_path / "m.bed")
    assert set(mp["name"]) <= set(motifs) and len(mp) > 0
    assert (mp["end"] - mp["start"]).isin([4, 5]).all()
    res = H.hint_differential([mp, mp], [bam, bam], genome, conditions=["a", "b"], bias=bias,
                              verbose=False)
    st = res.stats
    assert {"Num", "Protection_Score_a", "TC_b", "TF_Activity", "Z_score", "P_values"} <= set(st.columns)
    # the same BAM twice: zero activity difference
    assert np.allclose(st["TF_Activity"], 0)
    assert res.profiles[st.index[0]].shape == (2, 200)
    raw = H.hint_differential(mp, [bam, bam], genome, conditions=["a", "b"],
                              bias_correction=False, verbose=False)
    assert set(raw.stats.index) == set(st.index)
    from epione.tl._hint_plot import plot_hint_differential, plot_hint_profile
    plot_hint_differential(res, show=False)
    plot_hint_profile(res, list(st.index[:2]), genome=genome, show=False)


# ---------------------------------------------------------------------------
# Parity with official rgt-hint (skips without the reference outputs)
# ---------------------------------------------------------------------------

REF = os.environ.get("EPIONE_HINT_PARITY_DIR")
DATA = os.environ.get("EPIONE_HINT_DATA")
needs_ref = pytest.mark.skipif(not (REF and DATA and Path(REF).is_dir() and Path(DATA).is_dir()),
                               reason="EPIONE_HINT_PARITY_DIR / EPIONE_HINT_DATA not set")


@pytest.fixture(scope="module")
def parity():
    sys.path.insert(0, str(Path(__file__).parent / "parity" / "hint_atac"))
    import parity_check
    return parity_check


@needs_ref
def test_parity_tracks_h1_to_h5(parity):
    r = {}
    parity.gate_h1_to_h5(r)
    assert all(r[f"H{i}_pass"] for i in range(1, 6)), r


@needs_ref
def test_parity_footprints_h6_h7(parity, tmp_path, monkeypatch):
    monkeypatch.setattr(parity, "WORK", tmp_path)
    r = {}
    parity.gate_h6_h7(r, n_jobs=4)
    assert r["H6_pass"] and r["H7_pass"], r


@needs_ref
def test_parity_matching_h8(parity, tmp_path, monkeypatch):
    monkeypatch.setattr(parity, "WORK", tmp_path)
    r = {}
    parity.gate_h8(r)
    assert r["H8_pass"], r


@needs_ref
def test_parity_differential_h9(parity):
    r = {}
    parity.gate_h9(r, n_jobs=4)
    assert all(r[f"{g}_pass"] for g in ("H9a", "H9b", "H9c", "H9d", "H9e", "H9f")), r
