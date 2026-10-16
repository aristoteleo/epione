"""Equality tests for the parallel fragment pipeline.

``import_fragments`` / ``tsse`` / ``add_tile_matrix`` /
``add_gene_score_matrix`` stream the fragment file with
:mod:`epione.pp._fragment_io` (parallel BGZF inflate + pyarrow parsing).
Every test compares against the original line-by-line implementation
(kept verbatim below as ``_legacy_*``) on a small synthetic fragment file,
with the chunk size forced tiny so chunk / BGZF-block boundaries, comment
lines and the Python fallback parser are all exercised.

Run with:  pytest tests/test_fragment_pipeline.py -q
"""
from collections import defaultdict

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

pysam = pytest.importorskip("pysam")
pytest.importorskip("pyarrow")

from epione.pp import _fragment_io as fio  # noqa: E402
from epione.pp._data import _open_fragment_file  # noqa: E402

CHROM_SIZES = {"chr1": 400_000, "chr2": 250_000, "chrX": 150_000, "chrM": 16_569}


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

def _make_fragments(rng, n_cells=40):
    cells = [f"AAAC{i:04d}-1" for i in range(n_cells)]
    junk = [f"GGGT{i:04d}-1" for i in range(15)]           # not in obs
    rows = []
    for chrom, size in CHROM_SIZES.items():
        n = 1500 if chrom != "chrM" else 200
        starts = np.sort(rng.integers(0, size - 2000, n))
        for s in starts:
            length = int(rng.choice([1, 40, 180, 499, 500, 501, 1200, 3000]))
            bc = cells[rng.integers(n_cells)] if rng.random() < 0.85 else junk[rng.integers(15)]
            cnt = int(rng.integers(1, 6))
            rows.append((chrom, int(s), int(s) + length, bc, cnt))
    # chrom absent from chrom_sizes (must be ignored by every consumer)
    for s in range(1000, 50_000, 700):
        rows.append(("chrUn_x", s, s + 150, cells[s % n_cells], 1))
    # TSS-dense region so the pileup is non-trivial
    for k in range(800):
        s = 100_000 + int(rng.integers(-150, 150))
        rows.append(("chr1", s, s + int(rng.integers(30, 300)),
                     cells[int(rng.integers(n_cells))], 1))
    rows.sort(key=lambda r: (list(CHROM_SIZES).index(r[0]) if r[0] in CHROM_SIZES else 99,
                             r[1], r[2]))
    return cells, rows


@pytest.fixture(scope="module")
def frag_files(tmp_path_factory):
    rng = np.random.default_rng(7)
    cells, rows = _make_fragments(rng)
    d = tmp_path_factory.mktemp("frags")

    def write(path, messy):
        with open(path, "w") as fh:
            fh.write("# id=synthetic\n# pipeline=test\n")
            for i, (c, s, e, b, n) in enumerate(rows):
                if i == 777:
                    fh.write(f"{c}\t{s}\t{e}\t{b}\t.\n")       # non-digit count -> 1
                elif i == 2000:
                    fh.write(f"{c}\t{s}\t{e}\t{b}\n")          # 4 columns -> count 1
                elif messy and i == 3100:
                    fh.write("garbage line\n")                 # < 4 columns
                elif messy and i == 4200:
                    fh.write(f"{c}\tNA\t{e}\t{b}\t1\n")        # non-integer start
                elif messy and i == 4300:
                    fh.write(f"{c}\t {s}\t{e}\t{b}\t {n}\n")    # spaces
                elif messy and i == 4400:
                    fh.write(f"{c}\t{s}\t{e}\t{b}\t-{n}\n")     # negative count -> 1
                elif messy and i == 4500:
                    fh.write(f"#{c}\t{s}\t{e}\t{b}\t{n}\n")     # comment mid-file
                elif messy and i == 4600:
                    fh.write(f"{c}\t{s}\t{e}\t{b}\t{n}\r\n")   # CRLF
                else:
                    fh.write(f"{c}\t{s}\t{e}\t{b}\t{n}\n")

    import gzip
    plain = d / "fragments.tsv"
    write(plain, messy=False)
    gz = pysam.tabix_index(str(plain), preset="bed", force=True, keep_original=True)
    messy = d / "messy.tsv"
    write(messy, messy=True)
    messy_bgzf = d / "messy.tsv.gz"
    pysam.tabix_compress(str(messy), str(messy_bgzf), force=True)
    messy_gz = d / "messy_plaingzip.tsv.gz"
    with open(messy, "rb") as src, gzip.open(messy_gz, "wb") as dst:
        dst.write(src.read())
    return {"cells": cells, "bgzf": gz, "messy_plain": str(messy),
            "messy_bgzf": str(messy_bgzf), "messy_gzip": str(messy_gz)}


@pytest.fixture(params=[True, "disk", False], ids=["store-mem", "store-disk", "stream"])
def store(request):
    return request.param


@pytest.fixture(autouse=True)
def tiny_chunks(monkeypatch):
    # A few KB per task -> many tasks, most boundaries inside BGZF blocks.
    monkeypatch.setattr(fio, "_CHUNK_BYTES", 3000)


def _genes():
    return pd.DataFrame({
        "gene_name": ["g1", "g2", "g3", "g4", "g5", "g6", "g7"],
        "chrom": ["chr1", "chr1", "chr1", "chr2", "chr2", "chrX", "chrM"],
        "start": [100_001, 150_000, 300_000, 20_000, 120_000, 60_000, 3_000],
        "end": [120_000, 151_000, 380_000, 90_000, 121_000, 61_000, 4_000],
        "strand": ["+", "-", "+", "-", "+", "+", "+"],
    })


# ---------------------------------------------------------------------------
# Legacy (pre-parallel) reference implementations
# ---------------------------------------------------------------------------

def _legacy_qc(path, chrom_sizes, chrM=("chrM", "M", "chrMT", "MT")):
    chrM_set = set(chrM); valid = set(chrom_sizes)
    n_frag = defaultdict(int); n_uniq = defaultdict(int); n_mito = defaultdict(int)
    for chrom, start, end, bc, cnt in _open_fragment_file(path):
        if chrom not in valid and chrom not in chrM_set:
            continue
        if chrom in chrM_set:
            n_mito[bc] += 1
        else:
            n_frag[bc] += 1
            n_uniq[bc] += cnt
    barcodes = sorted(set(n_frag) | set(n_mito))
    return (barcodes, np.asarray([n_frag[b] for b in barcodes], dtype=np.int64),
            np.asarray([n_uniq[b] for b in barcodes], dtype=np.int64),
            np.asarray([n_mito[b] for b in barcodes], dtype=np.int64))


def _legacy_tile(path, barcodes, chrom_sizes, bin_size, strategy,
                 chrM=("chrM", "M", "chrMT", "MT")):
    offsets = {}; base = 0
    for chrom, length in chrom_sizes.items():
        offsets[chrom] = base
        base += (int(length) + bin_size - 1) // bin_size
    bc_to_idx = {b: i for i, b in enumerate(barcodes)}
    rows, cols, data = [], [], []
    for chrom, start, end, bc, cnt in _open_fragment_file(path):
        if bc not in bc_to_idx or chrom in chrM or chrom not in offsets:
            continue
        i = bc_to_idx[bc]; b0 = offsets[chrom]
        t1 = b0 + start // bin_size; t2 = b0 + (end - 1) // bin_size
        if strategy == "fragment":
            for t in range(t1, t2 + 1):
                rows.append(i); cols.append(t); data.append(cnt)
        else:
            rows.append(i); cols.append(t1); data.append(cnt)
            if t2 != t1:
                rows.append(i); cols.append(t2); data.append(cnt)
    X = sp.coo_matrix((data, (rows, cols)), shape=(len(barcodes), base),
                      dtype=np.int32).tocsr()
    X.sum_duplicates()
    return X


def _legacy_tss_mx(path, row_keys, tss_by_chrom_list, up, down):
    d = {k: v for k, v in zip(row_keys, range(len(row_keys)))}
    nf = up + down + 1
    mx = np.zeros((len(row_keys), nf), dtype=int)
    fragments = pysam.TabixFile(path, parser=pysam.asBed())
    for chrom, tss in tss_by_chrom_list:
        region_start = tss - up; region_end = tss + down + 1
        for fr in fragments.fetch(chrom, region_start, region_end):
            try:
                rowind = d[fr.name]
            except KeyError:
                continue
            for cut in (fr.start, fr.end - 1):
                col = cut - region_start
                if 0 <= col < nf:
                    mx[rowind, col] += 1
    fragments.close()
    return mx


def _legacy_gene_score(path, obs_names, genes, *, tile_size=500,
                       extend_upstream=(1000, 100000), extend_downstream=(1000, 100000),
                       gene_upstream=5000, gene_downstream=0, use_gene_boundaries=True,
                       gene_scale_factor=5.0, ceiling=4):
    bc_to_idx = {str(b): i for i, b in enumerate(obs_names)}
    n_cells = len(bc_to_idx); n_genes = len(genes)
    gs = genes.copy()
    gs["gene_idx_global"] = np.arange(n_genes, dtype=np.int64)
    gs = gs.sort_values(["chrom", "start", "end"], kind="stable").reset_index(drop=True)
    gene_score = np.zeros((n_cells, n_genes), dtype=np.float64)
    tbx = pysam.TabixFile(path)
    chroms_in_file = set(tbx.contigs)
    for chrom in [c for c in gs["chrom"].unique() if c in chroms_in_file]:
        g_chr = gs[gs["chrom"] == chrom].reset_index(drop=True)
        pos_list, cell_list = [], []
        for line in tbx.fetch(chrom):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            try:
                start = int(parts[1]); end = int(parts[2])
            except ValueError:
                continue
            cell_i = bc_to_idx.get(parts[3])
            if cell_i is None:
                continue
            ts = (start // tile_size) * tile_size
            te = ((end - 1) // tile_size) * tile_size
            pos_list.append(ts); cell_list.append(cell_i)
            if te != ts:
                pos_list.append(te); cell_list.append(cell_i)
        if not pos_list:
            continue
        pos_arr = np.asarray(pos_list, dtype=np.int64)
        cell_arr = np.asarray(cell_list, dtype=np.int64)
        uniq_ins, inv = np.unique(pos_arr, return_inverse=True)
        matGS = sp.coo_matrix((np.ones(len(pos_arr), dtype=np.float32), (inv, cell_arr)),
                              shape=(len(uniq_ins), n_cells)).tocsr()
        matGS.sum_duplicates()
        if ceiling is not None and ceiling > 0:
            np.clip(matGS.data, 0, ceiling, out=matGS.data)
        is_plus = g_chr["strand"].to_numpy() == "+"
        g_start = g_chr["start"].to_numpy(np.int64); g_end = g_chr["end"].to_numpy(np.int64)
        body_lo = np.where(is_plus, g_start - gene_upstream, g_start - gene_downstream).astype(np.int64)
        body_hi = np.where(is_plus, g_end + gene_downstream, g_end + gene_upstream).astype(np.int64)
        width = (body_hi - body_lo).astype(np.float64)
        width = np.where(width > 0, width, 1.0)
        m = 1.0 / width
        m_lo, m_hi = float(m.min()), float(m.max())
        if m_hi > m_lo:
            gene_weight = 1.0 + m * (gene_scale_factor - 1.0) / (m_hi - m_lo)
        else:
            gene_weight = np.ones_like(m)
        gene_weight = np.clip(gene_weight, 1.0, gene_scale_factor).astype(np.float32)
        u_min, u_max = extend_upstream; d_min, d_max = extend_downstream
        five_prime_max = np.where(is_plus, u_max, d_max).astype(np.int64)
        five_prime_min = np.where(is_plus, u_min, d_min).astype(np.int64)
        three_prime_max = np.where(is_plus, d_max, u_max).astype(np.int64)
        three_prime_min = np.where(is_plus, d_min, u_min).astype(np.int64)
        if use_gene_boundaries and len(g_chr) >= 2:
            prev_body_end = np.concatenate([[-np.inf], body_hi[:-1].astype(float)])
            next_body_start = np.concatenate([body_lo[1:].astype(float), [np.inf]])
        else:
            prev_body_end = np.full(len(g_chr), -np.inf)
            next_body_start = np.full(len(g_chr), np.inf)
        dom_lo = np.maximum(prev_body_end + tile_size, (body_lo - five_prime_max).astype(float))
        dom_lo = np.minimum((body_lo - five_prime_min).astype(float), dom_lo)
        dom_hi = np.minimum(next_body_start - tile_size, (body_hi + three_prime_max).astype(float))
        dom_hi = np.maximum((body_hi + three_prime_min).astype(float), dom_hi)
        dom_lo = np.maximum(dom_lo, 1.0).astype(np.int64)
        dom_hi = np.maximum(dom_hi.astype(np.int64), dom_lo + 1)
        tile_centres = uniq_ins + tile_size // 2
        W_rows, W_cols, W_vals = [], [], []
        for g in range(len(g_chr)):
            i0 = np.searchsorted(tile_centres, int(dom_lo[g]), side="left")
            i1 = np.searchsorted(tile_centres, int(dom_hi[g]), side="left")
            if i1 == i0:
                continue
            c = tile_centres[i0:i1]
            bl, bh = int(body_lo[g]), int(body_hi[g])
            x = np.maximum(0, np.maximum(bl - (c + tile_size // 2),
                                         (c - tile_size // 2) - bh)).astype(np.float64)
            w = (np.exp(-x / 5000.0) + np.exp(-1.0)).astype(np.float32) * gene_weight[g]
            W_rows.extend([g] * (i1 - i0)); W_cols.extend(range(i0, i1)); W_vals.extend(w.tolist())
        if not W_rows:
            continue
        W = sp.coo_matrix((np.asarray(W_vals, dtype=np.float32),
                           (np.asarray(W_rows, dtype=np.int64), np.asarray(W_cols, dtype=np.int64))),
                          shape=(len(g_chr), len(uniq_ins))).tocsr()
        chrom_score = (W @ matGS).astype(np.float32).toarray()
        gene_score[:, g_chr["gene_idx_global"].to_numpy(np.int64)] += chrom_score.T
    tbx.close()
    return gene_score.astype(np.float32)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _collect(path, n_jobs):
    out = []

    def fn(ch):
        names = np.asarray(ch.barcode_names.to_pylist(), dtype=object)
        chroms = np.asarray(ch.chrom_names, dtype=object)
        return list(zip(chroms[ch.chrom].tolist(), ch.start.tolist(), ch.end.tolist(),
                        names[ch.barcode].tolist(), ch.count.tolist()))
    for part in fio.map_fragment_chunks(path, fn, n_jobs=n_jobs):
        out.extend(part)
    return out


@pytest.mark.parametrize("kind", ["bgzf", "messy_plain", "messy_bgzf", "messy_gzip"])
@pytest.mark.parametrize("n_jobs", [1, 3])
def test_reader_matches_legacy_parser(frag_files, kind, n_jobs):
    path = frag_files[kind]
    assert _collect(path, n_jobs) == list(_open_fragment_file(path))


def test_reader_chunk_size_invariance(frag_files, monkeypatch):
    ref = list(_open_fragment_file(frag_files["messy_bgzf"]))
    for size in (500, 7000, 1 << 20):
        monkeypatch.setattr(fio, "_CHUNK_BYTES", size)
        assert _collect(frag_files["messy_bgzf"], 2) == ref


def test_sorted_reduce_paths_agree():
    rng = np.random.default_rng(0)
    k = rng.integers(0, 1000, 5000); w = rng.integers(0, 9, 5000)
    ku, wu = fio.sorted_reduce(k, w, 1000)                 # packed
    kr, wr = fio.sorted_reduce(k, w, 1 << 62)               # argsort fallback
    exp_k = np.unique(k)
    exp_w = np.array([w[k == x].sum() for x in exp_k])
    for a, b in ((ku, wu), (kr, wr)):
        np.testing.assert_array_equal(a, exp_k)
        np.testing.assert_array_equal(b, exp_w)


def test_import_fragments_matches_legacy(frag_files):
    import epione as epi
    ad = epi.pp.import_fragments(frag_files["bgzf"], chrom_sizes=CHROM_SIZES,
                                 min_num_fragments=1, n_jobs=3)
    bc, nf, nr, nm = _legacy_qc(frag_files["bgzf"], CHROM_SIZES)
    keep = nf >= 1
    bc = [b for b, k in zip(bc, keep) if k]
    nf, nr, nm = nf[keep], nr[keep], nm[keep]
    assert list(ad.obs_names) == bc
    np.testing.assert_array_equal(ad.obs["n_fragment"].to_numpy(), nf)
    np.testing.assert_array_equal(ad.obs["frac_dup"].to_numpy(),
                                  np.where(nr > 0, 1.0 - nf / nr, 0.0))
    np.testing.assert_array_equal(ad.obs["frac_mito"].to_numpy(),
                                  np.where(nf + nm > 0, nm / (nf + nm), 0.0))


def test_import_fragments_whitelist(frag_files):
    import epione as epi
    wl = frag_files["cells"][::3]
    ad = epi.pp.import_fragments(frag_files["bgzf"], chrom_sizes=CHROM_SIZES,
                                 min_num_fragments=1, whitelist=wl, n_jobs=2)
    full = epi.pp.import_fragments(frag_files["bgzf"], chrom_sizes=CHROM_SIZES,
                                   min_num_fragments=1, n_jobs=2)
    exp = full.obs.loc[[b for b in full.obs_names if b in set(wl)]]
    pd.testing.assert_frame_equal(ad.obs, exp)


@pytest.mark.parametrize("strategy", ["paired-insertion", "insertion", "fragment"])
@pytest.mark.parametrize("bin_size", [500, 5000])
def test_add_tile_matrix_matches_legacy(frag_files, strategy, bin_size, store):
    import epione as epi
    ad = epi.pp.import_fragments(frag_files["bgzf"], chrom_sizes=CHROM_SIZES,
                                 min_num_fragments=1, store_fragments=store)
    assert ("fragment_store" in ad.uns["files"]) == bool(store)
    # obs order != file order, plus a barcode with no fragments at all
    ad = ad[::-1].copy()
    epi.pp.add_tile_matrix(ad, bin_size=bin_size, counting_strategy=strategy, n_jobs=3)
    ref = _legacy_tile(frag_files["bgzf"], list(ad.obs_names), CHROM_SIZES, bin_size, strategy)
    X = ad.X
    assert X.dtype == ref.dtype == np.int32
    assert X.has_canonical_format
    np.testing.assert_array_equal(X.indptr, ref.indptr)
    np.testing.assert_array_equal(X.indices, ref.indices)
    np.testing.assert_array_equal(X.data, ref.data)
    labels = [f"{c}:{s}-{s + bin_size - 1}" for c, L in CHROM_SIZES.items()
              for s in range(0, ((L + bin_size - 1) // bin_size) * bin_size, bin_size)]
    assert list(ad.var_names) == labels


def test_tsse_matches_legacy_pileup(frag_files, store):
    import epione as epi
    ad = epi.pp.import_fragments(frag_files["bgzf"], chrom_sizes=CHROM_SIZES,
                                 min_num_fragments=1, store_fragments=store)
    ad = ad[::2].copy()                       # store must follow cell filtering
    genes = _genes()
    # duplicated TSS rows must be counted twice, like the per-TSS fetch loop
    genes = pd.concat([genes, genes.iloc[[0]]], ignore_index=True)
    feats = genes.rename(columns={"chrom": "Chromosome", "start": "Start",
                                  "end": "End", "strand": "Strand"})
    up = down = 300
    pile = epi.pp.tss_enrichment(ad, features=feats, extend_upstream=up,
                                 extend_downstream=down, n_tss=1_000_000, n_jobs=3)
    tss = np.where(feats["Strand"] == "-", feats["End"] - 1, feats["Start"])
    contigs = set(pysam.TabixFile(frag_files["bgzf"]).contigs)
    order = [(c, int(t)) for c, t in zip(feats["Chromosome"], tss) if c in contigs]
    mx = _legacy_tss_mx(frag_files["bgzf"], list(ad.obs_names), order, up, down)
    assert mx.sum() > 0
    assert pile.X.shape == mx.shape
    flank_means = np.hstack((mx[:, :100], mx[:, -100:])).mean(axis=1)
    flank_means[flank_means == 0] = flank_means.mean()
    centre = mx[:, (mx.shape[1] - 101) // 2: -((mx.shape[1] - 101) // 2)].mean(axis=1)
    np.testing.assert_array_equal(ad.obs["tss_score"].to_numpy(), centre / flank_means)


def test_tss_pileup_stream_exact(frag_files):
    from epione.pp._metric import _tss_pileup_stream
    cells = frag_files["cells"]
    genes = _genes()
    tss = np.where(genes["strand"] == "-", genes["end"] - 1, genes["start"]).astype(np.int64)
    by_chrom = {c: np.sort(tss[genes["chrom"].to_numpy() == c]) for c in genes["chrom"].unique()}
    up, down = 2000, 2000
    mx = np.zeros((len(cells), up + down + 1), dtype=int)
    _tss_pileup_stream(frag_files["bgzf"], mx, cells, by_chrom, up, down, n_jobs=3)
    order = [(c, int(t)) for c, t in zip(genes["chrom"], tss)]
    ref = _legacy_tss_mx(frag_files["bgzf"], cells, order, up, down)
    assert ref.sum() > 0
    np.testing.assert_array_equal(mx, ref)


def test_gene_score_matches_legacy(frag_files, store):
    import epione as epi
    ad = epi.pp.import_fragments(frag_files["bgzf"], chrom_sizes=CHROM_SIZES,
                                 min_num_fragments=1, store_fragments=store)
    ad = ad[np.random.default_rng(1).permutation(ad.n_obs)[:30]].copy()
    genes = _genes()
    epi.tl.add_gene_score_matrix(ad, gene_anno=genes, use_x="auto", n_jobs=3,
                                 verbose=False)
    from epione.tl._gene_score import _normalise_genes
    g = _normalise_genes(genes, exclude_chr=("chrY", "chrM"))
    raw = _legacy_gene_score(frag_files["bgzf"], ad.obs_names, g)
    row_sums = raw.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    ref = (raw * (10000.0 / row_sums)).astype(np.float32)
    assert ref.sum() > 0
    np.testing.assert_array_equal(ad.obsm["gene_score"], ref)


def test_threaded_csr_svd_bit_identical():
    import sklearn.utils.extmath
    from epione.tl._iterative_lsi import _randomized_svd
    rng = np.random.default_rng(3)
    for shape in ((700, 4000), (4000, 700)):          # transpose=True / False
        X = sp.random(*shape, density=0.02, format="csr", dtype=np.float32, random_state=4)
        X.data = np.log1p(rng.random(X.nnz).astype(np.float32) * 50).astype(np.float32)
        ref = sklearn.utils.extmath.randomized_svd(X, n_components=20, random_state=5,
                                                   n_iter="auto")
        for n_jobs in (1, 4):
            got = _randomized_svd(X, n_components=20, random_state=5, n_jobs=n_jobs)
            for a, b in zip(got, ref):
                np.testing.assert_array_equal(a, b)
    # X itself is left untouched (no transposed copy attached to the input)
    assert type(X) is sp.csr_matrix


def test_store_partial_cells_falls_back(frag_files):
    """Cells filtered out at import (min_num_fragments) are not in the
    store; asking for them must fall back to the file, not return zeros."""
    import epione as epi
    full = epi.pp.import_fragments(frag_files["bgzf"], chrom_sizes=CHROM_SIZES,
                                   min_num_fragments=1, store_fragments=False)
    thr = int(np.median(full.obs["n_fragment"]))
    ad = epi.pp.import_fragments(frag_files["bgzf"], chrom_sizes=CHROM_SIZES,
                                 min_num_fragments=thr)
    from epione.pp._fragment_store import get_fragment_store
    st = get_fragment_store(ad)
    assert st is not None and st.covering_row_map(full.obs_names) is None
    full.uns["files"]["fragment_store"] = ad.uns["files"]["fragment_store"]
    epi.pp.add_tile_matrix(full, bin_size=5000)
    ref = _legacy_tile(frag_files["bgzf"], list(full.obs_names), CHROM_SIZES, 5000,
                       "paired-insertion")
    assert (full.X != ref).nnz == 0
    epi.pp.release_fragment_store(ad)
    assert get_fragment_store(ad) is None


def test_lsi_kernels_bit_identical():
    from epione.tl import _lsi_kernels as K
    rng = np.random.default_rng(11)
    X = sp.random(300, 5000, density=0.03, format="csr", dtype=np.float32, random_state=2)
    X.data[:] = 1.0
    cols = np.sort(rng.choice(5000, 1700, replace=False))
    A = K.csr_select_columns(X, cols, n_jobs=3)
    B = X[:, cols]
    for f in ("indptr", "indices", "data"):
        np.testing.assert_array_equal(getattr(A, f), getattr(B, f))
    Q = rng.normal(size=(300, 7)).astype(np.float32)
    np.testing.assert_array_equal(K.csr_t_matmul(X, Q, n_jobs=3), X.T @ Q)
    Q64 = rng.normal(size=(300, 3))
    np.testing.assert_array_equal(K.csr_t_matmul(X, Q64, n_jobs=3), X.T @ Q64)
    codes = rng.integers(0, 4, 300)
    G = K.group_col_sums(X, codes, 4)
    for g in range(4):
        np.testing.assert_array_equal(G[g], np.asarray(X[codes == g].sum(axis=0)).ravel())
    inv = (1.0 / np.maximum(np.asarray(X.sum(1)).ravel(), 1)).astype(np.float32)
    idf = np.log1p(rng.random(5000)).astype(np.float32) + np.float32(0.1)
    got = K.tf_logidf_data(X, inv, idf, 10000.0, n_jobs=3)
    ref = X.multiply(inv[:, None]).multiply(idf).tocsr()
    ref.data = np.log1p(ref.data * 10000.0).astype(np.float32)
    np.testing.assert_array_equal(got.indices, ref.indices)
    np.testing.assert_array_equal(got.data, ref.data)


def test_add_tile_matrix_anndata_state(frag_files):
    """The in-place rebuild keeps obs / uns / obsm and yields the same object
    state as constructing a fresh AnnData."""
    import epione as epi
    from anndata import AnnData
    ad = epi.pp.import_fragments(frag_files["bgzf"], chrom_sizes=CHROM_SIZES,
                                 min_num_fragments=1)
    ad.obsm["emb"] = np.arange(ad.n_obs * 2, dtype=float).reshape(-1, 2)
    obs0 = ad.obs.copy(); uns0 = dict(ad.uns)
    epi.pp.add_tile_matrix(ad, bin_size=5000)
    ref = AnnData(X=ad.X, obs=obs0, var=ad.var, uns=uns0, obsm={"emb": ad.obsm["emb"]})
    pd.testing.assert_frame_equal(ad.obs, ref.obs)
    pd.testing.assert_frame_equal(ad.var, ref.var)
    assert ad.uns.keys() == ref.uns.keys()
    assert list(ad.obsm) == ["emb"] and not ad.is_view


@pytest.mark.parametrize("sample_cells_pre", [None, 150])
def test_iterative_lsi_fast_paths_bit_identical(sample_cells_pre):
    """Data-free binary view + threaded kernels == generic SciPy path
    (binarize=False on the pre-binarised float32 matrix, single thread)."""
    import epione as ep
    from anndata import AnnData
    from epione.tl._iterative_lsi import _binarize
    rng = np.random.default_rng(5)
    n, m = 400, 3000
    X = sp.random(n, m, density=0.05, format="csr", random_state=6, dtype=np.float64)
    X.data = rng.integers(1, 4, X.nnz).astype(np.int32)
    X = X.astype(np.int32)
    groups = np.repeat([0, 1], n // 2)
    X = X + sp.csr_matrix((np.ones(n * 40, np.int32),
                           (np.repeat(np.arange(n), 40),
                            (groups[:, None] * 1500 + rng.integers(0, 300, (n, 40))).ravel())),
                          shape=(n, m))
    X.sum_duplicates()
    obs = {"n_fragment": np.asarray(X.sum(1)).ravel()}
    kw = dict(n_components=10, iterations=2, var_features=800, total_features=2500,
              resolution=0.5, n_neighbors=10, sample_cells_pre=sample_cells_pre,
              depth_col="n_fragment", seed=3, verbose=False)
    a = AnnData(X=X, obs=obs)
    ep.tl.iterative_lsi(a, n_jobs=3, **kw)
    b = AnnData(X=_binarize(X), obs=obs)
    import epione.tl._iterative_lsi as L
    ep.tl.iterative_lsi(b, binarize=False, n_jobs=1, **kw)
    np.testing.assert_array_equal(a.obsm["X_iterative_lsi"], b.obsm["X_iterative_lsi"])
    for k in ("stdev", "feature_sums", "features_final", "kept_dims"):
        np.testing.assert_array_equal(a.uns["X_iterative_lsi"][k], b.uns["X_iterative_lsi"][k])


def test_store_long_fragments(tmp_path):
    """Fragments >= 65535 bp round-trip exactly through the uint16 length
    column + long-fragment side table."""
    import epione as epi
    rows = []
    for i in range(300):
        s = 1000 + 37 * i
        ln = [100, 65534, 65535, 70000, 900000][i % 5]
        rows.append(("chr1", s, s + ln, f"C{i % 7}", 1 + i % 3))
    rows.sort(key=lambda r: (r[1], r[2]))
    plain = tmp_path / "long.tsv"
    plain.write_text("".join(f"{c}\t{s}\t{e}\t{b}\t{n}\n" for c, s, e, b, n in rows))
    gz = pysam.tabix_index(str(plain), preset="bed", force=True, keep_original=True)
    cs = {"chr1": 2_000_000}
    for bin_size, strat in ((500, "paired-insertion"), (5000, "fragment")):
        ad = epi.pp.import_fragments(gz, chrom_sizes=cs, min_num_fragments=1)
        epi.pp.add_tile_matrix(ad, bin_size=bin_size, counting_strategy=strat)
        ref = _legacy_tile(gz, list(ad.obs_names), cs, bin_size, strat)
        assert (ad.X != ref).nnz == 0 and ad.X.nnz == ref.nnz


def test_tile_matrix_chrom_order_differs(frag_files):
    """chrom_sizes order != file order (tile bases decrease along the file):
    the per-cell merge must fall back to a full sort and stay exact."""
    import epione as epi
    cs = dict(reversed(list(CHROM_SIZES.items())))
    ad = epi.pp.import_fragments(frag_files["bgzf"], chrom_sizes=cs, min_num_fragments=1)
    epi.pp.add_tile_matrix(ad, bin_size=500)
    ref = _legacy_tile(frag_files["bgzf"], list(ad.obs_names), cs, 500, "paired-insertion")
    np.testing.assert_array_equal(ad.X.indptr, ref.indptr)
    np.testing.assert_array_equal(ad.X.indices, ref.indices)
    np.testing.assert_array_equal(ad.X.data, ref.data)
