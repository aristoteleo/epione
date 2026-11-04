"""EpiAgent cCRE vocabulary on any genome: liftover + official tokenisation from fragments.

EpiAgent's vocabulary is 1,355,445 hg38 cCREs (token id = 0-based cCRE index + 4). For another
genome the cCREs are lifted over and **keep their index**, so token ids stay the model's own;
cCREs that do not lift are never observed. Tokenisation follows the official recipe
(EpiAgent @ 9c4b790, demo "Data Preprocessing"):

1. cell x cCRE matrix: for each fragment overlapping a cCRE, add the fragment file's 5th column
   (duplicate count); a fragment overlapping k cCREs counts for each;
2. ``epiagent.preprocessing.global_TFIDF`` with the official cCRE document frequency;
3. tokens = nonzero cCREs sorted by descending TF-IDF (``np.argsort(-v)``), +4.

Validated on GSE205117 (mouse): 654,033 / 1,355,445 cCREs lift at minMatch 0.5; 0/50 token
mismatches against the official functions. Requires ``liftOver`` and ``bedtools`` executables.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

N_CCRE = 1355445
TOKEN_OFFSET = 4


def lift_ccre_bed(ccre_bed: str, chain: str, out_bed: str, *, min_match: float = 0.5,
                  len_ratio: tuple = (0.5, 2.0), chrom_regex: str = r"^chr[0-9XY]+$",
                  liftover: str = "liftOver") -> pd.DataFrame:
    """Lift EpiAgent's cCRE.bed to another genome, keeping each cCRE's index (4th column).

    Unique mappings only (liftOver default); mapped length must be within ``len_ratio`` of the
    source length; chromosomes are restricted to ``chrom_regex``.
    """
    src = pd.read_csv(ccre_bed, sep="\t", header=None, usecols=[0, 1, 2])
    if len(src) != N_CCRE:
        raise ValueError(f"expected {N_CCRE} cCREs, got {len(src)}")
    with tempfile.TemporaryDirectory() as td:
        idx_bed, raw, unm = (os.path.join(td, f) for f in ("in.bed", "out.bed", "unmapped.txt"))
        src.assign(idx=np.arange(N_CCRE)).to_csv(idx_bed, sep="\t", header=False, index=False)
        subprocess.run([liftover, f"-minMatch={min_match}", idx_bed, chain, raw, unm], check=True)
        m = pd.read_csv(raw, sep="\t", header=None, names=["chrom", "start", "end", "idx"])
    hl = (src[2] - src[1]).to_numpy()[m["idx"].to_numpy()]
    ml = (m["end"] - m["start"]).to_numpy()
    keep = (ml >= len_ratio[0] * hl) & (ml <= len_ratio[1] * hl) & m["chrom"].str.match(chrom_regex).to_numpy()
    m = m[keep].reset_index(drop=True)
    if not m["idx"].is_unique:
        raise AssertionError("a cCRE mapped twice")
    m.to_csv(out_bed, sep="\t", header=False, index=False)
    return m


def _count_sample(args):
    sample, frag, barcodes, bed, tmpdir, bedtools = args
    bc_file = os.path.join(tmpdir, f"{sample}.barcodes")
    Path(bc_file).write_text("\n".join(barcodes) + "\n")
    cmd = (f"gzip -dc {frag} | grep -v '^#' | awk -F'\\t' 'NR==FNR{{k[$1];next}} ($4 in k)' {bc_file} - | "
           f"{bedtools} intersect -a stdin -b {bed} -wa -wb | cut -f4,5,9")
    p = subprocess.Popen(["bash", "-o", "pipefail", "-c", cmd], stdout=subprocess.PIPE)
    df = pd.read_csv(p.stdout, sep="\t", header=None, names=["bc", "n", "idx"],
                     dtype={"bc": str, "n": np.float32, "idx": np.int64})
    if p.wait() != 0:
        raise RuntimeError(f"fragment/cCRE intersection failed for {sample}")
    return sample, df


def sentences_from_tfidf(T) -> list:
    """Official ``epiagent.tokenization.tokenization`` on CSR rows (same ordering, +4 offset)."""
    T = sp.csr_matrix(T); T.sort_indices()
    out = []
    for i in range(T.shape[0]):
        s, e = T.indptr[i], T.indptr[i + 1]
        idx, v = T.indices[s:e], T.data[s:e]
        k = v != 0
        out.append("[" + ",".join(map(str, idx[k][np.argsort(-v[k])] + TOKEN_OFFSET)) + "]")
    return out


def tokenize_fragments(fragments: Mapping[str, str], barcodes: Mapping[str, Sequence[str]], ccre_bed: str,
                       df_npy: str, *, cell_names: Optional[Mapping[str, Sequence[str]]] = None,
                       threads: int = 8, bedtools: str = "bedtools", verify: int = 0, seed: int = 0):
    """Cells x cCRE counts and EpiAgent cell sentences from per-sample fragment files.

    Parameters
    ----------
    fragments, barcodes
        ``{sample: fragments.tsv.gz}`` and ``{sample: [barcode, ...]}``.
    ccre_bed
        cCRE bed with the cCRE index in column 4 (EpiAgent's own bed with an index column, or the
        output of :func:`lift_ccre_bed`).
    df_npy
        EpiAgent's ``cCRE_document_frequency.npy``.
    cell_names
        Optional output names per sample (default ``f"{sample}#{barcode}"``).
    verify
        Re-run the official ``global_TFIDF`` + ``tokenization`` on this many cells and require
        identical sentences.

    Returns
    -------
    AnnData with ``X`` = counts (CSR, float32) and ``obs`` columns ``sample``, ``cell_sentences``,
    ``n_tokens``, ``n_ccre_counts``.
    """
    import anndata as ad
    from epiagent.preprocessing import global_TFIDF

    names = {s: list(cell_names[s]) if cell_names else [f"{s}#{b}" for b in barcodes[s]] for s in fragments}
    order = [n for s in fragments for n in names[s]]
    row_of = {n: i for i, n in enumerate(order)}
    rows, cols, vals = [], [], []
    with tempfile.TemporaryDirectory() as td, ProcessPoolExecutor(max_workers=threads) as ex:
        jobs = [(s, fragments[s], list(barcodes[s]), ccre_bed, td, bedtools) for s in fragments]
        for s, df in ex.map(_count_sample, jobs):
            name_of = dict(zip(barcodes[s], names[s]))
            rows.append(np.fromiter((row_of[name_of[b]] for b in df["bc"]), np.int64, len(df)))
            cols.append(df["idx"].to_numpy()); vals.append(df["n"].to_numpy())
    X = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(len(order), N_CCRE), dtype=np.float32).tocsr()
    X.sum_duplicates(); X.sort_indices()
    T = sp.csr_matrix(global_TFIDF(ad.AnnData(X=X), np.load(df_npy)).X)
    sentences = sentences_from_tfidf(T)
    if verify:
        from epiagent.tokenization import tokenization
        pick = np.sort(np.random.default_rng(seed).choice(X.shape[0], min(verify, X.shape[0]), replace=False))
        a = global_TFIDF(ad.AnnData(X=X[pick].copy()), np.load(df_npy)); tokenization(a)
        bad = sum(r != sentences[i] for r, i in zip(a.obs["cell_sentences"], pick))
        if bad:
            raise AssertionError(f"{bad}/{len(pick)} sentences differ from the official tokenisation")
    obs = pd.DataFrame(index=order)
    obs["sample"] = [s for s in fragments for _ in names[s]]
    obs["cell_sentences"] = sentences
    obs["n_tokens"] = np.diff(T.indptr)
    obs["n_ccre_counts"] = np.asarray(X.sum(1)).ravel()
    return ad.AnnData(X=X, obs=obs)
