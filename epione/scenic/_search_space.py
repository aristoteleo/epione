"""SCENIC+ region-to-gene search space (``scenicplus.data_wrangling.gene_search_space``).

Vectorised reimplementation of ``get_search_space`` for the SCENIC+ default
(``use_gene_boundaries=False``): every region whose interval overlaps a gene body
extended by ``upstream[1]`` bp upstream / ``downstream[1]`` bp downstream is paired
with that gene, and the signed distance from the region midpoint to the gene body is
reported.  Rows, their order and the ``Distance`` strings match scenicplus 1.0
(including its quirks: the distance column holds a bracketed list, midpoints are
rounded half-to-even, rows come grouped by natural-sorted chromosome / strand and then
sorted by gene and region name).
"""
from __future__ import annotations

import re
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = ["get_search_space", "gene_annotation_from_gtf"]


def gene_annotation_from_gtf(gtf, transcript_type: str | None = "protein_coding",
                             assembled_only: bool = True) -> pd.DataFrame:
    """Gene annotation table for :func:`get_search_space` from an Ensembl GTF.

    One row per (gene, TSS): ``Chromosome`` (UCSC style), gene-level ``Start``/``End`` (1-based,
    as biomart reports them), ``Strand``, ``Gene`` (gene name), ``Transcription_Start_Site``
    and ``Transcript_type`` — the columns of the biomart query scenicplus uses.
    """
    import gzip

    def attr(s, key):
        i = s.find(key + ' "')
        if i < 0:
            return None
        j = s.find('"', i + len(key) + 2)
        return s[i + len(key) + 2:j]

    opener = gzip.open if str(gtf).endswith(".gz") else open
    genes, tx = {}, []
    with opener(gtf, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if p[2] == "gene":
                genes[attr(p[8], "gene_id")] = (int(p[3]), int(p[4]))
            elif p[2] == "transcript":
                tss = int(p[3]) if p[6] == "+" else int(p[4])
                tx.append((p[0], attr(p[8], "gene_id"), p[6], attr(p[8], "gene_name"), tss,
                           attr(p[8], "transcript_biotype")))
    df = pd.DataFrame(tx, columns=["Chromosome", "gene_id", "Strand", "Gene",
                                   "Transcription_Start_Site", "Transcript_type"])
    df["Start"] = [genes[g][0] for g in df.gene_id]
    df["End"] = [genes[g][1] for g in df.gene_id]
    annot = df[["Chromosome", "Start", "End", "Strand", "Gene", "Transcription_Start_Site",
                "Transcript_type"]].drop_duplicates()
    if transcript_type is not None:
        annot = annot[annot.Transcript_type == transcript_type]
    if assembled_only:
        annot = annot[annot.Chromosome.isin([str(i) for i in range(1, 23)] + ["X", "Y", "MT"])]
    annot = annot.assign(Chromosome=["chrM" if c == "MT" else (c if c.startswith("chr") else "chr" + c)
                                     for c in annot.Chromosome])
    return annot.sort_values(["Chromosome", "Start", "Transcription_Start_Site", "Gene"],
                             kind="mergesort").reset_index(drop=True)


def _natural_key(s: str):
    parts = re.split(r"(\d+)", str(s))
    return tuple(int(p) if i % 2 else p for i, p in enumerate(parts))


def _regions_to_frame(regions: Iterable[str]) -> pd.DataFrame:
    names = [r for r in regions if ":" in r]
    chrom, rest = zip(*(r.split(":", 1) for r in names)) if names else ((), ())
    se = [r.split("-", 1) for r in rest]
    return pd.DataFrame({
        "Chromosome": np.asarray(chrom, dtype=object),
        "Start": np.fromiter((int(a) for a, _ in se), dtype=np.int64, count=len(se)),
        "End": np.fromiter((int(b) for _, b in se), dtype=np.int64, count=len(se)),
    })


def _pyranges_order(df: pd.DataFrame, stranded: bool = True) -> pd.DataFrame:
    """Row order of ``pr.PyRanges(df).df``: natural-sorted (Chromosome, Strand) groups,
    original order inside each group."""
    keys = ["Chromosome", "Strand"] if stranded else ["Chromosome"]
    groups = df.groupby(keys, sort=False, observed=True).indices
    order = sorted(groups, key=lambda k: tuple(_natural_key(x) for x in (k if isinstance(k, tuple) else (k,))))
    idx = np.concatenate([groups[k] for k in order]) if order else np.empty(0, dtype=np.int64)
    return df.iloc[idx]


def _overlap_pairs(r_start, r_end, g_start, g_end):
    """All (region, gene) index pairs with half-open overlap on one chromosome."""
    order = np.argsort(g_start, kind="stable")
    gs, ge = g_start[order], g_end[order]
    max_len = int((ge - gs).max()) if len(gs) else 0
    # candidate genes for region i: gs < r_end[i] and gs > r_start[i] - max_len
    lo = np.searchsorted(gs, r_start - max_len, side="right")
    hi = np.searchsorted(gs, r_end, side="left")
    counts = hi - lo
    ri = np.repeat(np.arange(len(r_start)), counts)
    offs = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    gi = np.repeat(lo, counts) + offs
    keep = ge[gi] > r_start[ri]
    return ri[keep], order[gi[keep]]


def get_search_space(
    regions: Iterable[str],
    genes: Iterable[str],
    gene_annotation: pd.DataFrame,
    *,
    upstream: Sequence[int] = (1000, 150000),
    downstream: Sequence[int] = (1000, 150000),
    extend_tss: Sequence[int] = (10, 10),
    use_gene_boundaries: bool = False,
    remove_promoters: bool = False,
) -> pd.DataFrame:
    """Pair regions with the genes whose (extended) body they fall in.

    Parameters
    ----------
    regions
        Region names ``chrom:start-end`` (e.g. the accessibility matrix columns).
    genes
        Gene names to keep (e.g. the expression matrix columns).
    gene_annotation
        Table with ``Chromosome, Start, End, Strand, Gene, Transcription_Start_Site``
        (one row per transcript, gene-level ``Start``/``End``), as produced for
        scenicplus from Ensembl.
    upstream, downstream
        ``(min, max)`` bp around the gene body; with ``use_gene_boundaries=False`` only the
        maximum is used, as in scenicplus.
    extend_tss
        bp around each TSS defining promoters (used by ``remove_promoters``).
    use_gene_boundaries
        Not supported (scenicplus' non-default mode that stops at neighbouring genes).
    remove_promoters
        Drop distal pairs (``Distance != 0``) whose region overlaps any promoter.

    Returns
    -------
    DataFrame with ``Name``, ``Gene`` and ``Distance`` (scenicplus' bracketed string).
    """
    if use_gene_boundaries:
        raise NotImplementedError("use_gene_boundaries=True is not implemented")
    genes = set(genes)
    ga = gene_annotation[gene_annotation["Gene"].isin(genes)].copy()
    ga["Gene"] = ga["Gene"].fillna("na")
    ga = ga.reset_index(drop=True)
    ga["Chromosome"] = ga["Chromosome"].astype(str)
    ga["Strand"] = ga["Strand"].astype(str)

    annot = _pyranges_order(ga[["Chromosome", "Start", "End", "Strand", "Gene"]])
    annot = annot[~annot.duplicated(["Chromosome", "Start", "End", "Strand"])]
    annot = annot.drop_duplicates(subset="Gene", keep="first")
    pos = (annot["Strand"] == "+").to_numpy()
    start = annot["Start"].to_numpy(np.int64)
    end = annot["End"].to_numpy(np.int64)
    ext_start = np.where(pos, start - upstream[1], start - downstream[1]).astype(np.int32).astype(np.int64)
    ext_end = np.where(pos, end + downstream[1], end + upstream[1]).astype(np.int32).astype(np.int64)
    annot = annot.assign(ExtStart=ext_start, ExtEnd=ext_end)
    annot = _pyranges_order(pd.concat([annot[pos], annot[~pos]]))
    annot = annot[~annot.duplicated(["Chromosome", "ExtStart", "ExtEnd", "Strand"])]

    reg = _regions_to_frame(regions)
    out = []
    for chrom, gsub in annot.groupby("Chromosome", sort=False):
        rsub = reg[reg["Chromosome"] == chrom]
        if rsub.empty:
            continue
        # pyranges' join (NCLS, ALL_POSITIVE_ORIENTATION) stores an interval with a negative
        # start as the reverse-orientation interval [-end, -start): genes whose extended
        # window runs past the chromosome start only catch regions with start < -ExtStart.
        es, ee = gsub["ExtStart"].to_numpy(), gsub["ExtEnd"].to_numpy()
        es, ee = np.where(es < 0, -ee, es), np.where(es < 0, -es, ee)
        ri, gi = _overlap_pairs(rsub["Start"].to_numpy(), rsub["End"].to_numpy(), es, ee)
        if len(ri) == 0:
            continue
        rs, re_ = rsub["Start"].to_numpy()[ri], rsub["End"].to_numpy()[ri]
        width = np.abs(re_ - rs).astype(np.int32)
        mid = np.round(rs + width / 2).astype(np.int32).astype(np.int64)
        gstart = gsub["Start"].to_numpy(np.int64)[gi]
        gend = gsub["End"].to_numpy(np.int64)[gi]
        strand = gsub["Strand"].to_numpy()[gi]
        sd = gstart - mid
        ed = gend - mid
        min_d = np.minimum(np.abs(sd), np.abs(ed))
        loc = np.zeros(len(ri), dtype=np.int64)
        loc[(sd > 0) & (ed > 0)] = 1
        loc[(sd < 0) & (ed < 0)] = -1
        loc *= np.where(strand == "+", 1, -1)
        out.append(pd.DataFrame({
            "Chromosome": chrom,
            "Start": rs, "End": re_,
            "Name": [f"{chrom}:{a}-{b}" for a, b in zip(rs, re_)],
            "Strand": strand,
            "Gene": gsub["Gene"].to_numpy()[gi],
            "Distance": loc * min_d.astype(np.int32),
        }))
    if not out:
        return pd.DataFrame({"Name": [], "Gene": [], "Distance": []})
    df = pd.concat(out, ignore_index=True)
    if remove_promoters:
        tss = ga["Transcription_Start_Site"].to_numpy(np.int64).astype(np.int32).astype(np.int64)
        tpos = (ga["Strand"] == "+").to_numpy()
        p_start = np.where(tpos, tss - extend_tss[0], tss - extend_tss[1])
        p_end = np.where(tpos, tss + extend_tss[1], tss + extend_tss[0])
        distal = df["Distance"].to_numpy() != 0
        hit = np.zeros(len(df), dtype=bool)
        for chrom in df["Chromosome"].unique():
            m = (df["Chromosome"] == chrom).to_numpy() & distal
            pm = (ga["Chromosome"] == chrom).to_numpy()
            if not m.any() or not pm.any():
                continue
            idx = np.flatnonzero(m)
            ri, _ = _overlap_pairs(df["Start"].to_numpy()[idx], df["End"].to_numpy()[idx],
                                   p_start[pm], p_end[pm])
            hit[idx[np.unique(ri)]] = True
        df = df[~hit]
    df = df.drop_duplicates(["Gene", "Name", "Distance"])
    agg = (df.groupby(["Gene", "Name"], sort=True)
             .agg(Chromosome=("Chromosome", "first"), Strand=("Strand", "first"),
                  Distance=("Distance", lambda x: "[" + ", ".join(str(int(v)) for v in x) + "]"))
             .reset_index())
    agg = _pyranges_order(agg)
    return agg[["Name", "Gene", "Distance"]].reset_index(drop=True)
