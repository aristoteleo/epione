"""Exonic masks for the v2 RNA head: gene -> which 32 bp bins of a window are exonic.

The RNA head of :mod:`._v2_decoder` pools the trunk embedding over a gene's exons and reads out one
scalar log-rate per gene. :func:`build_exon_masks` builds those pooling masks once from a GENCODE GFF3
on the genome the corpus was quantified against, and :class:`ExonMasks` answers them per window.

Why exons and not the gene body
-------------------------------
A 3'-biased UMI count measures mature transcript, so the quantity the head predicts is set by the
exons; pooling over introns would average in sequence that contributes nothing to the measured count
and would dilute the signal by the intron fraction, which for a typical human gene is most of the
locus.

Why symbols are the key
-----------------------
The store's ``rna.h5ad`` is symbol-keyed, because scGPT's vocabulary is symbols (the store builder sums
duplicate symbols on the way in). So the masks are keyed by symbol too, and a symbol appearing on more
than one contig keeps only its longest locus -- a symbol that maps to two places cannot be pooled as
one gene.

    from epione.models.aceso import build_exon_masks, ExonMasks
    build_exon_masks("gencode.v41.annotation.gff3.gz", "data/exon_masks")     # one-off, about a minute
    em = ExonMasks("data/exon_masks/exons_hg38.npz", gene_list=store.genes)
    cols, mask = em.window("chr1", 1_000_000, 1_196_608)
"""
from __future__ import annotations

import gzip
import json
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

MAIN_CONTIGS = {f"chr{i}" for i in range(1, 23)} | {"chrX"}


def build_exon_masks(gff, out_dir, *, name: str = "exons_hg38", contigs: Optional[set] = None,
                     log=print) -> str:
    """Collapse the GENCODE exons of a GFF3 to one merged interval set per gene symbol."""
    contigs = MAIN_CONTIGS if contigs is None else set(contigs)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    per = {}                                       # symbol -> {chrom, strand, [(s, e)]}
    n_lines = n_exon = 0
    with gzip.open(gff, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            n_lines += 1
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "exon" or f[0] not in contigs:
                continue
            attr = f[8]
            i = attr.find("gene_name=")
            if i < 0:
                continue
            sym = attr[i + 10:].split(";", 1)[0]
            s, e = int(f[3]) - 1, int(f[4])        # GFF is 1-based inclusive -> 0-based half-open
            d = per.setdefault(sym, {"chrom": f[0], "strand": f[6], "iv": []})
            if d["chrom"] != f[0]:
                # same symbol on two contigs: keep the one with more exonic bp, decided at the end
                d.setdefault("alt", {}).setdefault(f[0], []).append((s, e))
                continue
            d["iv"].append((s, e))
            n_exon += 1
    log(f"  {n_lines:,} annotation lines, {n_exon:,} exons on main contigs, "
        f"{len(per):,} gene symbols  [{time.time() - t0:.0f}s]")

    syms, chroms, strands, starts, ends, offs = [], [], [], [], [], [0]
    n_multi = 0
    for sym, d in per.items():
        iv = d["iv"]
        if "alt" in d:
            n_multi += 1
            best = max([(d["chrom"], iv)] + list(d["alt"].items()),
                       key=lambda kv: sum(e - s for s, e in kv[1]))
            d["chrom"], iv = best
        iv = sorted(iv)
        # merge overlapping exons across transcripts: the mask is "is this bin exonic in ANY isoform",
        # which is the right pooling domain when the head predicts one rate for the whole gene
        merged = []
        for s, e in iv:
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        syms.append(sym)
        chroms.append(d["chrom"])
        strands.append(d["strand"])
        for s, e in merged:
            starts.append(s)
            ends.append(e)
        offs.append(len(starts))
    log(f"  {n_multi:,} symbols appeared on more than one contig; kept the longest locus")

    npz = out_dir / f"{name}.npz"
    np.savez_compressed(
        npz,
        symbols=np.array(syms), chroms=np.array(chroms), strands=np.array(strands),
        starts=np.array(starts, np.int64), ends=np.array(ends, np.int64),
        offsets=np.array(offs, np.int64))
    json.dump(dict(gff=str(gff), n_symbols=len(syms), n_exons=len(starts),
                   n_multi_contig=n_multi, main_contigs=sorted(contigs)),
              open(out_dir / f"{name}.json", "w"), indent=1)
    log(f"  -> {npz}   {len(syms):,} genes, {len(starts):,} merged exons")
    return str(npz)


class ExonMasks:
    """Window -> (gene indices into a caller's gene list, mask ``[G, n_bins]``).

    Built once per process and queried per window. The per-chromosome sort plus ``searchsorted`` keeps a
    query at O(log n + k) rather than scanning 1.5e6 exons for every one of 10,116 windows.
    """

    def __init__(self, npz=None, gene_list: Optional[Sequence[str]] = None, bin_size: int = 32):
        if npz is None:
            from ._sites import site_path
            npz = site_path("exon_masks")
        z = np.load(npz, allow_pickle=False)
        self.sym = z["symbols"].astype(str)
        self.chrom = z["chroms"].astype(str)
        self.strand = z["strands"].astype(str)
        self.starts, self.ends, self.offs = z["starts"], z["ends"], z["offsets"]
        self.bin = bin_size
        # gene span, for the window overlap test
        self.g_start = np.array([self.starts[self.offs[i]:self.offs[i + 1]].min()
                                 for i in range(len(self.sym))])
        self.g_end = np.array([self.ends[self.offs[i]:self.offs[i + 1]].max()
                               for i in range(len(self.sym))])
        # only genes the caller's store actually has a column for; otherwise the head would predict a
        # rate with no target to compare it against
        self.keep = (np.isin(self.sym, np.asarray(gene_list)) if gene_list is not None
                     else np.ones(len(self.sym), bool))
        self.by_chrom = {}
        for c in np.unique(self.chrom):
            idx = np.flatnonzero((self.chrom == c) & self.keep)
            idx = idx[np.argsort(self.g_start[idx])]
            self.by_chrom[c] = idx
        self.col = ({g: i for i, g in enumerate(gene_list)} if gene_list is not None else None)

    def window(self, chrom, start, end):
        """-> (store column indices, mask ``[G, n_bins]`` float32). Empty arrays when no gene overlaps."""
        n_bins = (end - start) // self.bin
        idx = self.by_chrom.get(str(chrom))
        if idx is None or len(idx) == 0:
            return np.zeros(0, np.int64), np.zeros((0, n_bins), np.float32)
        gs, ge = self.g_start[idx], self.g_end[idx]
        lo = np.searchsorted(gs, end)                         # genes starting after the window end
        sel = idx[:lo][ge[:lo] > start]
        if len(sel) == 0:
            return np.zeros(0, np.int64), np.zeros((0, n_bins), np.float32)
        M = np.zeros((len(sel), n_bins), np.float32)
        for k, i in enumerate(sel):
            for s, e in zip(self.starts[self.offs[i]:self.offs[i + 1]],
                            self.ends[self.offs[i]:self.offs[i + 1]]):
                b0 = max(0, (s - start) // self.bin)
                b1 = min(n_bins, -(-(e - start) // self.bin))   # ceil
                if b1 > b0:
                    M[k, b0:b1] = 1.0
        keep = M.sum(1) > 0
        sel, M = sel[keep], M[keep]
        cols = (np.array([self.col[self.sym[i]] for i in sel], np.int64)
                if self.col is not None else sel)
        return cols, M
