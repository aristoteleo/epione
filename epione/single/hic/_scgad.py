"""Single-cell Gene Associating Domain (scGAD) score.

Implements the **scGAD** feature first defined by Shen, Zheng & Keleş
2022 (Bioinformatics) and used by Chang *et al.* 2024 *Nat Biotechnol*
to embed Droplet Hi-C cells into a UMAP. For each ``(gene i, cell j)``
pair, the score is the **sum of imputed contacts** in the
``(gene_body_bins) × (gene_body_bins)`` block of the cell's imputed
contact matrix at a chosen resolution. The result is a much smaller
``cell × gene`` feature matrix (~25 000 features) than the raw
``cell × bin-pair`` flattening (>10 M features at 100 kb mouse genome),
giving PCA + UMAP a vastly better signal-to-noise ratio for fine
celltype discrimination.

A pure-Python implementation: no scHiCluster R/Snakemake stack. Reads
per-cell imputed ``.npz`` files (output of
:func:`epione.single.hic.impute_cells`), uses the cooler bin layout to
map each gene to its overlapping bins, and aggregates.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd


def load_refflat_genes(
    refflat_path: Union[str, Path],
    *,
    chromosomes: Optional[Sequence[str]] = None,
    min_gene_length: int = 1_000,
    deduplicate: bool = True,
) -> pd.DataFrame:
    """Read a UCSC refFlat gene table → BED-style DataFrame.

    Arguments:
        refflat_path: path to ``refFlat.txt`` or ``refFlat.txt.gz`` from
            UCSC (``hgdownload.soe.ucsc.edu/goldenPath/<asm>/database``).
        chromosomes: keep only these chromosomes. Default = all.
        min_gene_length: drop tiny / spurious genes < this many bp.
        deduplicate: keep the longest isoform per ``(gene_name, chrom)``.

    Returns:
        ``DataFrame`` with columns ``gene, chrom, start, end, strand``.
    """
    df = pd.read_csv(refflat_path, sep="\t", header=None,
                     usecols=[0, 2, 3, 4, 5],
                     names=["gene", "chrom", "strand", "start", "end"],
                     dtype={"gene": str, "chrom": str, "strand": str,
                            "start": np.int64, "end": np.int64})
    if chromosomes is not None:
        df = df[df["chrom"].isin(list(chromosomes))]
    df = df[(df["end"] - df["start"]) >= int(min_gene_length)]
    if deduplicate:
        df = (df.sort_values(["gene", "chrom", "start"])
                .assign(_len=lambda x: x["end"] - x["start"])
                .sort_values("_len", ascending=False)
                .drop_duplicates(subset=["gene", "chrom"])
                .drop(columns="_len"))
    return df.reset_index(drop=True)


def _gene_to_chrom_bins(
    genes: pd.DataFrame,
    bins: pd.DataFrame,
    chromosomes: Sequence[str],
    flank_bins: int = 0,
) -> dict:
    """For each gene → (chrom, list[bin_idx_within_chrom]).

    ``bins`` is the cooler bin table (chrom, start, end). We compute the
    bin offset of each chromosome, then translate gene start/end → bin
    index within the chrom. ``flank_bins > 0`` extends the gene body by
    that many bins on each side — useful at 100 kb where most genes
    span only 1 bin (mean mouse gene ≈ 25 kb), making single-pixel
    scGAD too noisy.
    """
    gene_to_bins = {}
    bins = bins.reset_index(drop=True)
    for ch in chromosomes:
        sub = bins[bins["chrom"] == ch]
        if sub.empty:
            continue
        sub = sub.reset_index(drop=True)
        binsize = int(sub["end"].iloc[0] - sub["start"].iloc[0])
        n_chrom_bins = len(sub)
        ch_genes = genes[genes["chrom"] == ch]
        for _, row in ch_genes.iterrows():
            g_start = int(row["start"])
            g_end = int(row["end"])
            i0 = max(0, g_start // binsize - int(flank_bins))
            i1 = min(n_chrom_bins,
                     (g_end + binsize - 1) // binsize + int(flank_bins))
            if i1 <= i0:
                continue
            gene_to_bins[row["gene"]] = (ch, list(range(i0, i1)))
    return gene_to_bins


def _scgad_one_cell(args):
    """Worker: compute scGAD vector for a single cell from its npz."""
    cell_id, npz_path, gene_to_bins, gene_order = args
    try:
        z = np.load(npz_path)
    except Exception:
        return cell_id, None
    cache: dict = {}
    out = np.zeros(len(gene_order), dtype=np.float32)
    for gi, gene in enumerate(gene_order):
        rec = gene_to_bins.get(gene)
        if rec is None:
            continue
        ch, bin_idx = rec
        if ch not in cache:
            if ch not in z.files:
                cache[ch] = None
                continue
            cache[ch] = z[ch]
        P = cache[ch]
        if P is None:
            continue
        # Gene body sub-block sum — works for genes spanning 1+ bins.
        idx = np.asarray(bin_idx, dtype=np.intp)
        out[gi] = float(P[np.ix_(idx, idx)].sum())
    return cell_id, out


def scgad_score(
    adata,
    *,
    refflat_path: Union[str, Path],
    bins: pd.DataFrame,
    chromosomes: Optional[Sequence[str]] = None,
    min_gene_length: int = 1_000,
    flank_bins: int = 0,
    nproc: int = 1,
    progress: bool = True,
) -> pd.DataFrame:
    """Compute the cell × gene scGAD score matrix from imputed npz files.

    Arguments:
        adata: AnnData from :func:`load_cool_collection` after
            :func:`impute_cells` has been run. Reads
            ``adata.uns['hic']['imputed_dir']`` for the per-cell
            ``.npz`` files.
        refflat_path: path to a UCSC ``refFlat.txt(.gz)`` gene table.
        bins: cooler bin table (DataFrame with ``chrom, start, end``)
            — fetched from any of the imputation cools at the same
            resolution as the impute. Pass
            ``cooler.Cooler(cool).bins()[:]``.
        chromosomes: subset; default = all in the impute.
        min_gene_length: drop tiny genes.
        nproc: workers.
        progress: tqdm bar.

    Returns:
        ``DataFrame`` indexed by cell barcode, columns = gene names,
        values = scGAD score (sum of imputed contacts in gene-body block).
    """
    info = adata.uns.get("hic", {})
    imputed_dir = Path(info["imputed_dir"])
    chromosomes = list(chromosomes) if chromosomes is not None else list(
        info.get("impute_params", {}).get("chromosomes",
                                          info.get("chromosomes", []))
    )

    genes = load_refflat_genes(
        refflat_path, chromosomes=chromosomes,
        min_gene_length=min_gene_length, deduplicate=True,
    )
    gene_to_bins = _gene_to_chrom_bins(genes, bins, chromosomes, flank_bins=flank_bins)
    gene_order = [g for g in genes["gene"].tolist() if g in gene_to_bins]

    cell_ids = list(adata.obs_names)
    args_list = [
        (cid, imputed_dir / f"{cid}.npz", gene_to_bins, gene_order)
        for cid in cell_ids
    ]

    pbar = None
    if progress:
        try:
            from tqdm.auto import tqdm
            pbar = tqdm(total=len(args_list), desc="scgad_score")
        except ImportError:
            pass

    rows = {}
    if nproc <= 1:
        for args in args_list:
            cid, vec = _scgad_one_cell(args)
            if vec is not None:
                rows[cid] = vec
            if pbar:
                pbar.update(1)
    else:
        with ProcessPoolExecutor(max_workers=int(nproc)) as ex:
            futures = [ex.submit(_scgad_one_cell, a) for a in args_list]
            for fut in as_completed(futures):
                cid, vec = fut.result()
                if vec is not None:
                    rows[cid] = vec
                if pbar:
                    pbar.update(1)
    if pbar:
        pbar.close()

    df = pd.DataFrame.from_dict(rows, orient="index", columns=gene_order)
    df = df.loc[[c for c in cell_ids if c in df.index]]
    return df
