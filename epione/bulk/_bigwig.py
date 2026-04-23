
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc

import pandas as pd
from tqdm import tqdm
import re
import os
from scipy.sparse import csr_matrix
from scipy import sparse as sp
import anndata
from ._getScorePerBigWigBin import getScorePerBin
from ..utils import console
import multiprocessing as mp


_ANCHOR_CHOICES = ('center', '5p', '3p', 'body', 'start', 'end')


def _resolve_window(anchor, start, end, strand, upstream, downstream):
    """Return ``(r_start, r_end, strand_aware)`` for one record + anchor.

    ``strand_aware`` tells the caller whether downstream/upstream should be
    swapped on '-' strand so that bin 0 ends up at the 5' side after the
    automatic ``vals[::-1]`` step.
    """
    s, e = int(start), int(end)
    if anchor == 'center':
        a = (s + e) // 2
        return (max(0, a - upstream), a + downstream, False)
    if anchor == 'start':
        return (max(0, s - upstream), s + downstream, False)
    if anchor == 'end':
        return (max(0, e - upstream), e + downstream, False)
    if anchor == '5p':
        a = e if strand == '-' else s
        if strand == '-':
            return (max(0, a - downstream), a + upstream, True)
        return (max(0, a - upstream), a + downstream, True)
    if anchor == '3p':
        a = s if strand == '-' else e
        if strand == '-':
            return (max(0, a - downstream), a + upstream, True)
        return (max(0, a - upstream), a + downstream, True)
    if anchor == 'body':
        # Scan full region, rescaled to nbins. No flanks.
        return (max(0, s), e, False)
    raise ValueError(
        f"anchor must be one of {_ANCHOR_CHOICES}, got {anchor!r}"
    )


def _compute_region_chrom_task(args):
    """Worker for :meth:`bigwig.compute_matrix_region` parallel path.

    Processes one chromosome's records and returns, per requested anchor,
    a ``(n_records, nbins)`` matrix plus region start / end arrays. One
    pyBigWig handle per-call (each subprocess is short-lived).

    Args dict keys:
        bw_path, nbins, upstream, downstream, anchors (tuple), records
    where each record is ``(gene_id, seqname, start, end, strand)``.
    Output: ``{gene_ids, per_anchor: {anchor: {arr, rs, re}}}``.
    """
    import pyBigWig
    bw_path    = args["bw_path"]
    nbins      = args["nbins"]
    upstream   = args["upstream"]
    downstream = args["downstream"]
    anchors    = args["anchors"]   # tuple of strings
    records    = args["records"]

    bwh = pyBigWig.open(bw_path)
    chrom_lengths = bwh.chroms()

    n = len(records)
    per_anchor = {
        a: dict(
            arr=np.full((n, nbins), np.nan, dtype=np.float32),
            rs=np.zeros(n, dtype=np.int64),
            re=np.zeros(n, dtype=np.int64),
        ) for a in anchors
    }
    gene_ids = []

    for idx, (gene_id, chrom, start, end, strand) in enumerate(records):
        gene_ids.append(gene_id)

        chrom_use = chrom
        if chrom_use not in chrom_lengths:
            if chrom_use.startswith('chr') and chrom_use[3:] in chrom_lengths:
                chrom_use = chrom_use[3:]
            elif ('chr' + chrom_use) in chrom_lengths:
                chrom_use = 'chr' + chrom_use
        if chrom_use not in chrom_lengths:
            continue
        chrom_len = chrom_lengths[chrom_use]

        for anchor in anchors:
            r_start, r_end, _strand_aware = _resolve_window(
                anchor, start, end, strand, upstream, downstream,
            )
            if r_end > chrom_len:
                r_end = chrom_len

            # pyBigWig.stats(nBins=N) makes N internal queries — 130x
            # slower than a single values()+reshape. Fetch per-bp signal
            # and locally bin.
            try:
                raw = bwh.values(chrom_use, r_start, r_end, numpy=True)
            except Exception:
                raw = None
            if raw is None or len(raw) == 0:
                vals = np.zeros(nbins, dtype=np.float32)
            else:
                raw = np.asarray(raw, dtype=np.float32)
                np.nan_to_num(raw, copy=False)
                tgt = nbins * (len(raw) // nbins)
                if tgt == 0:
                    vals = np.full(nbins, raw.mean() if len(raw) else 0.0,
                                    dtype=np.float32)
                else:
                    if tgt < len(raw):
                        raw = raw[:tgt]
                    vals = raw.reshape(nbins, -1).mean(axis=1).astype(np.float32)

            if strand == '-':
                vals = vals[::-1]
            per_anchor[anchor]['arr'][idx, :] = vals
            per_anchor[anchor]['rs'][idx] = r_start
            per_anchor[anchor]['re'][idx] = r_end

    return dict(gene_ids=gene_ids, per_anchor=per_anchor)


def _compute_chrom_task(args):
    """Worker: compute TSS/TES/Body arrays for a set of genes on one chromosome.

    Args dict keys:
      - bw_path: path to bigwig file
      - nbins, upstream, downstream
      - records: list of (gene_id, seqname, start, end, strand)
    Returns dict with arrays and lists in the same order as records.
    """
    import pyBigWig
    bw_path = args["bw_path"]
    nbins = args["nbins"]
    upstream = args["upstream"]
    downstream = args["downstream"]
    records = args["records"]

    bwh = pyBigWig.open(bw_path)
    chrom_lengths = bwh.chroms()

    n = len(records)
    tss_array = np.full((n, nbins), np.nan, dtype=np.float32)
    tes_array = np.full((n, nbins), np.nan, dtype=np.float32)
    body_array = np.full((n, nbins), np.nan, dtype=np.float32)

    gene_ids = []
    tss_region_start_li=[]; tss_region_end_li=[]
    tes_region_start_li=[]; tes_region_end_li=[]
    body_region_start_li=[]; body_region_end_li=[]

    for idx, (gene_id, chrom, start, end, strand) in enumerate(records):
        gene_ids.append(gene_id)
        # normalize chrom name if needed
        chrom_use = chrom
        if chrom_use not in chrom_lengths:
            if chrom_use.startswith('chr') and chrom_use[3:] in chrom_lengths:
                chrom_use = chrom_use[3:]
            elif ('chr' + chrom_use) in chrom_lengths:
                chrom_use = 'chr' + chrom_use

        if strand == '-':
            tss_loc = end
            tes_loc = start
        else:
            tss_loc = start
            tes_loc = end

        tss_region_start = max(0, int(tss_loc - upstream))
        tss_region_end = int(tss_loc + downstream)
        tes_region_start = max(0, int(tes_loc - upstream))
        tes_region_end = int(tes_loc + downstream)
        body_region_start = int(start)
        body_region_end = int(end)

        chrom_len = chrom_lengths.get(chrom_use, None)
        if chrom_len is not None:
            if tss_region_end > chrom_len:
                tss_region_end = chrom_len
            if tes_region_end > chrom_len:
                tes_region_end = chrom_len

        # values() + numpy reshape (vs stats(nBins=N) which does N
        # separate internal queries — ~130x slower).
        def _binned(s, e):
            if e <= s:
                return np.zeros(nbins, dtype=np.float32)
            try:
                raw = bwh.values(chrom_use, s, e, numpy=True)
            except Exception:
                return np.zeros(nbins, dtype=np.float32)
            if raw is None or len(raw) == 0:
                return np.zeros(nbins, dtype=np.float32)
            raw = np.asarray(raw, dtype=np.float32)
            np.nan_to_num(raw, copy=False)
            tgt = nbins * (len(raw) // nbins)
            if tgt == 0:
                return np.full(nbins, raw.mean() if len(raw) else 0.0,
                               dtype=np.float32)
            if tgt < len(raw):
                raw = raw[:tgt]
            return raw.reshape(nbins, -1).mean(axis=1).astype(np.float32)

        tss_vals  = _binned(tss_region_start,  tss_region_end)
        tes_vals  = _binned(tes_region_start,  tes_region_end)
        body_vals = _binned(body_region_start, body_region_end)
        if strand == '-':
            tss_vals = tss_vals[::-1]
            tes_vals = tes_vals[::-1]
            body_vals = body_vals[::-1]

        tss_array[idx, :] = tss_vals
        tes_array[idx, :] = tes_vals
        body_array[idx, :] = body_vals

        tss_region_start_li.append(tss_region_start)
        tss_region_end_li.append(tss_region_end)
        tes_region_start_li.append(tes_region_start)
        tes_region_end_li.append(tes_region_end)
        body_region_start_li.append(body_region_start)
        body_region_end_li.append(body_region_end)

    return dict(
        gene_ids=gene_ids,
        tss=tss_array, tes=tes_array, body=body_array,
        tss_rs=tss_region_start_li, tss_re=tss_region_end_li,
        tes_rs=tes_region_start_li, tes_re=tes_region_end_li,
        body_rs=body_region_start_li, body_re=body_region_end_li,
    )


sc_color=['#7CBB5F','#368650','#A499CC','#5E4D9A','#78C2ED','#866017', '#9F987F','#E0DFED',
 '#EF7B77', '#279AD7','#F0EEF0', '#1F577B', '#A56BA7', '#E0A7C8', '#E069A6', '#941456', '#FCBC10',
 '#EAEFC5', '#01A0A7', '#75C8CC', '#F0D7BC', '#D5B26C', '#D5DA48', '#B6B812', '#9DC3C3', '#A89C92', '#FEE00C', '#FEF2A1']

import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle


def _draw_gene_lanes(
    ax,
    gtf_df,
    chrom,
    start,
    end,
    *,
    color='#000000',
    prefered_name='gene_name',
    jump_symbols=('ENSG',),
    max_label_genes=30,
    fontsize=8,
    n_lanes=2,
    min_gap_frac=0.02,
    draw_exons=True,
    draw_chevrons=True,
):
    """Draw a UCSC-style gene track with greedy lane packing.

    Collapses every GTF ``transcript`` row in ``[start, end]`` on ``chrom``
    to one gene-level record (min start, max end, strand, union of exons),
    then packs genes into ``n_lanes`` left-to-right so labels do not collide.
    Labels are suppressed when the window contains more than
    ``max_label_genes`` genes (otherwise they overlap illegibly on large
    regions such as a 5 Mb paper-UCSC snapshot).

    Arguments:
        ax: matplotlib Axes. Y-axis is set to (-0.1, 1.2) by this function;
            X-axis is set to (start, end).
        gtf_df: DataFrame from :func:`epione.utils.read_gtf` (needs columns
            ``seqname``, ``start``, ``end``, ``strand``, ``feature``, and a
            label column matching ``prefered_name``).
        chrom, start, end: the genomic window.
        color: gene-body / label colour.
        prefered_name: column to use for gene labels (typically
            ``'gene_name'``; falls back to ``'gene_id'``).
        jump_symbols: labels containing any of these substrings get skipped
            (e.g. default drops bare Ensembl IDs like ``ENSG00000...``).
        max_label_genes: if more than this many genes overlap the window,
            gene bodies are drawn but labels are suppressed.
        fontsize, n_lanes, min_gap_frac, draw_exons, draw_chevrons: see the
            rendering section.

    Returns:
        List of gene-level records actually drawn (one dict per gene).
    """
    if gtf_df is None or len(gtf_df) == 0:
        ax.set_xlim(start, end); ax.set_ylim(-0.1, 1.2); ax.axis('off')
        return []

    # Transcript-level rows within the window.
    sub = gtf_df[(gtf_df['seqname'] == chrom)
                 & (gtf_df['end'] > start)
                 & (gtf_df['start'] < end)]
    if 'feature' in sub.columns:
        tx = sub[sub['feature'] == 'transcript']
        ex = sub[sub['feature'] == 'exon'] if draw_exons else None
    else:
        tx = sub
        ex = None

    label_col = prefered_name if prefered_name in tx.columns else 'gene_id'
    # Fall back to gene_id when prefered is missing
    if label_col not in tx.columns:
        ax.set_xlim(start, end); ax.set_ylim(-0.1, 1.2); ax.axis('off')
        return []

    # Aggregate to gene level.
    agg = {'start': 'min', 'end': 'max', 'strand': 'first'}
    g_col = 'gene_name' if 'gene_name' in tx.columns else 'gene_id'
    genes = tx.groupby(g_col, sort=False).agg(agg).reset_index()
    genes = genes[(genes['end'] > start) & (genes['start'] < end)].copy()
    genes['label'] = genes[g_col].astype(str)
    # Drop jump_symbol labels and genes without a real name
    if jump_symbols:
        mask = ~genes['label'].apply(
            lambda s: any(js in s for js in jump_symbols)
        )
        genes = genes[mask]
    if len(genes) == 0:
        ax.set_xlim(start, end); ax.set_ylim(-0.1, 1.2); ax.axis('off')
        return []

    genes = genes.sort_values('start').reset_index(drop=True)

    # Per-gene merged exon intervals (clip to window).
    exons_by_gene = {}
    if ex is not None and len(ex):
        for g, grp in ex.groupby(g_col, sort=False):
            ivs = sorted(
                (max(float(s), start), min(float(e), end))
                for s, e in zip(grp['start'], grp['end'])
                if min(float(e), end) > max(float(s), start)
            )
            merged = []
            for s, e in ivs:
                if merged and s <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], e))
                else:
                    merged.append((s, e))
            exons_by_gene[g] = merged

    # Greedy n-lane packing.
    win = end - start
    min_gap = win * min_gap_frac
    last_ends = [-np.inf] * n_lanes
    lane_of = []
    for _, r in genes.iterrows():
        chosen = None
        for li in range(n_lanes):
            if r['start'] > last_ends[li] + min_gap:
                chosen = li; break
        if chosen is None:
            chosen = int(np.argmin(last_ends))
        last_ends[chosen] = float(r['end'])
        lane_of.append(chosen)
    genes['_lane'] = lane_of

    # Lane y-positions (stacked top-to-bottom).
    y_gap = 0.9 / max(n_lanes, 1)
    lane_y = {li: 0.85 - li * y_gap for li in range(n_lanes)}

    show_labels = len(genes) <= max_label_genes

    H_CDS = min(0.28, y_gap * 0.5)
    records = []
    for _, r in genes.iterrows():
        y = lane_y[int(r['_lane'])]
        g_s = max(float(r['start']), start)
        g_e = min(float(r['end']),   end)
        ax.hlines(y, g_s, g_e, color=color, linewidth=0.8, zorder=1)

        gname = r[g_col]
        merged_ex = exons_by_gene.get(gname, [])
        if merged_ex:
            for s, e in merged_ex:
                if e <= s: continue
                ax.add_patch(Rectangle(
                    (s, y - H_CDS / 2), e - s, H_CDS,
                    facecolor=color, edgecolor=color,
                    linewidth=0.3, zorder=2,
                ))
        else:
            # No exon data: show gene body as a thick bar.
            ax.add_patch(Rectangle(
                (g_s, y - H_CDS / 2), g_e - g_s, H_CDS,
                facecolor=color, edgecolor=color,
                linewidth=0.3, alpha=0.5, zorder=2,
            ))

        if draw_chevrons and (g_e - g_s) > win * 0.02:
            step = max(win * 0.01, 1500.0)
            xs = np.arange(g_s + step * 0.5, g_e, step)
            if len(xs):
                in_exon = np.zeros_like(xs, dtype=bool)
                for s, e in merged_ex:
                    in_exon |= (xs >= s) & (xs <= e)
                xs = xs[~in_exon]
                if len(xs):
                    marker = '>' if str(r.get('strand', '+')) == '+' else '<'
                    ax.scatter(xs, np.full_like(xs, y),
                               marker=marker, s=8, c=color,
                               linewidths=0, zorder=1.5)

        if show_labels:
            cx = 0.5 * (g_s + g_e)
            ax.text(cx, y - H_CDS / 2 - 0.06, str(r['label']),
                    ha='center', va='top',
                    fontsize=fontsize, color=color)

        records.append({
            'gene': gname, 'lane': int(r['_lane']),
            'start': g_s, 'end': g_e,
            'strand': r.get('strand', '+'),
        })

    ax.set_xlim(start, end)
    ax.set_ylim(-0.1, 1.2)
    ax.axis('off')
    return records
n_bins = 100
cmap_name = 'zebrafish_cmap'
colors=['#5478A4','#66B979','#F8F150']
xcmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

class bigwig(object):
    
    def __init__(self,bw_path_dict:str):
        """
        Initialize the bigwig object.

        Arguments:
            bw_path_dict: the dictionary of bigwig file path such as: {'bw_name':'bw_path'}
        
        """
        self.bw_path_dict=bw_path_dict
        self.bw_names=list(bw_path_dict.keys())
        self.bw_tss_scores_dict={}
        self.bw_tes_scores_dict={}
        self.bw_body_scores_dict={}
        self.bw_lens=len(self.bw_names)
        self.gtf=None
        self.scoreperbindata=None
    
    def read(self):
        """
        Read bigwig file from bw_path_dict.
        
        """
        import pyBigWig
        self.bw_dict={}
        with console.group_node("Load bigWig files", last=True, level=1):
            for i, bw_name in enumerate(self.bw_names):
                console.node(f"Loading {bw_name}...", last=(i == len(self.bw_names) - 1), level=2)
                self.bw_dict[bw_name]=pyBigWig.open(self.bw_path_dict[bw_name])
            
    
    def load_gtf(self,gtf_path:str):
        """
        Load gtf file.

        Arguments:
            gtf_path: the path of gtf file.
        """
        with console.group_node('Load GTF file', last=True, level=1):
            console.node('Reading GTF...', last=False, level=2)
            from ..utils import read_gtf
            # Parse only essential attributes and pre-filter for speed
            # Keep attribute string to support plotting labels fallback
            features = read_gtf(
                gtf_path,
                required_attrs=("gene_id", "gene_name"),
                feature_whitelist=("transcript", "exon", "3UTR", "5UTR"),
                chr_prefix="chr",
                keep_attribute=False,
            )
            self.gtf = features
            console.success('GTF loaded', level=2)

    def save_bw_result(self,save_path:str):
        """
        Save the computed matrix results of TSS/TES/Body.

        Arguments:
            save_path: the path of saving results.
        
        """
        for bw_name in self.bw_tss_scores_dict.keys():
            console.level2(f"Saving {bw_name} results...")
            if not os.path.exists(save_path+'/'+bw_name):
                os.mkdir(save_path+'/'+bw_name)
                console.success(f"Folder '{save_path+'/'+bw_name}' created.", level=2)
            else:
                pass
            
            if self.bw_tss_scores_dict[bw_name] is None:
                console.warn("You need to run the compute_matrix function first!")
            else:
                self.bw_tss_scores_dict[bw_name].write_h5ad(save_path+'/'+bw_name+'/tss_scores.h5ad')
            
            if self.bw_tes_scores_dict[bw_name] is None:
                console.warn("You need to run the compute_matrix function first!")
            else:
                self.bw_tes_scores_dict[bw_name].write_h5ad(save_path+'/'+bw_name+'/tes_scores.h5ad')

            if self.bw_body_scores_dict[bw_name] is None:
                console.warn("You need to run the compute_matrix function first!")
            else:
                self.bw_body_scores_dict[bw_name].write_h5ad(save_path+'/'+bw_name+'/body_scores.h5ad')
    
    def load_bw_result(self,load_path:str):

        """
        Load the computed matrix results of TSS/TES/Body.

        Arguments:
            load_path: the path of loading results.

        """
        with console.group_node('Load computed matrices', last=True, level=1):
            for j, bw_name in enumerate(self.bw_names):
                console.node(f"Loading {bw_name} results...", last=(j == len(self.bw_names) - 1), level=2)
                if not os.path.exists(load_path+'/'+bw_name+'/tss_scores.h5ad'):
                    console.warn("You need to run the compute_matrix function first for {}!".format(bw_name), level=3)
                else:
                    self.bw_tss_scores_dict[bw_name]=anndata.read_h5ad(load_path+'/'+bw_name+'/tss_scores.h5ad')
                
                if not os.path.exists(load_path+'/'+bw_name+'/tes_scores.h5ad'):
                    console.warn("You need to run the compute_matrix function first for {}!".format(bw_name), level=3)
                else:
                    self.bw_tes_scores_dict[bw_name]=anndata.read_h5ad(load_path+'/'+bw_name+'/tes_scores.h5ad')

                if not os.path.exists(load_path+'/'+bw_name+'/body_scores.h5ad'):
                    console.warn("You need to run the compute_matrix function first for {}!".format(bw_name), level=3)
                else:
                    self.bw_body_scores_dict[bw_name]=anndata.read_h5ad(load_path+'/'+bw_name+'/body_scores.h5ad')


    def compute_matrix(self,bw_name:str,nbins:int=100,
                          upstream:int=3000,downstream:int=3000,
                          n_jobs:int=1):
        """
        Compute the enrichment matrix of TSS/TES/Body.

        Arguments:
            bw_name: the name of bigwig file need to be computed.
            nbins: the number of bins.
            upstream: the upstream of TSS/TES.
            downstream: the downstream of TSS/TES.
            n_jobs: number of processes for per-chromosome parallelism
                    (1 = single-process, still uses the fast
                    groupby-once code path).

        """
        if bw_name in self.bw_tss_scores_dict:
            return
        return self._compute_matrix_parallel_impl(bw_name, nbins, upstream, downstream, n_jobs)

    def _compute_matrix_parallel_impl(self, bw_name: str, nbins: int, upstream: int, downstream: int, n_jobs: int):
        with console.group_node('Compute matrix: {}'.format(bw_name), last=True, level=1):
            console.node('Prepare features', last=False, level=2)
            features = self.gtf
            if 'feature' in features.columns:
                features = features.loc[features['feature']=='transcript']
            if 'seqname' in features.columns:
                features = features.loc[features['seqname'].str.contains('chr')]
            f_first = features.groupby('gene_id', sort=False).first().reset_index()
            tasks = []
            for chrom, sub in f_first.groupby('seqname', sort=False):
                records = list(zip(sub['gene_id'].tolist(),
                                   sub['seqname'].tolist(),
                                   sub['start'].astype(int).tolist(),
                                   sub['end'].astype(int).tolist(),
                                   sub['strand'].tolist()))
                if not records:
                    continue
                tasks.append(dict(bw_path=self.bw_path_dict[bw_name], nbins=nbins,
                                  upstream=upstream, downstream=downstream,
                                  records=records))

            console.node('Build matrices', last=False, level=2)
            if n_jobs and n_jobs > 1:
                with mp.Pool(processes=n_jobs) as pool:
                    results = []
                    with tqdm(total=len(tasks), desc='Chromosomes', unit='chr') as pbar:
                        for res in pool.imap_unordered(_compute_chrom_task, tasks):
                            results.append(res)
                            pbar.update(1)
            else:
                results = []
                for t in tqdm(tasks, desc='Chromosomes', unit='chr'):
                    results.append(_compute_chrom_task(t))

            gene_ids = []
            tss_list = []; tes_list = []; body_list = []
            tss_rs=[]; tss_re=[]; tes_rs=[]; tes_re=[]; body_rs=[]; body_re=[]
            for r in results:
                gene_ids.extend(r['gene_ids'])
                tss_list.append(r['tss']); tes_list.append(r['tes']); body_list.append(r['body'])
                tss_rs.extend(r['tss_rs']); tss_re.extend(r['tss_re'])
                tes_rs.extend(r['tes_rs']); tes_re.extend(r['tes_re'])
                body_rs.extend(r['body_rs']); body_re.extend(r['body_re'])

            tss_array = np.vstack(tss_list) if tss_list else np.zeros((0, nbins), dtype=np.float32)
            tes_array = np.vstack(tes_list) if tes_list else np.zeros((0, nbins), dtype=np.float32)
            body_array = np.vstack(body_list) if body_list else np.zeros((0, nbins), dtype=np.float32)

            console.node('Finalize', last=True, level=2)
            tss_array = np.nan_to_num(tss_array)
            tes_array = np.nan_to_num(tes_array)
            body_array = np.nan_to_num(body_array)
            order = np.argsort(-np.mean(tss_array, axis=1)) if len(tss_array) else np.array([], dtype=int)
            tss_array = tss_array[order]
            tes_array = tes_array[order]
            body_array = body_array[order]

            def _reorder(lst):
                return [lst[i] for i in order]

            gene_list_ordered = _reorder(gene_ids)
            tss_region_start_li = _reorder(tss_rs)
            tss_region_end_li = _reorder(tss_re)
            tes_region_start_li = _reorder(tes_rs)
            tes_region_end_li = _reorder(tes_re)
            body_region_start_li = _reorder(body_rs)
            body_region_end_li = _reorder(body_re)

            tss_csr=csr_matrix(tss_array)
            tes_csr=csr_matrix(tes_array)
            body_csr=csr_matrix(body_array)

            tss_adata=anndata.AnnData(tss_csr)
            tss_adata.obs.index=gene_list_ordered
            tss_adata.uns['range']=[0-downstream,upstream]
            tss_adata.uns['bins']=nbins
            tss_adata.obs['region_start']=tss_region_start_li
            tss_adata.obs['region_end']=tss_region_end_li

            tes_adata=anndata.AnnData(tes_csr)
            tes_adata.obs.index=gene_list_ordered
            tes_adata.uns['range']=[0-downstream,upstream]
            tes_adata.uns['bins']=nbins
            tes_adata.obs['region_start']=tes_region_start_li
            tes_adata.obs['region_end']=tes_region_end_li

            body_adata=anndata.AnnData(body_csr)
            body_adata.obs.index=gene_list_ordered
            body_adata.uns['range']=[0,upstream+downstream]
            body_adata.uns['bins']=nbins
            body_adata.obs['region_start']=body_region_start_li
            body_adata.obs['region_end']=body_region_end_li

            self.bw_tss_scores_dict[bw_name]=tss_adata
            self.bw_tes_scores_dict[bw_name]=tes_adata
            self.bw_body_scores_dict[bw_name]=body_adata
            console.success('{} matrix finished'.format(bw_name), level=2)
            console.level2('{} tss matrix in bw_tss_scores_dict[{}]'.format(bw_name,bw_name))
            console.level2('{} tes matrix in bw_tes_scores_dict[{}]'.format(bw_name,bw_name))
            console.level2('{} body matrix in bw_body_scores_dict[{}]'.format(bw_name,bw_name))
            return tss_adata, tes_adata, body_adata

    def compute_matrix_cis(self,bw_name:str,nbins:int=100,bw_type:str='TSS',
                           cis_distance:int=2000,
                          upstream:int=3000,downstream:int=3000):
        """
        Compute the enrichment matrix of TSS/TES/Body.

        Arguments:
            bw_name: the name of bigwig file need to be computed.
            bw_type: can be set as 'TSS','TES'.
            nbins: the number of bins.
            cis_distance: the distance to TSS/TES.
            upstream: the upstream of TSS/TES.
            downstream: the downstream of TSS/TES.

        """
        console.level2('Computing {} matrix'.format(bw_name))
        # Pre-aggregate: one row per gene_id (first transcript per gene).
        # The legacy code did ``features.loc[features.gene_id == g]`` inside
        # the per-gene loop — O(N_genes · N_feature_rows) pandas filters.
        features = self.gtf.loc[
            (self.gtf['feature'] == 'transcript') &
            (self.gtf['seqname'].str.contains('chr'))
        ]
        f_first = features.groupby('gene_id', sort=False).first()

        n_genes = len(f_first)
        gene_list = f_first.index.tolist()
        chrom_arr  = f_first['seqname'].to_numpy()
        start_arr  = f_first['start'].astype(int).to_numpy()
        end_arr    = f_first['end'].astype(int).to_numpy()
        strand_arr = f_first['strand'].to_numpy()

        # Preallocate dense output rather than DataFrame row-assign
        # (which triggers a full reindex on every ``.loc[g] = ...``).
        arr = np.full((n_genes, nbins), np.nan, dtype=np.float32)
        region_start_li = np.empty(n_genes, dtype=np.int64)
        region_end_li   = np.empty(n_genes, dtype=np.int64)

        bwh = self.bw_dict[bw_name]
        chrom_lengths = bwh.chroms()

        for idx, (g, chrom, s, e, strand) in enumerate(
            zip(tqdm(gene_list, desc='Processing genes', unit='gene'),
                chrom_arr, start_arr, end_arr, strand_arr)
        ):
            if strand == '-':
                tss_loc, tes_loc = e, s
            else:
                tss_loc, tes_loc = s, e

            if bw_type == 'TSS':
                r_start = max(0, tss_loc - cis_distance - upstream)
                r_end   = tss_loc - cis_distance + downstream
            elif bw_type == 'TES':
                r_start = max(0, tes_loc + cis_distance - upstream)
                r_end   = tes_loc + cis_distance + downstream
            else:
                raise ValueError(f"bw_type must be 'TSS' or 'TES', got {bw_type!r}")
            chrom_len = chrom_lengths.get(chrom)
            if chrom_len is not None and r_end > chrom_len:
                r_end = chrom_len

            vals = np.asarray(
                bwh.stats(chrom, r_start, r_end, nBins=nbins, type='mean'),
                dtype=np.float32,
            )
            if strand == '-':
                vals = vals[::-1]
            arr[idx, :] = vals
            region_start_li[idx] = r_start
            region_end_li[idx]   = r_end

        arr = np.nan_to_num(arr)
        order = np.argsort(-np.mean(arr, axis=1))
        arr = arr[order]
        gene_list_ordered = [gene_list[i] for i in order]

        tss_adata = anndata.AnnData(csr_matrix(arr))
        tss_adata.obs.index      = gene_list_ordered
        tss_adata.uns['range']   = [0 - downstream, upstream]
        tss_adata.uns['bins']    = nbins
        tss_adata.obs['region_start'] = region_start_li[order]
        tss_adata.obs['region_end']   = region_end_li[order]

        console.success('{} matrix finished'.format(bw_name), level=2)
        return tss_adata
    
    def compute_matrix_region(self, bw_name: str,
                              region: "pd.DataFrame | None" = None,
                              nbins: int = 100, upstream: int = 3000,
                              downstream: int = 3000, *,
                              anchor="center", sort: bool = True,
                              n_jobs: int = 1):
        """Per-region signal matrix (regions × bins) for one or more anchors.

        Arguments:
            bw_name: bigwig loaded via :meth:`read`.
            region: DataFrame with ``seqname``, ``start``, ``end`` (and
                optionally ``strand``, ``gene_id``, ``feature``). If
                ``gene_id`` is missing, a surrogate is generated; if
                ``strand`` is missing, ``'+'`` is assumed. When ``None``
                (default), falls back to ``self.gtf`` (same behaviour as
                :meth:`compute_matrix`).
            nbins: number of bins across the scanned window.
            upstream, downstream: bp flanks around the anchor point. For
                strand-aware anchors (``'5p'`` / ``'3p'``), these are
                interpreted in gene orientation (5' / 3'). Ignored for
                ``anchor='body'`` (which scans the full region).
            anchor: either a single anchor string or a list / tuple for
                batch output. Valid values:

                * ``'center'`` — midpoint of ``(start, end)``. Strand-ignorant.
                * ``'5p'`` — strand-aware 5' end: ``start`` on ``+``, ``end``
                  on ``-``. Same as the classic "TSS" anchor but named
                  without the gene-specific bias (works just as well on
                  any stranded interval).
                * ``'3p'`` — strand-aware 3' end.
                * ``'start'`` / ``'end'`` — raw genomic coordinate, ignoring
                  strand.
                * ``'body'`` — scan the full ``[start, end]`` region
                  rescaled to ``nbins``. No flanks.

                Pass a list to fetch several anchors in one call and get
                back a dict ``{anchor: AnnData}``. The underlying worker
                reuses the per-chromosome pyBigWig handle, so multi-anchor
                mode is cheaper than calling this function N times.
            sort: sort rows by descending mean of the *first* anchor's
                matrix. Pass ``sort=False`` when row order must align with
                the input (e.g. for feeding :meth:`plot_matrix_multi`).
            n_jobs: processes for per-chromosome parallelism. ``n_jobs=1``
                (default) runs sequentially but still benefits from the
                chrom-grouped layout.

        Returns:
            - When ``anchor`` is a string: one AnnData.
            - When ``anchor`` is a list / tuple: a dict ``{anchor: AnnData}``.

            Each AnnData has a ``(n_regions, nbins)`` sparse matrix and
            ``uns['range'] = [-upstream, downstream]`` (or
            ``[0, region_length]`` for ``'body'``).
        """
        # ---- input normalisation --------------------------------------
        if region is None:
            if getattr(self, 'gtf', None) is None:
                raise ValueError(
                    "region is None and no GTF loaded; call load_gtf() first "
                    "or pass an explicit region DataFrame"
                )
            region = self.gtf

        if isinstance(anchor, str):
            anchors = (anchor,)
            single = True
        else:
            anchors = tuple(anchor)
            single = False
        for a in anchors:
            if a not in _ANCHOR_CHOICES:
                raise ValueError(
                    f"anchor must be one of {_ANCHOR_CHOICES}, got {a!r}"
                )

        console.level2('Computing {} matrix (anchors={})'.format(
            bw_name, list(anchors)))

        features = region
        if 'feature' in features.columns:
            features = features.loc[features['feature'] == 'transcript']
        if 'gene_id' not in features.columns:
            features = features.assign(
                gene_id=[f'region_{i}' for i in range(len(features))]
            )
        # seqname/chrom alias
        chrom_col = 'seqname' if 'seqname' in features.columns else 'chrom'
        if chrom_col != 'seqname':
            features = features.rename(columns={chrom_col: 'seqname'})
        f_first = features.groupby('gene_id', sort=False).first().reset_index()

        n_rows = len(f_first)
        gene_order = f_first['gene_id'].tolist()
        gene_pos = {g: i for i, g in enumerate(gene_order)}
        has_strand = 'strand' in f_first.columns

        # ---- chrom-grouped task batches ------------------------------
        tasks = []
        for chrom, sub in f_first.groupby('seqname', sort=False):
            records = list(zip(
                sub['gene_id'].tolist(),
                sub['seqname'].tolist(),
                sub['start'].astype(int).tolist(),
                sub['end'].astype(int).tolist(),
                sub['strand'].tolist() if has_strand else ['+'] * len(sub),
            ))
            if not records:
                continue
            tasks.append(dict(
                bw_path=self.bw_path_dict[bw_name],
                nbins=nbins, upstream=upstream, downstream=downstream,
                anchors=anchors, records=records,
            ))

        if n_jobs and n_jobs > 1:
            with mp.Pool(processes=n_jobs) as pool:
                results = []
                with tqdm(total=len(tasks), desc='Chromosomes', unit='chr') as pbar:
                    for res in pool.imap_unordered(
                        _compute_region_chrom_task, tasks,
                    ):
                        results.append(res)
                        pbar.update(1)
        else:
            results = [
                _compute_region_chrom_task(t)
                for t in tqdm(tasks, desc='Chromosomes', unit='chr')
            ]

        # ---- reassemble per-anchor matrices in input order -----------
        per_anchor = {
            a: dict(
                arr=np.zeros((n_rows, nbins), dtype=np.float32),
                rs=np.zeros(n_rows, dtype=np.int64),
                re=np.zeros(n_rows, dtype=np.int64),
            ) for a in anchors
        }
        for r in results:
            rows = [gene_pos[g] for g in r['gene_ids']]
            for a in anchors:
                pa = r['per_anchor'][a]
                per_anchor[a]['arr'][rows] = pa['arr']
                per_anchor[a]['rs'][rows]  = pa['rs']
                per_anchor[a]['re'][rows]  = pa['re']

        # ---- sort order (same across anchors) ------------------------
        primary = anchors[0]
        primary_arr = np.nan_to_num(per_anchor[primary]['arr'])
        order = (np.argsort(-primary_arr.mean(axis=1)) if sort
                 else np.arange(n_rows))

        out = {}
        for a in anchors:
            arr = np.nan_to_num(per_anchor[a]['arr'])[order]
            rs  = per_anchor[a]['rs'][order]
            re  = per_anchor[a]['re'][order]
            ad = anndata.AnnData(csr_matrix(arr))
            ad.obs.index = [gene_order[i] for i in order]
            ad.obs['region_start'] = rs
            ad.obs['region_end']   = re
            ad.uns['bins']   = nbins
            ad.uns['anchor'] = a
            if a == 'body':
                ad.uns['range'] = [0, 0]  # body mode doesn't have ±flanks
            else:
                ad.uns['range'] = [-int(upstream), int(downstream)]
            out[a] = ad

        console.success('{} matrix finished'.format(bw_name), level=2)
        return out[primary] if single else out


    def plot_matrix(self,bw_name:str,bw_type:str='TSS',
                    figsize:tuple=(2,8),cmap:str='Greens',
                    vmax='auto',vmin='auto',fontsize:int=12,title:str='')->tuple:
        """
        Plot the enrichment matrix of TSS/TES/Body.

        Arguments:
            bw_name: the name of bigwig file need to be computed.
            bw_type: can be set as 'TSS','TES','body' or 'all'.
            figsize: the size of figure.
            cmap: the color map of figure.
            vmax: the max value of color bar. Default the 98% percentile of data.
            vmin: the min value of color bar. Default 0.
            fontsize: the fontsize of figure.
            title: the title of figure.
        
        Returns:
            fig: the figure object.
            ax: the axis object.

        
        """
        #from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        fig, ax = plt.subplots(figsize=figsize)
        #pm=np.clip(tss_array.loc[order], 0, 11)
        if bw_type=='TSS':
            adata=self.bw_tss_scores_dict[bw_name]
        elif bw_type=='TES':
            adata=self.bw_tes_scores_dict[bw_name]
        elif bw_type=='body':
            adata=self.bw_body_scores_dict[bw_name]
        elif bw_type=='all':
            adata=anndata.concat([self.bw_tss_scores_dict[bw_name][:,:50],
                self.bw_body_scores_dict[bw_name][:,:],
               self.bw_tes_scores_dict[bw_name][:,50:]],axis=1)
            adata.uns=self.bw_tss_scores_dict[bw_name].uns.copy()
            adata.uns['bins']=adata.shape[1]

        if vmax=='auto':
            vmax=np.percentile(adata.X.toarray() ,98)
        if vmin=='auto':
            vmin=0

        smp=ax.imshow(adata.X.toarray(),
                aspect='auto',
                interpolation='bilinear',
                origin='upper',
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,)
        cax=plt.colorbar(smp,shrink=0.5)
        cax.ax.tick_params(labelsize=fontsize)
        ax.yaxis.set_visible(False)
        #ax.xaxis.set_visible(False)
        #ax.set_xlabel('TSS')
        # 设置TSS和TES位置的刻度和标签
        if bw_type=='TSS':
            x=adata.uns['bins']
            tss_position = int(adata.shape[1] / 2)  # 1/4位置
            #tes_position = int(2 * 100 / 3)  # 后1/3位置

            ax.set_xticks([0,tss_position,adata.shape[1]])  # 设置刻度位置
            ax.set_xticklabels([format_number_with_k(adata.uns['range'][0]),
                    'TSS',format_number_with_k(adata.uns['range'][1])],fontsize=fontsize)  # 设置刻度标签
        elif bw_type=='TES':
            x=adata.uns['bins']
            tes_position = int(adata.shape[1] / 2)
            ax.set_xticks([0,tes_position,adata.shape[1]])  # 设置刻度位置
            ax.set_xticklabels([format_number_with_k(adata.uns['range'][0]),
                    'TES',format_number_with_k(adata.uns['range'][1])],fontsize=fontsize)  # 设置刻度标签
        elif bw_type=='body':
            x=adata.uns['bins']
            body_position = int(adata.shape[1] / 2)
            ax.set_xticks([0,body_position,adata.shape[1]])
            ax.set_xticklabels([format_number_with_k(adata.uns['range'][0]),
                    'Peak',format_number_with_k(adata.uns['range'][1])],fontsize=fontsize)
        elif bw_type=='all':
            x=adata.uns['bins']
            tss_position = int(adata.shape[1] / 4)
            tes_position = int(adata.shape[1] / 4 *3)
            ax.set_xticks([0,tss_position,tes_position,adata.shape[1]])
            ax.set_xticklabels([format_number_with_k(adata.uns['range'][0]),
                    'TSS','TES',format_number_with_k(adata.uns['range'][1])],fontsize=fontsize)
        plt.grid(False)
        plt.tight_layout()
        if title!='':
            plt.title(title,fontsize=fontsize)
        else:
            plt.title(bw_name,fontsize=fontsize)
        return fig,ax
    
    def plot_matrix_line(self,bw_name:str,bw_type:str='TSS',
                         figsize:tuple=(3,3),color:str='#a51616',
                         linewidth:int=2,fontsize:int=13,title:str='')->tuple:
        
        """
        Plot the enrichment hist of TSS/TES/Body.

        Arguments:  
            bw_name: the name of bigwig file need to be computed.
            bw_type: can be set as 'TSS','TES','body' or 'all'.
            figsize: the size of figure.
            color: the color of figure.
            linewidth: the linewidth of figure.
            fontsize: the fontsize of figure.
            title: the title of figure.

        Returns:
            fig: the figure object.
            ax: the axis object.

        """

        fig, ax = plt.subplots(figsize=figsize)

        if bw_type=='TSS':
            adata=self.bw_tss_scores_dict[bw_name]
        elif bw_type=='TES':
            adata=self.bw_tes_scores_dict[bw_name]
        elif bw_type=='body':
            adata=self.bw_body_scores_dict[bw_name]
        elif bw_type=='all':
            adata=anndata.concat([self.bw_tss_scores_dict[bw_name][:,:50],
                self.bw_body_scores_dict[bw_name][:,:],
               self.bw_tes_scores_dict[bw_name][:,50:]],axis=1)
            adata.uns=self.bw_tss_scores_dict[bw_name].uns.copy()
            adata.uns['bins']=adata.shape[1]

        ax.plot([i for i in range(adata.shape[1])],
                adata.X.toarray().mean(axis=0),linewidth=linewidth,color=color)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)

        if bw_type=='TSS':
            x=adata.uns['bins']
            tss_position = int(adata.shape[1] / 2)  # 1/4位置
            #tes_position = int(2 * 100 / 3)  # 后1/3位置

            ax.set_xticks([0,tss_position,adata.shape[1]])  # 设置刻度位置
            ax.set_xticklabels([format_number_with_k(adata.uns['range'][0]),
                    'TSS',format_number_with_k(adata.uns['range'][1])],fontsize=fontsize)  # 设置刻度标签
        elif bw_type=='TES':
            x=adata.uns['bins']
            tes_position = int(adata.shape[1] / 2)
            ax.set_xticks([0,tes_position,adata.shape[1]])  # 设置刻度位置
            ax.set_xticklabels([format_number_with_k(adata.uns['range'][0]),
                    'TES',format_number_with_k(adata.uns['range'][1])],fontsize=fontsize)  # 设置刻度标签
        elif bw_type=='body':
            x=adata.uns['bins']
            body_position = int(adata.shape[1] / 2)
            ax.set_xticks([0,body_position,adata.shape[1]])
            ax.set_xticklabels([format_number_with_k(adata.uns['range'][0]),
                    'Peak',format_number_with_k(adata.uns['range'][1])],fontsize=fontsize)
            
        elif bw_type=='all':
            x=adata.uns['bins']
            tss_position = int(adata.shape[1] / 4)
            tes_position = int(adata.shape[1] / 4 *3)
            ax.set_xticks([0,tss_position,tes_position,adata.shape[1]])
            ax.set_xticklabels([format_number_with_k(adata.uns['range'][0]),
                    'TSS','TES',format_number_with_k(adata.uns['range'][1])],fontsize=fontsize)
        ax.set_yticklabels(ax.get_yticklabels()[1:],fontsize=fontsize)
        
        plt.grid(False)
        plt.tight_layout()
        if title!='':
            plt.title(title,fontsize=fontsize)
        else:
            plt.title(bw_name,fontsize=fontsize)
        return fig,ax
        
        
    def plot_track(self,chrom:str,chromstart:int,chromend:int,nbins:int=700,
                   bp_per_bin:"int | None"=None,
                   value_type:str='mean',transform:str='no',
                   figwidth:int=6,figheight:int=6,plot_names=None,
                   color_dict=None,region_dict=None,
                   gtf_color:str='#000000',prefered_name:str='gene_id',
                   jump_symbols=['ENSG'],text_fontsize=12,text_height=1.5,
                   show_text='ylabel',text_rotation=0,
                   ylabel_rotation:float=0,ylabel_ha:str='right',
                   ylabel_va:str='center',ylabel_labelpad:float=8,
                   ymax=None,ymin=None,region_color='#c2c2c2',region_alpha=0.4)->tuple:
        
        """
        Plot the peak track of bigwig file.

        Arguments:
            chrom: the chromosome of region.
            chromstart: the start position of region.
            chromend: the end position of region.
            nbins: the number of bins. Ignored if ``bp_per_bin`` is set.
            bp_per_bin: bin width in base pairs. When given, overrides
                ``nbins`` with ``(chromend - chromstart) // bp_per_bin``.
                Use larger values (e.g. ``500``, ``1000``) to produce a
                coarser, smoother track.
            value_type: the type of value. Can be set as 'mean','max','min','coverage','std','sum'.
            transform: the transform of value. Can be set as 'log','log2','log10','log1p','-log' or 'no'.
            figwidth: the width of figure.
            figheight: the height of figure.
            plot_names: the name of bigwig file need to be plotted.
            color_dict: the color of bigwig file need to be plotted.
            region_dict: the region of interest.
            gtf_color: the color of gtf.
            prefered_name: the prefered name of gtf.
            jump_symbols: If the gene name contains characters within list, the text for that gene is not displayed.
            ymax: Since there are differences in our bigwig files for each type of cell, we can specify the maximum value to ensure a high degree of consistency when visualising. Pass a scalar to apply the same ymax to every track, or a ``{bw_name: ymax}`` dict to give each track its own y-axis range (matches the per-row scales used by paper-style UCSC browser snapshots).
            ymin: Lower bound of the y-axis. Same scalar-or-dict form as ``ymax``. Useful to crop out background (e.g. paper UCSC browsers often display ATAC with ymin>0 so only peaks above a baseline are shown).
            text_fontsize: the fontsize of text in figures
            ylabel_rotation: degrees for the per-track left label. Default 0
                (horizontal, IGV / paper style). Set to 90 for matplotlib's
                classic vertical label orientation.
            ylabel_ha, ylabel_va: horizontal / vertical alignment of the left
                label — defaults place it right-aligned next to the axis.
            ylabel_labelpad: extra padding (points) between axis and label.

        Returns:
            fig: the figure object.
            ax: the axis object.
        
        """
        if bp_per_bin is not None:
            nbins = max(1, (int(chromend) - int(chromstart)) // int(bp_per_bin))
        self.scores_per_bin_dict={}
        for bw_name in self.bw_names:

            score_list=np.array(self.bw_dict[bw_name].stats(chrom,
                                                       chromstart,
                                                    chromend, nBins=nbins,
                                                     type=value_type)).astype(float)
            
            if transform in ['log', 'log2', 'log10']:
                score_list=eval('np.' + transform + '(score_list)')
                #score_list[score_list < 0] = 0
            elif transform == 'log1p':
                score_list=np.log1p(score_list)
                #score_list[score_list < 0] = 0
            elif transform == '-log':
                score_list =- np.log(score_list)
                #score_list[score_list < 0] = 0
            else:
                pass
            np.nan_to_num(score_list,0)
            self.scores_per_bin_dict[bw_name]=score_list


        if color_dict is None:
            color_dict={}
            if len(self.bw_names)<=len(sc_color):
                color_dict=dict(zip(self.bw_names,sc_color))
            elif len(self.bw_names)<102:
                 color_dict=dict(zip(self.bw_names,sc.pl.palettes.default_102))

        if self.gtf is None:
            bw_lens=self.bw_lens
        else:
            bw_lens=self.bw_lens+1

        figheight=figheight/len(self.bw_names)

        fig, axes = plt.subplots(bw_lens,1,figsize=(figwidth,bw_lens*figheight))
        if plot_names is None:
            plot_names=self.bw_names
        
        if bw_lens==1:
            axes=[axes]
        
        if self.gtf is None:
            if bw_lens>1:
                plot_axes=axes[:-1]
            else:
                plot_axes=axes
        else:
            plot_axes=axes
       

        for ax,bw_name,plot_name in zip(plot_axes,self.bw_names,plot_names):
            x_values = np.linspace(chromstart, chromend, nbins)
            ax.plot(x_values, self.scores_per_bin_dict[bw_name], '-',color=color_dict[bw_name])
            ax.fill_between(x_values, self.scores_per_bin_dict[bw_name], linewidth=0.1,
                            color=color_dict[bw_name])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            #ax.axis('off')
            ax.grid(False)
            ax.get_xaxis().get_major_formatter().set_scientific(False)

            #ax.set_xticklabels([start_region,chromEnd],fontsize=11)
            ax.set_yticklabels([])
            ax.set_xticklabels([],fontsize=11)
            if show_text=='ylabel':
                # Default: horizontal (paper/IGV style); pass
                # ylabel_rotation=90 for matplotlib's classic vertical label.
                ax.set_ylabel(plot_name, fontsize=12,
                              rotation=ylabel_rotation,
                              ha=ylabel_ha, va=ylabel_va,
                              labelpad=ylabel_labelpad)
    
            if ymax is not None or ymin is not None:
                # Allow per-track ymax/ymin: dict keyed by bw_name, else scalar applied to all.
                ymax_row = ymax.get(bw_name, None) if isinstance(ymax, dict) else ymax
                ymin_row = ymin.get(bw_name, None) if isinstance(ymin, dict) else ymin
                lo = 0 if ymin_row is None else ymin_row
                hi = ymax_row
                if hi is not None:
                    ax.set_ylim(lo, hi)
                    ax.set_yticks([lo, hi])
                    ax.set_yticklabels([lo, hi], fontsize=max(7, text_fontsize-5))
                    if show_text == 'rowlabel':
                        ax.text(chromstart, (lo+hi)*0.5, plot_name,
                                fontsize=text_fontsize, rotation=text_rotation)
                elif ymin_row is not None:
                    ax.set_ylim(bottom=lo)

            if region_dict is not None:
                for region in region_dict:
                    # region_color can be a scalar or {region_name: color} dict.
                    if isinstance(region_color, dict):
                        rc = region_color.get(region, '#c2c2c2')
                    else:
                        rc = region_color
                    ax.axvspan(region_dict[region][0],region_dict[region][1],
                               alpha=region_alpha,color=rc)
        
        if self.gtf is not None:
            # Lane-packed UCSC-style gene track. Collapses per-transcript
            # rows to gene-level, packs into 2 lanes left-to-right to avoid
            # label collisions, and suppresses labels on very dense windows
            # (e.g. a multi-Mb browser snapshot where individual names would
            # overlap illegibly).
            _draw_gene_lanes(
                axes[-1], self.gtf, chrom, chromstart, chromend,
                color=gtf_color, prefered_name=prefered_name,
                jump_symbols=tuple(jump_symbols),
                fontsize=text_fontsize,
                max_label_genes=30, n_lanes=2,
            )
        
        plt.suptitle('{}:{:,}-{:,}'.format(chrom,chromstart,chromend),x=0.9,fontsize=12,horizontalalignment='right')
        #plt.tight_layout()
        return fig,axes
    
    def getscoreperbin(self,bin_size=10000,number_thread=1,):
        test=getScorePerBin(list(self.bw_path_dict.values()),bin_size,number_thread,
                            out_file_for_raw_data='test.tab')
        k=0
        data=pd.DataFrame()
        for _values, tempFileName in test:
            bed_data=pd.read_csv(tempFileName,sep='\t',header=None)
            if k==0:
                data=bed_data
            else:
                data=pd.concat([data,bed_data],axis=0)
            k+=1
            os.remove(tempFileName)
        data.columns=['chrom','start','end']+list(self.bw_path_dict.keys())
        data.index=[i for i in range(data.shape[0])]
        self.scoreperbindata=data
        return data
    
    def compute_correlation(self,bw_names:list,method:str='pearson'):
        from scipy.stats import gaussian_kde
        score_data=self.scoreperbindata.fillna(0)
        x=score_data[bw_names[0]].values
        y=score_data[bw_names[1]].values

        xy = np.vstack([x,y])  #  将两个维度的数据叠加
        z = gaussian_kde(xy)(xy)  # 建立概率密度分布，并计算每个样本点的概率密度

        # Sort the points by density, so that the densest points are plotted last
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
        self.plot_x=x
        self.plot_y=y
        self.plot_z=z
        self.plot_bw_names=bw_names
        from scipy.stats import pearsonr
        console.level1('The correlation between {} and {} is {:.2}'.format(bw_names[0],bw_names[1],pearsonr(x,y)[0]))
        console.level2('Now you can use plot_correlation() to plot the correlation scatter plot')
        return pearsonr(x,y)[0]
    
    def plot_correlation_bigwig(self,figsize=(3.5,3),
                                cmap='',scatter_size=1,scatter_alpha=0.8,
                                fontsize=14,title='')->tuple:

        if self.scoreperbindata is None:
            console.warn('Please run getscoreperbin() first')
            return None
        
        from scipy.stats import pearsonr
        fig, ax = plt.subplots(figsize=figsize)
        if cmap=='':
            cmap=xcmap
        smp=ax.scatter(self.plot_x, self.plot_y,c=self.plot_z,s=scatter_size, 
                       cmap=cmap,alpha=scatter_alpha) # c表示标记的颜色
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['bottom'].set_position(('outward', 10))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)

        ax.grid(False)
        plt.xlim(0,20)
        plt.ylim(0,20)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        cax=plt.colorbar(smp,shrink=0.5)
        cax.ax.set_yticks([np.min(self.plot_z),np.max(self.plot_z)])
        cax.ax.set_yticklabels(['Low','High'],fontsize=fontsize)
        if title=='':
            ax.set_title('Pearson:{:.2}'.format(pearsonr(self.plot_x,self.plot_y)[0]),fontsize=fontsize)
        else:
            ax.set_title(title,fontsize=fontsize)
        ax.set_xlabel(self.plot_bw_names[0],fontsize=fontsize)
        ax.set_ylabel(self.plot_bw_names[1],fontsize=fontsize)

        return fig,ax

    # ------------------------------------------------------------------
    # Multi-group / multi-locus helpers used by the OTX2 case-study
    # notebooks (tutorials/case/otx2/fig3*.ipynb). Kept inside the class
    # so they share the self.bw_dict / self.gtf state.
    # ------------------------------------------------------------------
    def plot_matrix_line_compare(
        self,
        bw_name: str,
        groups: "dict[str, anndata.AnnData] | None" = None,
        *,
        colors: dict = None,
        figsize: tuple = (3.4, 3.1),
        linewidth: float = 1.5,
        xlabel: str = 'Distance (kb)',
        ylabel: str = 'Normalized RPKM',
        title: str = '',
        fontsize: int = 10,
        clip_percentile: float = None,
        normalize: str = None,
        normalize_ref: str = None,
        flank_kb: float = 1.5,
        flank_target: float = 0.5,
    ):
        """Overlay multiple mean-signal lines on one axes.

        Each entry of ``groups`` is a ``{label: AnnData}`` produced by
        :meth:`compute_matrix_region` / :meth:`compute_matrix_cis`. All
        matrices must share the same column count + ``uns['range']`` so the
        x-axis is well-defined.

        Arguments:
            bw_name: name of the bigwig the matrices came from (used only for
                the default title).
            groups: mapping ``{label: adata}``. Columns must be identical
                across matrices.
            colors: optional ``{label: color}`` mapping.
            figsize, linewidth, xlabel, ylabel, title, fontsize: standard.
            clip_percentile: if given (e.g. 99), per-group winsorise each
                matrix at the k-th percentile before averaging. Suppresses a
                handful of super-bound outlier rows that would otherwise
                dominate a 100-200 gene metaplot (classic CUT&RUN artefact).
            normalize: ``None`` (default) plots raw per-bin means. ``'flanks'``
                rescales every group's mean curve by a single factor so the
                reference group's flank (``|x| > flank_kb``) averages to
                ``flank_target``. This matches the "Normalized RPKM" y-axis
                convention used in CUT&RUN metaplots where both groups are
                anchored to a common background level.
            normalize_ref: group label to use as the flank reference when
                ``normalize='flanks'``. Defaults to the last group in
                ``groups`` (usually the negative / nonmaternal / random set).
            flank_kb: half-width of the TSS-proximal region excluded from the
                flank when computing the scaling factor.
            flank_target: value the reference group's flank mean is scaled to.

        Returns:
            ``(fig, ax)``.
        """
        if not groups:
            raise ValueError('groups must be a non-empty {label: adata} mapping')
        colors = dict(colors or {})

        # 1. Extract dense matrices, optionally winsorising per-group outliers.
        mats = {}
        x_range = None
        for label, adata in groups.items():
            if x_range is None:
                x_range = adata.uns.get(
                    'range', [-adata.shape[1] // 2, adata.shape[1] // 2]
                )
            m = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
            if clip_percentile is not None:
                m = np.clip(m, 0, np.percentile(m, clip_percentile))
            mats[label] = m

        # 2. Per-bin mean across genes.
        curves = {lbl: m.mean(axis=0) for lbl, m in mats.items()}
        n_bins = next(iter(curves.values())).shape[0]
        x_kb = np.linspace(x_range[0], x_range[1], n_bins) / 1000

        # 3. Flank-anchored rescaling (used by CUT&RUN-style metaplots).
        if normalize == 'flanks':
            ref = normalize_ref or list(curves.keys())[-1]
            if ref not in curves:
                raise ValueError(f"normalize_ref={ref!r} not in groups")
            mask = np.abs(x_kb) > flank_kb
            denom = curves[ref][mask].mean()
            if denom > 0:
                scale = flank_target / denom
                for lbl in curves:
                    curves[lbl] = curves[lbl] * scale
        elif normalize is not None:
            raise ValueError(f"unsupported normalize={normalize!r}")

        # 4. Plot.
        fig, ax = plt.subplots(figsize=figsize)
        for label, y in curves.items():
            ax.plot(x_kb, y, lw=linewidth,
                    color=colors.get(label), label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title or bw_name, fontsize=fontsize)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(frameon=False, fontsize=fontsize - 1, loc='upper right')
        ax.set_xlim(x_range[0] / 1000, x_range[1] / 1000)
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        return fig, ax

    def plot_matrix_multi(
        self,
        matrices: "dict[str, anndata.AnnData]",
        *,
        cluster_labels=None,
        cluster_order: list = None,
        cluster_colors: dict = None,
        max_per_cluster: int = None,
        cmap: str = 'Reds',
        vmax_percentile: float = 98,
        figsize: tuple = (6, 8),
        title: str = '',
        bw_order: list = None,
        sort_by: "str | None" = None,
    ):
        """Render a precomputed multi-condition × multi-cluster heatmap.

        This function is **pure rendering**. Callers are expected to have
        already produced one ``AnnData`` per bigwig via
        :meth:`compute_matrix_region` (or any other source) and to pass
        them in as ``matrices={bw_label: adata, ...}``. All AnnData must
        share the same ``n_obs`` (same regions in the same order) and the
        same ``uns['range']`` / ``uns['bins']`` so the x-axis is well-defined.

        Arguments:
            matrices: ``{bw_label: AnnData}`` mapping. Each AnnData is
                typically the return value of ``compute_matrix_region(...,
                sort=False)`` so the row order matches the original regions
                DataFrame. Keys become column titles.
            cluster_labels: optional array-like of length ``n_obs`` assigning
                each region to a cluster name. When None, all rows are
                plotted as one contiguous block (no side-bar).
            cluster_order: explicit top→bottom ordering of cluster labels.
                Clusters not listed are appended alphabetically.
            cluster_colors: ``{cluster_label: color}`` for the left sidebar.
            max_per_cluster: optional cap on rows per cluster (keeps the
                figure responsive on 100k+ peak sets). Rows are kept by
                their descending primary-bw mean rank.
            cmap: matplotlib colormap, applied per column.
            vmax_percentile: per-column colour-clip percentile (default 98).
            figsize, title: standard.
            bw_order: explicit column ordering. Defaults to ``matrices.keys()``.
            sort_by: bigwig label whose per-row mean drives the within-cluster
                sort. Defaults to the first key in ``bw_order``.

        Returns:
            ``(fig, axes)`` — ``axes`` is the list of per-bigwig axes
            (length ``len(matrices)``).

        Example:
            >>> # compute once per bigwig (explicit)
            >>> adapted = regions.rename(columns={'chrom':'seqname'}).assign(
            ...     strand='+', feature='transcript',
            ...     gene_id=[f'peak_{i}' for i in range(len(regions))])
            >>> m_4C  = bw.compute_matrix_region('4C',  adapted, sort=False)
            >>> m_ESC = bw.compute_matrix_region('ESC', adapted, sort=False)
            >>> m_dEC = bw.compute_matrix_region('dEC', adapted, sort=False)
            >>> # plot — just the dict
            >>> fig, axes = bw.plot_matrix_multi(
            ...     {'4C': m_4C, 'ESC': m_ESC, 'dEC': m_dEC},
            ...     cluster_labels=regions['cluster'],
            ...     cluster_order=['4C-specific','ESC-specific',
            ...                    'dEC-specific','4C/ESC shared',
            ...                    '4C/ESC/dEC shared'],
            ...     cluster_colors={...},
            ...     max_per_cluster=3000,
            ...     title='Distal OTX2 binding',
            ... )
        """
        if not matrices:
            raise ValueError("matrices must be a non-empty {bw: AnnData} mapping")
        bws = list(bw_order) if bw_order is not None else list(matrices.keys())
        for bw in bws:
            if bw not in matrices:
                raise ValueError(f"bw_order references missing key: {bw}")

        # Extract dense arrays + validate shapes.
        dense = {}
        n_obs = None; nbins = None; x_range = None
        for bw in bws:
            ad = matrices[bw]
            m = ad.X.toarray() if sp.issparse(ad.X) else np.asarray(ad.X)
            dense[bw] = m.astype(np.float32, copy=False)
            if n_obs is None:
                n_obs = m.shape[0]; nbins = m.shape[1]
                x_range = ad.uns.get('range', None)
            elif m.shape != (n_obs, nbins):
                raise ValueError(
                    f"matrix shapes differ: {bw} is {m.shape}, "
                    f"expected ({n_obs}, {nbins})"
                )

        # Build per-cluster row index ranges.
        if cluster_labels is None:
            groups = {'_all': np.arange(n_obs)}
            order_labels = ['_all']
        else:
            labs = np.asarray(list(cluster_labels))
            if len(labs) != n_obs:
                raise ValueError(
                    f"cluster_labels length {len(labs)} != n_obs {n_obs}"
                )
            uniq = list(pd.unique(labs[pd.notna(labs)]))
            if cluster_order is not None:
                order_labels = [c for c in cluster_order if c in uniq] + \
                               [c for c in uniq if c not in cluster_order]
            else:
                order_labels = sorted(uniq)
            groups = {c: np.where(labs == c)[0] for c in order_labels}

        # Primary bigwig drives the within-cluster sort order.
        sort_by = sort_by or bws[0]
        primary = dense[sort_by]

        # Within each cluster, order by descending primary-bw mean; optionally
        # cap rows per cluster.
        row_order_by_cluster = {}
        for c, idxs in groups.items():
            if len(idxs) == 0:
                row_order_by_cluster[c] = idxs
                continue
            means = primary[idxs].mean(axis=1)
            ord_local = np.argsort(-means)
            idxs = idxs[ord_local]
            if max_per_cluster and len(idxs) > max_per_cluster:
                idxs = idxs[:max_per_cluster]
            row_order_by_cluster[c] = idxs

        cluster_colors = dict(cluster_colors or {})
        total_rows = sum(len(idxs) for idxs in row_order_by_cluster.values())
        n_col = len(bws)

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(
            nrows=1, ncols=n_col + 1,
            width_ratios=[0.06] + [1.0] * n_col,
            left=0.15, right=0.97, top=0.94, bottom=0.08, wspace=0.06,
        )

        # Left cluster bar (skip when there's only one group).
        ax_bar = fig.add_subplot(gs[0, 0])
        ax_bar.set_axis_off()
        ax_bar.set_ylim(0, total_rows); ax_bar.invert_yaxis()
        if cluster_labels is not None:
            y0 = 0
            for c in order_labels:
                n = len(row_order_by_cluster[c])
                if n == 0: continue
                col = cluster_colors.get(c, '#888888')
                ax_bar.add_patch(plt.Rectangle(
                    (0, y0), 1, n, color=col, clip_on=False))
                ax_bar.text(-0.25, y0 + n / 2, str(c),
                             ha='right', va='center', fontsize=7, color=col)
                y0 += n
        ax_bar.set_xlim(0, 1)

        axes = []
        for j, bw in enumerate(bws):
            ax = fig.add_subplot(gs[0, j + 1])
            stacked = np.concatenate(
                [dense[bw][row_order_by_cluster[c]] for c in order_labels], axis=0,
            )
            vmax = float(np.percentile(stacked, vmax_percentile)) if stacked.size else 1.0
            ax.imshow(
                stacked, aspect='auto', cmap=cmap, vmin=0, vmax=vmax,
                interpolation='bilinear', origin='upper',
            )
            ax.set_title(bw, fontsize=10)
            ax.set_xticks([0, nbins // 2, nbins - 1])
            if x_range is not None:
                ax.set_xticklabels([
                    f'{x_range[0] // 1000}', '0', f'{x_range[1] // 1000}',
                ])
                ax.set_xlabel('kb', fontsize=8)
            else:
                ax.set_xticklabels(['-', '0', '+'])
            ax.set_yticks([])
            y = 0
            for c in order_labels:
                y += len(row_order_by_cluster[c])
                if 0 < y < total_rows:
                    ax.axhline(y - 0.5, color='white', lw=0.4)
            axes.append(ax)

        if title:
            fig.suptitle(title, y=0.99, fontsize=11)
        return fig, axes

    def plot_track_multi(
        self,
        loci: list,
        bw_names: list = None,
        *,
        color_dict: dict = None,
        ymax: "dict | None" = None,
        nbins_per_100bp: int = 1,
        bp_per_bin: "int | None" = None,
        value_type: str = 'max',
        figwidth: float = 14,
        figheight: float = 7,
        gene_row: bool = True,
        region_dict_by_locus: "dict | None" = None,
        region_colors_by_locus: "dict | None" = None,
        region_alpha: float = 0.30,
        title: str = '',
    ):
        """Horizontal multi-locus genomic browser (one panel per locus).

        Arguments:
            loci: list of ``(name, chrom, start, end)`` (or ``(chrom, start, end)``)
                tuples. Panel widths are proportional to ``end - start``.
            bw_names: bigwig tracks to stack vertically, default ``self.bw_dict``
                order.
            color_dict: ``{bw_name: color}`` for the fill colour of each track.
            ymax: optional ``{bw_name: float}`` fixed y-axis max. If omitted
                epione auto-detects per locus.
            nbins_per_100bp: plot pixels per 100-bp bigwig bin. ``1`` (default)
                matches the native resolution; raise for smoother-looking
                signal at cost of speed. Ignored if ``bp_per_bin`` is set.
            bp_per_bin: bin width in base pairs, overriding ``nbins_per_100bp``
                when given. Use a larger value (e.g. ``500`` or ``1000``) to
                produce a coarser, smoother track — useful when ``value_type``
                is ``'max'`` and narrow spikes dominate at 100-bp resolution.
            value_type: pyBigWig stats aggregator, ``'max'`` or ``'mean'``.
            figwidth, figheight: overall figure size.
            gene_row: append a thin gene-model row at the bottom (uses
                ``self.gtf`` if loaded).
            region_dict_by_locus: ``{locus_name: {region_label: (start, end)}}``
                - highlight boxes per panel.
            region_colors_by_locus: ``{locus_name: {region_label: color}}``.
            region_alpha: highlight alpha.
            title: optional suptitle.

        Returns:
            ``(fig, axes)`` — axes is a dict keyed by ``(locus_name, bw_name)``.
        """
        # Normalise loci to (name, chrom, start, end)
        norm = []
        for L in loci:
            if len(L) == 4:
                norm.append(tuple(L))
            else:
                c, s, e = L
                norm.append((f'{c}:{s}-{e}', c, s, e))
        loci = norm
        bw_names = bw_names or list(self.bw_dict.keys())
        color_dict = dict(color_dict or {})
        ymax = dict(ymax or {})

        widths = np.array([e - s for _, _, s, e in loci], float)
        widths = widths / widths.sum() * len(loci)
        n_rows = len(bw_names) + (1 if gene_row else 0)
        n_cols = len(loci)

        fig = plt.figure(figsize=(figwidth, figheight))
        gs = fig.add_gridspec(
            nrows=n_rows,
            ncols=n_cols,
            width_ratios=list(widths),
            height_ratios=[1.0] * len(bw_names) + ([0.5] if gene_row else []),
            left=0.08, right=0.99, top=0.96, bottom=0.07,
            wspace=0.12, hspace=0.12,
        )

        axes = {}
        # pre-fetch per-panel ymax when not supplied explicitly
        for li, (lname, chrom, start, end) in enumerate(loci):
            if bp_per_bin is not None:
                nbins = max(1, (end - start) // int(bp_per_bin))
            else:
                nbins = max(1, (end - start) // (100 // nbins_per_100bp))
            # region highlights for this locus
            hl = {}
            if region_dict_by_locus and lname in region_dict_by_locus:
                hl = region_dict_by_locus[lname]
            hl_colors = (region_colors_by_locus or {}).get(lname, {}) if region_dict_by_locus else {}

            for ti, bw in enumerate(bw_names):
                ax = fig.add_subplot(gs[ti, li])
                # bw may be a string key or a list/tuple of keys to average
                if isinstance(bw, (list, tuple)):
                    name = bw[0]
                    acc = np.zeros(nbins, dtype=np.float32)
                    for b in bw:
                        try:
                            vv = self.bw_dict[b].stats(
                                chrom, int(start), int(end), nBins=nbins, type=value_type)
                        except Exception:
                            vv = [0] * nbins
                        acc += np.asarray(
                            [0.0 if v is None else float(v) for v in vv],
                            dtype=np.float32)
                    vals = acc / len(bw)
                else:
                    name = bw
                    try:
                        vals = self.bw_dict[bw].stats(
                            chrom, int(start), int(end), nBins=nbins, type=value_type)
                    except Exception:
                        vals = [0] * nbins
                    vals = np.asarray([0.0 if v is None else float(v) for v in vals])
                bw = name
                x = np.linspace(start, end, len(vals))
                color = color_dict.get(bw, '#555555')
                ax.fill_between(x, 0, vals, color=color, alpha=0.85, linewidth=0)
                ax.plot(x, vals, color=color, lw=0.3)
                for label, (hs, he) in hl.items():
                    ax.axvspan(
                        max(hs, start), min(he, end),
                        color=hl_colors.get(label, '#f5a9b8'),
                        alpha=region_alpha, zorder=0,
                    )
                ym = ymax.get(bw)
                if ym is None:
                    ym = max(1.0, float(vals.max()) * 1.05)
                ax.set_ylim(0, ym)
                ax.set_xlim(start, end)
                ax.set_yticks([0, ym])
                ax.tick_params(axis='y', labelsize=7, length=2, pad=1)
                ax.set_xticks([])
                for sp in ['top', 'right', 'bottom']:
                    ax.spines[sp].set_visible(False)
                ax.spines['left'].set_linewidth(0.6)
                if li == 0:
                    ax.set_ylabel(
                        bw, rotation=0, ha='right', va='center',
                        fontsize=9, labelpad=14,
                    )
                axes[(lname, bw)] = ax

            if gene_row:
                axg = fig.add_subplot(gs[-1, li])
                axg.set_xlim(start, end)
                axg.set_ylim(-1.6, 1.6)
                axg.set_xticks([]); axg.set_yticks([])
                for sp in ['top', 'right', 'left', 'bottom']:
                    axg.spines[sp].set_visible(False)
                axg.text(
                    (start + end) / 2, -1.3, lname,
                    ha='center', va='top', fontsize=9, fontstyle='italic',
                )
                if li == 0:
                    axg.set_ylabel(
                        'gene', rotation=0, ha='right', va='center',
                        fontsize=8, labelpad=14,
                    )
                # Draw GTF genes with proper lane packing + labels (only
                # when the locus is narrow enough that ≤ max_label_genes
                # overlap; wider panels show body-only to avoid clutter).
                if hasattr(self, 'gtf') and self.gtf is not None:
                    _draw_gene_lanes(
                        axg, self.gtf, chrom, start, end,
                        color='#2C3E50', prefered_name='gene_name',
                        fontsize=7, n_lanes=2, max_label_genes=20,
                    )
                axes[(lname, '_gene')] = axg

        if title:
            fig.suptitle(title, fontsize=11, y=0.99)
        return fig, axes




def gene_expression_from_bigwigs(
    gtf: pd.DataFrame,
    bigwigs,
    *,
    stat: str = 'mean',
    min_len: int = 200,
    quiet: bool = False,
):
    """Gene × sample expression matrix from (pre-normalised) RNA bigwigs.

    Takes a gene annotation (one row per gene, e.g. the output of
    :func:`epione.utils.get_gene_annotation`) plus a mapping ``{sample_name:
    bigwig_path}`` and returns a DataFrame indexed by gene_name whose
    columns are per-sample mean-over-gene-body bigwig values. When the
    bigwigs are already library-normalised (for example using ``RPKM`` or
    ``BPM`` scaling, which the common pre-processed deposits use), the
    mean is an approximation of gene-level FPKM / CPM.

    Arguments:
        gtf: DataFrame with at least ``chrom, start, end, gene_name``
            (and optionally ``strand``). Typically the result of
            :func:`epione.utils.get_gene_annotation` which is already
            filtered to protein-coding + one row per gene.
        bigwigs: ``{sample_name: bigwig_path}``.
        stat: pyBigWig aggregation. ``'mean'`` → FPKM-like (if bigwig is
            already RPKM-normalised); ``'sum'`` → total coverage;
            ``'max'`` / ``'min'`` also supported.
        min_len: drop genes shorter than this (bp) to avoid noise on very
            short annotations.
        quiet: suppress progress.

    Returns:
        ``pd.DataFrame`` indexed by gene_name; columns match
        ``bigwigs.keys()``. Genes missing on any bigwig's chromosome list
        get NaN in that sample.

    Example:
        >>> import epione as epi
        >>> gtf = epi.utils.get_gene_annotation('gencode.v41lift37.gtf.gz')
        >>> expr = epi.bulk.gene_expression_from_bigwigs(
        ...     gtf,
        ...     {'GV': '.../h_GV_rep12_merge_RNA_10bp_rpkm.bw',
        ...      '4C': '.../h_4C_rep12_merge_RNA_10bp_rpkm.bw'},
        ... )
        >>> activated_4c = expr[(expr['4C'] > 1)
        ...                     & (expr['4C'] > 4 * expr[['GV','MII']].max(1))]
    """
    import pyBigWig

    need_cols = {'chrom', 'start', 'end', 'gene_name'}
    missing = need_cols - set(gtf.columns)
    if missing:
        raise ValueError(f"gtf is missing columns: {sorted(missing)}")

    g = (gtf.dropna(subset=['chrom', 'start', 'end', 'gene_name'])
            .drop_duplicates('gene_name'))
    g = g[(g['end'].astype(int) - g['start'].astype(int)) >= int(min_len)]
    g = g.reset_index(drop=True)

    chroms   = g['chrom'].astype(str).to_numpy()
    starts   = g['start'].astype(int).to_numpy()
    ends     = g['end'].astype(int).to_numpy()
    names    = g['gene_name'].astype(str).to_numpy()
    n_genes  = len(g)

    out = pd.DataFrame(np.full((n_genes, len(bigwigs)), np.nan, dtype=np.float32),
                       index=names, columns=list(bigwigs.keys()))
    for si, (sample, path) in enumerate(bigwigs.items()):
        bwh = pyBigWig.open(str(path))
        bw_chroms = bwh.chroms()
        col = np.full(n_genes, np.nan, dtype=np.float32)
        it = range(n_genes)
        if not quiet:
            it = tqdm(it, desc=f'{sample}', unit='gene', leave=False)
        for gi in it:
            c = chroms[gi]
            cl = bw_chroms.get(c)
            if cl is None:
                continue
            s = max(0, starts[gi])
            e = min(cl, ends[gi])
            if e <= s:
                continue
            try:
                v = bwh.stats(c, s, e, type=stat)
            except Exception:
                continue
            if v and v[0] is not None:
                col[gi] = float(v[0])
        out.iloc[:, si] = col
        bwh.close()
        if not quiet:
            console.level2(f'{sample}: {np.isfinite(col).sum()}/{n_genes} genes '
                           f'with signal; median={np.nanmedian(col):.2f}')
    return out


def format_number_with_k(number):
    units = ['', 'kb', 'Mb', 'Gb', 'Tb']
    unit_index = 0
    
    # 处理负数情况
    sign = ""
    if number < 0:
        sign = "-"
        number = abs(number)
    
    # 依次除以1000，直到达到合适的单位
    while number >= 1000 and unit_index < len(units) - 1:
        number /= 1000.0
        unit_index += 1
    
    return f"{sign}{number:.0f}{units[unit_index]}"                    
                         
        

def plot_matrix(adata,bw_type='body',
                figsize=(2,8),cmap='Greens',
                vmax='auto',vmin='auto',fontsize=12,title=''):
    #from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    fig, ax = plt.subplots(figsize=figsize)
    #pm=np.clip(tss_array.loc[order], 0, 11)

    if vmax=='auto':
        vmax=np.percentile(adata.X.toarray() ,98)
    if vmin=='auto':
        vmin=0

    smp=ax.imshow(adata.X.toarray(),
            aspect='auto',
            interpolation='bilinear',
            origin='upper',
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,)
    cax=plt.colorbar(smp,shrink=0.5)
    cax.ax.tick_params(labelsize=fontsize)
    ax.yaxis.set_visible(False)
    #ax.xaxis.set_visible(False)
    #ax.set_xlabel('TSS')
    # 设置TSS和TES位置的刻度和标签
    if bw_type=='TSS':
        x=adata.uns['bins']
        tss_position = int(adata.shape[1] / 2)  # 1/4位置
        #tes_position = int(2 * 100 / 3)  # 后1/3位置
        ax.set_xticks([0,tss_position,adata.shape[1]])  # 设置刻度位置
        ax.set_xticklabels([format_number_with_k(adata.uns['range'][0]),
                'TSS',format_number_with_k(adata.uns['range'][1])],fontsize=fontsize)  # 设置刻度标签
    elif bw_type=='TES':
        x=adata.uns['bins']
        tes_position = int(adata.shape[1] / 2)
        ax.set_xticks([0,tes_position,adata.shape[1]])  # 设置刻度位置
        ax.set_xticklabels([format_number_with_k(adata.uns['range'][0]),
                'TES',format_number_with_k(adata.uns['range'][1])],fontsize=fontsize)  # 设置刻度标签
    elif bw_type=='body':
        x=adata.uns['bins']
        body_position = int(adata.shape[1] / 2)
        ax.set_xticks([0,body_position,adata.shape[1]])
        ax.set_xticklabels([format_number_with_k(adata.uns['range'][0]),
                'Peak',format_number_with_k(adata.uns['range'][1])],fontsize=fontsize)
    elif bw_type=='all':
        x=adata.uns['bins']
        tss_position = int(adata.shape[1] / 4)
        tes_position = int(adata.shape[1] / 4 *3)
        ax.set_xticks([0,tss_position,tes_position,adata.shape[1]])
        ax.set_xticklabels([format_number_with_k(adata.uns['range'][0]),
                'TSS','TES',format_number_with_k(adata.uns['range'][1])],fontsize=fontsize)
    plt.grid(False)
    plt.tight_layout()
    plt.title(title,fontsize=fontsize)
    return fig,ax

def plot_matrix_line(adata,bw_type='TSS',
                        figsize=(3,3),color='#a51616',
                        linewidth=2,fontsize=13,title=''):

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot([i for i in range(adata.shape[1])],
            adata.X.toarray().mean(axis=0),linewidth=linewidth,color=color)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    if bw_type=='TSS':
        x=adata.uns['bins']
        tss_position = int(adata.shape[1] / 2)  # 1/4位置
        #tes_position = int(2 * 100 / 3)  # 后1/3位置

        ax.set_xticks([0,tss_position,adata.shape[1]])  # 设置刻度位置
        ax.set_xticklabels([format_number_with_k(adata.uns['range'][0]),
                'TSS',format_number_with_k(adata.uns['range'][1])],fontsize=fontsize)  # 设置刻度标签
    elif bw_type=='TES':
        x=adata.uns['bins']
        tes_position = int(adata.shape[1] / 2)
        ax.set_xticks([0,tes_position,adata.shape[1]])  # 设置刻度位置
        ax.set_xticklabels([format_number_with_k(adata.uns['range'][0]),
                'TES',format_number_with_k(adata.uns['range'][1])],fontsize=fontsize)  # 设置刻度标签
    elif bw_type=='body':
        x=adata.uns['bins']
        body_position = int(adata.shape[1] / 2)
        ax.set_xticks([0,body_position,adata.shape[1]])
        ax.set_xticklabels([format_number_with_k(adata.uns['range'][0]),
                'Peak',format_number_with_k(adata.uns['range'][1])],fontsize=fontsize)
    elif bw_type=='all':
        x=adata.uns['bins']
        tss_position = int(adata.shape[1] / 4)
        tes_position = int(adata.shape[1] / 4 *3)
        ax.set_xticks([0,tss_position,tes_position,adata.shape[1]])
        ax.set_xticklabels([format_number_with_k(adata.uns['range'][0]),
                'TSS','TES',format_number_with_k(adata.uns['range'][1])],fontsize=fontsize)
    ax.set_yticklabels(ax.get_yticklabels()[1:],fontsize=fontsize)
    
    plt.grid(False)
    plt.tight_layout()
    plt.title(title,fontsize=fontsize)
    return fig,ax  


class plotloc(object):
    def __init__(self,chrom,start,end):
        self.chr=chrom
        self.start=int(start)
        self.end=int(end)
        self.length=int(end)-int(start)
        #print(self.length)
        self.length_len=len(str(int(self.length)))
        #print(self.length_len)
        
    def cal_start(self):
        #print((np.power(10,self.length_len)))
        new_start=(self.start//(np.power(10,self.length_len-1)))*(np.power(10,self.length_len-1))
        return new_start
        
    def cal_end(self):
        new_end=(self.end//(np.power(10,self.length_len-1)))*(np.power(10,self.length_len-1))
        return new_end
    
    def cal(self):
        new_start=(self.start//(np.power(10,self.length_len-1)))*(np.power(10,self.length_len-1))
        new_end=(self.end//(np.power(10,self.length_len-1)))*(np.power(10,self.length_len-1))
        le=new_end-new_start
        return new_start-le,new_end+le
