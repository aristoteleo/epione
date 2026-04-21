from ..utils._genome import Genome
import gzip
import numpy as np
from pathlib import Path
from ..utils import console
import os
from typing import Union, Optional, Literal
import pandas as pd
from anndata import AnnData
from ..utils.genome import Genome


from tqdm import tqdm


def _obs_columns(adata) -> list:
    """Return the list of ``obs`` column names in a way that works on both
    plain ``anndata.AnnData`` and snapATAC2-backed AnnData. The snap
    backend exposes an ``obs`` object whose ``.columns`` / ``.keys()``
    attributes don't mirror pandas, so fall back to iterating.
    """
    try:
        return list(adata.obs.columns)
    except Exception:
        pass
    try:
        return list(adata.obs.keys())
    except Exception:
        pass
    return []


def frag_size_distr(
    adata: AnnData,
    *,
    max_recorded_size: int = 1000,
    add_key: str = "frag_size_distr",
    inplace: bool = True,
    whitelist_barcodes: bool = True,
) -> np.ndarray | None:
    """Compute the fragment-size distribution over the dataset.

    Scans the bgzipped fragments.tsv.gz referenced in
    ``adata.uns['files']['fragments']`` and builds a length histogram
    from 0 up to ``max_recorded_size`` (inclusive). Fragments longer
    than ``max_recorded_size`` fall into the index-0 overflow bin
    (matches the snapATAC2 convention).

    Parameters
    ----------
    adata
        AnnData carrying ``uns['files']['fragments']``
        (as produced by :func:`epi.pp.import_fragments`).
    max_recorded_size
        Longest fragment length tracked in a per-length bin.
    add_key
        Key used to store the resulting vector in ``adata.uns``.
    inplace
        When True the histogram is written to ``adata.uns[add_key]``;
        otherwise it is returned.
    whitelist_barcodes
        When True restrict counting to the barcodes present in
        ``adata.obs_names`` (the post-QC cell set). When False every
        fragment in the file contributes — useful for the
        uncurated library-level distribution.
    """
    frag_file = adata.uns.get("files", {}).get("fragments")
    if frag_file is None:
        raise ValueError(
            "adata.uns['files']['fragments'] not set — run epi.pp.import_fragments first"
        )
    console.level1("Computing fragment size distribution for adata...")

    whitelist = None
    if whitelist_barcodes:
        try:
            whitelist = set(map(str, adata.obs_names))
        except Exception:
            whitelist = None

    hist = np.zeros(max_recorded_size + 1, dtype=np.int64)
    opener = gzip.open if str(frag_file).endswith(".gz") else open
    with opener(str(frag_file), "rt") as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            try:
                start = int(parts[1]); end = int(parts[2])
            except ValueError:
                continue
            if whitelist is not None and parts[3] not in whitelist:
                continue
            size = end - start
            if size <= 0:
                continue
            if size > max_recorded_size:
                hist[0] += 1          # overflow bin (snapATAC2 convention)
            else:
                hist[size] += 1

    if inplace:
        adata.uns[add_key] = hist
        console.level2(f"Added fragment size distribution to adata.uns['{add_key}']")
        return None
    return hist

def tsse(
    adata: AnnData,
    gene_anno: Genome | Path,
    *,
    exclude_chroms: list[str] | str | None = ["chrM", "M", "chrMT", "MT"],
    flank_size: int = 2000,
    inplace: bool = True,
) -> dict | None:
    """Per-cell TSS enrichment score (ENCODE-style), pure Python.

    Reuses :func:`epi.pp.tss_enrichment` internally — reads TSS
    coordinates from a GTF/GFF or ``Genome.annotation``, pulls fragments
    around each TSS via pysam, and computes the enrichment of insertion
    density at the TSS over flanking background. No snapATAC2.

    Parameters
    ----------
    adata
        AnnData with ``uns['files']['fragments']`` set.
    gene_anno
        A :class:`epione.utils.Genome` or a path to a GTF/GFF annotation
        file. Used as the TSS source.
    exclude_chroms
        Chromosomes skipped when sampling TSSes.
    flank_size
        Half-width of the pileup window (ENCODE uses 2 kb).
    inplace
        When True, stores ``adata.obs['tsse']``, ``adata.uns['TSS_profile']``,
        ``adata.uns['library_tsse']``, ``adata.uns['frac_overlap_TSS']``.

    Returns
    -------
    dict | None
        When ``inplace=False`` returns a dict with the four fields above.
    """
    # Resolve the annotation path (Genome → GTF/GFF path).
    if isinstance(gene_anno, Genome):
        gene_anno = gene_anno.annotation
    # Load features for TSS pileup via tss_enrichment below.
    from ..utils import read_features
    features = read_features(str(gene_anno))
    console.level1("Computing TSS enrichment score for adata...")
    result = tss_enrichment(
        adata, features=features,
        extend_upstream=flank_size, extend_downstream=flank_size,
        n_tss=None, return_tss=True,
    )
    # ``tss_enrichment`` already stores ``tss_score`` in ``obs``.
    # Build the snapATAC2-equivalent exports on top.
    tss_scores = np.asarray(adata.obs["tss_score"].to_numpy()) if "tss_score" in adata.obs else None
    out = {
        "tsse": tss_scores,
        "library_tsse": float(np.nanmean(tss_scores)) if tss_scores is not None else None,
        "frac_overlap_TSS": None,
        "TSS_profile": adata.uns.get("TSS_profile"),
    }
    if inplace:
        if tss_scores is not None:
            adata.obs["tsse"] = tss_scores
        if out["library_tsse"] is not None:
            adata.uns["library_tsse"] = out["library_tsse"]
        console.level2("Added TSS enrichment score to adata.obs['tsse']")
        return None
    return out


def ensure_tabix_index(path, preset="bed"):
    """
    Ensure that the given .tsv.gz file has a corresponding .tbi index file.
    If not, create one by calling pysam.tabix_index.
    
    Parameters
    ----------
    path : str
        The path of the bgzip compressed file (.tsv.gz)
    preset : str
        The preset of the index, default "bed". Can be "vcf", "gff" etc.
    """
    import pysam
    index_path = path + ".tbi"
    if os.path.exists(index_path):
        print(f"Index already exists: {index_path}")
    else:
        print(f"Index not found, creating: {index_path}")
        # pysam will automatically check if the file is bgzip compressed
        pysam.tabix_index(path, preset=preset, force=True)
        print("Index created!")


def tss_enrichment(
    adata: AnnData,
    features: Optional[pd.DataFrame] = None,
    extend_upstream: int = 2000,
    extend_downstream: int = 2000,
    n_tss: int = 10000,
    return_tss: bool = True,
    random_state=None,
    barcodes: Optional[str] = None,
) -> np.ndarray | list[np.ndarray] | None:
    """
    Calculate TSS enrichment according to ENCODE guidelines. Adds a column `tss_score` to the `.obs` DataFrame and
    optionally returns a tss score object.

    Parameters
    ----------
    data
        AnnData object with peak counts or multimodal MuData object with 'atac' modality.
    features
        A DataFrame with feature annotation, e.g. genes.
        Annotation has to contain columns: Chromosome, Start, End.
    extend_upsteam
        Number of nucleotides to extend every gene upstream (2000 by default to extend gene coordinates to promoter regions)
    extend_downstream
        Number of nucleotides to extend every gene downstream (0 by default)
    n_tss
        How many randomly chosen TSS sites to pile up. The fewer the faster. Default: 2000.
    return_tss
        Whether to return the TSS pileup matrix. Needed for enrichment plots.
    random_state : int, array-like, BitGenerator, np.random.RandomState, optional
        Argument passed to pandas.DataFrame.sample() for sampling features.
    barcodes
        Column name in the .obs of the AnnData
        with barcodes corresponding to the ones in the fragments file.

    Returns
    ----------
    AnnData
        AnnData object with a 'tss_score' column in the .obs slot.

    """
    import pysam
    from tqdm import tqdm
    console.level1("Computing TSS enrichment score for adata...")
    if features is None:
        # Try to gene gene annotation in the data.mod['rna']
        raise ValueError(
                "Argument `features` is required. It should be a BED-like DataFrame with gene coordinates and names."
            )

    if features.shape[0] > n_tss:
        # Only use n_tss randomly chosen sites to make function faster
        features = features.sample(n=n_tss, random_state=random_state)

    n = adata.n_obs
    n_features = extend_downstream + extend_upstream + 1

    # Dictionary with matrix positions
    if barcodes and barcodes in _obs_columns(adata):
        d = {k: v for k, v in zip(adata.obs.loc[:, barcodes], range(n))}
    else:
        d = {k: v for k, v in zip(list(adata.obs_names), range(n))}

    # Not sparse since we expect most positions to be filled
    mx = np.zeros((n, n_features), dtype=int)

    if "files" not in adata.uns:
        raise ValueError("'files' not found in adata.uns")
    if "fragments" not in adata.uns["files"]:
        raise ValueError("'fragments' not found in adata.uns['files']")

    #dertermine the fragments tbi file
    fragments_tbi = adata.uns["files"]["fragments"] + ".tbi"
    if not os.path.exists(fragments_tbi):
        console.level2("Index not found, creating: {}".format(fragments_tbi))
        ensure_tabix_index(adata.uns["files"]["fragments"])

    fragments = pysam.TabixFile(adata.uns["files"]["fragments"], parser=pysam.asBed())

    # Subset the features to the chromosomes present in the fragments file
    chromosomes = fragments.contigs
    features = features[features.Chromosome.isin(chromosomes)]

    # logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Counting fragments in {n} cells for {features.shape[0]} features...")

    for i in tqdm(
        range(features.shape[0]), desc="Fetching Regions..."
    ):  # iterate over features (e.g. genes)
        f = features.iloc[i]
        tss_start = f.Start - extend_upstream  # First position of the TSS region
        for fr in fragments.fetch(
            f.Chromosome, f.Start - extend_upstream, f.Start + extend_downstream
        ):
            try:
                rowind = d[fr.name]  # cell barcode (e.g. GTCAGTCAGTCAGTCA-1)
                score = int(fr.score)  # number of cuts per fragment (e.g. 2)
                colind_start = max(fr.start - tss_start, 0)
                colind_end = min(fr.end - tss_start, n_features)  # ends are non-inclusive in bed
                mx[rowind, colind_start:colind_end] += score
            except:
                pass

    fragments.close()

    anno = pd.DataFrame(
        {"TSS_position": range(-extend_upstream, extend_downstream + 1)},
    )
    anno.index = anno.index.astype(str)
    tss_pileup=AnnData(X=mx, obs=adata.obs, var=anno, dtype=int)

    flank_means, center_means = _calculate_tss_score(data=tss_pileup)

    tss_pileup.X = tss_pileup.X / flank_means[:, None]

    tss_scores = center_means / flank_means

    adata.obs["tss_score"] = tss_scores
    console.level2("Added TSS enrichment score to adata.obs['tss_score']")
    tss_pileup.obs["tss_score"] = tss_scores
    console.level1("Created TSS enrichment score")
    console.level2("Added TSS enrichment score to tss_pileup.obs['tss_score']")
    if return_tss:
        console.level2("Returned TSS enrichment score")
        return tss_pileup
    else:
        return None



def _calculate_tss_score(data, flank_size: int = 100, center_size: int = 1001):
    """
    Calculate TSS enrichment scores (defined by ENCODE) for each cell.

    Parameters
    ----------
    data
        AnnData object with TSS positons as generated by `tss_pileup`.
    flank_size
        Number of nucleotides in the flank on either side of the region (ENCODE standard: 100bp).
    center_size
        Number of nucleotides in the center on either side of the region (ENCODE standard: 1001bp).
    """
    region_size = data.X.shape[1]

    if center_size > region_size:
        raise ValueError(
            f"`center_size` ({center_size}) must smaller than the piled up region ({region_size})."
        )

    if center_size % 2 == 0:
        raise ValueError(f"`center_size` must be an uneven number, but is {center_size}.")

    # Calculate flank means
    flanks = np.hstack((data.X[:, :flank_size], data.X[:, -flank_size:]))
    flank_means = flanks.mean(axis=1)

    # Replace 0 means with population average (to not have 0 division after)
    flank_means[flank_means == 0] = flank_means.mean()

    # Calculate center means
    center_dist = (region_size - center_size) // 2  # distance from the edge of data region
    centers = data.X[:, center_dist:-center_dist]
    center_means = centers.mean(axis=1)

    return flank_means, center_means



    


def nucleosome_signal(
    adata: AnnData,
    n: Union[int, float] = None,
    nucleosome_free_upper_bound: int = 147,
    mononuleosomal_upper_bound: int = 294,
    barcodes: Optional[str] = None,
):
    """
    Computes the ratio of nucleosomal cut fragments to nucleosome-free fragments per cell.
    Nucleosome-free fragments are shorter than 147 bp while mono-mucleosomal fragments are between
    147 bp and 294 bp long.

    Parameters
    ----------
    data
        AnnData object with peak counts or multimodal MuData object with 'atac' modality.
    n
        Number of fragments to count. If `None`, 1e4 fragments * number of cells.
    nucleosome_free_upper_bound
        Number of bases up to which a fragment counts as nucleosome free. Default: 147
    mononuleosomal_upper_bound
        Number of bases up to which a fragment counts as mononuleosomal. Default: 294
    barcodes
        Column name in the .obs of the AnnData
        with barcodes corresponding to the ones in the fragments file.
    """
    console.level1("Computing nucleosome signal for adata...")

    if "files" not in adata.uns or "fragments" not in adata.uns["files"]:
        raise KeyError(
            "There is no fragments file located yet. Run muon.atac.tl.locate_fragments first."
        )

    try:
        import pysam
    except ImportError:
        raise ImportError(
            "pysam is not available. It is required to work with the fragments file. Install pysam from PyPI (`pip install pysam`) or from GitHub (`pip install git+https://github.com/pysam-developers/pysam`)"
        )

    if "files" not in adata.uns or "fragments" not in adata.uns["files"]:
        raise KeyError(
            "There is no fragments file located yet. "
        )

    fragments_tbi = adata.uns["files"]["fragments"] + ".tbi"
    if not os.path.exists(fragments_tbi):
        console.level2("Index not found, creating: {}".format(fragments_tbi))
        ensure_tabix_index(adata.uns["files"]["fragments"])

    fragments = pysam.TabixFile(adata.uns["files"]["fragments"], parser=pysam.asBed())

    # Dictionary with matrix row indices. ``adata.obs_names`` works on
    # both plain anndata.AnnData and snapATAC2-backed AnnData, whereas
    # ``adata.obs.index`` only exists on the former.
    if barcodes and barcodes in _obs_columns(adata):
        d = {k: v for k, v in zip(adata.obs.loc[:, barcodes], range(adata.n_obs))}
    else:
        d = {k: v for k, v in zip(list(adata.obs_names), range(adata.n_obs))}
    mat = np.zeros(shape=(adata.n_obs, 2), dtype=int)

    fr = fragments.fetch()

    if n is None:
        n = int(adata.n_obs * 1e4)
    else:
        n = int(n)  # Cast n to int

    for i in tqdm(range(n), desc="Reading Fragments"):
        try:
            f = next(fr)
            length = f.end - f.start
            row_ind = d[f.name]
            if length < nucleosome_free_upper_bound:
                mat[row_ind, 0] += 1
            elif length < mononuleosomal_upper_bound:
                mat[row_ind, 1] += 1
        except StopIteration:
            break
        except KeyError:
            pass
        # if i % 1000000 == 0:
        #     print(f"Read {i/1000000} Mio. fragments.", end='\r')

    # Prevent division by 0
    mat[mat[:, 0] == 0, :] += 1

    # Calculate nucleosome signal
    nucleosome_enrichment = mat[:, 1] / mat[:, 0]
    # nucleosome_enrichment[mat[:,0] == 0] = 0

    adata.obs["nucleosome_signal"] = nucleosome_enrichment

    # Message for the user
    console.level2('Added a "nucleosome_signal" column to the .obs slot of the AnnData object')
    console.level1("Created nucleosome signal")
    console.level2("Added nucleosome signal to adata.obs['nucleosome_signal']")
    return None


def _find_most_accessible_features(
    feature_count,
    filter_lower_quantile,
    filter_upper_quantile,
    total_features,
) -> np.ndarray:
    idx = np.argsort(feature_count)
    for i in range(idx.size):
        if feature_count[idx[i]] > 0:
            break
    idx = idx[i:]
    n = idx.size
    n_lower = int(filter_lower_quantile * n)
    n_upper = int(filter_upper_quantile * n)
    idx = idx[n_lower:n-n_upper]
    return idx[::-1][:total_features]
 
def _chunked_column_sum(X, chunk: int = 5000) -> np.ndarray:
    """Per-column sum over X, chunked row-wise so anndataoom's backed
    arrays don't materialise the full matrix."""
    import scipy.sparse as sp
    n_vars = X.shape[1]
    out = np.zeros(n_vars, dtype=np.float64)
    if hasattr(X, "chunked"):
        for start, end, block in X.chunked(chunk):
            out += np.asarray(block.sum(axis=0)).ravel()
    else:
        # plain ndarray / scipy.sparse
        if sp.issparse(X):
            out += np.asarray(X.sum(axis=0)).ravel()
        else:
            out += np.asarray(X).sum(axis=0)
    return out


def _bed_overlap_mask(var_names, bed_path: Path) -> np.ndarray:
    """Return a boolean mask over ``var_names`` (``chrN:s-e`` format)
    marking which ones overlap any interval in ``bed_path``. Pure Python
    replacement for snapatac2's internal.intersect_bed.
    """
    import re, gzip
    from collections import defaultdict
    _re = re.compile(r"^([^:]+):(\d+)-(\d+)$")

    # Parse BED intervals.
    opener = gzip.open if str(bed_path).endswith(".gz") else open
    intervals = defaultdict(list)
    with opener(str(bed_path), "rt") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            try:
                intervals[parts[0]].append((int(parts[1]), int(parts[2])))
            except ValueError:
                continue
    for chrom in intervals:
        intervals[chrom].sort()

    mask = np.zeros(len(var_names), dtype=bool)
    for i, name in enumerate(var_names):
        m = _re.match(str(name))
        if m is None:
            continue
        chrom, s, e = m.group(1), int(m.group(2)), int(m.group(3))
        ivs = intervals.get(chrom)
        if not ivs:
            continue
        # Binary-search for first interval whose start >= e.
        lo, hi = 0, len(ivs)
        while lo < hi:
            mid = (lo + hi) // 2
            if ivs[mid][0] >= e:
                hi = mid
            else:
                lo = mid + 1
        k = lo - 1
        while k >= 0 and ivs[k][0] < e:
            if ivs[k][1] > s:
                mask[i] = True
                break
            k -= 1
    return mask


def select_features(
    adata: AnnData,
    n_features: int = 500000,
    filter_lower_quantile: float = 0.005,
    filter_upper_quantile: float = 0.005,
    whitelist: Optional[Path] = None,
    blacklist: Optional[Path] = None,
    inplace: bool = True,
    verbose: bool = True,
) -> np.ndarray | None:
    """Pick the top-``n_features`` most-accessible features.

    Pure-Python replacement for the old snapATAC2-backed implementation.
    Features are ranked by per-column sum across cells. ``filter_*_quantile``
    removes rare and ubiquitous outliers before picking the top-N. Optional
    whitelist / blacklist BEDs intersect with the var names
    (``chrN:start-end`` format).
    """
    count = _chunked_column_sum(adata.X)
    if inplace:
        try:
            adata.var['count'] = count
        except Exception:
            pass

    selected = _find_most_accessible_features(
        count, filter_lower_quantile, filter_upper_quantile, n_features,
    )

    # Optional BED overlap filters.
    if blacklist is not None:
        mask = _bed_overlap_mask(adata.var_names, blacklist)
        selected = selected[~mask[selected]]

    result = np.zeros(adata.n_vars, dtype=bool)
    result[selected] = True

    if whitelist is not None:
        wl = _bed_overlap_mask(adata.var_names, whitelist)
        wl &= count != 0
        result |= wl

    if verbose:
        console.level1(f"Selected {int(result.sum()):,} features.")

    if inplace:
        try:
            adata.var["selected"] = result
        except Exception:
            pass
        return None
    return result

