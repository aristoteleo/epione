"""Turn Epione objects into the inputs a scooby-style model expects.

Upstream scooby is driven by four side-car artefacts, produced by a
chain of notebooks: a cell-embedding parquet, a k-NN graph, a BED of
genomic windows carrying a fold label, and (for evaluation) pseudobulk
bigWigs. This module produces them from an Epione ``AnnData`` plus a
``{chrom: length}`` mapping, so no part of the pipeline needs the forked
snapATAC2 or a genome-wide insertion matrix.

Fold assignment is handled here rather than inherited from Borzoi's
precomputed human BED, because the datasets Epione targets are often
mouse or non-standard assemblies. It is also the step most easily got
wrong: the model reads a 524,288 bp input to predict a 196,608 bp
target, so two adjacent target windows have inputs overlapping by
163,840 bp on each flank. Assigning windows to folds at random lets a
training window's *input* cover a validation window's *target*, which
leaks. Folds are therefore assigned to contiguous genomic blocks, and
windows lying within one flank-width of a block edge are dropped so
that no retained training input reaches into another fold's target.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Union

import numpy as np

# Borzoi/scooby geometry: 524,288 bp in, 6,144 bins x 32 bp out.
INPUT_LENGTH = 524_288
TARGET_LENGTH = 196_608
FLANK = (INPUT_LENGTH - TARGET_LENGTH) // 2  # 163,840

__all__ = [
    "FLANK",
    "INPUT_LENGTH",
    "TARGET_LENGTH",
    "expected_fragments_per_window",
    "export_embedding",
    "export_neighbors",
    "groups_from_obs",
    "make_fold_bed",
]


# ---------------------------------------------------------------------------
# Cell-side artefacts
# ---------------------------------------------------------------------------

def export_embedding(
    adata,
    obsm_key: str,
    out_path: Union[str, Path],
    *,
    n_dims: Optional[int] = None,
    drop_first: bool = False,
    scale: bool = False,
) -> "object":
    """Write ``adata.obsm[obsm_key]`` as a scooby embedding parquet.

    The on-disk schema is upstream's: one row per cell, column
    ``embedding`` holding the vector as a list and column ``obs_names``
    holding the barcode. Row order follows ``adata.obs_names``, which is
    the order the dataset and the model's hypernetwork both assume.

    Parameters
    ----------
    adata
        AnnData carrying the embedding.
    obsm_key
        Key in ``.obsm`` — e.g. ``'X_lsi'`` from
        :func:`epione.tl.iterative_lsi`, or a joint embedding.
    out_path
        Destination ``.pq``.
    n_dims
        Keep only the leading ``n_dims`` components.
    drop_first
        Drop component 0 before anything else. LSI's first component
        tracks sequencing depth rather than biology; ArchR and Signac
        both discard it by convention. Applied before ``n_dims``.
    scale
        Z-score each retained component across cells. Useful when the
        embedding has very uneven per-component scale, which makes the
        hypernetwork's first linear layer badly conditioned.

    Returns
    -------
    pandas.DataFrame
        What was written, for inspection.
    """
    import pandas as pd

    if obsm_key not in adata.obsm:
        raise KeyError(
            f"adata.obsm has no key {obsm_key!r}; available: "
            f"{list(adata.obsm.keys())}"
        )
    X = np.asarray(adata.obsm[obsm_key], dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"obsm[{obsm_key!r}] must be 2-D, got {X.shape}")

    if drop_first:
        if X.shape[1] < 2:
            raise ValueError(
                "drop_first needs at least 2 components, "
                f"obsm[{obsm_key!r}] has {X.shape[1]}"
            )
        X = X[:, 1:]
    if n_dims is not None:
        if n_dims > X.shape[1]:
            raise ValueError(
                f"n_dims={n_dims} exceeds available {X.shape[1]} components"
            )
        X = X[:, :n_dims]
    if scale:
        mu = X.mean(0, keepdims=True)
        sd = X.std(0, keepdims=True)
        X = (X - mu) / np.where(sd > 0, sd, 1.0)

    if not np.isfinite(X).all():
        raise ValueError(
            "embedding contains NaN/Inf; clean it before training"
        )

    df = pd.DataFrame(
        {
            "embedding": list(X.astype(np.float32)),
            "obs_names": list(map(str, adata.obs_names)),
        }
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(out_path)
    except ImportError as exc:
        # Parquet is upstream scooby's embedding format, but neither
        # engine is a core Epione dependency.
        raise ImportError(
            "writing the embedding needs a parquet engine; "
            "`pip install pyarrow`. The DataFrame itself is returned "
            "regardless and can be passed to ScoobyATACDataset directly."
        ) from exc
    return df


def export_neighbors(
    adata,
    out_path: Union[str, Path],
    *,
    obsp_key: str = "distances",
    include_self: bool = False,
) -> "object":
    """Write a k-NN graph as the ``.npz`` the dataset consumes.

    Upstream uses ``adata.obsp['distances']`` from
    ``scanpy.pp.neighbors`` with a large ``n_neighbors`` and treats any
    non-zero entry as a neighbour, aggregating a cell with its
    neighbours to make its target less sparse.

    How many neighbours are needed is a property of the data, not a
    style choice. At ~1.5e4 fragments per cell over a 2.7e9 bp genome a
    single cell puts ~1 fragment into a 196 kb window, so a graph with
    k in the low tens still yields an almost empty target. Check
    :func:`expected_fragments_per_window` before choosing k.
    """
    import scipy.sparse as sp

    if obsp_key not in adata.obsp:
        raise KeyError(
            f"adata.obsp has no key {obsp_key!r}; run a neighbors step "
            f"first. Available: {list(adata.obsp.keys())}"
        )
    G = sp.csr_matrix(adata.obsp[obsp_key])
    if include_self:
        G = (G + sp.eye(G.shape[0], format="csr")).tocsr()
    G.eliminate_zeros()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(str(out_path), G)
    return G


def groups_from_obs(
    adata,
    key: str,
    *,
    min_cells: int = 1,
) -> tuple[list[np.ndarray], list[str]]:
    """Group cells by an ``.obs`` column, for pseudobulk targets.

    Returns ``(groups, names)`` where ``groups[i]`` holds the integer
    row indices of the cells in ``names[i]``. Feed ``groups`` straight
    to :meth:`FragmentInsertionReader.pseudobulk`.

    This is the mode to prefer on sparse data: instead of one noisy
    target per cell, one deep target per biological group (cell type,
    timepoint, spatial domain, or their interaction).
    """
    import pandas as pd

    if key not in adata.obs:
        raise KeyError(
            f"adata.obs has no column {key!r}; available: "
            f"{list(adata.obs.columns)}"
        )
    labels = pd.Series(np.asarray(adata.obs[key])).astype(str)
    groups: list[np.ndarray] = []
    names: list[str] = []
    for name, idx in labels.groupby(labels).groups.items():
        rows = np.asarray(idx, dtype=np.int64)
        if rows.size < min_cells:
            continue
        groups.append(rows)
        names.append(str(name))
    order = np.argsort(names)
    return [groups[i] for i in order], [names[i] for i in order]


def expected_fragments_per_window(
    n_fragments_per_cell: float,
    genome_length: float,
    n_cells_per_group: int = 1,
    target_length: int = TARGET_LENGTH,
) -> float:
    """Expected fragments a group contributes to one target window.

    A sanity check to run *before* training. Values below ~1e3 mean the
    per-window profile is dominated by sampling noise and the group is
    too small, however good the model is.
    """
    return (
        float(n_fragments_per_cell)
        * float(n_cells_per_group)
        * float(target_length)
        / float(genome_length)
    )


# ---------------------------------------------------------------------------
# Genome-side artefacts
# ---------------------------------------------------------------------------

def make_fold_bed(
    chrom_sizes: Mapping[str, int],
    out_path: Union[str, Path],
    *,
    target_length: int = TARGET_LENGTH,
    flank: int = FLANK,
    n_folds: int = 8,
    block_windows: int = 16,
    edge_margin: int = 100,
    chroms: Optional[Sequence[str]] = None,
    exclude: Iterable[str] = ("chrM", "M", "chrMT", "MT", "chrY"),
    seed: int = 0,
) -> "object":
    """Tile the genome into fold-labelled target windows.

    Windows are ``target_length`` wide and non-overlapping. Consecutive
    runs of ``block_windows`` windows form a block; blocks are shuffled
    and dealt round-robin to ``n_folds`` folds. Windows within ``flank``
    bp of a block boundary are dropped, so a retained window's full
    ``target_length + 2 * flank`` input never reaches a target window
    belonging to a different fold.

    Parameters
    ----------
    chrom_sizes
        ``{chrom: length}``.
    out_path
        Destination BED. Columns are ``chrom, start, end, fold``, which
        is the layout ``enformer_pytorch.GenomeIntervalDataset`` reads
        as ``column_1..column_4``.
    n_folds
        Number of folds. Upstream convention names them ``fold0`` ..
        ``fold{n-1}`` and selects train/val/test by name.
    block_windows
        Windows per contiguous block. The block must be comfortably
        wider than ``flank`` or nearly everything is dropped; the
        default gives blocks of ~3.1 Mb against a 164 kb flank.
    edge_margin
        Keep windows at least this far from either end of a contig.
        Upstream scooby cannot produce a target for a window ending
        within ``custom_read_length + 10`` (= 100) bp of a contig end:
        its trim becomes ``dense[100:0]``, an empty array, and the
        pooling step then raises. The default keeps the two pipelines
        comparable; raise it if a different ``custom_read_length`` is
        in use.
    chroms
        Restrict to these chromosomes, in this order. Defaults to every
        key of ``chrom_sizes`` not in ``exclude``.
    exclude
        Chromosome names to skip entirely.

    Returns
    -------
    pandas.DataFrame
        The written table.
    """
    import pandas as pd

    if block_windows * target_length <= 2 * flank:
        raise ValueError(
            f"block of {block_windows} x {target_length} bp is not wider "
            f"than the {2 * flank} bp dropped at its edges; raise "
            "block_windows"
        )

    excl = set(exclude)
    if chroms is None:
        chroms = [c for c in chrom_sizes if c not in excl]
    else:
        chroms = [c for c in chroms if c not in excl]

    # Enumerate blocks genome-wide first, so folds are balanced across
    # chromosomes rather than within each one.
    blocks: list[tuple[str, int, int]] = []  # (chrom, first_win, n_win)
    for chrom in chroms:
        length = int(chrom_sizes[chrom])
        n_win = length // target_length
        if n_win == 0:
            continue
        for b0 in range(0, n_win, block_windows):
            blocks.append((chrom, b0, min(block_windows, n_win - b0)))

    if not blocks:
        raise ValueError(
            "no window fits; is target_length larger than every "
            "chromosome in chrom_sizes?"
        )

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(blocks))

    rows = []
    for rank, bi in enumerate(perm):
        chrom, b0, n_win = blocks[bi]
        fold = f"fold{rank % n_folds}"
        block_start = b0 * target_length
        block_end = (b0 + n_win) * target_length
        chrom_len = int(chrom_sizes[chrom])
        for w in range(b0, b0 + n_win):
            start = w * target_length
            end = start + target_length
            # Drop windows whose input context would cross the block
            # edge into a neighbouring, differently-folded block.
            if start - flank < block_start or end + flank > block_end:
                continue
            # Keep clear of contig ends (see ``edge_margin``).
            if start < edge_margin or end > chrom_len - edge_margin:
                continue
            rows.append((chrom, start, end, fold))

    if not rows:
        raise ValueError(
            "every window was dropped by the flank buffer; raise "
            "block_windows or lower flank"
        )

    df = pd.DataFrame(rows, columns=["chrom", "start", "end", "fold"])
    df = df.sort_values(["chrom", "start"]).reset_index(drop=True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", header=False, index=False)
    return df
