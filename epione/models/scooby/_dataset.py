"""Torch datasets feeding a scooby-style model from Epione fragments.

Two sampling modes, selected by which argument is given:

``groups=``
    Pseudobulk mode. Every item returns a target for each group, and a
    group's conditioning vector is the mean embedding of its members.
    This is the mode to use when per-cell coverage is thin, which on
    spatial ATAC it almost always is.

``neighbors=``
    Per-cell mode, reproducing upstream scooby: sample
    ``cell_sample_size`` cells per item and aggregate each with its
    k-NN neighbourhood.

Both return ``(sequence, rc_flag, targets, embeddings)`` with
``sequence`` of shape ``[input_length, 4]``, ``targets`` of shape
``[n_bins, n_groups]`` and ``embeddings`` of shape
``[n_groups, emb_dim]`` — the layout upstream's model and
``poisson_multinomial`` loss expect.

Sequence extraction goes through ``pysam.FastaFile`` rather than
``enformer_pytorch``'s ``FastaInterval``. pysam is already required for
the fragment reads, so this drops a dependency and avoids upstream's
need to re-create the reader in every worker to dodge a pyfaidx/fork
interaction.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Union

import numpy as np

from ._prepare import FLANK, INPUT_LENGTH, TARGET_LENGTH
from ._targets import ATAC_SCALE, BIN_SIZE, FragmentInsertionReader, groups_to_matrix

__all__ = [
    "ScoobyATACDataset",
    "one_hot_dna",
    "read_fold_bed",
    "worker_init",
]

# Row order matches Borzoi's one-hot convention.
_BASES = "ACGT"
_LUT = np.zeros((256, 4), dtype=np.float32)
for _i, _b in enumerate(_BASES):
    _LUT[ord(_b), _i] = 1.0
    _LUT[ord(_b.lower()), _i] = 1.0
# Anything else (N, IUPAC codes, masked lowercase runs) stays all-zero,
# which is how Borzoi represents an uninformative position.


def one_hot_dna(seq: str) -> np.ndarray:
    """One-hot encode DNA as ``[len(seq), 4]`` in ACGT order.

    Unknown characters encode as an all-zero row rather than raising,
    matching Borzoi's handling of ``N``.
    """
    arr = np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8)
    return _LUT[arr]


def read_fold_bed(
    bed_path: Union[str, Path],
    folds: Optional[Iterable[str]] = None,
    *,
    exclude_folds: Optional[Iterable[str]] = None,
):
    """Read a fold-labelled BED written by :func:`make_fold_bed`.

    Parameters
    ----------
    bed_path
        Four-column BED: ``chrom, start, end, fold``.
    folds
        Keep only these fold labels. ``None`` keeps all.
    exclude_folds
        Drop these fold labels. Applied after ``folds``.

    Returns
    -------
    pandas.DataFrame
        Columns ``chrom, start, end, fold``.
    """
    import pandas as pd

    df = pd.read_csv(
        bed_path,
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "fold"],
        dtype={"chrom": str, "start": int, "end": int, "fold": str},
    )
    if folds is not None:
        keep = set(map(str, folds))
        df = df[df["fold"].isin(keep)]
    if exclude_folds is not None:
        drop = set(map(str, exclude_folds))
        df = df[~df["fold"].isin(drop)]
    if df.empty:
        raise ValueError(
            f"no intervals left in {bed_path} after fold filtering"
        )
    return df.reset_index(drop=True)


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - env dependent
        raise ImportError(
            "ScoobyATACDataset needs torch. Install it in a dedicated "
            "environment; the target layer in epione.models.scooby._targets "
            "works without it."
        ) from exc
    return torch


class ScoobyATACDataset:
    """ATAC targets and DNA for one fragments file.

    Deliberately not a subclass of ``torch.utils.data.Dataset``:
    ``DataLoader`` duck-types map-style datasets on ``__len__`` and
    ``__getitem__``, so staying unsubclassed keeps ``import
    epione.models.scooby`` working in an environment without torch.

    Parameters
    ----------
    reader
        A :class:`~epione.models.scooby._targets.FragmentInsertionReader`
        over the sample's fragments, built with the same barcode order
        as ``embedding``.
    fasta
        Path to an indexed reference FASTA (``.fai`` next to it).
    intervals
        DataFrame from :func:`read_fold_bed`, or any frame with
        ``chrom, start, end``. Rows are the *target* windows; the input
        window is these plus ``flank`` on each side.
    embedding
        ``[n_cells, emb_dim]`` array aligned to ``reader.barcodes``, or
        the DataFrame written by
        :func:`~epione.models.scooby._prepare.export_embedding`.
    groups
        Pseudobulk mode: a sequence of cell-index iterables.
    neighbors
        Per-cell mode: ``[n_cells, n_cells]`` sparse adjacency. A cell's
        target aggregates the cell and its non-zero neighbours.
    cell_sample_size
        Cells sampled per item in per-cell mode.
    cell_weights
        Sampling probabilities over cells in per-cell mode, e.g. to
        up-weight a rare population. Must sum to 1.
    rc_aug
        Randomly reverse-complement. The target is reversed to match.
        Valid for unstranded ATAC; a stranded RNA target would also
        need its strand pair swapped, which this class does not model.
    shift_aug
        Maximum jitter in bp applied coherently to input *and* target.
        Default 0 reproduces upstream, which shifts only the sequence
        and leaves the target in place — harmless there because the
        shift is far below one 32 bp bin, but not something to copy.
    seed
        Seed for cell sampling and augmentation.
    """

    def __init__(
        self,
        reader: FragmentInsertionReader,
        fasta: Union[str, Path],
        intervals,
        embedding,
        *,
        groups: Optional[Sequence[Iterable[int]]] = None,
        neighbors=None,
        cell_sample_size: int = 64,
        cell_weights: Optional[np.ndarray] = None,
        input_length: int = INPUT_LENGTH,
        target_length: int = TARGET_LENGTH,
        bin_size: int = BIN_SIZE,
        scale: float = ATAC_SCALE,
        rc_aug: bool = False,
        shift_aug: int = 0,
        seed: int = 0,
    ) -> None:
        if (groups is None) == (neighbors is None):
            raise ValueError(
                "pass exactly one of groups= (pseudobulk mode) or "
                "neighbors= (per-cell mode)"
            )
        if target_length % bin_size:
            raise ValueError(
                f"target_length ({target_length}) must be a multiple of "
                f"bin_size ({bin_size})"
            )
        if (input_length - target_length) % 2:
            raise ValueError(
                "input_length - target_length must be even so the target "
                "sits centred in the input"
            )

        self.reader = reader
        self.fasta_path = str(fasta)
        self.input_length = int(input_length)
        self.target_length = int(target_length)
        self.flank = (self.input_length - self.target_length) // 2
        self.bin_size = int(bin_size)
        self.scale = float(scale)
        self.rc_aug = bool(rc_aug)
        self.shift_aug = int(shift_aug)
        self.cell_sample_size = int(cell_sample_size)
        self.cell_weights = cell_weights
        self._seed = int(seed)

        self.intervals = intervals.reset_index(drop=True)
        for col in ("chrom", "start", "end"):
            if col not in self.intervals.columns:
                raise ValueError(f"intervals is missing column {col!r}")

        self.embedding = self._coerce_embedding(embedding, reader)

        self.groups = None
        self.group_matrix = None
        self.group_embedding = None
        self.neighbors = None

        if groups is not None:
            self.groups = [np.asarray(list(g), dtype=np.int64) for g in groups]
            self.group_matrix = groups_to_matrix(self.groups, reader.n_cells)
            # A group's conditioning vector is the mean of its members.
            self.group_embedding = np.vstack(
                [
                    self.embedding[g].mean(0) if g.size else
                    np.zeros(self.embedding.shape[1], dtype=np.float32)
                    for g in self.groups
                ]
            ).astype(np.float32)
            empty = [i for i, g in enumerate(self.groups) if g.size == 0]
            if empty:
                raise ValueError(
                    f"groups {empty} are empty; drop them before training"
                )
        else:
            import scipy.sparse as sp

            self.neighbors = sp.csr_matrix(neighbors)
            if self.neighbors.shape[0] != reader.n_cells:
                raise ValueError(
                    f"neighbors has {self.neighbors.shape[0]} rows but "
                    f"reader has {reader.n_cells} cells"
                )
            if self.cell_weights is not None:
                w = np.asarray(self.cell_weights, dtype=np.float64)
                if w.shape != (reader.n_cells,):
                    raise ValueError(
                        f"cell_weights must have shape ({reader.n_cells},)"
                    )
                if not np.isclose(w.sum(), 1.0):
                    raise ValueError("cell_weights must sum to 1")
                self.cell_weights = w

        self._fasta = None
        self._rng = None

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _coerce_embedding(embedding, reader) -> np.ndarray:
        """Accept an array or the parquet DataFrame; verify cell order."""
        if hasattr(embedding, "columns"):
            cols = set(embedding.columns)
            if not {"embedding", "obs_names"} <= cols:
                raise ValueError(
                    "embedding DataFrame needs 'embedding' and 'obs_names' "
                    f"columns, got {sorted(cols)}"
                )
            names = list(map(str, embedding["obs_names"]))
            if names != [str(b) for b in reader.barcodes]:
                raise ValueError(
                    "embedding row order does not match reader.barcodes; "
                    "reindex the embedding before constructing the dataset "
                    "— a silent mismatch here trains every cell against "
                    "another cell's coverage"
                )
            X = np.vstack(list(embedding["embedding"].values))
        else:
            X = np.asarray(embedding)
        X = np.ascontiguousarray(X, dtype=np.float32)
        if X.shape[0] != reader.n_cells:
            raise ValueError(
                f"embedding has {X.shape[0]} rows but reader has "
                f"{reader.n_cells} cells"
            )
        return X

    def _ensure_open(self):
        if self._fasta is None:
            import pysam

            self._fasta = pysam.FastaFile(self.fasta_path)
            self.reader.reopen()
        return self._fasta

    def _get_rng(self, idx: int):
        # Per-item RNG so that workers do not share a stream and an
        # epoch is reproducible given the seed.
        return np.random.default_rng((self._seed, idx))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_fasta"] = None
        state["_rng"] = None
        return state

    def __len__(self) -> int:
        return len(self.intervals)

    # -- item --------------------------------------------------------------

    def __getitem__(self, idx: int):
        torch = _torch()
        fasta = self._ensure_open()
        rng = self._get_rng(idx)

        row = self.intervals.iloc[idx]
        chrom = str(row["chrom"])
        t_start = int(row["start"])
        t_end = int(row["end"])

        shift = 0
        if self.shift_aug:
            shift = int(rng.integers(-self.shift_aug, self.shift_aug + 1))
        t_start += shift
        t_end += shift

        i_start = t_start - self.flank
        i_end = t_end + self.flank

        # Sequence, zero-padded where the input runs off the contig.
        chrom_len = fasta.get_reference_length(chrom)
        fetch_s = max(0, i_start)
        fetch_e = min(chrom_len, i_end)
        seq = fasta.fetch(chrom, fetch_s, fetch_e) if fetch_e > fetch_s else ""
        onehot = np.zeros((self.input_length, 4), dtype=np.float32)
        if seq:
            onehot[fetch_s - i_start : fetch_s - i_start + len(seq)] = one_hot_dna(seq)

        # Targets over the centred window.
        if self.groups is not None:
            targets = self.reader.pseudobulk(
                chrom, t_start, t_end, self.group_matrix,
                bin_size=self.bin_size, scale=self.scale,
            )
            emb = self.group_embedding
        else:
            if self.cell_weights is not None:
                cells = rng.choice(
                    self.reader.n_cells, size=self.cell_sample_size,
                    p=self.cell_weights,
                )
            else:
                cells = rng.choice(
                    self.reader.n_cells, size=self.cell_sample_size,
                )
            groups = [
                np.append(self.neighbors[c].nonzero()[1], c) for c in cells
            ]
            targets = self.reader.pseudobulk(
                chrom, t_start, t_end, groups,
                bin_size=self.bin_size, scale=self.scale,
            )
            emb = self.embedding[cells]

        rc = False
        if self.rc_aug and bool(rng.integers(0, 2)):
            rc = True
            # Reverse-complement: flip bases (ACGT -> TGCA is a reversal
            # of the channel order) and reverse positions. The ATAC
            # target is unstranded, so reversing bins is sufficient.
            onehot = onehot[::-1, ::-1].copy()
            targets = targets[:, ::-1].copy()

        return (
            torch.from_numpy(onehot),
            torch.tensor(rc),
            torch.from_numpy(np.ascontiguousarray(targets.T)),
            torch.from_numpy(np.ascontiguousarray(emb)),
        )


def worker_init(worker_id: int):  # pragma: no cover - needs a DataLoader
    """``DataLoader(worker_init_fn=...)`` hook.

    A forked worker inherits the parent's pysam handles, which then
    return corrupt reads. This drops them so each worker opens its own.
    """
    import torch

    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    ds = info.dataset
    if isinstance(ds, ScoobyATACDataset):
        ds._fasta = None
        ds.reader.reopen()
