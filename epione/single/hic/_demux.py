"""Per-barcode / per-cluster demultiplexing of a single-cell pairs file.

Droplet Hi-C (Chang 2024) and similar single-cell Hi-C protocols
co-encapsulate many cells per droplet, so the aligned ``.pairs`` file
holds reads from every cell mixed together — each line carries a
cell-barcode tag (Chang's convention: appended to ``readID`` as
``read123_<16bp-barcode>``). To get per-celltype contact maps you need
two passes:

* split that mixed pairs file into one pairs file per celltype
  (:func:`demux_pairs_by_barcode`); then
* run ``pairs_to_cool`` on each, giving one ``.cool`` per celltype
  (:func:`pseudobulk_by_celltype`).

This module is the upstream of Chang 2024 Fig 1d/e/f/g/h — the per-
celltype compartment / loop / correlation comparisons.
"""
from __future__ import annotations

import gzip
import io
import re
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Union

import pandas as pd


def _open_text(path: Union[str, Path], mode: str = "rt"):
    """Stream-open a ``.pairs`` or ``.pairs.gz`` file as text."""
    p = str(path)
    if p.endswith(".gz") or p.endswith(".bgz"):
        return gzip.open(p, mode)
    return open(p, mode)


def _default_extract_barcode(read_id: str) -> str:
    """Last underscore-delimited token of ``read_id`` (Chang 2024 layout).

    Example: ``"NS500_ATAC123_AAACCTGAGAACTGTA-1"`` → ``"AAACCTGAGAACTGTA-1"``.
    Override via :func:`demux_pairs_by_barcode`'s ``extract_barcode=``.
    """
    return read_id.rsplit("_", 1)[-1] if "_" in read_id else read_id


def demux_pairs_by_barcode(
    pairs_path: Union[str, Path],
    barcode_to_group: Mapping[str, str],
    output_dir: Union[str, Path],
    *,
    barcode_column: Optional[int] = None,
    extract_barcode: Optional[Callable[[str], str]] = None,
    gzip_output: bool = True,
    drop_unassigned: bool = True,
) -> Dict[str, Path]:
    """Split a multi-cell ``.pairs`` file into one pairs file per group.

    Each input line is read, its cell-barcode is extracted, and the
    line is appended to the output stream of the group that barcode
    maps to (e.g. ``"AAACCTGAGAACTGTA-1" → "T_cell"``). The 4DN
    pairs header (``##`` + ``#chromsize:`` + ``#columns:`` lines) is
    propagated verbatim into every output file.

    Arguments:
        pairs_path: input ``.pairs`` or ``.pairs.gz``.
        barcode_to_group: mapping ``barcode → group_name`` (e.g.
            celltype label from a clustering). Pass a ``dict`` or a
            ``pandas.Series``.
        output_dir: directory for ``{group}.pairs.gz``. Created if
            absent.
        barcode_column: 1-based column index in the pairs body
            (``readID`` is column 1). If ``None``, ``extract_barcode``
            is applied to ``readID`` instead.
        extract_barcode: callable ``readID → barcode``. Default =
            split on the last ``_`` (Chang 2024 convention). Ignored
            if ``barcode_column`` is given.
        gzip_output: write ``.pairs.gz`` (default) or plain ``.pairs``.
        drop_unassigned: skip reads whose barcode isn't in
            ``barcode_to_group``. If ``False``, write them to a
            ``"_unassigned.pairs.gz"`` file.

    Returns:
        ``{group_name: output_path}``. Writes one file per group that
        actually got at least one read.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(barcode_to_group, pd.Series):
        bc2g: Dict[str, str] = barcode_to_group.to_dict()
    else:
        bc2g = dict(barcode_to_group)

    if extract_barcode is None:
        extract_barcode = _default_extract_barcode

    suffix = ".pairs.gz" if gzip_output else ".pairs"
    handles: Dict[str, io.TextIOBase] = {}
    paths: Dict[str, Path] = {}
    header_lines: list[str] = []

    def _open_group(group: str) -> io.TextIOBase:
        if group in handles:
            return handles[group]
        out = output_dir / f"{group}{suffix}"
        fh = _open_text(out, "wt")
        for h in header_lines:
            fh.write(h)
        handles[group] = fh
        paths[group] = out
        return fh

    with _open_text(pairs_path, "rt") as fin:
        for line in fin:
            if line.startswith("#"):
                header_lines.append(line)
                continue
            fields = line.rstrip("\n").split("\t")
            if barcode_column is not None:
                bc = fields[barcode_column - 1]
            else:
                bc = extract_barcode(fields[0])
            group = bc2g.get(bc)
            if group is None:
                if drop_unassigned:
                    continue
                group = "_unassigned"
            _open_group(group).write(line)

    for fh in handles.values():
        fh.close()
    return paths


def pseudobulk_by_celltype(
    pairs_path: Union[str, Path],
    barcode_to_group: Mapping[str, str],
    chrom_sizes: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    binsize: int = 100_000,
    barcode_column: Optional[int] = None,
    extract_barcode: Optional[Callable[[str], str]] = None,
    keep_pairs: bool = False,
    balance: bool = True,
) -> Dict[str, Path]:
    """End-to-end: demux a multi-cell pairs file then build one cool per group.

    Pipeline:

    1. :func:`demux_pairs_by_barcode` splits ``pairs_path`` into one
       pairs file per group.
    2. ``epione.upstream.pairs_to_cool`` builds ``{group}.cool`` from
       each pairs file at the chosen ``binsize``.
    3. (Optional) ``epione.bulk.hic.balance_cool`` ICE-balances each
       cool — needed for downstream :func:`cluster_correlation` /
       :func:`epione.bulk.hic.compartments`.
    4. Intermediate per-group pairs files are deleted unless
       ``keep_pairs=True``.

    Arguments:
        pairs_path, barcode_to_group, barcode_column, extract_barcode:
            forwarded to :func:`demux_pairs_by_barcode`.
        chrom_sizes: tab-separated ``chrom\\tsize`` file, used by
            ``pairs_to_cool``.
        output_dir: where ``{group}.cool`` go (and the temporary
            pairs files).
        binsize: cool resolution in bp (Chang 2024 uses 100 kb for
            the per-celltype maps in Fig 1).
        keep_pairs: keep the demuxed ``.pairs.gz`` files alongside
            the cools. Default deletes them.
        balance: ICE-balance each cool. Default ``True``.

    Returns:
        ``{group_name: cool_path}``.
    """
    from epione.upstream import pairs_to_cool
    from epione.bulk.hic import balance_cool

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs_paths = demux_pairs_by_barcode(
        pairs_path, barcode_to_group, output_dir,
        barcode_column=barcode_column,
        extract_barcode=extract_barcode,
        gzip_output=True, drop_unassigned=True,
    )

    cool_paths: Dict[str, Path] = {}
    for group, ppath in pairs_paths.items():
        cool = output_dir / f"{group}.cool"
        pairs_to_cool(ppath, chrom_sizes, cool, binsize=int(binsize))
        if balance:
            balance_cool(cool, mad_max=10, min_nnz=1, ignore_diags=0)
        cool_paths[group] = cool
        if not keep_pairs:
            try:
                ppath.unlink()
            except OSError:
                pass

    return cool_paths
