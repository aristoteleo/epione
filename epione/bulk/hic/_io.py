"""Read HiCExplorer ``.h5`` matrices.

:func:`h5_to_cool` writes the same ``.cool`` as ``hicConvertFormat
--inputFormat h5 --outputFormat cool`` (HiCExplorer 3.7.6 / HiCMatrix;
deeptools, GPL-3.0): the stored correction factors are inverted into a
multiplicative ``weight`` column and the pixels divided back to raw scale, so
``Cooler.matrix(balance=True)`` (cooltools) and HiCExplorer both see the
corrected values the ``.h5`` held.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from scipy.sparse import triu


def h5_to_cool(h5_path: Union[str, Path], cool_path: Union[str, Path], *, nan_masked: bool = False) -> str:
    """Convert a HiCExplorer ``.h5`` contact matrix to ``.cool``.

    Arguments:
        h5_path: HiCExplorer matrix (``hicBuildMatrix`` / ``hicCorrectMatrix``
            output, PyTables HDF5; Blosc compression is handled by PyTables).
        cool_path: output ``.cool``.
        nan_masked: write ``NaN`` instead of ``0`` as the ``weight`` of
            masked bins. ``hicConvertFormat`` writes ``0``; cooler and
            cooltools only treat ``NaN`` weights as masked, so use ``True``
            before cooltools analyses. HiCExplorer reads both the same way.

    Returns:
        ``str`` path of the written ``.cool``.

    Notes:
        Masked bins (``nan_bins``) keep no pixels when both ends are masked, as
        in ``hicConvertFormat``. Needs ``tables`` (PyTables).
    """
    import cooler
    import tables
    from scipy.sparse import csr_matrix

    with tables.open_file(str(h5_path), "r") as f:
        parts = {p: getattr(f.root.matrix, p).read() for p in ("data", "indices", "indptr", "shape")}
        matrix = csr_matrix((parts["data"], parts["indices"], parts["indptr"]), shape=parts["shape"])
        chrom = [c.decode() if isinstance(c, (bytes, np.bytes_)) else str(c)
                 for c in f.root.intervals.chr_list.read()]
        start = f.root.intervals.start_list.read()
        end = f.root.intervals.end_list.read()
        nan_bins = f.root.nan_bins.read() if hasattr(f.root, "nan_bins") else np.array([])
        cf = None
        if hasattr(f.root, "correction_factors"):
            cf = np.array(f.root.correction_factors.read())
            cf[np.isnan(cf)] = 0
            cf[np.isinf(cf)] = 0

    # HiCMatrix cool.create_cooler_input with fileWasH5=True, pApplyCorrection=True
    matrix.eliminate_zeros()
    if nan_bins is not None and len(nan_bins) > 0:
        ones = np.ones(matrix.shape[0])
        ones[nan_bins] = 0
        matrix.sort_indices()
        inst, feat = matrix.nonzero()
        both = np.logical_not(np.logical_or(ones[inst], ones[feat]))
        matrix.data[both] = 0
        matrix.eliminate_zeros()
    matrix.data[np.isnan(matrix.data)] = 0
    matrix.eliminate_zeros()
    matrix = triu(matrix, format="csr")
    matrix.eliminate_zeros()
    bins = pd.DataFrame({"chrom": chrom, "start": start, "end": end})
    dtypes = {"bin1_id": np.int32, "bin2_id": np.int32, "count": np.int32}
    if cf is not None:
        dtypes["weight"] = np.float32
        with np.errstate(divide="ignore"):
            cf = 1 / np.array(cf).flatten()
        cf[np.isnan(cf)] = 0
        cf[np.isinf(cf)] = 0
        weight = cf.copy()
        weight[np.isnan(weight)] = 1
        if nan_masked:
            weight = np.where(weight == 0, np.nan, weight)
        bins = bins.assign(weight=weight)
        inst, feat = matrix.nonzero()
        factors = cf[inst]
        factors *= cf[feat]
        matrix.data = matrix.data.astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            matrix.data /= factors
        matrix.eliminate_zeros()
    inst, feat = matrix.nonzero()
    pixels = pd.DataFrame(inst, columns=["bin1_id"], dtype=np.int32)
    pixels = pixels.assign(bin2_id=feat)
    pixels = pixels.assign(count=matrix.data)
    if matrix.dtype not in [np.int32, int]:
        dtypes["count"] = matrix.dtype
    cool_path = str(cool_path)
    cooler.create_cooler(cool_uri=cool_path, bins=bins, pixels=pixels, dtypes=dtypes,
                         ordered=True, metadata={"generated-by": "epione.bulk.hic.h5_to_cool"})
    return cool_path
