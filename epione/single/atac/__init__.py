"""Single-cell ATAC-seq analysis — counterpart to :mod:`epione.bulk.atac`.

Future home (during the v0.4 refactor PR 2):

    * chromVAR (currently :mod:`epione.tl._chromvar`)
    * Gene activity score (currently :mod:`epione.tl._gene_score`)
    * Peak co-accessibility (currently :mod:`epione.tl._coaccessibility`)
    * Per-barcode BAM split, MACS2 wrapping, pseudobulk
      (currently :mod:`epione.single._bam_split`, ``_call_peaks``,
      ``_pseudobulk``)

Empty in PR 1; populated when :mod:`epione.single`'s ATAC-flavoured
``_*.py`` files migrate here.
"""
from __future__ import annotations

__all__: list[str] = []
