"""Pure-format readers / writers (no subprocess, no analysis).

Future home of:

    * BED / BedGraph / narrowPeak / broadPeak readers (currently in
      :mod:`epione.utils._read`)
    * GTF / FASTA / FASTQ readers (currently in :mod:`epione.utils`)
    * Cool / pairs file readers (currently in :mod:`epione.bulk.hic`)
    * AnnData read / write helpers

Populated incrementally during the v0.4 architecture refactor;
this stub exists so :mod:`epione.io` is a stable import target
from PR 1 onward.
"""
from __future__ import annotations

__all__: list[str] = []
