"""Pre-built reference registries — instances of :class:`epione.core.genome.Genome`.

The :class:`Genome` class lives in :mod:`epione.core.genome`; this
module instantiates the canonical genomes (GRCh37/38, GRCm38/39) so
they're available without recreating fetcher closures every time.

Example::

    import epione as epi
    genome = epi.data.GRCh38
    fasta_path = genome.fasta()  # downloads on first call, caches

Distinguished from :mod:`epione.datasets` (downloadable example bio
data for tutorials).
"""
from __future__ import annotations

from .genomes import GRCh37, GRCh38, GRCm38, GRCm39, hg19, hg38, mm10, mm39

__all__ = [
    "GRCh37",
    "GRCh38",
    "GRCm38",
    "GRCm39",
    "hg19",
    "hg38",
    "mm10",
    "mm39",
]
