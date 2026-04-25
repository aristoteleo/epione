"""Pre-built reference registries — instances of :class:`epione.core.genome.Genome`,
PWM databases, etc.

Use cases:

    * ``epi.data.GRCh38`` — pre-built ``Genome`` for human;
      pulls FASTA / GTF / chrom.sizes on first access, caches under
      ``~/.cache/epione/``.
    * ``epi.data.GRCm39`` — mouse counterpart.
    * ``epi.data.JASPAR_2024`` — PWM database object ready to feed
      ``epione.core.motifs.find_motifs_genome``.

Distinguished from :mod:`epione.datasets` (example bio data for
tutorials) — *this* module is for foundational references that nearly
every analysis uses.

Populated during the v0.4 refactor when :mod:`epione.utils.genome`
splits class definitions (→ :mod:`epione.core.genome`) from instances
(→ here).
"""
from __future__ import annotations

__all__: list[str] = []
