# Aggregate footprints (`epi.tl.get_footprints`) vs ArchR

Moved from the notes of the `t_footprint` tutorial (2026-10-26).

`epi.tl.get_footprints(..., groupby='celltype', flank=250)` follows ArchR's `getFootprints`.
To compare, run `getFootprints(ArchRProj, positions=..., groupBy='celltype', flank=250)` with the
same motif BEDs and cell-type assignments: the per-cell-type Pearson correlation of the
bias-subtracted ("Subtract") curves is at least 0.98 for well-represented cell types on the
3-sample BMMC data of the tutorial.
