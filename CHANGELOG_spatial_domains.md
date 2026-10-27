# Breaking change: default spatial domains

`epi.tl.spatial_domains(adata, n_domains)` now runs `method="aegle"` (`epi.tl.aegle`) on the
spots x peaks matrix. The previous default, mclust EEE on an embedding (tutorial:
`use_rep="X_birmt_smooth"`, `use_pca=False`), is still available as `method="mclust"`.

Pre-registered held-out evaluation (paper/fig2/spatial/native/PREREG.md), ARI vs anatomical labels:

| method | E15.5 | E18.5 | mean |
|---|---|---|---|
| AEGLE | 0.463 (5 seeds) | 0.442 (1) | 0.452 |
| CellCharter (X_birmt) | 0.454 (1) | 0.448 (1) | 0.451 |
| previous default | 0.268 (1) | 0.363 (1) | 0.316 |
| SnapATAC2 (non-spatial) | - | 0.268 (1) | - |

Caveats: tie with CellCharter; STAGATE / GraphST / SpaceFlow / SpatialGlue and the R/TF ports
did not finish or were not available; the speed rule (not slower than the previous default)
was not met on an oversubscribed node and not fairly measured. Default switched by user decision.
