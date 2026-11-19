# TF binding score (`epi.tl.tf_binding`, occupancy), evaluation

`t_tf_binding_k562.ipynb` is the former `t_tf_binding` tutorial: the evaluation of the default
per-site binding score (`method='occupancy'`, `epi.tl.tf_occupancy`) against ChIP-seq, kept here
with its outputs when the tutorial was rewritten as a PBMC application (2026-10-26).
`t_tf_binding_data.py` builds its inputs (ENCODE K562 snATAC chr19, six K562 ChIP-seq sets);
`t_tf_binding_k562_build.py` regenerates the notebook.

## What it shows

Median per-TF AUROC, accessible motif matches, bound = site inside the factor's ENCODE ChIP-seq
peak. Constants frozen on K562; GM12878, HepG2 and MCF-7 held out.

| | occupancy (default) | TOBIAS | Tn5 count ±100 bp | PRINT | HINT-ATAC |
|---|---|---|---|---|---|
| K562 (design, 88 TFs) | 0.890 | 0.851 | 0.845 | 0.725 | 0.644 |
| GM12878 (36 TFs) | 0.911 | 0.895 | 0.905 | 0.774 | 0.651 |
| HepG2 (131 TFs) | 0.909 | 0.898 | 0.904 | 0.747 | 0.673 |
| MCF-7 (38 TFs) | 0.887 | 0.843 | 0.847 | 0.725 | 0.644 |

On the chr19 slice in the notebook (six TFs): occupancy median AUROC 0.908, PRINT 0.836, Tn5
count 0.870, PWM alone 0.687; PRINT is best for CTCF only (0.938 vs 0.900). Ablations: the site
alone 0.901, β = 0 (no motif term) 0.900. Scoring 15,113 sites took 7.3 s on 8 workers, PRINT
3,974 s on a shared node (same-node benchmarks: about 70x).
