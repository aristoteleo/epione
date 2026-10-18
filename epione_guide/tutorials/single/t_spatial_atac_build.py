"""Generate t_spatial_atac.ipynb (spatial ATAC / ATAC-RNA analysis).

Execute with ``jupyter nbconvert --to notebook --execute --inplace``; needs
``torch`` and ``pymclustR`` and about 40 min on 8 CPU cores, most of it the
tile matrix, the motif scan and mclust on 9k spots. Data go to
``$EPIONE_SPATIAL_DIR`` (default ``./data_spatial``); the hg38 genome to
``$EPIONE_DATA_DIR``.
"""
import pathlib

import nbformat as nbf

CELLS = []


def md(src):
    CELLS.append(nbf.v4.new_markdown_cell(src.strip("\n")))


def code(src):
    CELLS.append(nbf.v4.new_code_cell(src.strip("\n")))


md(r"""
# Spatial ATAC-seq and spatial ATAC-RNA-seq

DBiT-seq based assays put barcodes on a grid of microfluidic channels and so
give every 10-50 um pixel of a tissue section its own chromatin accessibility
profile: spatial-ATAC-seq (Deng et al., *Nature* 2022), and spatial
ATAC-RNA-seq and CUT&Tag-RNA-seq, which add the transcriptome of the same
pixel (Zhang et al., *Nature* 2023). Each pixel ("spot") is a mini-bulk of a
few cells. This tutorial goes from fragments to spatial domains:

| Step | Epione function |
|---|---|
| fragments + barcode -> (x, y) table + tissue image -> AnnData / MuData | `epi.io.read_spatial_atac`, `epi.io.read_spatial_atac_rna`, `epi.io.add_spatial` |
| per-spot QC on the tissue | `epi.pp.tsse`, `epi.pl.spatial_qc` |
| spot maps of any value | `epi.pl.spatial` |
| spot graph, neighbourhood smoothing, Moran's I | `epi.tl.spatial_neighbors`, `epi.tl.spatial_smooth`, `epi.tl.spatial_moran` |
| spatial domains | `epi.tl.spatial_domains` |
| ATAC + RNA integration | `epi.tl.spatialglue` (port of SpatialGlue, Long et al., *Nature Methods* 2024) |

The AnnData follow the scanpy / squidpy layout (`obsm["spatial"]`,
`uns["spatial"][library_id]`), so `sc.pl.spatial` and `squidpy.gr.*` work on
them too.

**Data.** A human hippocampus section profiled by spatial ATAC-RNA-seq at
50 um (Zhang et al. 2023, GEO GSE205055: 2,500 spots, 45 M fragments), read
from the raw GEO files. In the last part we rerun SpatialGlue on the mouse
brain spatial ATAC-RNA-seq data released with the SpatialGlue paper.
""")

md("## Setup")
code(r"""
import os
import pathlib
import warnings

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import epione as epi

warnings.filterwarnings("ignore")
epi.pl.plot_set()
DATA = pathlib.Path(os.environ.get("EPIONE_SPATIAL_DIR", "data_spatial"))
genome = epi.utils.genome.hg38
""")

md(r"""
## Read fragments, RNA and spot positions

`zhang2023_hippocampus_atac_rna` downloads the three GEO files (fragments,
RNA counts, and the Space Ranger style `spatial/` folder with
`tissue_positions_list.csv`, `scalefactors_json.json` and the tissue image).
`read_spatial_atac_rna` scans the fragments of the on-tissue spots only (the
tissue mask in `in_tissue`), reads the RNA table, aligns the two modalities on
the spot barcode (the fragments carry a `-1` suffix, the RNA table does not)
and attaches the coordinates and the image. A bare DBiT whitelist with
columns `barcode, x, y` works as `positions` too.
""")
code(r"""
paths = epi.datasets.zhang2023_hippocampus_atac_rna(data_dir=DATA / "zhang2023_hippocampus")
mdata = epi.io.read_spatial_atac_rna(
    paths["fragments"], paths["rna"], paths["positions"], genome,
    spatial_dir=paths["spatial_dir"], library_id="hippocampus",
)
atac, rna = mdata["atac"], mdata["rna"]
mdata
""")

md(r"""
## Spot QC on the tissue

TSS enrichment and unique fragments per spot, drawn as 50 um squares on the
section. The straight lines in the fragment map follow the microfluidic
channels: in DBiT data one row or column of the barcode grid can be deeper
than its neighbours. This is a technical effect that downstream steps
should not mistake for biology.
""")
code(r"""
genes = epi.utils.get_gene_annotation(genome)
epi.pp.tsse(atac, genes)
atac.obs[["n_fragment", "tsse", "frac_dup"]].describe().round(2)
""")
code(r"""
epi.pl.spatial_qc(atac, keys=("n_fragment", "tsse", "frac_dup"), show=False)
plt.show()
""")
code(r"""
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
epi.pl.spatial(atac, None, alpha=0.0, ax=axs[0], title="H&E (hires image)", show=False)
epi.pl.spatial(atac, "tsse", alpha=0.6, ax=axs[1], show=False)
plt.tight_layout(); plt.show()
""")

md(r"""
## Tile matrix, cell embedding and the spot graph

The usual scATAC steps apply unchanged to spots: a 500 bp tile matrix, the
50,000 most accessible tiles, and Epione's default spot embedding, BiRMT-2
(`epi.tl.birmt`, `obsm['X_birmt']`). The spot graph is the
4-connected DBiT grid (`coord_type="grid"`); `coord_type="generic"` builds a
k-nearest-neighbour graph on the coordinates instead, for Visium or
irregular layouts.
""")
code(r"""
epi.pp.add_tile_matrix(atac, bin_size=500)
epi.pp.select_features(atac, n_features=50000)
tiles = atac[:, atac.var["selected"].to_numpy(bool)].copy()
epi.tl.birmt(tiles)
epi.tl.spatial_neighbors(tiles, coord_type="grid", n_neighs=4)
tiles
""")

md(r"""
## Neighbourhood smoothing and spatial domains

A 50 um spot of a 45 M-fragment section holds about 10,000 fragments, so
spot-level embeddings are noisy. `spatial_smooth` replaces every spot by
`(1 - lam) * itself + lam * mean of its grid neighbours`. It is a plain
smoother, not a published method. `spatial_domains` clusters an embedding into a fixed number of
domains: PCA, then a Gaussian mixture (mclust, `EEE` covariance, the routine
SpatialGlue uses, via the pure-Python pymclustR port), a Leiden resolution
search, or k-means. Moran's I of the one-hot domain labels measures how
spatially coherent the domains are (smoothing makes domains more coherent
partly by construction, so this is a description, not a score of
correctness). With about 10,000 fragments per spot, only four components of the
ATAC embedding rise above the Marchenko-Pastur noise edge. Without smoothing
the domains are mostly salt-and-pepper apart from one coherent band; one round
of smoothing makes them more coherent (mean Moran's I 0.20 -> 0.38). The joint
RNA + ATAC model further down resolves the hippocampal layers.
""")
code(r"""
epi.tl.spatial_smooth(tiles, use_rep="X_birmt", lam=0.5)
for rep, key in [("X_birmt", "domain_atac"), ("X_birmt_smooth", "domain_atac_smooth")]:
    # BiRMT-2 keeps only the components above the noise edge (few here), so no PCA step
    epi.tl.spatial_domains(tiles, 8, use_rep=rep, method="mclust", use_pca=False, key_added=key)

def domain_moran(a, key):
    lab = a.obs[key].astype(str)
    a.obsm["_onehot"] = pd.get_dummies(lab).to_numpy(float)
    return float(epi.tl.spatial_moran(a, use_rep="_onehot").mean())

pd.Series({k: domain_moran(tiles, k) for k in ["domain_atac", "domain_atac_smooth"]},
          name="mean Moran's I of domains").round(3)
""")
code(r"""
epi.pl.spatial(tiles, ["domain_atac", "domain_atac_smooth"], show=False)
plt.show()
""")

md(r"""
## Gene scores next to expression

ArchR gene scores summarise accessibility around each gene. Spatial
ATAC-RNA-seq measures expression in the same spot, so the two can be put side
by side: SNAP25 (neurons), PROX1 (dentate gyrus granule cells), MBP
(oligodendrocytes, white matter). With about 10,000 fragments per spot a
single gene's score is sparse, so we also show it after one round of grid
smoothing (`spatial_smooth` works on matrices as well as embeddings). The
spot-level agreement with RNA is weak for these genes (table below), which
is typical for gene scores at this depth; domains and joint embeddings,
which pool many regions, are more robust than single genes.
""")
code(r"""
epi.tl.add_gene_score_matrix(atac, genes)
gs = ad.AnnData(np.log1p(atac.obsm["gene_score"]), obs=atac.obs[["array_row", "array_col"]].copy(),
                var=pd.DataFrame(index=pd.Index(atac.uns["gene_score_gene_names"]).astype(str)))
gs.var_names_make_unique()
gs.obsm["spatial"], gs.uns["spatial"] = atac.obsm["spatial"], atac.uns["spatial"]
gs.obsp["spatial_connectivities"] = tiles.obsp["spatial_connectivities"]
markers = ["SNAP25", "PROX1", "MBP"]
gs = gs[:, markers].copy()
epi.tl.spatial_smooth(gs, use_rep=None, lam=0.5)          # -> gs.layers["smoothed"]

rna_n = rna.copy()
sc.pp.normalize_total(rna_n, target_sum=1e4); sc.pp.log1p(rna_n)
fig, axs = plt.subplots(3, 3, figsize=(11, 10))
for j, g in enumerate(markers):
    epi.pl.spatial(gs, g, ax=axs[0, j], title=f"{g} gene score", show=False)
    epi.pl.spatial(gs, g, layer="smoothed", ax=axs[1, j], title=f"{g} gene score, smoothed", show=False)
    epi.pl.spatial(rna_n, g, ax=axs[2, j], title=f"{g} RNA", show=False)
plt.tight_layout(); plt.show()
""")
code(r"""
from scipy.stats import spearmanr
rows = []
for g in markers:
    y = rna_n[:, g].X.toarray().ravel()
    rows.append((g, spearmanr(gs[:, g].X.ravel(), y).statistic,
                 spearmanr(gs[:, g].layers["smoothed"].ravel(), y).statistic))
pd.DataFrame(rows, columns=["gene", "Spearman vs RNA (raw score)", "Spearman vs RNA (smoothed score)"]).round(3)
""")

md(r"""
## TF motif activity on the section

Per-spot TF activity of JASPAR motifs over the 50,000 tiles with Epione's
default `epi.tl.tf_activity` (MOODS scan, then a conditional count model on
GC x accessibility bias strata; no background tiles needed; the value is the
shrunken log fold change of the motif's tiles), drawn on the tissue: SOX10
for oligodendrocyte lineage, NEUROD2 for neurons, and CTCF as a ubiquitous
control. Moran's I measures spatial structure; smoothing raises it for every
motif by construction (the CTCF control included), so a smoothed motif map
should be compared with the control rather than with the raw value.
(chromVAR-compatible Z-scores: `epi.tl.add_background_peaks` +
`epi.tl.compute_deviations`.)
""")
code(r"""
epi.tl.add_motif_matrix(tiles, genome_fasta=str(genome.fasta), verbose=False)
epi.tl.tf_activity(tiles, verbose=False)                   # -> tiles.obsm["tf_activity"]
# spots x motifs activities, as an AnnData that keeps the spot positions
dev = ad.AnnData(np.asarray(tiles.obsm["tf_activity"], dtype=np.float32),
                 obs=tiles.obs[["array_row", "array_col"]].copy(),
                 var=pd.DataFrame(index=pd.Index(tiles.uns["tf_activity_names"]).astype(str)))
dev.var_names_make_unique()
dev.obsm["spatial"], dev.uns["spatial"] = tiles.obsm["spatial"], tiles.uns["spatial"]
dev.obsp["spatial_connectivities"] = tiles.obsp["spatial_connectivities"]
epi.tl.spatial_smooth(dev, use_rep=None, lam=0.5)          # -> dev.layers["smoothed"]
dev
""")
code(r"""
def find(tf):
    # motif names are "<JASPAR id>_<TF>"; take the first matrix of that TF
    return next(v for v in dev.var_names if v.split("_", 1)[-1].upper() == tf)
show = [find(t) for t in ["SOX10", "NEUROD2", "CTCF"]]
epi.pl.spatial(dev, show, layer="smoothed", cmap="RdBu_r", vmin="p2", vmax="p98", ncols=3,
               title=[f"{m} (smoothed)" for m in show], show=False)
plt.show()
pd.DataFrame({"Moran's I, raw": epi.tl.spatial_moran(dev, features=show),
              "Moran's I, smoothed": epi.tl.spatial_moran(dev, features=show, layer="smoothed")}).round(3)
""")

md(r"""
## Joint ATAC + RNA domains with SpatialGlue

SpatialGlue encodes each modality with a graph convolution on a spatial graph
and on a feature graph, mixes the two with an attention layer, mixes the
modalities with a second attention layer, and trains by reconstructing both
inputs. `epi.tl.spatialglue_preprocess` runs the official tutorial's
preprocessing (RNA: 3,000 HVGs, scaled, PCA 50; ATAC: LSI). The modality
weights `alpha` tell, per spot, whether RNA or ATAC drives the joint
embedding.
""")
code(r"""
rna_sg = rna.copy()
atac_sg = tiles.copy()
del atac_sg.obsm["X_birmt_smooth"]
rna_sg, atac_sg = epi.tl.spatialglue_preprocess(rna_sg, atac_sg)
import mudata as md
sg = md.MuData({"rna": rna_sg, "atac": atac_sg})
sg.obsm["spatial"] = rna_sg.obsm["spatial"]
sg.uns["spatial"] = rna_sg.uns["spatial"]
out = epi.tl.spatialglue(sg, modalities=("rna", "atac"), device="cpu", verbose=False)
epi.tl.spatial_domains(sg, 8, use_rep="SpatialGlue", method="mclust", key_added="domain_spatialglue")
sg.obs["alpha_rna"] = sg.obsm["SpatialGlue_alpha"][:, 0]
tiles_sub = tiles[sg.obs_names].copy()          # spots kept by the RNA filter
sg.obsp["spatial_connectivities"] = tiles_sub.obsp["spatial_connectivities"]
pd.Series({"ATAC smoothed BiRMT-2": domain_moran(tiles_sub, "domain_atac_smooth"),
           "SpatialGlue (RNA+ATAC)": domain_moran(sg, "domain_spatialglue")},
          name="mean Moran's I of domains").round(3)
""")
code(r"""
epi.pl.spatial(sg, ["domain_spatialglue", "alpha_rna"], show=False)
plt.show()
""")

md(r"""
## SpatialGlue on its own published dataset

The mouse brain (P22) spatial ATAC-RNA-seq data of the SpatialGlue paper,
from Zenodo, run with the official tutorial settings (1,600 epochs, 18
domains). The Epione port reproduces the official implementation exactly:
with the same seed and inputs the joint embeddings are identical, and over
three seeds the agreement of the domains with the original authors' ATAC and
RNA clusters matches (gates and numbers in
`tests/parity/spatial_atac/MANIFEST.md`).
""")
code(r"""
from sklearn.metrics import adjusted_rand_score
mrna, matac = epi.datasets.spatialglue_mouse_brain(data_dir=DATA / "spatialglue")
mrna, matac = epi.tl.spatialglue_preprocess(mrna, matac, random_seed=2022)
mb = md.MuData({"rna": mrna, "atac": matac})
mb.obsm["spatial"] = mrna.obsm["spatial"]
epi.tl.spatialglue(mb, modalities=("rna", "atac"), random_seed=2022, device="cpu", verbose=False)
epi.tl.spatial_domains(mb, 18, use_rep="SpatialGlue", method="mclust", key_added="SpatialGlue")
pd.Series({ref: adjusted_rand_score(mrna.obs[ref].astype(str), mb.obs["SpatialGlue"])
           for ref in ["ATAC_clusters", "RNA_clusters"]}, name="ARI").round(3)
""")
code(r"""
fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))
mrna.obs["SpatialGlue"] = pd.Categorical(mb.obs["SpatialGlue"].astype(int).to_numpy())
epi.pl.spatial(mrna, "ATAC_clusters", ax=axs[0], title="original ATAC clusters", show=False)
epi.pl.spatial(mrna, "SpatialGlue", ax=axs[1], title="SpatialGlue domains", show=False)
plt.tight_layout(); plt.show()
""")

md(r"""
## Notes

* Visium-style releases (Space Ranger `spatial/` folder, e.g. AtlasXomics
  DBiT data on GEO) and bare DBiT barcode whitelists are both read by
  `epi.io.read_spatial_positions`. `epi.io.add_spatial` attaches positions to
  any AnnData built elsewhere, such as a peak matrix.
* `epi.tl.spatialglue` is a port of SpatialGlue v1.1.5 and keeps its
  AGPL-3.0 licence (the rest of Epione is GPL-3.0). R is not needed: mclust
  runs through pymclustR.
* A spot is a mixture of cells, so spot domains mix composition and cell state.
* Epione's RNA-to-ATAC translators (`epi.translate`) can predict chromatin
  accessibility for spatial transcriptomics panels that have no ATAC
  measurement. Sections profiled like this one give spot-matched RNA and ATAC
  to check such predictions against.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = CELLS
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
out = pathlib.Path(__file__).with_name("t_spatial_atac.ipynb")
nbf.write(nb, out)
print("wrote", out)
