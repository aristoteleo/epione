"""Generate t_doublets.ipynb (doublet detection for scATAC-seq).

Execute with ``jupyter nbconvert --to notebook --execute --inplace``; needs
``pip install 'epione[doublets]'`` and about 30 min on 4 cores, most of it
building the tile matrix from 68 M fragments.
"""
import pathlib

import nbformat as nbf

CELLS = []


def md(src):
    CELLS.append(nbf.v4.new_markdown_cell(src.strip("\n")))


def code(src):
    CELLS.append(nbf.v4.new_code_cell(src.strip("\n")))


md(r"""
# Doublet detection

Two cells captured in one droplet share a barcode. In scATAC-seq such
doublets show up as spurious intermediate states and as clusters that bridge
real ones. Epione offers several detectors that see different things:

| Function | Idea | Needs |
|---|---|---|
| `epi.pp.doublets` → `epi.pp.bayes_doublets` (**default**) | generative model: a doublet's profile is a mixture of two singlet-type profiles; Bayes factor vs the best singlet type, doublet rate calibrated by EM | tile or peak matrix |
| `epi.pp.scrublet` | simulated doublets, kNN classifier (Scrublet / SnapATAC2) | tile or peak matrix |
| `epi.pp.archr_doublets` | ArchR's `addDoubletScores`: simulated doublets projected into iterative LSI + UMAP | tile matrix |
| `epi.pp.scdblfinder` | scDblFinder's scATAC mode: aggregated features, gradient-boosted classifier | tile or peak matrix |
| `epi.pp.amulet` | AMULET: too many loci covered by more than two fragments; also catches *homotypic* doublets | fragments |
| `epi.pp.combine_doublet_scores` | Fisher combination of AMULET and scDblFinder, which the scDblFinder authors found most robust | both |
| `epi.pp.filter_doublets` | removes cells: by default the `bayes_doublets` call; or ArchR's rule, a threshold, a class column | any of the above |

`scdblfinder`, `amulet` and `clamulet` wrap
[pyscdblfinder](https://github.com/omicverse/py-scDblFinder), the omicverse
port of R scDblFinder (`pip install 'epione[doublets]'`). `archr_doublets` is
a port of ArchR 1.0.3. Its agreement with ArchR is recorded in
`tests/parity/doublets/MANIFEST.md`, next to ArchR's agreement with itself
across seeds, which is low.

**Data with ground truth.** The ArchR paper (Granja et al. 2021, GEO
GSE162690) loaded a mixture of 11 genotyped cell lines into a 10x chip at a
high concentration on purpose, and called every barcode with demuxlet. A
doublet of two different lines is visible in the genotypes, so we can measure
each method. Demuxlet cannot see two cells of the *same* line, so a few
"singlets" are really homotypic doublets. We use the low-loading sample, with
about 21% genotype doublets among the cells with at least 1,000 fragments.
""")

md("## Data")
code(r"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import epione as epi

d = epi.datasets.granja2021_cellline_mix("low", data_dir=os.environ.get("EPIONE_DOUBLETS_DIR"))
dmx = d["demuxlet"]
dmx["call"].value_counts()
""")

md(r"""
Demuxlet called every barcode, including empty droplets. We keep barcodes
with at least 1,000 unique fragments, ArchR's default `minFrags`, and build
the 500-bp tile matrix that ArchR and Scrublet use.
""")
code(r"""
adata = epi.pp.import_fragments(d["fragments"], chrom_sizes=d["chrom_sizes"],
                                whitelist=list(dmx.index), min_num_fragments=1000)
epi.pp.add_tile_matrix(adata, bin_size=500)
adata.obs["demuxlet"] = dmx.loc[adata.obs_names, "call"].to_numpy()
adata.obs["cell_line"] = dmx.loc[adata.obs_names, "cell_line"].to_numpy()
print(adata)
adata.obs["demuxlet"].value_counts()
""")

md(r"""
## Default: generative doublet model (`bayes_doublets`)

`epi.pp.doublets(adata)` runs `epi.pp.bayes_doublets`. It clusters the cells
into singlet types (LSI + Leiden on the 100k most frequently accessible
tiles) and asks, for every cell, whether its fragments are better explained by
one type profile or by a mixture `α·p_k + (1-α)·p_l` of two, integrating over
the unknown fraction α. The log posterior odds is the ranking score; the
doublet rate is started from Poisson loading statistics and calibrated by EM,
so `bayes_doublet_probability` is a calibrated posterior. Clusters that are
themselves populations of doublets are detected and not used as singlet types.
No doublets are simulated, so the result does not depend on a random seed
except through the clustering.
""")
code(r"""
epi.pp.doublets(adata)                      # = epi.pp.bayes_doublets(adata)
print(adata.uns["bayes_doublet"])
adata.obs[["bayes_doublet_probability", "bayes_doublet_score", "bayes_doublet_class",
           "bayes_doublet_pair"]].head()
""")

md("## Scrublet")
code(r"""
sc = adata.copy()
epi.pp.select_features(sc)
epi.pp.scrublet(sc)
adata.obs["scrublet_score"] = sc.obs["doublet_score"].to_numpy()
adata.obs["scrublet_is_doublet"] = sc.obs["is_doublet"].to_numpy()
del sc
""")

md(r"""
## ArchR doublet scores

`archr_doublets` runs ArchR's recipe per sample. It fits iterative LSI on the
binarised tiles. It then makes `nTrials × n/1000` rounds of 1,000 synthetic
doublets: two random cells, each keeping half of its open tiles. The doublets
are projected into the LSI and the UMAP, and every real cell is scored by how
often it is among the 10 nearest neighbours of a simulated doublet:

* `archr_doublet_enrichment` is the neighbour count per 10,000 simulated doublets
  (ArchR's `DoubletEnrichment`);
* `archr_doublet_score` is the −log10 Bonferroni-adjusted binomial p-value
  (ArchR's `DoubletScore`).
""")
code(r"""
epi.pp.archr_doublets(adata, seed=1)
adata.uns["archr_doublet"]
""")

md(r"""
## scDblFinder, AMULET and their combination

These come from pyscdblfinder. `scdblfinder` uses the scATAC-seq recipe of the
scDblFinder vignette (`aggregate_features=True, n_features=25,
processing="normFeatures"`). `amulet` reads the fragments of the cells in
`adata` and excludes chrM, chrX and chrY by default. For real projects, also
pass a repeat filter for your genome as `regions_to_exclude`.
""")
code(r"""
epi.pp.scdblfinder(adata)
epi.pp.amulet(adata, d["fragments"])
epi.pp.combine_doublet_scores(adata)        # Fisher(amulet_p, 1 - scDblFinder_score)
adata.obs[["scDblFinder_score", "scDblFinder_class", "amulet_nAbove2", "amulet_q",
           "doublet_combined_p"]].describe(include="all").T
""")

md(r"""
## How well does each method find the genotype doublets?

Each method gives one number per cell; higher means more doublet-like
(p-values become −log10 p). AUPRC, the area under the precision–recall curve,
is the informative metric when doublets are a minority (here ~21%): a random score
gets an AUPRC equal to the doublet fraction.
""")
code(r"""
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

ev = adata.obs[adata.obs["demuxlet"] != "AMB"]
y = (ev["demuxlet"] == "DBL").to_numpy()
scores = {
    "bayes_doublets (default)": ev["bayes_doublet_score"],
    "scrublet": ev["scrublet_score"],
    "ArchR (enrichment)": ev["archr_doublet_enrichment"],
    "scDblFinder": ev["scDblFinder_score"],
    "AMULET": -np.log10(ev["amulet_p"].clip(lower=1e-300)),
    "Fisher(AMULET, scDblFinder)": -np.log10(ev["doublet_combined_p"]),
}
tab = pd.DataFrame({m: dict(AUROC=roc_auc_score(y, s), AUPRC=average_precision_score(y, s))
                    for m, s in scores.items()}).T
tab.loc["random"] = [0.5, y.mean()]
tab.round(3).sort_values("AUPRC", ascending=False)
""")
code(r"""
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for m, s in scores.items():
    p, r, _ = precision_recall_curve(y, s)
    ax[0].plot(r, p, label=f"{m} ({average_precision_score(y, s):.2f})")
ax[0].axhline(y.mean(), color="grey", ls=":", label=f"random ({y.mean():.2f})")
ax[0].set(xlabel="recall", ylabel="precision", title="Genotype doublets, low loading")
ax[0].legend(fontsize=8)
nf = ev["n_fragment"]
ax[1].scatter(nf[~y], ev["amulet_nAbove2"][~y], s=2, alpha=0.3, label="singlet")
ax[1].scatter(nf[y], ev["amulet_nAbove2"][y], s=4, color="crimson", label="genotype doublet")
ax[1].set(xscale="log", yscale="symlog", xlabel="unique fragments",
          ylabel="loci with >2 fragments", title="What AMULET counts")
ax[1].legend()
plt.tight_layout()
""")

md(r"""
Note the right panel: doublets have more fragments *and* more loci covered by
more than two fragments. AMULET's power depends on depth. Its authors
recommend it for libraries with a median of 10–15k fragments per cell, and
this sample is shallower.
""")

md(r"""
## Removing doublets

`filter_doublets` removes cells by one of four rules:

* `method="bayes"` (default) removes the cells that `bayes_doublets` calls
  doublets (posterior probability > 0.5). The posterior is calibrated to the
  Poisson-loading doublet rate of a standard 10x channel (here 11%). This chip
  was overloaded on purpose (21% genotype doublets), so pass the known rate,
  `epi.pp.doublets(adata, expected_doublet_rate=0.25)`, to remove more of them.
* `method="archr"` reproduces ArchR's `filterDoublets`. Within each sample it
  removes the top `n²/100000` cells by enrichment, the expected number of
  doublets for a 10x chip. It matches ArchR exactly on ArchR's own scores
  (gate F1 in the MANIFEST).
* `method="threshold"` removes cells with `score >= threshold`. For p- or
  q-value columns it removes cells with values `<= threshold`.
* `method="class"` removes cells labelled `"doublet"`, e.g. `scDblFinder_class`.
""")
code(r"""
res = {}
for name, kw in {
    "bayes_doublets (default)": dict(),
    "ArchR rule": dict(method="archr"),
    "scDblFinder class": dict(method="class", score_key="scDblFinder_class"),
    "AMULET q < 0.01": dict(method="threshold", score_key="amulet_q", threshold=0.01),
    "Fisher p < 0.01": dict(method="threshold", score_key="doublet_combined_p", threshold=0.01),
}.items():
    a = adata.copy()
    epi.pp.filter_doublets(a, **kw, key_added="removed")
    rm = adata.obs_names.difference(a.obs_names)
    called = adata.obs.loc[rm, "demuxlet"]
    res[name] = dict(removed=len(rm), of_which_DBL=int((called == "DBL").sum()),
                     DBL_remaining=int((a.obs["demuxlet"] == "DBL").sum()))
pd.DataFrame(res).T
""")
code(r"""
epi.pp.filter_doublets(adata)                # default: bayes_doublet_class == "doublet"
adata
""")

md(r"""
## What the benchmark shows

On these QC-filtered cells (≥ 1,000 fragments, no prior doublet removal), the default `bayes_doublets` ranks the genotype doublets best, followed by Scrublet; the ArchR score and scDblFinder come next. AMULET is weak here because the cells are shallow (median ≈ 4k fragments), below the 10–15k its authors recommend, and cell lines are aneuploid, so "more than two copies" is common in singlets too. Fisher-combining it with scDblFinder therefore lowers scDblFinder's AUPRC rather than raising it.

This dataset was used to *develop* `bayes_doublets`, so its numbers here are optimistic. On three held-out datasets with genotype or species truth (a 3-donor PBMC pool demultiplexed by genotype, and two 10x human–mouse mixtures) it had the best mean AUPRC of all detectors (0.78 vs 0.72 for SnapATAC2's scrublet, 0.70 for Epione's scrublet), with near-perfect scores on the species mixtures. On the PBMC pool, where cell states are continuous rather than discrete cell lines, SnapATAC2's scrublet ranked doublets better (AUPRC 0.42 vs 0.34), so check a second detector (`epi.pp.doublets(adata, method="scrublet")`) on primary tissue.

The pre-registered benchmark in `tests/parity/doublets/MANIFEST.md` also scores the authors' *published* cell sets of both loadings. Those sets turn out to be already doublet-filtered: they keep about 14% of the genotype doublets that pass QC but about 91% of the singlets. Every method looks poor on them, because the remaining doublets are the hard ones. Use QC-only cells like these when you compare methods.

On a single, shallow dataset of cell lines, no ranking should be over-generalised. Cell lines differ more than cell types do, which favours methods based on simulated doublets.
""")


nb = nbf.v4.new_notebook()
nb["cells"] = CELLS
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
out = pathlib.Path(__file__).with_name("t_doublets.ipynb")
nbf.write(nb, out)
print("wrote", out)
