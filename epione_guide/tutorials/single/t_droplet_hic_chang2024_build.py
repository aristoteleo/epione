"""Generate t_droplet_hic_chang2024.ipynb (Droplet Hi-C, Chang et al. 2024, Fig. 1d-f).

Execute with ``jupyter nbconvert --to notebook --execute --inplace``; needs
``pip install 'epione[hic]'``, scHiCluster, the GEO files of sample LC462
(GSE253407) and an mm10 FASTA. The first run takes about 2.5 h on 30 cores:
1.5 h to split the 25 GB pairs file by cell, the rest for the scool, the
pseudobulks and the 100 kb imputation. Later runs read the cached files.
"""
import pathlib

import nbformat as nbf

CELLS = []


def md(src):
    CELLS.append(nbf.v4.new_markdown_cell(src.strip("\n")))


def code(src):
    CELLS.append(nbf.v4.new_code_cell(src.strip("\n")))


md(r"""
# Case study: Droplet Hi-C

Droplet Hi-C (Chang *et al.* 2024, *Nat. Biotechnol.*,
[doi:10.1038/s41587-024-02447-1](https://www.nature.com/articles/s41587-024-02447-1))
captures chromatin contacts of thousands of single nuclei in droplets; every
contact carries its cell's barcode. This case study reproduces Figure 1d–f of
the paper, adult mouse cortex, with `epione`:

| Panel | What | epione |
|---|---|---|
| 1f | contact maps and A/B compartments of each cell type on chr1 | `merge_cells`, `compartment_model`, `project_compartments` |
| 1e | correlation of compartment scores between cell types | `project_compartments` |
| 1d | cell embedding from single-cell contacts | `impute_cells` (scHiCluster, 100 kb), `embedding(method="schicluster")` |

**Data.** Sample LC462 of GEO GSE253407: `GSM8020244_LC462_mm10.sc.pairs.gz`
(all contacts, 25 GB; the barcode is the `barcode1` column) and the paper's
cell table `GSE253407_Droplet_Hi-C_mouse_ctx_metadata.xls.gz` (tab-separated
despite the name). LC462 has 3,329 nuclei in 20 cortical cell types
(`rnatype`); the labels are used to group cells and to colour and score the
embedding, never to build it.
""")

md("## 1 · Setup")
code(r"""
import os
import pathlib
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')     # before importing epione: silences tqdm's IProgress notice
import matplotlib.pyplot as plt
import scanpy as sc

import epione as epi
from epione.single import hic

epi.pl.plot_set()

# Folder with the GEO files, mm10.chrom.sizes and mm10.fa; override with EPIONE_CHANG2024_DIR.
DATA = pathlib.Path(os.environ.get('EPIONE_CHANG2024_DIR', 'data/chang2024-droplet'))
OUT = pathlib.Path('result/t_droplet_hic_chang2024')
OUT.mkdir(parents=True, exist_ok=True)
NPROC = int(os.environ.get('EPIONE_NPROC', 16))

PAIRS = DATA / 'GSM8020244_LC462_mm10.sc.pairs.gz'
META = DATA / 'GSE253407_Droplet_Hi-C_mouse_ctx_metadata.xls.gz'
FASTA = DATA / 'mm10.fa'
RES = 100_000
CHROMS = [f'chr{i}' for i in range(1, 20)] + ['chrX']
sizes = pd.read_csv(DATA / 'mm10.chrom.sizes', sep='\t', header=None, index_col=0)[1].loc[CHROMS]
for p in (PAIRS, META, FASTA):
    assert p.exists(), f'missing: {p}'
print(f'pairs: {PAIRS.stat().st_size / 1e9:.1f} GB')
""")

md(r"""
## 2 · Cells and cell types

The `rnatype` column gives the paper's cell-type label of each barcode. The
order below is the legend of Fig. 1d: glutamatergic neurons (`GL`),
GABAergic neurons (`GA`, `MSN`) and glia.
""")
code(r"""
meta = pd.read_csv(META, sep='\t')
meta = meta[meta['sample'] == 'LC462'].set_index('bc')
PAPER_ORDER = ['ITL23GL', 'ITL45GL', 'ITL5GL', 'ITL6GL', 'CTGL', 'PTGL', 'NPGL', 'CLAGL',
               'OBGL', 'D12MSN', 'STRGA', 'OBGA', 'PVGA', 'SSTGA', 'VIPGA',
               'ASC', 'OGC', 'OPC', 'MGL', 'VLMC']
MAJOR = {**{t: 'Glu' for t in PAPER_ORDER[:9]}, **{t: 'GABA' for t in PAPER_ORDER[9:15]},
         **{t: 'glia' for t in PAPER_ORDER[15:]}}
TYPES = [t for t in PAPER_ORDER if t in set(meta['rnatype'])]
print(f'{len(meta):,} cells, {len(TYPES)} cell types')
meta['rnatype'].value_counts().reindex(TYPES).rename('cells').to_frame().T
""")

md(r"""
## 3 · One pass over the pairs file: one pairs file per cell

`demux_pairs_by_barcode` reads the 25 GB file once and writes every contact
to the file of its cell (`barcode_column=9` is `barcode1`). Everything below,
the cell-type pseudobulks included, is built from these per-cell files, so the
big file is read only once. This is the slow step (about 1.5 h); it is cached.
""")
code(r"""
CELL_PAIRS = OUT / 'per_cell_pairs'
if not CELL_PAIRS.exists():
    tmp = OUT / 'per_cell_pairs.tmp'
    hic.demux_pairs_by_barcode(PAIRS, {bc: bc for bc in meta.index}, tmp, barcode_column=9)
    tmp.rename(CELL_PAIRS)
cell_files = {bc: CELL_PAIRS / f'{bc}.pairs.gz' for bc in meta.index if (CELL_PAIRS / f'{bc}.pairs.gz').exists()}
print(f'{len(cell_files):,} per-cell pairs files')
""")

md(r"""
## 4 · Raw single-cell contacts at 100 kb

`contacts_to_scool` bins every cell's pairs at 100 kb into one `.scool`
(chromosome and position columns 1–4 of the pairs format; duplicate contacts
and cis contacts closer than 2.5 kb are dropped, as in scHiCluster).
""")
code(r"""
SCOOL = OUT / 'lc462_raw_100kb.scool'
cells = list(cell_files)
if not SCOOL.exists():
    hic.contacts_to_scool([cell_files[c] for c in cells], SCOOL, chrom_sizes=sizes, resolution=RES,
                          cell_ids=cells, columns={'chrom1': 1, 'pos1': 2, 'chrom2': 3, 'pos2': 4},
                          nproc=NPROC)
adata = hic.load_scool_cells(SCOOL, cell_names=cells, obs=meta.loc[cells, ['rnatype', 'total', 'cis']])
adata.obs['major'] = adata.obs['rnatype'].map(MAJOR)
adata
""")

md(r"""
## 5 · Cell-type pseudobulks

`merge_cells` sums the raw contacts of the cells of each type (and of all
cells, for the compartment model). Only contacts within a chromosome are kept
(`cis_only=True`): the maps and compartments below use nothing else. The
cell-type maps are ICE-balanced for display.
""")
code(r"""
PB = OUT / 'pseudobulk_100kb'
adata.obs['merged'] = 'merged'
pb = hic.merge_cells(adata, 'rnatype', PB, cis_only=True, nproc=8)
pb_all = hic.merge_cells(adata, 'merged', PB, cis_only=True)['merged']
for p in pb.values():
    epi.bulk.hic.balance_cool(p)
import cooler
pd.Series({t: cooler.Cooler(pb[t]).info['sum'] for t in TYPES}, name='cis contacts').to_frame().T
""")

md(r"""
## 6 · Compartments on one axis for all cell types

`compartment_model` fits one PCA model on the contacts of all cells (bins with
enough contacts, distance-normalised observed/expected, correlation, PCA); the
component is signed so that A is CpG-rich, using `cpg_ratio` from the mm10
sequence. `project_compartments` then scores every cell type on the same
loadings, so a bin that is A in one type and B in another changes sign, and
scores are comparable across types. (Eigenvectors computed separately for each
cell type have an arbitrary sign on every chromosome and cannot be compared
without this phasing.)
""")
code(r"""
cpg = hic.cpg_ratio(FASTA, chrom_sizes=sizes, resolution=RES)
model = hic.compartment_model(pb_all, cpg)
comp = hic.project_compartments(pb, model, nproc=8)
print(f"bins kept: {model['keep'].sum():,} of {len(model):,}; "
      f"PC used: {model.groupby('chrom', sort=False)['pc'].first().map({0: 'PC1', 1: 'PC2'}).value_counts().to_dict()}")
""")

md(r"""
### Fig 1f — contact maps and compartments on chr1

Contact maps of seven cell types on chr1, and below them each type's
compartment score on the shared axis (red A, blue B). A whole chromosome at
100 kb is too sparse to read for the smaller types, so the maps are drawn from
500 kb copies of the pseudobulks (`cooler.coarsen_cooler`, then ICE); the
compartment tracks stay at 100 kb.
""")
code(r"""
FOCAL = [t for t in ['ITL23GL', 'ITL5GL', 'CTGL', 'D12MSN', 'ASC', 'OGC', 'MGL'] if t in pb]
coarse = {}
for t in FOCAL:
    coarse[t] = PB / f'{t}.500kb.cool'
    if not coarse[t].exists():
        cooler.coarsen_cooler(str(pb[t]), str(coarse[t]), factor=5, chunksize=10_000_000)
        epi.bulk.hic.balance_cool(coarse[t])
fig, axes = plt.subplots(2, len(FOCAL), figsize=(2.5 * len(FOCAL), 3.6), height_ratios=[5, 1.3])
for j, t in enumerate(FOCAL):
    epi.pl.plot_contact_matrix(coarse[t], region='chr1', balance=True, log=True, cmap='YlOrRd',
                               ax=axes[0, j], colorbar=False, title=f"{t} ({(adata.obs['rnatype'] == t).sum()})")
    axes[0, j].set_xticks([]); axes[0, j].set_yticks([]); axes[0, j].set_xlabel(''); axes[0, j].set_ylabel('')
    epi.pl.plot_compartments(comp, chromosome='chr1', track_column=t, ax=axes[1, j], title='')
    axes[1, j].set_xlabel('chr1 (Mb)' if j == 0 else ''); axes[1, j].set_ylabel('')
plt.tight_layout(); plt.show()
""")

md(r"""
Every type shows the checkerboard of compartments, and the large blocks are
shared: most of chr1 beyond about 120 Mb is A, most of the first 120 Mb is B
with A islands. Types with few cells (ASC, MGL; 70–90 cells) give sparser maps
and tracks with a smaller range.
""")

md(r"""
### Fig 1e — compartment-score correlation between cell types

Spearman correlation of the compartment scores of every pair of cell types,
over all kept bins (the paper's SCC).
""")
code(r"""
scc = comp[TYPES].corr(method='spearman')
fig, ax, img = epi.pl.plot_correlation_heatmap(scc, vmin=float(scc.min().min()), vmax=1.0,
                                               annot_fontsize=5.5, figsize=(7.5, 6.5),
                                               title='Fig 1e — compartment-score SCC')
fig.axes[-1].set_ylabel('Spearman r')
plt.show()
neuron = pd.Series({t: MAJOR[t] != 'glia' for t in TYPES})
def mean_scc(a, b):
    return np.mean([scc.loc[x, y] for x in TYPES for y in TYPES
                    if x != y and neuron[x] == a and neuron[y] == b])
print(f'mean SCC  neuron-neuron {mean_scc(True, True):.3f}   neuron-glia {mean_scc(True, False):.3f}'
      f'   glia-glia {mean_scc(False, False):.3f}')
""")

md(r"""
As in the paper, neuronal types are more alike than neurons and glia: the mean
SCC is 0.879 between neuronal types, 0.796 between a neuronal and a glial type
and 0.762 between glial types. The tightest block is the cortical
glutamatergic types (ITL23GL to CTGL, 0.91–0.98). The lowest values involve
MGL and VLMC, the types with the fewest contacts (1.7 and 1.3 million), so
part of their distance from the others is sampling noise.
""")

md(r"""
## 7 · Fig 1d — cell embedding from imputed 100 kb contacts

`impute_cells` runs scHiCluster's imputation on every cell and chromosome at
100 kb, from each cell's raw pairs file, as `hicluster impute-cell` does with a
contact file: scHiCluster bins the contacts itself (`resolution`, pairs
columns 1–4), convolves with a 3 × 3 kernel (`pad=1`), runs a random walk with
restart (`rp=0.5`, `tol=0.01`) over whole chromosomes (`window_size` larger
than any chromosome) and normalises by square-root vanilla coverage. It keeps
imputed contacts up to 10 Mb (`output_dist`); the cut comes after the
normalisation, so the values kept are the same as with a longer
`output_dist`. `load_contact_tables` indexes the raw files without binning
them.

`embedding(method="schicluster")` is scHiCluster's `hicluster embedding`: per
chromosome, the imputed contacts within 1 Mb of the diagonal are reduced by a
50-component truncated SVD, and the chromosome reductions are reduced again.
The settings are those of `hicluster embedding --dim 50 --dist 1000000
--resolution 100000 --scale_factor 100000 --norm_sig`: with `norm_sig=True`
every component is divided by its singular value, at both steps. Only
contacts go into the embedding; the labels only colour it.
""")
code(r"""
raw = hic.load_contact_tables([cell_files[c] for c in cells], chrom_sizes=sizes, resolution=RES,
                              cell_ids=cells, obs=adata.obs[['rnatype', 'major', 'total']],
                              columns={'chrom1': 1, 'pos1': 2, 'chrom2': 3, 'pos2': 4},
                              chromosomes=CHROMS)
IMPUTED = OUT / 'imputed_100kb_contacts'
hic.impute_cells(raw, out_dir=IMPUTED, resolution=RES, pad=1, std=1.0, rp=0.5, tol=0.01,
                 window_size=500_000_000, step_size=500_000_000, output_dist=10_000_000,
                 compress=True, nproc=NPROC, progress=False)
emb = hic.embedding(raw, method='schicluster', dim=50, dist=1_000_000, scale_factor=100_000, norm_sig=True)
emb.obs['major'] = emb.obs['major'].astype('category')
emb.obs['rnatype'] = pd.Categorical(emb.obs['rnatype'], categories=TYPES)
sc.pp.neighbors(emb, n_neighbors=15, use_rep='X_schicluster')
sc.tl.umap(emb, random_state=0)
emb.obs['log10 contacts'] = np.log10(emb.obs['total'].astype(float))
fig, axes = plt.subplots(1, 3, figsize=(19, 5))
sc.pl.umap(emb, color='major', ax=axes[0], show=False, title='Fig 1d — Glu / GABA / glia')
sc.pl.umap(emb, color='rnatype', ax=axes[1], show=False, title='Fig 1d — 20 cell types')
sc.pl.umap(emb, color='log10 contacts', ax=axes[2], show=False, title='contacts per cell (log10)')
plt.tight_layout(); plt.show()
X = emb.obsm['X_schicluster']
r = [np.corrcoef(X[:, k], emb.obs['log10 contacts'])[0, 1] for k in range(3)]
print('correlation of components 1-3 with log10 contacts:', np.round(r, 2))
""")

md(r"""
The embedding separates glia from neurons: oligodendrocytes (OGC), OPC and
microglia (MGL) each form a group of their own, and astrocytes (ASC) sit at the
edge of the neurons. Glu and GABA neurons, and the neuronal subtypes, share one
large group. Within it the main gradient is sequencing depth: components 1 and
3 correlate with log10 contacts at r = 0.60 and 0.72.
""")

md(r"""
How much of the labels does the embedding recover? A cross-validated
15-nearest-neighbour classifier on the 50 components predicts each cell's
label from its neighbours; Leiden clusters are compared with the labels by
the adjusted Rand index (ARI).
""")
code(r"""
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import adjusted_rand_score
cv = StratifiedKFold(5, shuffle=True, random_state=0)
rows = {}
for level in ('major', 'rnatype'):
    y = emb.obs[level].astype(str)
    acc = cross_val_score(KNeighborsClassifier(15), emb.obsm['X_schicluster'], y, cv=cv)
    rows[level] = {'15-NN accuracy': acc.mean(), 'largest class': y.value_counts(normalize=True).iloc[0]}
sc.tl.leiden(emb, resolution=1.0, random_state=0, key_added='leiden')
for level in ('major', 'rnatype'):
    rows[level]['ARI (Leiden)'] = adjusted_rand_score(emb.obs[level].astype(str), emb.obs['leiden'])
pd.DataFrame(rows).T.round(3)
""")

md(r"""
The 15-NN classifier gets the major class right for 71 % of cells, against
63 % for always guessing Glu, and the cell type for 26 %, against 17 %. The
Leiden clusters hardly match the labels (ARI 0.03 and 0.06). On the
scool-binned contacts, removing depth (regressing log10 contacts out of the
components, or keeping the deepest half of the cells) or widening `dist` to
5 Mb left the 15-NN accuracy between 0.70 and 0.79.

So with contacts alone, the 100 kb scHiCluster embedding of LC462 separates
glia from neurons but not the neuronal types. One reason is depth: a median
LC462 cell has 22,000 cis contacts longer than 1 kb (most of its 167,000
contacts are shorter), against about 200,000 long-range cis contacts in an
HCEA cell, where the same embedding separates six blood types (see the HCEA
embedding case study). The paper's Fig 1d labels come from co-embedding with
snRNA-seq (Droplet Paired-Tag), not from contacts alone.
""")

md(r"""
## Summary

| Chang 2024 panel | epione |
|---|---|
| Fig 1f (chr1 maps, compartments) | `demux_pairs_by_barcode` → `contacts_to_scool` → `merge_cells` → `compartment_model` / `project_compartments` |
| Fig 1e (compartment SCC) | `project_compartments` → Spearman |
| Fig 1d (cell embedding) | `impute_cells` (scHiCluster, 100 kb) → `embedding(method="schicluster")` → UMAP |

Not covered: Fig 1g (25 kb insulation, TAD boundaries) and Fig 1h (10 kb
zoom). Build the pseudobulks at those resolutions with `contacts_to_scool` /
`merge_cells` and use `epi.bulk.hic.insulation` / `tad_boundaries`. The paper's
UMAP combines both samples (LC462 and LC716, 6,235 cells); this case study uses
LC462 only.
""")


nb = nbf.v4.new_notebook()
nb["cells"] = CELLS
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
out = pathlib.Path(__file__).with_name("t_droplet_hic_chang2024.ipynb")
nbf.write(nb, out)
print("wrote", out)
