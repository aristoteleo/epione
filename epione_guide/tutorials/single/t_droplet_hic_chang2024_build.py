"""Generate t_droplet_hic_chang2024.ipynb (Droplet Hi-C, Chang et al. 2024, Fig. 1d-f).

Execute with ``jupyter nbconvert --to notebook --execute --inplace``; needs
``pip install 'epione[hic]'``, scHiCluster, the GEO files of sample LC462
(GSE253407), the frontal-cortex RNA of GSE224560, a GENCODE vM25 GTF and an
mm10 FASTA. The first run takes about 2.5 h on 30 cores, most of it to split
the 25 GB pairs file by cell; later runs read the cached files and take
about 20 min.
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
| 1d | cells co-embedded with snRNA-seq through gene scores | `gene_score`, `epi.tl.integrate(method='cca')`, `epi.tl.transfer_labels` |

**Data.** Sample LC462 of GEO GSE253407: `GSM8020244_LC462_mm10.sc.pairs.gz`
(all contacts, 25 GB; the barcode is the `barcode1` column) and the paper's
cell table `GSE253407_Droplet_Hi-C_mouse_ctx_metadata.xls.gz` (tab-separated
despite the name). LC462 has 3,329 nuclei in 20 cortical cell types
(`rnatype`). The labels group the cells for the pseudobulks; for Fig 1d we
transfer labels from an RNA reference ourselves and compare them with these.
Fig 1d also needs the RNA reference (GEO GSE224560, section 8) and a GENCODE
vM25 gene annotation (`gencode.vM25.*annotation.gtf`).
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
sc.settings.verbosity = 0
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# Folder with the GEO files, mm10.chrom.sizes and mm10.fa; override with EPIONE_CHANG2024_DIR.
DATA = pathlib.Path(os.environ.get('EPIONE_CHANG2024_DIR', 'data/chang2024-droplet'))
OUT = pathlib.Path('result/t_droplet_hic_chang2024')
OUT.mkdir(parents=True, exist_ok=True)
NPROC = int(os.environ.get('EPIONE_NPROC', 16))

PAIRS = DATA / 'GSM8020244_LC462_mm10.sc.pairs.gz'
META = DATA / 'GSE253407_Droplet_Hi-C_mouse_ctx_metadata.xls.gz'
FASTA = DATA / 'mm10.fa'
GTF = DATA / 'gencode.vM25.chr_patch_hapl_scaff.annotation.gtf'
RES = 100_000
CHROMS = [f'chr{i}' for i in range(1, 20)] + ['chrX']
sizes = pd.read_csv(DATA / 'mm10.chrom.sizes', sep='\t', header=None, index_col=0)[1].loc[CHROMS]
for p in (PAIRS, META, FASTA, GTF):
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
## 7 · Gene scores

At 100 kb a single LC462 cell is too sparse to place by its contact map
alone. The paper summarises each cell at the gene level instead, with
scHiCluster's gene score (`hicluster gene-score`): for every gene, the
contacts whose two ends both fall inside the gene body, in 10 kb bins.
`gene_score` computes the same score from the raw per-cell pairs files,
indexed without binning by `load_contact_tables`. Genes are the
protein-coding and lncRNA genes of GENCODE vM25.

Most LC462 contacts are shorter than 1 kb: a median cell has 159,000 cis
contacts, of which only 22,000 are 1 kb or longer. Inside a gene body the
short ones mostly measure read coverage rather than folding. `min_pos_dist=1000` keeps only contacts at least 1 kb
apart, which separated the cell types better than all contacts in our tests.
""")
code(r"""
GS_RES = 10_000
gtf = epi.io.read_gtf(GTF, required_attrs=('gene_id', 'gene_name', 'gene_type'),
                      feature_whitelist=['gene'], keep_attribute=False)
gtf = gtf[gtf['gene_type'].isin(['protein_coding', 'lncRNA']) & gtf['seqname'].isin(CHROMS)]
genes = pd.DataFrame({'chrom': gtf['seqname'].to_numpy(), 'start': gtf['start'].to_numpy() - 1,
                      'end': gtf['end'].to_numpy(), 'gene_name': gtf['gene_name'].to_numpy()},
                     index=gtf['gene_id'].str.split('.').str[0].to_numpy())
genes = genes[~genes.index.duplicated()]

contacts = hic.load_contact_tables([cell_files[c] for c in cells], chrom_sizes=sizes, resolution=GS_RES,
                                   cell_ids=cells, obs=adata.obs[['rnatype', 'major', 'total']],
                                   columns={'chrom1': 1, 'pos1': 2, 'chrom2': 3, 'pos2': 4},
                                   chromosomes=CHROMS)
gs = hic.gene_score(contacts, genes, resolution=GS_RES, min_pos_dist=1000, nproc=NPROC, progress=False)
per_cell = np.asarray(gs.X.sum(axis=1)).ravel()
print(f'{gs.n_obs:,} cells x {gs.n_vars:,} genes; median {np.median(per_cell):,.0f} contacts per cell inside genes')
""")

md(r"""
Do the gene scores carry cell-type information? Each cell has only a few
thousand contacts inside genes. A Wilcoxon test of each type against all other
cells ranks genes by how much higher their (log-normalised) scores are in that
type. The top five genes per type:
""")
code(r"""
gsl = gs.copy()
sc.pp.normalize_total(gsl, target_sum=1e4)
sc.pp.log1p(gsl)
gsl.var_names = gsl.var['gene_name'].astype(str)
gsl.var_names_make_unique()
gsl.obs['rnatype'] = pd.Categorical(gsl.obs['rnatype'], categories=TYPES)
sc.tl.rank_genes_groups(gsl, 'rnatype', method='wilcoxon')
top = sc.get.rank_genes_groups_df(gsl, None)
top = top[top['logfoldchanges'] > 0].groupby('group', observed=True).head(5)
top.groupby('group', observed=True)['names'].apply(', '.join).reindex(TYPES).to_frame('top genes')
""")

md(r"""
Known type genes come out on top without a marker list: Rasgrf2 for layer
2/3 IT neurons, Hs3st4 for CT neurons, Rarb and Foxp1 for striatal neurons
(D12MSN), Adarb2 and Gad2 for VIP neurons, Cacna2d2 for PV neurons and Sox6 for
OPC. The signal comes from long genes; short markers such as Sst (1.4 kb) or
Vip (8 kb) span one or two 10 kb bins and collect almost no contacts.
""")

md(r"""
## 8 · Fig 1d: co-embedding with snRNA-seq

The UMAP of the paper's Fig 1d is not built from Hi-C alone. The authors
co-embedded the Hi-C gene scores with snRNA-seq of the same tissue (frontal
cortex Droplet Paired-Tag RNA, GEO GSE224560; 22,054 nuclei) and gave each
Hi-C cell the label of its nearest RNA nuclei. The `rnatype` column of the
cell table is the result. The RNA labels (`Anno0.8_yel`) are the same 20
types.

We repeat the co-embedding with epione: canonical correlation analysis (CCA)
between the two gene-level matrices (`epi.tl.integrate(method='cca')`), then a
distance-weighted vote of the 15 nearest RNA nuclei (`epi.tl.transfer_labels`).
As in the paper, the features are the 2,000 genes whose gene scores vary
most across cells (standard deviation after depth normalisation) among the
genes measured in the RNA. Both matrices are depth-normalised and
log-transformed; `integrate` z-scores each feature before the CCA.

**RNA data.** The six frontal-cortex RNA matrices of GSE224560
(`GSM7025954`, `GSM7025955`, `GSM7025960`–`GSM7025963`,
`*_raw_feature_bc_matrix.h5`) and the cell table in
`GSE224560_FC_Droplet_Paired-Tag_metadata.tar.gz`. Nuclei with
`RNA_keep_cells` are the 22,054 used in the paper.
""")
code(r"""
import tarfile
import anndata as ad
with tarfile.open(DATA / 'GSE224560_FC_Droplet_Paired-Tag_metadata.tar.gz') as t:
    rmeta = pd.read_csv(t.extractfile('metadata/FC_RNA_merged_rmScrublet.meta.xls'), sep='\t', index_col=0)
rmeta = rmeta[rmeta['RNA_keep_cells'] == True]
parts = []
for gsm in ('GSM7025954_YX796', 'GSM7025955_YX797', 'GSM7025960_YX839',
            'GSM7025961_YX840', 'GSM7025962_YX841', 'GSM7025963_YX842'):
    r = sc.read_10x_h5(DATA / f'{gsm}_raw_feature_bc_matrix.h5')
    r.var_names = r.var['gene_ids'].astype(str)
    r.obs_names = gsm.split('_')[1] + ':' + r.obs_names
    parts.append(r[r.obs_names.isin(rmeta.index)].copy())
rna = ad.concat(parts, merge='first')
rna.obs = rmeta.loc[rna.obs_names, ['Anno0.8_yel']].copy()
print(f'RNA: {rna.n_obs:,} nuclei x {rna.n_vars:,} genes')
rna.obs['Anno0.8_yel'].value_counts().reindex(TYPES).rename('nuclei').to_frame().T
""")
code(r"""
hic_gs = gs[:, gs.var_names.isin(rna.var_names)].copy()
hic_gs = hic_gs[:, np.asarray((hic_gs.X > 0).sum(axis=0)).ravel() > 10].copy()
depth = np.asarray(hic_gs.X.sum(axis=1)).ravel()
sd = (hic_gs.X.toarray() / depth[:, None] * np.median(depth)).std(axis=0)
features = list(hic_gs.var_names[np.argsort(-sd)[:2000]])

for a in (hic_gs, rna):
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
epi.tl.integrate(hic_gs, rna, method='cca', features=features, num_cc=30)
epi.tl.transfer_labels(hic_gs, rna, reference_label='Anno0.8_yel', k=15, key_added='transferred')

pred, pub = hic_gs.obs['transferred'].astype(str), hic_gs.obs['rnatype'].astype(str)
print(f"same cell type as the paper's label: {np.mean(pred == pub):.3f}")
print(f"same major class (Glu / GABA / glia): {np.mean(pred.map(MAJOR) == pub.map(MAJOR)):.3f}")
""")
code(r"""
colors = rmeta.drop_duplicates('Anno0.8_yel').set_index('Anno0.8_yel')['Subclass.color']
joint = ad.concat([rna, hic_gs], label='modality', keys=['RNA', 'Hi-C'])
joint.obs['cell type'] = pd.Categorical(np.r_[rna.obs['Anno0.8_yel'].astype(str), pred.to_numpy()],
                                        categories=TYPES)
joint.obs['paper label'] = pd.Categorical(np.r_[rna.obs['Anno0.8_yel'].astype(str), pub.to_numpy()],
                                          categories=TYPES)
sc.pp.neighbors(joint, n_neighbors=25, use_rep='X_cca')
sc.tl.umap(joint, random_state=0)
for key in ('cell type', 'paper label'):
    joint.uns[f'{key}_colors'] = [colors[t] for t in TYPES]
is_rna = (joint.obs['modality'] == 'RNA').to_numpy()
fig, axes = plt.subplots(1, 4, figsize=(26, 6))
joint.obs['modality'] = pd.Categorical(joint.obs['modality'].astype(str), categories=['RNA', 'Hi-C'])
joint.uns['modality_colors'] = ['#BBBBBB', '#D62728']
sc.pl.umap(joint, color='modality', ax=axes[0], show=False, size=4, title='RNA and Hi-C')
sc.pl.umap(joint[is_rna], color='cell type', ax=axes[1], show=False, size=4, legend_loc='on data',
           legend_fontsize=7, title='RNA: Anno0.8_yel')
sc.pl.umap(joint[~is_rna], color='cell type', ax=axes[2], show=False, size=8, legend_loc='on data',
           legend_fontsize=7, title='Hi-C: transferred label')
sc.pl.umap(joint[~is_rna], color='paper label', ax=axes[3], show=False, size=8, legend_loc='on data',
           legend_fontsize=7, title="Hi-C: paper's label (rnatype)")
plt.tight_layout(); plt.show()
""")

md(r"""
The Hi-C cells fall among the RNA nuclei rather than in a block of their own.
Glia and striatal neurons land on their RNA clusters; many cortical neurons
sit between the IT clusters. How often does the transferred label agree with
the paper's?
""")
code(r"""
agree = pd.Series({t: np.mean(pred[pub == t] == t) for t in TYPES}, name='agreement')
fig, axes = plt.subplots(1, 2, figsize=(17, 6.5), gridspec_kw={'width_ratios': [1, 2.2]})
agree.plot.barh(ax=axes[0], color=[colors[t] for t in TYPES])
axes[0].invert_yaxis(); axes[0].set_xlim(0, 1); axes[0].set_xlabel("fraction with the paper's label")
cm = epi.tl.confusion_matrix(pub.to_numpy(), pred.to_numpy()).reindex(index=TYPES, columns=TYPES).fillna(0)
epi.pl.plot_confusion_matrix(cm, normalize='row', ax=axes[1])
axes[1].set_ylabel("paper's label"); axes[1].set_xlabel('transferred label')
plt.tight_layout(); plt.show()
""")

md(r"""
The transferred label matches the paper's for 46 % of the cells, and the
major class (Glu, GABA or glia) for 74 %. Agreement follows how distinct a type
is in the RNA. Microglia, OPC, striatal neurons (D12MSN), astrocytes and
oligodendrocytes agree for 73–97 % of cells, PV interneurons and
corticothalamic neurons (CTGL) for about 60–70 %. The IT neurons of
neighbouring layers, the claustrum (CLAGL) and striatal GABA neurons (STRGA)
agree least; their cells mostly go to another IT or neighbouring neuronal
type. None of the 68 VLMC cells is called VLMC.

The paper's labels are not ground truth: they came from the same kind of
transfer, but with gene scores from imputed 10 kb contacts (see below) and
both samples (6,235 cells). The comparison shows that raw gene scores of
LC462 recover the glial and the most distinct neuronal types; finer neuronal
subtypes need more signal per cell than raw contacts give.
""")

md(r"""
### Gene scores from imputed 10 kb contacts

The paper computed its gene scores from contacts imputed at 10 kb, not from
raw contacts. Imputation spreads each contact over neighbouring bins, which
can help cells as sparse as these. With scHiCluster it is two commands: impute
every cell at 10 kb (this writes a Snakemake workflow that produces one
imputed `.cool` per cell), then score the genes on the imputed cools
(`--cell_table_path`: two columns, cell id and imputed cool path):

```
hicluster prepare-impute --cell_table cell_table.tsv --batch_size 1536 --cpu_per_job 96 \
    --pad 2 --chr1 1 --pos1 2 --chr2 3 --pos2 4 --resolution 10000 \
    --window_size 30000000 --step_size 10000000 --output_dist 5050000 \
    --chrom_size_path mm10.chrom.sizes --output_dir impute/10K/
hicluster gene-score --mode impute --resolution 10000 --cell_table_path imputed_cools.tsv \
    --gene_meta_path genes.bed --chrom_size_path mm10.chrom.sizes \
    --output_hdf_path geneimputescore.hdf --cpu 48
```

The gene-score table can then replace `gs` above. At 10 kb the imputed
matrices take about 1 GB per cell (about 3.5 TB for LC462) and several hours
on a cluster, so this case study scores raw contacts.
""")

md(r"""
## Summary

| Chang 2024 panel | epione |
|---|---|
| Fig 1f (chr1 maps, compartments) | `demux_pairs_by_barcode` → `contacts_to_scool` → `merge_cells` → `compartment_model` / `project_compartments` |
| Fig 1e (compartment SCC) | `project_compartments` → Spearman |
| gene scores | `load_contact_tables` → `gene_score` (10 kb, contacts ≥ 1 kb) |
| Fig 1d (co-embedding with RNA) | `epi.tl.integrate(method='cca')` → `epi.tl.transfer_labels` → UMAP |

Not covered: Fig 1g (25 kb insulation, TAD boundaries) and Fig 1h (10 kb
zoom). Build the pseudobulks at those resolutions with `contacts_to_scool` /
`merge_cells` and use `epi.bulk.hic.insulation` / `tad_boundaries`. The paper's
UMAP combines both samples (LC462 and LC716, 6,235 cells) and gene scores from
imputed 10 kb contacts; this case study uses LC462 and raw contacts.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = CELLS
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
out = pathlib.Path(__file__).with_name("t_droplet_hic_chang2024.ipynb")
nbf.write(nb, out)
print("wrote", out)
