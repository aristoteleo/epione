"""Generate t_sc_hic_loop.ipynb (HCEA blood cells: differential chromatin loops).

Execute with ``jupyter nbconvert --to notebook --execute --inplace``; needs
``pip install 'epione[hic]'`` and the HCEA loop calls of the six blood major
types (per type ``{ct}.loop.bedpe`` and ``{ct}.{Q,Q2,E,E2,T,T2}.cool``, about
2.4 GB each), the hg38 loop blacklist and a GENCODE hg38 GTF. Runs in about
30 min on 24 cores; later runs read the cached ANOVA.
"""
import pathlib

import nbformat as nbf

CELLS = []


def md(src):
    CELLS.append(nbf.v4.new_markdown_cell(src.strip("\n")))


def code(src):
    CELLS.append(nbf.v4.new_code_cell(src.strip("\n")))


md(r"""
# Case study: HCEA differential loops

Chromatin loops are pairs of distant loci that contact each other more than
their surroundings. In single-cell Hi-C they are called per cell group, from
contacts imputed at 10 kb: scHiCluster normalises every cell's imputed matrix
by its distance background (E) and by the local background around each pixel
(T), averages them over the cells of the group, and calls pixels whose E and T
are significantly positive. This case study takes the loops that the Human Cell
Epigenome Atlas (HCEA) called this way for six blood major types and finds the
loops that differ between them, following the atlas (Fig. 4).

| Step | epione | HCEA / scHiCluster |
|---|---|---|
| call loops per group | `call_loops` (not run here) | `hicluster loop` |
| merge the groups' loops | `merge_loops` | `merged_loop` (fig4/07) |
| ANOVA of Q / E / T across groups | `loop_anova` | `compute_anova` (fig4/07) |
| statistics at each loop, differential loops | `loop_values`, `differential_loops` | fig4/07, fig4/08 |
| group-specific loops, aggregate peak analysis | `specific_loops`, `loop_apa` | fig4/07 |
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
import seaborn as sns
import cooler
from scipy.stats import zscore, norm

import epione as epi
from epione.single import hic

epi.pl.plot_set()

# HCEA folder with loop/majortype/ and metadata/; override with EPIONE_HCEA_DIR.
DATA = pathlib.Path(os.environ.get('EPIONE_HCEA_DIR', 'data/hcea'))
LOOPS = DATA / 'loop/majortype'
BLACKLIST = pathlib.Path(os.environ.get('EPIONE_LOOP_BLACKLIST', 'data/blacklist/hg38_bismark_loop_blacklist.bed'))
GTF = pathlib.Path(os.environ.get('EPIONE_HG38_GTF', 'data/gencode_hg38/gencode.v30.annotation.gtf'))
OUT = pathlib.Path('result/t_sc_hic_loop')
OUT.mkdir(parents=True, exist_ok=True)
NPROC = int(os.environ.get('EPIONE_NPROC', 16))

CHROMS = [f'chr{i}' for i in range(1, 23)]
L1 = pd.read_csv(DATA / 'metadata/L1color.tsv', sep='\t', index_col=0)
CODES = {'B': 'c7', 'Mast': 'c24', 'Myeloid': 'c0', 'NK': 'c27', 'Tmem': 'c1', 'Tnaive': 'c15'}
TYPES = list(CODES)
COLORS = dict(zip(TYPES, ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#17becf']))
PREFIX = {t: LOOPS / c / c for t, c in CODES.items()}       # PREFIX[t].Q.cool, .Q2.cool, ...
for t in TYPES:
    assert pathlib.Path(f'{PREFIX[t]}.Q.cool').exists(), f'missing loop calls of {t}'
""")

md(r"""
## 2 · From single cells to loop calls (not run here)

The loop calls this case study starts from were made by the atlas with
scHiCluster, in two steps that epione wraps:

1. **Impute every cell at 10 kb** (`impute_cells`): convolution with a 5 x 5
   kernel (`pad=2`), random walk with restart in 30 Mb windows moved by 10 Mb,
   imputed contacts kept within 5.05 Mb of the diagonal. epione writes one
   imputed `.cool` per cell (`output='cool'`), the input of loop calling.
2. **Call loops per type** (`call_loops`, scHiCluster `hicluster loop`): each
   cell's imputed matrix is normalised by its distance background (E) and by
   the local background around each pixel (T); per type, the means over cells
   of Q (imputed contacts), E and T and of their squares are written as
   `<type>.{Q,Q2,E,E2,T,T2}.cool`. Pixels whose E and T are significantly
   positive across cells (t-tests, FDR from shuffled cells) and that stand out
   from their neighbourhood are the loop pixels (`<type>.loop.bedpe`); one
   summit per cluster of adjacent pixels is kept (`<type>.loop_summit.bedpe`).

For the 23,967 cells of the six types this takes about 1,400 CPU-hours (about
3.5 min per cell for the 10 kb imputation) and several TB of disk, so the code
below is switched off and the rest of the case study reads the atlas's
outputs (`data/hcea/loop/majortype/<type>/`). With `RUN_LOOP_CALLING = True`
it recomputes them from the contact tables and the later sections use the new
files.
""")
code(r"""
RUN_LOOP_CALLING = False
if RUN_LOOP_CALLING:
    FASTA = pathlib.Path(os.environ.get('EPIONE_HG38_FASTA', 'data/hg38/hg38.fa'))
    sizes = pd.read_csv(f'{FASTA}.fai', sep='\t', header=None, index_col=0)[1].loc[CHROMS]
    files = {f.name.split('.3C.')[0]: f for f in sorted(DATA.glob('sc_contact/*/*.3C.contact.rmbkl.tsv.gz'))}
    meta = pd.read_csv(DATA / 'metadata/5kCG100k3C_summary.csv.gz', index_col=0)
    meta = meta.loc[meta.index.isin(list(files)) & meta['majortype'].isin(list(CODES.values()))]
    meta['celltype'] = meta['majortype'].map({c: t for t, c in CODES.items()})
    cells = list(meta.index)
    # 1. impute every cell at 10 kb; one imputed .cool per cell
    raw = hic.load_contact_tables([files[c] for c in cells], chrom_sizes=sizes, resolution=10_000,
                                  cell_ids=cells, obs=meta[['celltype']], chromosomes=CHROMS)
    hic.impute_cells(raw, out_dir=OUT / 'imputed_10kb', resolution=10_000, pad=2, std=1.0, rp=0.5, tol=0.01,
                     window_size=30_000_000, step_size=10_000_000, output_dist=5_050_000, nproc=NPROC)
    # 2. call loops per type (scHiCluster); returns {type: prefix of its Q/E/T cools and loops}
    PREFIX = hic.call_loops(raw, groupby='celltype', out_dir=OUT / 'loops', black_list=BLACKLIST,
                            shuffle=True, cpu_per_job=NPROC)
print({t: str(p) for t, p in PREFIX.items()})
""")

md(r"""
## 3 · Loops called in each type

Each type has its loop pixels (`loop.bedpe`, 10 kb pixels that pass the
tests) and its loop summits (`loop_summit.bedpe`, one pixel per cluster of
adjacent loop pixels, the one with the highest E). `group_n_cells` is the
number of cells averaged into the type's matrices.
""")
code(r"""
stats = {}
for t in TYPES:
    px = pd.read_csv(f'{PREFIX[t]}.loop.bedpe', sep='\t', header=None)
    sm = pd.read_csv(f'{PREFIX[t]}.loop_summit.bedpe', sep='\t', header=None)
    stats[t] = {'cells': cooler.Cooler(f'{PREFIX[t]}.Q.cool').info['group_n_cells'],
                'loop pixels': len(px), 'loop summits': len(sm),
                'median summit distance (kb)': np.median(sm[4] - sm[1]) / 1e3}
pd.DataFrame(stats).T.astype({'cells': int, 'loop pixels': int, 'loop summits': int})
""")

md(r"""
## 4 · One set of loops for all six types

`merge_loops` takes the union of the six types' loop pixels, so every type is
measured at the same positions; a column per type records which types called
each pixel.
""")
code(r"""
loops = hic.merge_loops({t: f'{PREFIX[t]}.loop.bedpe' for t in TYPES}, chromosomes=CHROMS)
n_called = loops[TYPES].sum(axis=1)
print(f'{len(loops):,} merged loop pixels')
ax = n_called.value_counts().sort_index().plot.bar(color='0.4', figsize=(4.5, 3))
ax.set_xlabel('number of types that call the pixel'); ax.set_ylabel('loop pixels')
plt.tight_layout(); plt.show()
""")

md(r"""
## 5 · Which loops differ between the types?

`loop_anova` runs a one-way ANOVA at every pixel, separately for Q (imputed
contacts), E (enrichment over the distance background) and T (enrichment over
the local background). It needs only each type's mean (`Q.cool`), mean of
squares (`Q2.cool`) and number of cells: the F statistic is
`(SS_total / SS_within - 1) * (N - k) / (k - 1)`. Pixels in blacklisted regions
and outside 50 kb-5 Mb are left out, as in the atlas.

`differential_loops` reads the three F statistics at the merged loops, turns
them into p-values (F distribution, `k - 1 = 5` and `N - k` degrees of
freedom) and Benjamini-Hochberg FDRs, and applies the atlas rule: a loop is
differential when the z-scores of log10 F for both Q and T exceed
`norm.isf(0.15)` (the top ~15 % of each), it is 100 kb-4.95 Mb long and away
from the chromosome ends. We also require FDR < 0.01 for Q, E and T (atlas
Fig. 4, subtype level).
""")
code(r"""
ANOVA = OUT / 'anova'
anova = {m: ANOVA / f'{m}pv.cool' for m in 'QET'}
if not all(p.exists() for p in anova.values()):
    anova = hic.loop_anova(PREFIX, ANOVA, chromosomes=CHROMS, black_list=BLACKLIST, nproc=NPROC)
diff = hic.differential_loops(loops, anova, fdr=0.01, nproc=NPROC)
print(f"{diff['differential'].sum():,} differential loop pixels of {len(diff):,}")

thr = norm.isf(0.15)
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
for ax, m in zip(axes, 'QT'):
    v = diff.loc[diff[f'{m}anova'] > 0, f'{m}anova']
    ax.hist(zscore(np.log10(v)), bins=100, range=(-4, 4), color='0.5')
    ax.axvline(thr, color='r', lw=0.8)
    ax.set_xlabel(f'z-score of log10 F ({m})'); ax.set_ylabel('loops')
plt.tight_layout(); plt.show()
""")

md(r"""
## 6 · How the differential loops differ

At the differential loops we read each type's Q and T (`loop_values`) and
z-score every loop across the six types. Clustering 2,000 random differential
loops shows blocks of loops that are strong in one type or in a few related
types.
""")
code(r"""
dl = diff.loc[diff['differential']].reset_index(drop=True)
q = hic.loop_values(dl, {t: f'{PREFIX[t]}.Q.cool' for t in TYPES}, nproc=NPROC)
t_ = hic.loop_values(dl, {t: f'{PREFIX[t]}.T.cool' for t in TYPES}, nproc=NPROC)
zq, zt = zscore(q.to_numpy(), axis=1), zscore(t_.to_numpy(), axis=1)
rng = np.random.default_rng(0)
sel = rng.choice(len(dl), min(2000, len(dl)), replace=False)
cg = sns.clustermap(zq[sel], metric='cosine', col_cluster=False, cmap='bwr', vmin=-3, vmax=3)
plt.close(cg.fig)
order = sel[cg.dendrogram_row.reordered_ind]
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(6.5, 5))
for ax, z, m in zip(axes, (zq, zt), 'QT'):
    im = ax.imshow(z[order], cmap='bwr', vmin=-3, vmax=3, aspect='auto', interpolation='none')
    ax.set_xticks(range(len(TYPES))); ax.set_xticklabels(TYPES, rotation=90); ax.set_title(m)
axes[0].set_yticks([]); axes[0].set_ylabel(f'{len(sel):,} differential loops')
fig.colorbar(im, ax=axes, shrink=0.5, label='z-score across types')
plt.show()
""")

md(r"""
## 7 · Type-specific loops

`specific_loops` calls a differential loop specific to a type when its Q in
that type is more than 1.2 times the mean of the other five, and its T more
than 1.5 times (atlas thresholds). Aggregate peak analysis (`loop_apa`) then
averages each type's distance-normalised contacts around each type's specific
loops: a loop set should stand out (a bright centre) only in the type it was
called for.
""")
code(r"""
spec = hic.specific_loops(q, t_)
print(spec.sum().rename('specific loops').to_frame().T.to_string())
sets = {t: dl.loc[spec[t]] for t in TYPES}
APA = OUT / 'apa.npz'
if APA.exists():
    z = np.load(APA)
    apa = {(a, b): z[f'{a}|{b}'] for a in TYPES for b in TYPES}
else:
    apa = {}
    for a in TYPES:                                   # contacts of type a
        res = hic.loop_apa(f'{PREFIX[a]}.Q.cool', sets, chromosomes=CHROMS)
        for b in TYPES:                               # loops specific to type b
            apa[(a, b)] = res[b]
    np.savez(APA, **{f'{a}|{b}': v for (a, b), v in apa.items()})

fig, axes = plt.subplots(6, 6, figsize=(8, 8))
for i, a in enumerate(TYPES):
    for j, b in enumerate(TYPES):
        ax = axes[i, j]
        ax.imshow(apa[(a, b)], cmap='Oranges', vmin=0.3, vmax=0.75)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0: ax.set_title(b, fontsize=9)
        if j == 0: ax.set_ylabel(a, fontsize=9)
fig.supylabel('contacts of'); fig.suptitle('loops specific to')
plt.tight_layout(); plt.show()

score = pd.DataFrame({b: {a: apa[(a, b)][10, 10] / apa[(a, b)][-5:, :5].mean() for a in TYPES} for b in TYPES})
fig, ax, img = epi.pl.plot_correlation_heatmap(score, cmap='Oranges', vmin=1.5, vmax=4.5, figsize=(5.2, 4.4),
                                               annot_fontsize=7, title='APA score (rows: contacts, columns: loops)')
fig.axes[-1].set_ylabel('APA score')
plt.show()
""")

md(r"""
Each type's specific loops stand out most in that type's own contacts: in every
column the APA score is highest on the diagonal (3.4-4.5, against 1.5-3.9
elsewhere). The lymphoid types share part of their loops: memory-T-specific
loops are also strong in naive T and NK cells (3.9 and 3.7), and NK-specific
loops in memory T cells (3.2). Myeloid contacts show the mast-specific loops
almost as strongly as mast cells themselves (3.3 against 3.4).
""")

md(r"""
## 8 · An example: a B-cell loop at MS4A1

MS4A1 encodes CD20, the classic B-cell surface marker. Several B-specific
loops are anchored at it; we show the strongest (highest T in B cells) with
each type's T around it: red pixels contact each other more than their
neighbourhood, and loops appear as red dots.
""")
code(r"""
gtf = epi.io.read_gtf(GTF, required_attrs=('gene_name', 'gene_type'), feature_whitelist=['gene'],
                      keep_attribute=False)
gene = gtf[gtf['gene_name'] == 'MS4A1'].iloc[0]
near = lambda pos: (pos >= gene['start'] - 20_000) & (pos <= gene['end'] + 20_000)
cand = dl.loc[spec['B'] & (dl['chrom1'] == gene['seqname']) & (near(dl['start1']) | near(dl['start2']))]
best = cand.loc[t_.loc[cand.index, 'B'].idxmax()]
c, x, y = best['chrom1'], int(best['start1']), int(best['start2'])
print(f"{len(cand)} B-specific loops at MS4A1; strongest {c}:{x:,}-{y + 10_000:,} ({(y - x) / 1e3:.0f} kb)")
print('T at the loop:', t_.loc[best.name].round(2).to_dict())

lo, hi = x - 250_000, y + 250_000
mats = {t: cooler.Cooler(f'{PREFIX[t]}.T.cool').matrix(balance=False).fetch(f'{c}:{lo}-{hi}') for t in TYPES}
n = len(next(iter(mats.values())))
ii, jj = np.indices((n, n))
near_diag = np.abs(ii - jj) < 5                      # the first 50 kb carry no loop signal
v = max(np.nanpercentile(np.abs(m[~near_diag]), 99.5) for m in mats.values())
fig, axes = plt.subplots(1, 6, figsize=(18, 3.4), sharey=True)
axes[0].set_ylabel(f'{c} (Mb)')
for ax, t in zip(axes, TYPES):
    m = mats[t].astype(float)
    m[near_diag] = np.nan
    im = ax.imshow(m, cmap='RdBu_r', vmin=-v, vmax=v, extent=(lo / 1e6, hi / 1e6, hi / 1e6, lo / 1e6))
    ax.scatter([y / 1e6 + 0.005], [x / 1e6 + 0.005], s=160, facecolors='none', edgecolors='k', lw=1)
    ax.set_title(t); ax.set_xlabel(f'{c} (Mb)')
fig.colorbar(im, ax=axes, shrink=0.8, label='T (local enrichment)')
plt.show()
""")

md(r"""
The loop joins MS4A1 to a site about 110 kb away. It is a clear dot in B cells
(T = 0.41) and much weaker in the other types (0.15-0.31, highest in myeloid
cells). The T maps also show loops shared by several types nearby.
""")

md(r"""
## Summary

| Step | epione |
|---|---|
| loops per group | `call_loops` (scHiCluster, from imputed `.cool` cells) |
| one loop set | `merge_loops` |
| ANOVA across groups | `loop_anova` |
| differential loops | `loop_values`, `differential_loops` |
| type-specific loops, APA | `specific_loops`, `loop_apa` |

`loop_anova`, `merge_loops`, `loop_values` and the FDR reproduce the atlas's
differential-loop analysis of all 35 major types
(`tests/parity/schicluster_loop`). To call loops on your own cells, impute
them at 10 kb (`impute_cells(resolution=10_000, pad=2, window_size=30_000_000,
step_size=10_000_000, output_dist=5_050_000)`, which writes one `.cool` per
cell) and run `call_loops(adata, groupby=..., out_dir=..., black_list=...)`;
its output prefixes go straight into `loop_anova` and `merge_loops`.
""")


nb = nbf.v4.new_notebook()
nb["cells"] = CELLS
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
out = pathlib.Path(__file__).with_name("t_sc_hic_loop.ipynb")
nbf.write(nb, out)
print("wrote", out)
