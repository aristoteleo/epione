"""Generate t_seq2print.ipynb (seq2PRINT in Epione).

Execute with ``jupyter nbconvert --to notebook --execute --inplace`` on a
GPU node; training takes about an hour on an H100.
"""
import pathlib

import nbformat as nbf

CELLS = []


def md(src):
    CELLS.append(nbf.v4.new_markdown_cell(src.strip("\n")))


def code(src):
    CELLS.append(nbf.v4.new_code_cell(src.strip("\n")))


md(r"""
# seq2PRINT — predicting footprints from DNA

[PRINT](t_print_footprint.ipynb) reads footprints off measured Tn5 cuts.
**seq2PRINT** (from [scPrinter](https://github.com/buenrostrolab/scPrinter))
learns to predict them from sequence: a dilated CNN reads 1,840 bp of DNA
and outputs PRINT's footprint z-score at 99 scales across the central
800 bp, plus the window's log coverage. Fine-tuned per cell state with
LoRA, one model serves every state — a sequence-to-footprint model you can
then perturb, attribute or compare across states.

```python
model = epi.models.seq2print.build_seq2print()                    # scPrinter's architecture
data  = epi.models.seq2print.Seq2PRINTDataset(peaks, fasta=..., reader=..., bias=...)
res   = epi.models.seq2print.fit(model, data.split(train), data.split(valid))
model.add_lora(group_embeddings)                                  # per-state fine-tuning
```

**What is scPrinter's, and what is not.** The network blocks and the
training targets are scPrinter's, vendored verbatim: the same weights give
**bit-identical** predictions, and checkpoints load in either direction
(last section). The training path is not. scPrinter trains through a JSON
config, a CLI script of ~770 lines, a `fit` method bound to wandb, and one
exported insertion bigWig per cell group plus a genome-wide bias bigWig.
Here windows are read from the fragments file and an on-demand bias, and
training is one function call.

| | scPrinter | Epione |
|---|---|---|
| inputs | per-group bigWigs + genome-wide bias bigWig | fragments file + `Tn5Bias` |
| model | built from a JSON config | `build_seq2print(**kwargs)`, scPrinter's defaults |
| LoRA | five copy-pasted wrapping blocks | `model.add_lora(...)`, same layout and scaling |
| training | `seq2print_train` CLI, wandb | `fit()`; same loss, metrics, AdamW, EMA, early stopping |
| folds | `genome.splits[fold]` | `chromosome_folds('mm10', fold)` — the same tables |

The data are the same as in the other tutorials: GSE308623 `P2S1`, mouse
brain at P2, spatial ATAC, four dorsal→ventral bands of spots.
""")

md("## Part 1 · Setup")

code(r"""
%matplotlib inline
import os, sys, pathlib, time, json
import warnings
warnings.filterwarnings('ignore', module='tqdm')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pysam
import torch


def _find_repo(start):
    p = pathlib.Path(start).resolve()
    for cand in (p, *p.parents):
        if (cand / 'epione' / '__init__.py').exists():
            return cand
    return None


CODE = (pathlib.Path(os.environ['EPIONE_SRC']) if os.environ.get('EPIONE_SRC')
        else _find_repo(pathlib.Path.cwd()))
if CODE is not None:
    sys.path.insert(0, str(CODE))

import epione as epi
from epione._insertions import InsertionReader
s2p = epi.models.seq2print

DATA  = pathlib.Path(os.environ.get('SCOOBY_TUTORIAL_DATA', (CODE or pathlib.Path.cwd()).parent))
FRAG  = DATA / 'scooby_bench' / 'P2S1.fragments.tsv.gz'
POS   = DATA / 'scooby_bench' / 'P2S1.positions.csv'
FASTA = DATA / 'genome' / 'mm10' / 'mm10.fa'          # whole genome: folds are by chromosome
WORK  = DATA / 'seq2print_tutorial'; WORK.mkdir(exist_ok=True)
for p in (FRAG, POS, FASTA):
    if not p.exists():
        raise FileNotFoundError(f'{p} — see "Getting the inputs" in t_scooby_bridge; '
                                'mm10.fa is hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', DEVICE, torch.cuda.get_device_name(0) if DEVICE == 'cuda' else '')

C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
plt.rcParams.update({
    'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb',
    'axes.edgecolor': '#c3c2b7', 'axes.linewidth': 0.8, 'axes.labelcolor': INK2,
    'axes.titlesize': 11, 'axes.titleweight': 'bold', 'axes.titlecolor': INK,
    'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'grid.color': GRID, 'grid.linewidth': 0.6, 'legend.frameon': False,
    'legend.fontsize': 9, 'font.size': 10, 'lines.linewidth': 2.0, 'figure.dpi': 110,
})
""")

md(r"""
## Part 2 · Spots, groups, peaks and folds

Same spots and bands as the PRINT tutorial. Peaks are stand-ins — the
10,000 densest 500-bp bins genome-wide, summit at the bin centre — and the
split is **scPrinter's fold 0 for mm10**: whole chromosomes for train,
validation and test, so no held-out window shares sequence with training.
""")

code(r"""
pos = (pd.read_csv(POS, header=None, names=['bc', 'in_tissue', 'row', 'col', 'pr', 'pc'])
         .query('in_tissue == 1').reset_index(drop=True))
barcodes = (pos['bc'] + '-1').tolist()
edges = np.quantile(pos['row'], [0, .25, .5, .75, 1])
band = np.clip(np.digitize(pos['row'], edges[1:-1]), 0, 3)
bands = [f'band {b + 1}' for b in range(4)]
groups = [np.flatnonzero(band == b) for b in range(4)]
print({n: len(g) for n, g in zip(bands, groups)})

folds = s2p.chromosome_folds('mm10', 0)
{k: ', '.join(v) for k, v in folds.items()}
""")

code(r"""
%%time
N_PEAKS = 10_000
peak_file = WORK / f'peaks_top{N_PEAKS}.tsv'
if peak_file.exists():
    peaks = pd.read_csv(peak_file, sep='\t')
else:
    keep, cnt = set(barcodes), {}
    tbx = pysam.TabixFile(str(FRAG))
    for c in [f'chr{i}' for i in range(1, 20)] + ['chrX']:
        for line in tbx.fetch(c):
            f = line.split('\t', 4)
            if f[3] in keep:
                for p in (int(f[1]), int(f[2])):
                    cnt[(c, p // 500)] = cnt.get((c, p // 500), 0) + 1
    top = sorted(cnt.items(), key=lambda kv: -kv[1])[:N_PEAKS]
    peaks = pd.DataFrame({'chrom': [k[0] for k, _ in top],
                          'summit': [k[1] * 500 + 250 for k, _ in top],
                          'cuts': [n for _, n in top]})
    peaks.to_csv(peak_file, sep='\t', index=False)
peaks['split'] = np.select([peaks.chrom.isin(folds[s]) for s in ('train', 'valid', 'test')],
                           ['train', 'valid', 'test'], 'unused')
peaks['split'].value_counts()
""")

md(r"""
## Part 3 · Training windows

`Seq2PRINTDataset` reads each window once, straight from the fragments —
1,840 bp of sequence and a 1-kb track of insertions and Tn5 bias around the
summit — and applies scPrinter's augmentation (shared jitter of up to
±128 bp, reverse complement). The footprint target is computed from those
two tracks on the fly at every step, on 30 random scales.

The bias is PRINT's CNN. Here it runs on the GPU with `exact=False` (one
pass per window): fast, and equal to scPrinter's genome-wide bias up to
float32 rounding. The bit-exact mode reproduces scPrinter's CPU chunking
and is what the PRINT tutorial uses; for training targets the speed
matters more.
""")

code(r"""
%%time
bias = epi.tl.tn5_bias(FASTA, device=DEVICE, exact=False)
model = s2p.build_seq2print()
print(f'model: {sum(p.numel() for p in model.parameters()):,} parameters; '
      f'{model.dna_len} bp in -> {model.output_len} bp out, 99 scales')


def windows(groups=None):
    reader = InsertionReader(FRAG, barcodes, shift=bias.shift)
    return s2p.Seq2PRINTDataset(peaks, fasta=FASTA, reader=reader, bias=bias,
                                groups=groups, dna_len=model.dna_len)


bulk = windows()                         # all spots as one pseudobulk
train, valid, test = (bulk.split(folds[s]) for s in ('train', 'valid', 'test'))
print({'train': len(train), 'valid': len(valid), 'test': len(test)})
""")

code(r"""
X, y = test.centred()[0]
print('X', tuple(X.shape), '(one-hot DNA)   y', tuple(y.shape), '(insertions, bias)')
dispersion = epi.tl.DispersionModel.load()
target_model = s2p.footprint_target_model(dispersion)
z = s2p.multiscaleFoot(y[None, 0], y[None, 1], np.arange(2, 101), target_model)[0]
print('target', tuple(z.shape), f'(scales x positions); {torch.isnan(z).float().mean():.1%} NaN '
      '(no cuts in a flank: masked in the loss)')
""")

md(r"""
## Part 4 · Train the bulk model

scPrinter's recipe: AdamW (lr 3×10⁻³, weight decay 10⁻³), bf16, an
exponential moving average of the weights (the EMA is what is validated
and kept), gradient clipping, early stopping after 5 epochs without
improvement in validation loss.

seq2PRINT is heavy — every window runs 1,840 bp through eight 1,024-channel
dilated blocks, ~130 GFLOP forward and backward — so on an H100 an epoch
here takes ~15 s and training to convergence ~10 minutes.

So by default the tutorial **loads the trained model from Hugging Face**
([`starlitnightly/epione-seq2print`](https://huggingface.co/starlitnightly/epione-seq2print)),
~47 MB, pinned to a revision and md5-checked. Its `manifest.json` carries
the training history, so every figure below is the same either way. Set
`TRAIN = True` to retrain; a finished run is cached in `WORK`.
""")

code(r"""
%%time
TRAIN = False                                     # True: train instead of downloading
ckpt, hist_file = WORK / 'bulk_fold0.pt', WORK / 'bulk_fold0_history.tsv'
if not TRAIN and 'p2s1_fold0_bulk' in s2p.PRETRAINED:
    model, manifest = s2p.load_pretrained('p2s1_fold0_bulk', return_manifest=True)
    hist = pd.DataFrame(manifest['history'])
    print('loaded p2s1_fold0_bulk from Hugging Face')
elif ckpt.exists() and hist_file.exists():
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    hist = pd.read_csv(hist_file, sep='\t')
    print('loaded cached model')
else:
    res = s2p.fit(model, train, valid, epochs=60, patience=5, num_workers=6,
                  device=DEVICE, target_model=target_model)
    model, hist = res.model, res.to_frame()
    torch.save(model.state_dict(), ckpt)          # only after a complete run
    hist.to_csv(hist_file, sep='\t', index=False)
best_epoch = int(hist.valid_loss.idxmin())
print(f'best epoch {best_epoch} of {len(hist)}')
""")

code(r"""
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.6))
a1.plot(hist.epoch, hist.train_loss, color=C[1], lw=1.6, label='train (augmented, 30 scales)')
a1.plot(hist.epoch, hist.valid_loss, color=C[0], lw=1.8, label='valid (centred, 99 scales)')
a1.axvline(best_epoch, color=MUTED, lw=0.8, ls=':')
a1.set_xlabel('epoch'); a1.set_ylabel('loss'); a1.legend(); a1.grid(alpha=0.6)
a1.set_title('Loss')
for col, c, lab in (('valid_profile_pearson', C[0], 'footprint profile r'),
                    ('valid_coverage_pearson', C[2], 'coverage r')):
    a2.plot(hist.epoch, hist[col], color=c, lw=1.8, label=lab)
    a2.annotate(f'{hist[col].iloc[best_epoch]:.2f}', xy=(best_epoch, hist[col].iloc[best_epoch]),
                xytext=(5, 5), textcoords='offset points', color=INK, fontsize=9, fontweight='bold')
a2.axvline(best_epoch, color=MUTED, lw=0.8, ls=':')
a2.set_xlabel('epoch'); a2.set_ylabel('Pearson r (validation chromosomes)'); a2.legend()
a2.grid(alpha=0.6); a2.set_title('Validation metrics')
plt.tight_layout(); plt.show()
""")

md(r"""
## Part 5 · Held-out chromosomes

`evaluate` computes scPrinter's four validation metrics on centred
windows and all 99 scales — here on the **test** chromosomes, which the
model never saw.
""")

code(r"""
%%time
model = model.to(DEVICE)
metrics_bulk = s2p.evaluate(model, test, target_model=target_model, device=DEVICE)
pd.Series(metrics_bulk).round(3)
""")

code(r"""
# One held-out window: observed footprint vs prediction.
i = int(np.argsort(test.coverage[:, 0])[-50])       # a well-covered test peak
X, y = test.centred()[i]
with torch.no_grad():
    pred, _ = model.eval()(X[None].to(DEVICE))
obs = s2p.multiscaleFoot(y[None, 0].to(DEVICE), y[None, 1].to(DEVICE),
                         np.arange(2, 101), target_model)[0]   # fit() moved it to DEVICE
pred = pred[0].float().cpu().numpy(); obs = obs.cpu().numpy()
r = np.corrcoef(np.nan_to_num(obs).ravel(), pred.ravel())[0, 1]
chrom, summit = test.peaks.chrom[i], int(test.peaks.summit[i])

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
ext = [-400, 400, 4, 200]
vm = np.nanpercentile(np.abs(obs), 99)
for ax, M, t in ((axes[0], obs, 'observed (PRINT z, from cuts)'), (axes[1], pred, 'predicted from sequence')):
    im = ax.imshow(M, aspect='auto', origin='lower', cmap='RdBu_r', vmin=-vm, vmax=vm,
                   extent=ext, interpolation='nearest')
    ax.set_title(t); ax.set_xlabel('position from summit (bp)')
axes[0].set_ylabel('footprint size (bp)')
cb = fig.colorbar(im, ax=axes, shrink=0.85, label='footprint z'); cb.outline.set_visible(False)
fig.suptitle(f'{chrom}:{summit:,} (test chromosome) — Pearson r = {r:.2f}',
             fontsize=11, fontweight='bold', color=INK, y=1.02)
plt.show()
""")

code(r"""
# Coverage head across all test windows
loader = torch.utils.data.DataLoader(test.centred(), batch_size=128)
pc, oc = [], []
with torch.no_grad():
    for Xb, yb in loader:
        pc.append(model(Xb.to(DEVICE))[1].float().cpu()); oc.append(torch.log1p(yb[:, 0].sum(-1)))
pc, oc = torch.cat(pc).numpy(), torch.cat(oc).numpy()
fig, ax = plt.subplots(figsize=(4.8, 4.5))
ax.scatter(oc, pc, s=3, color=C[0], alpha=0.3, lw=0)
lo, hi = np.percentile(np.r_[oc, pc], [0.5, 99.5])
ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.8, ls=':')
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel('observed log1p(cuts in 1 kb)'); ax.set_ylabel('predicted')
ax.set_title(f'Coverage, {len(oc):,} test windows — r = {np.corrcoef(oc, pc)[0, 1]:.2f}')
ax.grid(alpha=0.6); plt.tight_layout(); plt.show()
""")

md(r"""
## Part 6 · Per-band fine-tuning with LoRA

To make the model state-specific, scPrinter freezes the trained backbone
and adds LoRA to every convolution: a small network turns a cell state's
embedding into a low-rank weight update, so one backbone serves every
state. `add_lora` does the same, with the same layout and embedding
rescaling (so checkpoints still load into scPrinter).

**This tutorial uses a light LoRA.** With scPrinter's defaults (rank 32,
hidden width 256, every convolution wrapped) the networks that *generate*
the LoRA matrices hold 414 million parameters — 35× the 12-million-
parameter backbone they adapt, since each emits a full A and B per layer —
and fine-tuning four bands takes the better part of an hour. Here rank 8
and hidden width 64 give 26 million parameters and ~15 minutes of
training. Because it is much smaller and trains briefly, it uses a higher
learning rate (10⁻⁴, no gradient accumulation) than scPrinter's
3×10⁻⁵ × 8. A benchmark against scPrinter should use its defaults:
`add_lora(emb, coverages=cov)` and
`fit(..., lr=3e-5, accumulate=8, coverage_warming=5)`.

**Use coverage warming.** A footprint z-score grows with depth (roughly
as √n), and each band has about a quarter of the pooled cuts, so the bulk
model overshoots every band's targets. LoRA can reduce that error by
*bending the shape* of its predictions instead. scPrinter's remedy,
reproduced by `coverage_warming=`, first trains only a small
coverage-adjustment head to absorb the scale, then freezes it and trains
LoRA. (A first attempt here without warming, and at lr 10⁻³, halved
per-band accuracy; we did not separate the two causes.)

Each band's embedding here is simply its one-hot identity; a real analysis
would use the band's mean LSI or joint embedding. The coverage covariate
is the band's log cuts in peaks, z-scored, as scPrinter supplies.

""")

code(r"""
%%time
per_band = windows(groups)               # every (peak, band) pair; items carry the band index
tr4, va4, te4 = (per_band.split(folds[s]) for s in ('train', 'valid', 'test'))
cov = np.log10(per_band.coverage.sum(0))[:, None]  # log cuts in peaks, per band
cov = ((cov - cov.mean()) / cov.std()).astype(np.float32)
emb = np.eye(4, dtype=np.float32)

import copy
lora = copy.deepcopy(model).cpu().add_lora(emb, coverages=cov, rank=8, hidden_dim=64)
n_train = sum(p.numel() for p in lora.parameters() if p.requires_grad)
print(f'LoRA: {n_train:,} trainable of {sum(p.numel() for p in lora.parameters()):,} parameters '
      '(backbone frozen; the trainable part is the LoRA hypernetworks)')
lora_ckpt, lora_hist = WORK / 'lora_fold0.pt', WORK / 'lora_fold0_history.tsv'
if not TRAIN and 'p2s1_fold0_lora_bands' in s2p.PRETRAINED:
    lora = s2p.load_pretrained('p2s1_fold0_lora_bands')
    print('loaded p2s1_fold0_lora_bands from Hugging Face')
elif lora_ckpt.exists() and lora_hist.exists():
    lora.load_state_dict(torch.load(lora_ckpt, map_location='cpu'))
    print('loaded cached LoRA model')
else:
    res_lora = s2p.fit(lora, tr4, va4, epochs=8, patience=3, lr=1e-4, coverage_warming=2,
                       num_workers=6, device=DEVICE, target_model=target_model)
    lora = res_lora.model
    torch.save(lora.state_dict(), lora_ckpt)
    res_lora.to_frame().to_csv(lora_hist, sep='\t', index=False)
lora = lora.to(DEVICE).eval()
""")

code(r"""
%%time
# Held-out chromosomes, one band at a time. Three predictors per band:
#   bulk            the model before fine-tuning
#   LoRA            conditioned on the band itself
#   LoRA, wrong band  the same windows, conditioned on the *opposite* band
# If LoRA only helped because fine-tuning is more training, the wrong-band
# score would match the right-band one. It is the control that says
# whether band identity is actually used.
class AsBand(torch.utils.data.Dataset):
    def __init__(self, base, idx, band):
        self.base, self.idx, self.band = base, idx, band
    def __len__(self):
        return len(self.idx)
    def __getitem__(self, i):
        X, y, _ = self.base[self.idx[i]]
        return X, y, torch.tensor(self.band)

rows = []
centred = te4.centred()
for g, name in enumerate(bands):
    idx = np.arange(g, len(te4), 4)
    for label, m, cond in (('bulk', model, g), ('LoRA', lora, g),
                           ('LoRA, wrong band', lora, 3 - g)):
        r = s2p.evaluate(m, AsBand(centred, idx, cond), target_model=target_model, device=DEVICE)
        rows.append({'band': name, 'model': label, **r})
per_band_metrics = pd.DataFrame(rows)
per_band_metrics.pivot(index='band', columns='model',
                       values=['profile_pearson', 'coverage_pearson']).round(3)
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=False)
w = 0.27
xb = np.arange(4)
for ax, col, t in ((axes[0], 'profile_pearson', 'Footprint profile r'),
                   (axes[1], 'coverage_pearson', 'Coverage r')):
    for k, (label, colr) in enumerate((('bulk', MUTED), ('LoRA', C[0]),
                                        ('LoRA, wrong band', C[1]))):
        v = per_band_metrics.query('model == @label')[col].to_numpy()
        bars = ax.bar(xb + (k - 1) * w, v, width=w - 0.03, color=colr, label=label)
        for xx, vv in zip(xb + (k - 1) * w, v):
            ax.annotate(f'{vv:.2f}', xy=(xx, vv), xytext=(0, 3), textcoords='offset points',
                        ha='center', fontsize=8, color=INK2)
    ax.set_xticks(xb); ax.set_xticklabels(bands)
    ax.set_title(t + ' — test chromosomes'); ax.grid(axis='y', alpha=0.6); ax.set_axisbelow(True)
axes[0].legend(loc='lower right')
plt.tight_layout(); plt.show()
""")

md(r"""
**Read the control before the gain.** LoRA lifts every band's footprint
accuracy (r ≈ 0.43 → 0.47) and coverage accuracy (≈ 0.14 → 0.19). But
conditioning on the *wrong* band scores the same, so none of that gain is
band identity: it is the generic effect of adapting the bulk model to
thinner, per-band data — mostly the depth scaling absorbed during warming
(validation r was already 0.46 after the two warming epochs, 0.47 at the
end).

That is a fact about these groups, not a bug: four quartiles of array row
each mix many cell types, so their footprints differ mainly in depth. Cell
states for LoRA should be groups that differ in chromatin — clusters, or
the per-spot embedding scPrinter uses — and the wrong-state control should
be run with them too. Without it, a per-state gain cannot be attributed to
the state.
""")

md(r"""
## Part 7 · Exactness and checkpoint compatibility

The network blocks are scPrinter's own, vendored unchanged, and the
assembled model has scPrinter's layout. So the bulk model trained above
loads into **scPrinter's** `seq2PRINT` class with `strict=True`, and the
two must predict identically — checked here on real held-out windows.
The training targets get the same check. `tests/test_seq2print.py` covers
the LoRA model, the EMA (against `ema_pytorch`) and the loss as well.

Set `SCPRINTER_SRC` to the `scprinter/` directory of a scPrinter clone to
run it.
""")

code(r"""
SRC = os.environ.get('SCPRINTER_SRC')
if not SRC:
    print('SCPRINTER_SRC not set — skipping.')
else:
    import ast, importlib, shutil, types, tempfile
    root = pathlib.Path(tempfile.mkdtemp()); pkg = root / 'scp_up'
    (pkg / 'seq').mkdir(parents=True)
    (pkg / '__init__.py').write_text(''); (pkg / 'seq' / '__init__.py').write_text('')
    for f in ('Models.py', 'Modules.py', 'Functions.py', 'minimum_footprint.py'):
        shutil.copy(pathlib.Path(SRC) / 'seq' / f, pkg / 'seq' / f)
    text = (pathlib.Path(SRC) / 'utils.py').read_text(); lines = text.splitlines(keepends=True)
    fn = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'rz_conv_torch')
    (pkg / 'utils.py').write_text('import torch\n\n' + ''.join(lines[fn.lineno - 1:fn.end_lineno]))
    sys.modules.setdefault('wandb', types.ModuleType('wandb')); sys.path.insert(0, str(root))
    UM = importlib.import_module('scp_up.seq.Modules')
    UModels = importlib.import_module('scp_up.seq.Models')
    Umf = importlib.import_module('scp_up.seq.minimum_footprint')

    # scPrinter's construct_model_from_config, with its published defaults
    dil = lambda x: 2 ** (x + 1)
    theirs = UModels.seq2PRINT(
        dna_cnn_model=UM.DNA_CNN(n_filters=1024),
        hidden_layer_model=UM.DilatedCNN(n_filters=1024, bottleneck=1024, n_layers=8, kernel_size=3,
                                         groups=8, activation=torch.nn.GELU(), batch_norm=True,
                                         residual=True, rezero=False, dilation_func=dil,
                                         batch_norm_momentum=0.1, inception=[True] * 8,
                                         inception_version=2),
        profile_cnn_model=UM.Footprints_head(1024, kernel_size=1, n_scales=99),
        dna_len=model.dna_len, output_len=800)
    theirs.load_state_dict(model.state_dict(), strict=True)
    theirs = theirs.to(DEVICE).eval(); model.eval()

    same_pred = same_tgt = True
    tm_up = Umf.dispModel({str(r): {**{k: p[k] for k in ('featureMean', 'featureSD', 'targetMean', 'targetSD')},
                                    'modelWeights': [torch.from_numpy(w).float() for w in p['weights']]}
                           for r, p in dispersion.params.items()}).to(DEVICE)
    tm = target_model.to(DEVICE)
    loader = torch.utils.data.DataLoader(test.centred(), batch_size=64)
    n = 0
    with torch.no_grad():
        for b, (Xb, yb) in enumerate(loader):
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            a, c = model(Xb), theirs(Xb)
            same_pred &= torch.equal(a[0], c[0]) and torch.equal(a[1], c[1])
            ta = s2p.multiscaleFoot(yb[:, 0], yb[:, 1], np.arange(2, 101), tm)
            tb = Umf.multiscaleFoot(yb[:, 0], yb[:, 1], np.arange(2, 101), tm_up)
            same_tgt &= torch.equal(torch.nan_to_num(ta, 123.), torch.nan_to_num(tb, 123.))
            n += len(Xb)
            if b == 9:
                break
    print(f'{n} held-out windows: predictions identical to scPrinter: {same_pred}; '
          f'targets identical: {same_tgt}')
""")

md(r"""
## Summary

| | |
|---|---|
| Data | GSE308623 `P2S1` (same as the scooby and PRINT tutorials), 10,000 stand-in peaks |
| Split | scPrinter's mm10 fold 0 — whole chromosomes |
| Inputs | fragments + PRINT bias on demand; no bigWig export |
| Model | scPrinter's architecture and weights layout; checkpoints load both ways |
| Exactness | predictions and targets equal to scPrinter's on real held-out windows |
| Runtime | bulk ~12 min, light LoRA ~15 min on an H100 (scPrinter's LoRA defaults ~1 h); trained weights download from Hugging Face |
| LoRA on spatial bands | +0.04 profile r over bulk, but the wrong-band control matches it: the gain is depth adaptation, not band identity |

Not included yet: scPrinter's **attribution and sequence-TFBS** stages
(DeepLIFT on the trained model, then its pretrained TFBS heads), which turn
a trained seq2PRINT into base-resolution TF binding calls.
""")

nb = nbf.v4.new_notebook(cells=CELLS)
nb.metadata.update({
    "kernelspec": {"display_name": "epi-scooby", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
})
out = pathlib.Path(__file__).with_name("t_seq2print.ipynb")
nbf.write(nb, str(out))
print(f"wrote {out} ({len(CELLS)} cells)")
