"""Generate t_pretrained_models.ipynb (the pretrained-model registry).

Execute with ``jupyter nbconvert --to notebook --execute --inplace`` in an
environment with torch; it downloads about 50 MB (PRINT, HINT and one
seq2PRINT checkpoint) and runs on a CPU in under a minute.
"""
import pathlib

import nbformat as nbf

CELLS = []


def md(src):
    CELLS.append(nbf.v4.new_markdown_cell(src.strip("\n")))


def code(src):
    CELLS.append(nbf.v4.new_code_cell(src.strip("\n")))


md(r"""
# Pretrained models — one registry for every family

Epione loads published weights for several model families: ChromBPNet's
ENCODE releases, PRINT's footprinting networks, HINT-ATAC's bias tables and
HMMs, Borzoi backbones, the published scooby models and Epione's own
seq2PRINT checkpoints. They are all entries of a single registry,
`epi.models.registry`:

```python
epi.models.list_pretrained(family=None, **filters)   # -> DataFrame
epi.models.load_pretrained(name, **loader_kwargs)    # -> model
epi.models.fetch_pretrained(name)                    # -> verified local file(s)
epi.models.register(entry)                           # add your own
```

Each entry records what the model is (family, task, organism, genome,
biosample), where it comes from — a Hugging Face repository **at a pinned
commit**, an ENCODE accession, a Zenodo record, a Git commit — every file
with an **md5 or sha256**, its license and citation, and the loader that
builds it. Nothing is re-hosted: files are downloaded from the original
publisher on first use, checked against the pinned checksum and cached. A
file that no longer matches is refused rather than silently loaded, so a
script keeps loading the model it was written against.

The per-family functions still work and now go through the registry:
`epi.models.chrombpnet.load_pretrained('encode_k562', 'bias')`,
`epi.models.seq2print.load_pretrained('p2s1_fold0_bulk')`,
`epi.tl.fetch_print_model('tn5_bias')`, `epi.tl.fetch_hint_file(...)`.
""")

md("## Setup")

code(r"""
import os, sys, pathlib


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

import warnings
warnings.filterwarnings('ignore', module='tqdm')
import numpy as np
import pandas as pd
import epione as epi

pd.set_option('display.max_colwidth', 60)
""")

md(r"""
## What is registered

Listing needs no torch and touches no network. `cached` says whether the
files are already on this machine.
""")

code(r"""
models = epi.models.list_pretrained()
models[['family', 'task', 'organism', 'genome', 'biosample', 'source', 'size_mb', 'license', 'cached']]
""")

md(r"""
Filters are column names. Strings match case-insensitively, a list matches
any member, a function is applied to the column:
""")

code(r"""
epi.models.list_pretrained('chrombpnet', biosample=['K562', 'hepg2'])[['biosample', 'source', 'size_mb']]
""")

code(r"""
epi.models.list_pretrained(organism='human', size_mb=lambda mb: mb < 10)[['family', 'task', 'source']]
""")

md(r"""
## One entry in full

`get_pretrained` returns the entry: its pinned source, files and checksums,
license, citation and how its loader is configured. Older names still
resolve as aliases (`'encode_k562'` → `'chrombpnet_encode_k562'`).
""")

code(r"""
e = epi.models.get_pretrained('encode_k562')
print(e.name, '|', e.task, '|', e.organism, e.genome, e.biosample)
print('source :', e.source)
print('files  :', e.files)
print('url    :', e.url(next(iter(e.files))))
print('license:', e.license)
print('cite   :', e.citation)
""")

md(r"""
## Loading a model

`load_pretrained` downloads what is missing, verifies it and hands the
files to the family's loader; keyword arguments go to that loader. The
small PRINT models make the point — the Tn5 bias CNN is a TorchScript
module that maps one-hot DNA to a bias track:
""")

code(r"""
import torch

tn5 = epi.models.load_pretrained('print_tn5_bias')
rng = np.random.default_rng(0)
seq = torch.nn.functional.one_hot(torch.from_numpy(rng.integers(0, 4, 400)), 4).T[None].float()
with torch.no_grad():
    out = tn5(seq)
print(type(tn5).__name__, tuple(seq.shape), '->', tuple(out.shape))
""")

md(r"""
The same call builds non-network models too: PRINT's dispersion model
(the footprint null) and HINT-ATAC's HMM, which ships with Epione.
""")

code(r"""
disp = epi.models.load_pretrained('print_dispersion')
hmm = epi.models.load_pretrained('hint_atac_hmm_paired')
print(type(disp).__name__, disp.version, f'{len(disp.params)} radii')
print(type(hmm).__name__, hmm.name, 'states x tracks:', np.asarray(hmm.means).shape)
""")

md(r"""
A seq2PRINT checkpoint trained in the [seq2PRINT tutorial](t_seq2print.ipynb)
(47 MB from Hugging Face, commit `e8a36d4`); `return_manifest=True` also
returns its training record:
""")

code(r"""
model, manifest = epi.models.load_pretrained('seq2print_p2s1_fold0_bulk', return_manifest=True)
x = torch.nn.functional.one_hot(torch.from_numpy(rng.integers(0, 4, model.dna_len)), 4).T[None].float()
with torch.no_grad():
    footprint, coverage = model(x)
print('input', tuple(x.shape), '-> footprints', tuple(footprint.shape), '+ coverage', tuple(coverage.shape))
print('manifest records:', ', '.join(manifest))
""")

md(r"""
The large models load the same way — they are not run here because each
is 0.5–0.8 GB:

```python
bias  = epi.models.load_pretrained('chrombpnet_encode_k562', kind='bias', fold=0)  # BPNet
full  = epi.models.load_pretrained('chrombpnet_encode_k562')                        # ChromBPNet
borz  = epi.models.load_pretrained('borzoi_replicate_0')        # needs borzoi-pytorch
scoob = epi.models.load_pretrained('scooby_neurips')            # needs upstream scooby
```

The scooby entries carry the constructor arguments the checkpoints need
(`cell_emb_dim`, `n_tracks`, read off the weights), so no upstream script is
needed to rebuild them:
""")

code(r"""
for name in epi.models.list_pretrained('scooby').index:
    print(f'{name:22s}', epi.models.get_pretrained(name).loader_kwargs)
""")

md(r"""
## Cache, checksums and offline use

Files go to `EPIONE_MODELS_DIR` if set, else `EPIONE_DATA_DIR`, else the OS
cache, one directory per family — the same places the per-family loaders
always used, so earlier downloads are reused. Each verified file gets a
small `.verified` stamp (checksum, size, mtime) so a 700 MB checkpoint is
not re-hashed on every load; any change to the file invalidates it.

With `EPIONE_OFFLINE=1` (or `HF_HUB_OFFLINE=1`, or `offline=True`) nothing
is downloaded: cached files are verified and used, and a missing one raises
an error that says where it is expected. Fetch on a machine with network
(`epi.models.fetch_pretrained(name)`) and copy the cache across.
""")

code(r"""
path = epi.models.fetch_pretrained('print_tn5_bias')
print('<cache>/' + str(path.relative_to(epi.models.pretrained_cache_dir())))
print(epi.models.registry.file_checksum(path, 'md5'), '==', epi.models.get_pretrained('print_tn5_bias').files[path.name])

epi.models.load_pretrained('print_tn5_bias', offline=True)          # cached: fine offline
try:
    epi.models.fetch_pretrained('borzoi_replicate_3', offline=True)  # not cached
except FileNotFoundError as err:
    print('offline:', str(err).split(' at ')[0], '...')
""")

md(r"""
## Registering your own model

Any trained model can be registered and then loaded by name — by you, by a
collaborator from a shared directory, or by a downstream package that
publishes weights. Here the seq2PRINT model above is saved with its
family's `save_pretrained`, which returns the checksums, and registered as
a local entry. A new model family registers its entries the same way,
passing `loader='package.module:function'`; the loader is called as
`loader(entry, **kwargs)` and fetches files with `entry.fetch(filename)`.
""")

code(r"""
import tempfile

out = pathlib.Path(tempfile.mkdtemp()) / 'my_checkpoint'
files = epi.models.seq2print.save_pretrained(model, out, manifest={
    'architecture': manifest['architecture'], 'note': 'copy of seq2print_p2s1_fold0_bulk'})
epi.models.register({
    'name': 'my_seq2print', 'family': 'seq2print', 'task': 'sequence -> multi-scale footprints',
    'organism': 'mouse', 'genome': 'mm10', 'license': 'GPL-3.0',
    'source': {'type': 'local', 'path': str(out)}, 'files': files,
})
mine = epi.models.load_pretrained('my_seq2print')
with torch.no_grad():
    print('identical predictions:', torch.equal(mine(x)[0], footprint))
epi.models.list_pretrained(name='my_seq2print')[['family', 'source', 'files', 'cached']]
""")

md(r"""
## Summary

| Family | Entries | Source | Loader returns |
|---|---|---|---|
| `chrombpnet` | 5 ENCODE cell lines × 5 folds × {full, nobias, bias} | ENCODE accession + md5 | `ChromBPNet` / `BPNet` (no TensorFlow) |
| `seq2print` | 2 Epione checkpoints (mouse P2 brain) | Hugging Face commit + md5 | `Seq2PRINT` (+ manifest) |
| `print` | dispersion, Tn5 bias, TF binding | Zenodo record + md5 | `DispersionModel`, TorchScript |
| `hint` | 8-mer bias tables, 2 HMMs | RGT Git commit + md5 / shipped | tables or `HintBias`, `HintHMM` |
| `borzoi` | 4 human replicates | Hugging Face commit + sha256 | `Borzoi` |
| `scooby` | NeurIPS BMMC, OneK1K, epicardioids | Hugging Face commit + sha256 | upstream `Scooby` |
| `translate`, `higashi` | your own checkpoints | local path (+ checksum) | translation model, Higashi |

Torch pickles (`translate`, `higashi`) execute code when loaded: register
only files you trust — the checksum then guarantees it is that file.
""")

nb = nbf.v4.new_notebook(cells=CELLS)
nb.metadata.update({
    "kernelspec": {"display_name": "epi-scooby", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
})
out = pathlib.Path(__file__).with_name("t_pretrained_models.ipynb")
nbf.write(nb, str(out))
print(f"wrote {out} ({len(CELLS)} cells)")
