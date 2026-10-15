"""Built-in entries of the pretrained-model registry.

Plain data, read by :mod:`epione.models.registry` on first use. Every
remote file is pinned twice: by the publisher's own version (Hugging Face
commit, ENCODE accession, Zenodo record, Git commit) and by a checksum.
Hugging Face weights use the ``sha256`` the Hub records for LFS files, so
no file had to be downloaded to pin it; the rest use the publisher's md5.

To add a model, append an entry here (built-in) or call
:func:`epione.models.register` from the model's own package.
"""
from __future__ import annotations

from typing import Dict, List

#: family -> default loader (``module:function``), called as ``loader(entry, **kw)``.
FAMILY_LOADERS: Dict[str, str] = {
    "seq2print": "epione.models.seq2print._pretrained:_registry_load",
    "chrombpnet": "epione.models.chrombpnet._pretrained:_registry_load",
    "print": "epione.models._loaders:load_print",
    "hint": "epione.models._loaders:load_hint",
    "scooby": "epione.models._loaders:load_scooby_checkpoint",
    "borzoi": "epione.models._loaders:load_borzoi",
    # Families without published weights: register a model you trained
    # (``source={'type': 'local', 'path': ...}``) and load it by name.
    "translate": "epione.models._loaders:load_translate",
    "higashi": "epione.models._loaders:load_higashi",
}

_SEQ2PRINT_REV = "e8a36d4ca434e15d1de8002e796a61b8352529b3"
_SEQ2PRINT = dict(
    family="seq2print", task="sequence -> multi-scale footprints",
    organism="mouse", genome="mm10", license="GPL-3.0",
    citation=("Epione (this package); architecture from Hu et al., scPrinter / seq2PRINT, "
              "bioRxiv 2025. Data: GEO GSE308623."),
    homepage="https://huggingface.co/starlitnightly/epione-seq2print",
)

_CHROMBPNET = dict(
    family="chrombpnet", task="sequence -> base-resolution ATAC profile + counts",
    organism="human", genome="GRCh38",
    license="ENCODE data-use policy (free reuse with citation); chrombpnet code MIT",
    citation=("Pampari et al. ChromBPNet: bias factorized, base-resolution deep learning "
              "models of chromatin accessibility. bioRxiv (2024). ENCODE Project Consortium."),
)
_CHROMBPNET_RELEASES = [
    # key, biosample, models accession, md5, experiment, annotation, MB
    ("encode_gm12878", "GM12878", "ENCFF772UBB", "47a4549c40d25bb2440f4fc9b07eec21",
     "ENCSR637XSC", "ENCSR081YFO", 504.1),
    ("encode_k562", "K562", "ENCFF008TMU", "a1b9870fcb52bba03ea5ef71c8acf897",
     "ENCSR868FGK", "ENCSR893SUD", 508.1),
    ("encode_hepg2", "HepG2", "ENCFF839MZM", "952774ac9f31b5765b224fc85540dabb",
     "ENCSR291GJU", "ENCSR476FKK", 506.2),
    ("encode_imr90", "IMR-90", "ENCFF419JCT", "eaf764676ca1000ae5d296a6ce4764d1",
     "ENCSR200OML", "ENCSR585BFP", 595.5),
    ("encode_tcell", "T-cell", "ENCFF358NCZ", "6c1471fcc3f8606678a044686adb09bd",
     "ENCSR102RXG", "ENCSR752EUR", 505.9),
]

_PRINT_ZENODO = "https://zenodo.org/records/15224770/files/"
_PRINT = dict(
    family="print", organism="any", genome="any", license="CC-BY-4.0",
    citation=("Hu et al. Multiscale footprints reveal the organization of cis-regulatory "
              "elements. Nature (2025). Weights: Zenodo 10.5281/zenodo.15224770."),
    homepage="https://zenodo.org/records/15224770",
)

_RGT_COMMIT = "66f5fbb33f199424d09272c628287955e383f7bd"
_HINT = dict(
    family="hint", organism="human", genome="any", license="GPL-3.0 (RGT)",
    citation=("Li et al. Identification of transcription factor binding sites using "
              "ATAC-seq. Genome Biology 20, 45 (2019)."),
    homepage="https://github.com/CostaLab/reg-gen",
)

_BORZOI = dict(
    family="borzoi", task="sequence -> RNA-seq/ATAC/ChIP coverage (Borzoi backbone)",
    organism="human", genome="hg38", biosample="multi-tissue (ENCODE/GTEx/FANTOM)",
    license="CC-BY-4.0",
    citation=("Linder et al. Predicting RNA-seq coverage from DNA sequence as a unifying "
              "model of gene regulation. Nature Genetics (2025). PyTorch port: "
              "johahi/borzoi-pytorch."),
)
_BORZOI_REPLICATES = [
    ("521db103372769581b8d1d616e6a3b496da490c5",
     "d839593c3350bc3aea93b7a6d1fc3ce2b1629e0f4322ecb3c6b6743d14932f75",
     "36abbcd288215e1df7172b16aae099faf8a5bced15d6e4a3b37f3fc347cc9d35"),
    ("ad05f7a1c7d5b033149e4bd68620c4b54c557b2d",
     "bcfa7a55e9a0c9ecb6c4c7ced4014f65a2c98cf06146c717fb08229bb6b8a5e5",
     "6c96d3b9a5b3c025eab2da6538e791ec3e45eb49d5f6eebc2215622fe60a045d"),
    ("871a31b59e976b17a11d1f221b0907b0c593a57a",
     "a6785049ee34a0f075e3950313351d9d8b177f5be886997af0076c77accc861a",
     "6c96d3b9a5b3c025eab2da6538e791ec3e45eb49d5f6eebc2215622fe60a045d"),
    ("da3b900237f3914cf36859ff455cd63f5a7ecc9c",
     "9e6260bf948db2f3ec602c1cabca5ac8dc04814261cdbf34dcf2b7de64cd3acf",
     "6c96d3b9a5b3c025eab2da6538e791ec3e45eb49d5f6eebc2215622fe60a045d"),
]

_SCOOBY = dict(
    family="scooby", task="sequence + cell state -> single-cell RNA/ATAC profiles",
    organism="human", genome="hg38",
    license="no license on the model card; scooby code MIT",
    citation=("Hingerl, Martens et al. scooby: modeling multimodal genomic profiles from "
              "DNA sequence at single-cell resolution. Nature Methods (2026)."),
)
# (name, repo, revision, weights sha256, config sha256, MB, cell_emb_dim, n_tracks,
#  biosample, description). cell_emb_dim / n_tracks are read off the
#  checkpoints' cell_state_to_conv shapes; tracks are upstream's layout.
_SCOOBY_MODELS = [
    ("scooby_neurips", "johahi/neurips-scooby", "1591d15696c65f14053e4495e52c74812a86fd8d",
     "8f2e2de3e86016378116c90accf94572e75963d5445beb6cabd585c8372170dd",
     "26862280d15319c004a062e1823699d9e1150e753c1afcc6c40ffe9676e68eba", 773.0, 14, 3,
     "bone marrow mononuclear cells (NeurIPS 2021 multiome)",
     "scooby on the NeurIPS 2021 BMMC multiome; tracks RNA+, RNA-, ATAC per cell."),
    ("scooby_onek1k", "lauradmartens/onek1k-scooby", "dd7a2b7be54c906b43de4bc0159b169caaf057c5",
     "2ae335c651182040462d965bb879118be40a6cb174550a3686fb10b04a74f7cb",
     "e4802fb5c2d5c74a4f3fc852436c3cb5a5f41ded5c80a4e2b1dcc43a0d3f47d9", 765.1, 10, 2,
     "PBMC (OneK1K scRNA-seq)",
     "scooby on OneK1K PBMC scRNA-seq; tracks RNA+, RNA- per cell (no ATAC)."),
    ("scooby_epicardioids", "lauradmartens/epicardioids-scooby",
     "cefc3a626f5080baaf83878754ea924df6bd7e8d",
     "f945e1937a46ddd5cb31249ef71e721e7a7b1381aa207fb2de449f23ac75a9e8",
     "f74372c396e1dd3d7d139eae6fed6fbfc6742130061e7e0cdaa11d93a00e700e", 773.0, 50, 3,
     "iPSC-derived epicardioids (multiome)",
     "scooby on epicardioid multiome; tracks RNA+, RNA-, ATAC per cell."),
]


def _entries() -> List[dict]:
    out: List[dict] = [
        dict(_SEQ2PRINT, name="seq2print_p2s1_fold0_bulk", aliases=["p2s1_fold0_bulk"],
             biosample="mouse brain section P2S1 (spatial ATAC), all spots pooled",
             description=("seq2PRINT bulk backbone, 10,000 stand-in peaks, scPrinter mm10 "
                          "fold 0 (t_seq2print tutorial)."),
             source={"type": "huggingface", "repo": "starlitnightly/epione-seq2print",
                     "revision": _SEQ2PRINT_REV, "subfolder": "p2s1_fold0_bulk",
                     "layout": "flat"},
             files={"model.safetensors": "md5:4b278474646f7e66df0d0f37bb9a0059",
                    "manifest.json": "md5:b05f15a3e1dd89e1ac0753e86c2d4b9a"},
             size_mb=47.2),
        dict(_SEQ2PRINT, name="seq2print_p2s1_fold0_lora_bands",
             aliases=["p2s1_fold0_lora_bands"],
             biosample="mouse brain section P2S1 (spatial ATAC), four spatial bands",
             description="The bulk backbone + LoRA (rank 8, hidden 64) for four spatial bands.",
             source={"type": "huggingface", "repo": "starlitnightly/epione-seq2print",
                     "revision": _SEQ2PRINT_REV, "subfolder": "p2s1_fold0_lora_bands",
                     "layout": "flat"},
             files={"model.safetensors": "md5:cb4f77aef16b220e2188746982851a70",
                    "manifest.json": "md5:d054d44fddb97ad94e6fa61de1fa94de"},
             size_mb=152.2),
    ]
    for key, bios, acc, md5, exp, ann, mb in _CHROMBPNET_RELEASES:
        out.append(dict(
            _CHROMBPNET, name=f"chrombpnet_{key}", aliases=[key], biosample=bios,
            description=(f"Official ENCODE ChromBPNet release for {bios} ATAC-seq: 5 folds x "
                         "{full, bias-corrected, scaled bias}; pick with kind=, fold=."),
            source={"type": "encode", "accession": acc},
            files={f"{acc}.tar.gz": f"md5:{md5}"}, size_mb=mb,
            homepage=f"https://www.encodeproject.org/annotations/{ann}/",
            extra={"key": key, "experiment": exp, "annotation": ann}))
    for key, fname, md5, mb, task, desc in [
        ("dispersion", "dispersion_model_py_v2.h5", "341a011610c7db9a1418917e359d3169", 0.74,
         "footprint null (dispersion MLP)",
         "PRINT's dispersion model (v2): the learned null of footprint ratios."),
        ("tn5_bias", "Tn5_NN_model_py_v2.pt", "5ea6c26fcd54ad2aff3d9705ed54dd3c", 0.10,
         "Tn5 sequence bias",
         "PRINT's Tn5 bias CNN (v2, TorchScript); epi.tl.tn5_bias(method='print')."),
        ("tf_binding", "TFBS_model_py.pt", "5bd79a9c4f3374241a6f4341eb39fe2c", 0.65,
         "TF binding score",
         "PRINT's TF binding-score network (TorchScript); epi.tl.tf_binding_score."),
    ]:
        out.append(dict(
            _PRINT, name=f"print_{key}", aliases=[], task=task, description=desc,
            biosample="cell-type agnostic",
            source={"type": "url", "base_url": _PRINT_ZENODO, "label": "zenodo:15224770",
                    "legacy_flat_env": True},
            files={fname: f"md5:{md5}"}, size_mb=mb, extra={"key": key}))
    out.append(dict(
        _HINT, name="hint_atac_bias_tables", task="Tn5 sequence bias (8-mer tables)",
        biosample="ATAC-seq (RGT estimate)",
        description=("HINT-ATAC's forward/reverse 8-mer Tn5 bias tables; "
                     "epi.tl.tn5_bias(method='hint')."),
        source={"type": "url",
                "base_url": f"https://raw.githubusercontent.com/CostaLab/reg-gen/{_RGT_COMMIT}"
                            "/data/fp_hmms/",
                "label": f"github:CostaLab/reg-gen@{_RGT_COMMIT[:7]}", "legacy_flat_env": True},
        files={"atac_bias_table_F.txt": "md5:75dcd7c1c21ae23ce45cb50d636e4ca9",
               "atac_bias_table_R.txt": "md5:0212670fc325736397ca85ac030c16ff"},
        size_mb=2.4))
    for model, md5, desc in [
        ("paired", "82ff52f95251b0c45e5dcdeee2028b8e", "9 states x 8 tracks, paired-end ATAC"),
        ("single", "b8387418ddf0e4da7086c3d03c0ab232", "7 states x 4 tracks, single-end ATAC"),
    ]:
        out.append(dict(
            _HINT, name=f"hint_atac_hmm_{model}", task="footprint HMM", genome="any",
            biosample="ATAC-seq", loader_kwargs={"model": model},
            description=f"HINT-ATAC's trained footprint HMM ({desc}); ships with Epione.",
            source={"type": "package", "path": "data/hint"},
            files={f"atac_{model}_hmm.npz": f"md5:{md5}"}, size_mb=0.01))
    for i, (rev, w, cfg) in enumerate(_BORZOI_REPLICATES):
        out.append(dict(
            _BORZOI, name=f"borzoi_replicate_{i}",
            description=(f"Borzoi replicate {i} (human head), PyTorch weights; the backbone "
                         "scooby fine-tunes (epi.models.scooby.load_scooby)."),
            source={"type": "huggingface", "repo": f"johahi/borzoi-replicate-{i}",
                    "revision": rev},
            files={"config.json": f"sha256:{cfg}", "model.safetensors": f"sha256:{w}"},
            size_mb=743.7, homepage=f"https://huggingface.co/johahi/borzoi-replicate-{i}"))
    for (name, repo, rev, w, cfg, mb, emb, tracks, bios, desc) in _SCOOBY_MODELS:
        out.append(dict(
            _SCOOBY, name=name, biosample=bios, description=desc,
            source={"type": "huggingface", "repo": repo, "revision": rev},
            files={"config.json": f"sha256:{cfg}", "model.safetensors": f"sha256:{w}"},
            size_mb=mb, homepage=f"https://huggingface.co/{repo}",
            loader_kwargs={"cell_emb_dim": emb, "n_tracks": tracks}))
    return out


BUILTIN_ENTRIES: List[dict] = _entries()
