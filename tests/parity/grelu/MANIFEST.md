# gReLU bridge, pre-registered validation

Written before the bridge code was run against any of the checks below.

## Scope and discovery

- **gReLU** (Genentech, MIT), version 1.1.0, is an optional dependency (`epione[grelu]`)
  and is imported lazily. Nothing is vendored.
- **What gReLU provides:**
  - a Lightning wrapper (`LightningModel`) with a uniform prediction API;
  - variant-effect prediction (`grelu.variant`);
  - ISM and attributions (`grelu.interpret.score`, built on captum and tangermeme);
  - motif and marginalisation tools;
  - a model zoo (Borzoi, Enformer trunks).
- gReLU has no BPNet/ChromBPNet model.
- **What the bridge adds:**
  1. `wrap(model, head=...)`: any Epione sequence model, i.e. a ChromBPNet BPNet,
     ChromBPNet or Seq2PRINT, or any `nn.Module` taking `(N, 4, L)` one-hot input, becomes
     a gReLU `LightningModel`. All of gReLU's tooling then runs on it.
  2. `predict_variant_effects`, `ism` and `attributions`: one API across Epione models and
     native gReLU models, using local FASTA files and no genomepy download.
  3. `pseudobulk_anndata`: a cells × peaks AnnData aggregated into a groups × peaks AnnData
     in the layout gReLU's `AnnDataSeqDataset` trains on.
- scooby is not wrapped. Its forward pass needs a per-cell embedding and a 524-kb input,
  so it does not fit gReLU's single-input interface.

## Checks

**V1, adapter exactness (bit-exact, CPU).** For a ChromBPNet `BPNet`, a full
`ChromBPNet` and a `Seq2PRINT`, the wrapped model's output must equal the Epione model's
own output for every head: `torch.equal` on 8 random one-hot inputs.

**V2, bridge = direct gReLU (bit-exact).** On the same wrapped model and inputs:

- `epione.models.grelu.predict_variant_effects` must equal
  `grelu.variant.predict_variant_effects`;
- `ism` must equal `grelu.interpret.score.ISM_predict`;
- `attributions(method="saliency")` must equal `get_attributions(method="saliency")`;
- `attributions(method="deepshap", seed=0)` must equal `get_attributions(method="deepshap",
  seed=0)`.

All with `np.array_equal`.

**V3, biological sanity (sign test).**

- **Model:** ENCODE GM12878 ChromBPNet (ENCFF772UBB), fold 0, the bias-corrected `nobias`
  model, log-count head, loaded with `epione.models.chrombpnet.load_pretrained`.
- **Regions:** MACS3 peaks of 10x PBMC 3k on chr1, chr3 and chr6, the test chromosomes of
  fold 0. PBMC lymphocytes and GM12878 share CTCF occupancy.
- **Motif variants:**
  - Find matches of the CTCF core regex `CCA[CG][CT]AG[AG][GT]GG[CA]` within ±200 bp of
    peak summits, on either strand. Take the first 300 by genomic order.
  - Each variant mutates the 10th core base, the conserved G in `AG[AG]GG`
    (C on the reverse strand), to its complement.
- **Controls:** one random SNV per matched peak, at least 50 bp from any regex match, with
  a random alternative base (seed 0).
- **Effect:** alt − ref predicted log-counts, computed with `predict_variant_effects`
  (`compare_func="subtract"`).
- **Pass if all hold:**
  - motif variants have median effect < 0 and at least 70% negative;
  - controls have |median| < 0.01 and between 40% and 60% negative.

**V4, dataset builder (exact).** The X of `pseudobulk_anndata` equals a direct pandas
groupby-sum of the same cells × peaks matrix. `var` carries `chrom`, `start` and `end`.

## Results

(Appended after the runs.)
