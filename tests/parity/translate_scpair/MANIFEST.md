# scPair parity (metric level), pre-registered

Written before any Epione scPair code or run existed.

**Reference:** scTranslation's vendored scPair, at sctranslation@8bf1d63. It is an older copy of quon-titative-biology/scPair (MIT), and upstream is at c585949.

**Port:** `epione/translate/_scpair.py` adapts the MIT-licensed scPair code and keeps its notice. The ZINB log-likelihood follows scvi-tools (BSD-3), as scPair's does.

## How scTranslation runs scPair (from its runner and config)

Each run trains **two independent supervised predictors**, RNA→ATAC and ATAC→RNA, each an encoder plus a likelihood decoder. It then trains an embedding-to-embedding mapping, which `predict()` doesn't use.

| setting | value |
|---|---|
| encoder | FCNN [input, 800, 30] with layers Linear → LayerNorm → LeakyReLU → BatchNorm → Dropout(0.1), then an extra Linear(30, 30) |
| library-size branch | in the encoder when the output is ATAC (`infer_library_size_atac=True`) |
| RNA decoder | ZINB: mean = softmax(ReLU(Linear)) × library size (the observed library in training, `sample_factor_rna=True`); dispersion = exp(pre-softmax output); zero-inflation logits from a G×G Linear |
| ATAC decoder | Bernoulli: sigmoid(Linear) × sigmoid(inferred library) |
| optimiser | AdamW, learning rate 1e-3; weight decay 1e-9 on non-output weight matrices |
| batches | 130, shuffled each epoch |
| stopping | at most 1,000 epochs; early stop after 25 epochs without improvement on the full validation loss; best checkpoint kept |
| initialisation | Xavier-normal weights, zero biases |
| inputs | RNA as **raw counts** of the 3,000 HVGs (no log transform); ATAC as the peaks open in ≥ 49 cells, **binarised** inside scPair (`check_distribution`) for both input and target |
| predictions | RNA→ATAC: probabilities. ATAC→RNA: the ZINB *scale*, i.e. per-cell **proportions** summing to 1, because `predict()` passes library size 1 |
| truth files | RNA→ATAC: the **raw peak counts** of the test cells, not binarised (maximum 75). ATAC→RNA: the raw HVG counts |

**Vendored vs upstream.** The semantic diff (`ast.unparse`) shows only memory and engineering changes in upstream: gradient checkpointing, optional AMP, lazy GPU loading. The vendored copy also renames the modality "Gene Expression" to "Gene_Expression", consistently throughout. **No deviation that changes the trained model or its predictions was found**, unlike scButterfly's unshuffled loader.

**Epione differences, all in execution.**
- The mapping phase is omitted. It doesn't affect predictions; it only affects the optional cross-modal embeddings, which Epione doesn't expose yet. This saves time, so any speed comparison with the reference is reported with that caveat.
- Sparse matrices are densified batch by batch.

## Setting

| | |
|---|---|
| data | 10x Multiome PBMC 10k |
| split | fold0: 6,933 train, 771 validation, 1,927 test |
| seeds | 0, 1, 2 on both sides |
| RNA input | `inputs/rna_counts_hvg.h5ad` (raw counts) |
| ATAC input | `prepare_atac(inputs/atac_counts.h5ad, binarize=False, use_tfidf=False, scale=False)`: the ≥ 49-cell peaks as counts; scPair binarises them internally |

## Gates

The structure, tolerances and floors are those of the scButterfly and BABEL MANIFESTs.

- **G0 · preprocessing.**
  - rna2atac: Epione's filtered peak counts equal the reference truth exactly.
  - atac2rna: Epione's HVG counts equal the reference truth exactly.
- **G1 · metric implementation.** Epione's metrics on the reference predictions reproduce the reference `metrics.json` under `convention="sctranslation"`, with the same per-metric tolerances.
- **G2 · training parity.**
  - For every (metric, direction), pass if `|mean_Epione − mean_reference| ≤ max(2 × SD_reference, floor)`, using 3 seeds a side.
  - AUROC is not gated for atac2rna, so there are 23 gated pairs; G2 passes if all 23 pass.
  - If G2 fails, it is reported. Any amendment must be registered here before its run, must come from config, log or code facts, and is limited to one.
- **G3 · fast mode.**
  - Setting: `fast=True` is bf16 autocast with batch 520 (4×); the learning rate is unchanged.
  - Metric gate: the same rule as G2, against Epione's default.
  - Speed gate: the median default/fast ratio of training time is ≥ 1.5.
  - If it fails, fast mode is documented as **not equivalent** to the default, with no amendments.

## Results

Filled in after the runs.
