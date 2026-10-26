# Footprints and sequence models

Per-site TF binding (the default `epi.tl.tf_binding`, a thermodynamic motif-accessibility score), TF footprinting, PRINT, ChromBPNet and sequence-to-profile deep learning models, variant effects and attributions with gReLU, and the registry of pretrained weights shared by all of them.

**Which tutorial for which question**

| Question | Tutorial | Method |
|---|---|---|
| Which motif sites does a factor bind, in which cell type? | [t_tf_binding](t_tf_binding.ipynb) | `epi.tl.tf_binding`: occupancy score per site, the default |
| Is a factor bound in a cell type, on average over its sites? | [t_footprint](t_footprint.ipynb) | `epi.tl.get_footprints`: aggregate footprint, as in ArchR |
| What protects one locus, a factor or a nucleosome? | [t_print_footprint](t_print_footprint.ipynb) | PRINT multi-scale footprints and binding probability |
| What would a cell type's footprints be, from sequence alone? | [t_seq2print](t_seq2print.ipynb) | seq2PRINT: a sequence model per cell type, TF binding scores |
| Which bases and motifs does a sequence model use? | [t_sequence_attribution](t_sequence_attribution.ipynb) | attributions and TF-MoDISco through gReLU |
| A base-resolution accessibility model for a sample? | [t_chrombpnet](t_chrombpnet.ipynb) | ChromBPNet, with Tn5 bias factored out |
| What does a variant do to accessibility? | [t_grelu_variants](t_grelu_variants.ipynb) | variant effects with ChromBPNet or seq2PRINT |

```{toctree}
:maxdepth: 1
:titlesonly: true

t_tf_binding
t_footprint
t_print_footprint
t_seq2print
t_chrombpnet
t_grelu_variants
t_sequence_attribution
t_scooby_bridge
t_pretrained_models
```
