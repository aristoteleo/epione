# PRINT tutorial checks (archived)

`tutorial_checks.ipynb` holds the cells that were moved out of
`epione_guide/tutorials/single/t_print_footprint.ipynb` when the tutorial
was rewritten around PBMCs. They are kept with the outputs from their last
run. They come from the earlier version of the tutorial, which used a
spatial ATAC section of P2 mouse brain (GSE308623 `P2S1`, mm10, chr19) split
into four bands. The setup, peak, footprint and binding-score cells of that
version are included so that the checks run on their own.

| check | what it shows |
|---|---|
| PRINT CNN vs hexamer table (Part 3) | The two Tn5 bias models correlate weakly at base resolution (r = 0.33 over 200 bp). They also use different cut conventions, `(0, 0)` for PRINT and `(0, -1)` for the hexamer table. The dispersion null and the binding model were fitted to the CNN, so Epione refuses to combine them with the hexamer table. |
| Binomial vs learned dispersion null (Part 5) | On the same cuts and bias, a binomial null calls 0.268 of the heatmap at p < 0.01, PRINT's dispersion null 0.167. Tn5 counts are overdispersed, so the binomial null overcalls footprints. |
| Exactness against scPrinter (Part 7) | scPrinter's own `multiscale_footprints` and `getRegionBindingScore`, lifted from its source (`SCPRINTER_SRC`), give results identical (`np.array_equal`) to Epione's on real insertion and bias tracks: a (4, 50, 1201) multi-scale footprint and (4, 80) TF binding scores. |

The automated version of these checks is `tests/test_print.py`. It tests the
CNN bias, the dispersion model, every scale's p-values, the multi-scale
score and the TF binding score against scPrinter bit for bit, along with the
public API and the plots.

To re-run the notebook, set `SCOOBY_TUTORIAL_DATA` to the folder with
`scooby_bench/P2S1.fragments.tsv.gz`, `scooby_bench/P2S1.positions.csv` and
`genome/mm10/chr19.fa` (see `t_scooby_bridge`). Set `SCPRINTER_SRC` as well
for the exactness check.
