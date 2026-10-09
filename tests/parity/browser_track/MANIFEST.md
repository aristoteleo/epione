# Parity manifest — `epi.pl.plot_browser_track` vs ArchR 1.0.3 `plotBrowserTrack`

Gates were pre-registered before the first comparison and never loosened.
Row 6 was **added after** the first comparison. It is stricter than the gate it
complements (exact equality instead of a correlation).

Fixture: ArchR Hematopoiesis tutorial project (3 samples, 10,660 cells,
`Clusters` C1–C11), fragments `HemeFragments/*.fragments.tsv.gz`.
Genes: CD34, GATA1, MS4A1, LYZ, CD3D (±50 kb around the TSS).
Parameters: `groupBy="Clusters"`, `tileSize=250`, `normMethod="ReadsInTSS"`,
`minCells=25`, `maxCells=500`, `addArchRThreads(8)` (→ `RNGkind("L'Ecuyer-CMRG")`),
`set.seed(1)`.

| # | Output | R source | Class | Metric | Threshold | Result (worst of 5 genes) |
|---|---|---|---|---|---|---|
| 1 | Gene → plotting region | `plotBrowserTrack` (resize + `extendGR`) | 1 | exact equality | 0 | identical ✅ |
| 2 | Group coverage `y` per (group, tile), ArchR's cells injected | `.groupRegionSumArrows` | 1 | max_abs_err, identical rows/order | 1e-8 | 4.9e-15 ✅ |
| 3 | Clipped plot data (`ylim = q(0.999)`) | `.bulkTracks` | 1 | max_abs_err | 1e-8 | 4.9e-15 ✅ |
| 4 | Group coverage, free-running numpy RNG | `.groupRegionSumArrows` | 2 | Pearson r per tile | ≥ 0.95 | **0.937 ❌** (see note) |
| 5 | Peaks overlapping region | `.featureTracks` | 1 | identical interval set | 0 | identical ✅ |
| 6 | Group coverage, free-running `rng="archr"`, `seed=1` (no injection) | `.groupRegionSumArrows` | 1 | same cells + max_abs_err | 1e-8 | same cells, 4.9e-15 ✅ |
| 7 | Gene rows (packing) + clipped coordinates | `.geneTracks` | 1 | exact equality of (symbol, start, end, strand, cluster) | 0 | identical ✅ |
| 8 | Loop arc geometry (x, y per arc) | `.loopTracks` / `getArchDF` | 1 | max_abs_err given identical loops | 1e-6 | 5.4e-7 ✅ (R `fwrite` text precision at ~1e8 bp) |
| 9 | Group palette | `paletteDiscrete` | 1 | identical hex strings | 0 | identical ✅ |

**Note on row 4.** `maxCells=500` subsampling uses R's RNG. Drawing a different
random 500 cells per cluster changes the per-tile signal by more than the gate
allows. ArchR is not reproducible to that level either. ArchR's own draws under
`set.seed(1)` and `set.seed(2)` correlate at r = 0.925–0.980 across the 5 genes.
Five numpy seeds against ArchR give min 0.916 / mean 0.930–0.982. The gate stays
failed as registered; the threshold sits above ArchR's own seed-to-seed
reproducibility. Exact equality is reached instead by porting R's RNG
(`RRandom`: `set.seed` scrambling, Mersenne-Twister / L'Ecuyer-CMRG, R ≥ 3.6
rejection sampling in `sample()`). That is row 6.

Reproduce:

```bash
Rscript r_reference.R <fixture>/ArchR_base <ref_dir> 8
python compare.py <ref_dir> <fixture> <HemeFragments> parity.json
```
