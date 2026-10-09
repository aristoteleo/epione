# Parity manifest — peak annotation enrichment (ArchR 1.0.3)

Pre-registered before the first comparison; thresholds are read-only.

Fixture: ArchR Hematopoiesis tutorial project (10,660 cells, 141,369 peaks,
11 clusters, JASPAR2020 CORE vertebrate motif matches), hg19.
Reference: `r_reference.R` (R 4.4.3, ArchR 1.0.3). Comparison: `compare.py`.

The marker test (`getMarkerFeatures`, random bias-matched background) is an
*input* here: ArchR's marker assays are exported and fed to Epione so the
enrichment layer itself is tested deterministically.

| Output | ArchR function | Class | Metric | Threshold |
|---|---|---|---|---|
| peak × annotation matrix (custom BED sets) | `addPeakAnnotations` | 1 deterministic | exact equality of sparse pattern + names | 0 mismatches |
| peak × annotation matrix (ArchR db: EncodeTFBS, ATAC) | `addArchRAnnotations` | 1 deterministic | exact equality of sparse pattern + names | 0 mismatches |
| all 9 numeric enrichment assays (motif, bg = all) | `peakAnnoEnrichment` | 1 deterministic (standard) | max abs err | ≤ 1e-8 |
| all 9 numeric enrichment assays (motif, bg = bgdPeaks) | `peakAnnoEnrichment(background="bgdPeaks")` | 1 | max abs err | ≤ 1e-8 |
| all 9 numeric enrichment assays (OR cutoff, region, EncodeTFBS, ATAC) | `peakAnnoEnrichment` | 1 | max abs err | ≤ 1e-8 |
| unrounded −log10 p (diagnostic) | `phyper(..., log.p=TRUE)` | 1 (bounded) | max rel err | ≤ 1e-10 |
| table + row order | `customEnrichment` (all / bgd) | 1 | max abs err, identical row order | ≤ 1e-8 |
| heatmap matrix values, row names, row/col order (6 configs + bgd) | `plotEnrichHeatmap(returnMatrix=TRUE)` | 1 | max abs err + identical dimnames | ≤ 1e-8 |
| hclust leaf order (30 random inputs) | `hclust(dist(x))$order` | 1 | identical permutation | 30/30 |
| end-to-end with `epi.tl.find_marker_features` markers (motif mlog10Padj) | full pipeline | 8 statistical inference | Spearman on mlog10Padj over (motif, cluster) | ≥ 0.90 |
| end-to-end top-10 motifs per cluster | full pipeline | 5 ranked | mean top-10 Jaccard | ≥ 0.5 |

Notes:
* `mlog10p` is rounded to 4 decimals by ArchR (`round(4)`); Epione reproduces
  the rounding, so class-1 comparison is on the rounded values.
* The end-to-end rows use a different (Epione) marker test, so only
  class 5/8 metrics apply there.

## Results (2026-10-09; thresholds above unchanged)

Run: `Rscript r_reference.R <fixture> <ref> ArchR-Hg19-v1.Anno` then
`python compare.py <fixture> <ref> ArchR-Hg19-v1.Anno --e2e`.

| Output | Value | Gate |
|---|---|---|
| addPeakAnnotations matrix (4 sets, 1 dropped for 0 overlaps) | 0 mismatches, names identical | PASS |
| addArchRAnnotations EncodeTFBS (141,369 × 689) / ATAC (× 96) | 0 mismatches, names identical | PASS |
| peakAnnoEnrichment motif, bg = all / bgdPeaks / OR-cutoff | max abs err 6.8e-13 / 3.4e-13 / 3.4e-13 | PASS |
| peakAnnoEnrichment Region / EncodeTFBS / ATAC | 2.8e-13 / 9.1e-13 / 9.1e-13 | PASS |
| unrounded −log10 p | max rel err 1.5e-13 | PASS |
| customEnrichment all / bgd | 3.4e-13, identical row order | PASS |
| plotEnrichHeatmap(returnMatrix) × 7 configs | ≤ 6.4e-14, identical dimnames | PASS |
| hclust leaf order | 30/30 | PASS |
| end-to-end Spearman(mlog10Padj), Epione markers vs ArchR markers | 0.854 | **FAIL** (≥ 0.90) |
| end-to-end mean top-10 Jaccard | 0.620 | PASS |

The end-to-end failure is upstream of this port: `epi.tl.find_marker_features`
(raw counts, ≤ 2000 cells/group, own background matching, Log2FC pseudocount
1e-6) is not a 1:1 port of ArchR `getMarkerFeatures` (`scaleTo = 1e4` depth
normalisation, `maxCells = 500`, Log2FC = log2((m1 + 1)/(m2 + 1)) on normalised
means), so `FDR <= 0.1 & Log2FC >= 0.5` selects 1.4–4.3× more peaks per cluster
in Epione. Given identical marker inputs, the enrichment layer
is exact (rows above).
