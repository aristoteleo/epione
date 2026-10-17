# Changelog

All notable changes to this project will be documented in this file.

The format roughly follows Keep a Changelog.


## Unreleased

### Added
- `epi.single.scan_peaks` (CellScan): a scATAC-native peak caller. Each window is scored by the number of
  distinct cells with a Tn5 insertion, against a depth-aware null and a Tn5-bias-corrected local background.
- `epi.single.call_peaks(adata, groupby, method=...)`: one entry point that returns a non-overlapping,
  fixed-width peak set ready for `epi.pp.make_peak_matrix`.

### Changed (breaking)
- **Default peak caller is now CellScan** (`call_peaks(method='cellscan')`). The previous behaviour is
  `call_peaks(method='macs3')`, i.e. `epi.single.macs3` + `merge_peaks`, unchanged.
  - Held-out accuracy: on ENCODE HepG2 / MCF-7 / A549 scATAC with bulk ATAC + DNase truth, F1 of the top 50k
    peaks was 0.884 / 0.850 / 0.876 vs 0.875 / 0.837 / 0.866 for MACS3 + merge. It was also above ArchR,
    SnapATAC2, Signac, HMMRATAC and Genrich.
  - Speed: 10–30x faster (100k cells: 6 min vs 2 h 40 min, 8 threads).
  - Known weaknesses: less precise CTCF summits than Genrich / SnapATAC2 / MACS3; lower recall of peaks
    specific to a rare cluster than ArchR; peak memory 3–4x higher.
