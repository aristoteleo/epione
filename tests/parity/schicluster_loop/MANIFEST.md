# Differential loops (HCEA): parity (pre-registered)

Written before any comparison was run. Thresholds are fixed here; later changes are dated
amendments below the original text, written before the rerun.

## Reference

Human Cell Epigenome Atlas, `fig4/07.diffloop_majortype` (group `all`, the 35 major types
`c0`-`c34`): `compute_anova` per chromosome and matrix (Q, E, T) from each type's
`{ct}.{m}.cool` / `{ct}.{m}2.cool` and `group_n_cells`, blacklist
`hg38_bismark_loop_blacklist.bed` (±7 bins), `5 < j - i < 500` bins; the merged loops
(`{ct}.loop.bedpe` concatenated, column 6 dropped, de-duplicated on chrom/start1/start2, sorted);
`load_Q` (Q and T of each type at each merged loop); and the F-test FDR of `fig4/08` / `10`
(`f.sf(F, k - 1, N - k)`, Benjamini-Hochberg).

Reference outputs: `Hic_data/HCEA/analysis/diff_loop/all/` — `majortype_{Q,E,T}pv.cool`,
`merged_loop.bedpe`, `merged_loop.hdf` (`Qanova`, `Eanova`, `Tanova`, `Qfdr`, `Efdr`, `Tfdr`),
`loop_Q.hdf`, `loop_T.hdf`.

Inputs: `zhoujt/project/ENTEx/loop/majortype/{ct}/{ct}.{Q,Q2,E,E2,T,T2}.cool` and
`{ct}.loop.bedpe` (read only).

## Gates

| Gate | Epione | Reference | Class | Threshold |
|---|---|---|---|---|
| L1 | `loop_anova` (35 types, Q/E/T, chr1-22) | `majortype_{Q,E,T}pv.cool` | 1, deterministic (same float64 operations, stored float32) | identical pixel sets; values equal (NaN matching NaN), or max relative error ≤ 1e-6 if not bit-identical |
| L2 | `merge_loops` (35 `loop.bedpe`) | `merged_loop.bedpe` | exact | identical rows in identical order |
| L3 | `loop_values` of each type's Q and T cool | `loop_Q.hdf`, `loop_T.hdf` (by column name) | exact | identical arrays |
| L4 | `differential_loops`: `Qanova`/`Eanova`/`Tanova` at the merged loops | `merged_loop.hdf` | 1 | as L1 |
| L5 | `differential_loops`: `Qfdr`/`Efdr`/`Tfdr` (k = 35, N = sum of `group_n_cells`) | `merged_loop.hdf` | 1 | max relative error ≤ 1e-6 on finite values; same NaN positions |

L1 and L4 compare float32 values written by the same float64 arithmetic; a mismatch beyond
float32 rounding means the port differs.

## First run (2026-10-22)

L2 ✅ (3,154,101 rows identical), L3 ✅ (identical), L4 ✅ (max rel 4.3e-7). L1: same pixels; Q
max rel 5.3e-7 ✅, E 1.5e-5 ❌, T 4.4e-6 ❌. L5 ❌: max rel 1.0; the 0.9 % (Q) of loops that differ
all have a reference FDR between 5e-324 and 2.4e-274, where epione's is exactly 0. The degrees of
freedom are right: the reference FDRs imply k - 1 = 34 and N - k = 86,654 (N = 86,689 cells).

## Amendment L-1 (2026-10-22, after the first run, before the rerun)

1. **Group order (L1, L4).** HCEA adds the types in the row order of `L1color.tsv`
   (`c33, c3, c22, …`), the first run in `c0 … c34`. Summation order changes the float64 sums,
   and `SS_total / SS_within - 1` amplifies that where the two are nearly equal. The rerun uses
   HCEA's order. Thresholds unchanged.
2. **FDR underflow (L5).** Below about 1e-270 the F-distribution tail of SciPy 1.17 returns 0 where
   the reference (an older SciPy) returned subnormal numbers; both mean "zero" for any threshold.
   L5 is now evaluated on loops whose reference FDR is ≥ 1e-270 (threshold unchanged there), and
   the number of loops below is reported. This changes the gate after seeing the first result;
   it is recorded here as such.

## Rerun after Amendment L-1 (2026-10-22)

| Gate | Measured | Threshold | Pass |
|---|---|---|---|
| L1 Q | same pixels; max rel 5.3e-7 | ≤ 1e-6 | ✅ |
| L1 E | same pixels; max rel 1.5e-5; 53 % of pixels differ by ≥ 1 float32 ulp (max 232 ulp) | ≤ 1e-6 | ❌ |
| L1 T | same pixels; max rel 4.4e-6; 17 % of pixels differ (max 36 ulp) | ≤ 1e-6 | ❌ |
| L2 merged loops | 3,154,101 rows identical, same order | exact | ✅ |
| L3 Q, T at loops | identical | exact | ✅ |
| L4 F at loops | max rel 4.3e-7 (Q), 2.1e-7 (E), 3.3e-7 (T) | ≤ 1e-6 | ✅ |
| L5 FDR (reference ≥ 1e-270) | max rel 1.2e-4 (Q, T), 6.2e-5 (E); NaN positions identical | ≤ 1e-6 | ❌ |

Group order does not explain L1: the rerun in HCEA's order gives identical numbers (sums of
float32 values times integer cell counts are exact in float64). The E and T differences are at
large F (median 150-480 among the pixels above 1e-6), not near zero, so they are not cancellation;
the reference cools appear to come from a computation that differs from the `fig4/07` code in
floating-point detail. L5 follows from L4: one-ulp differences in F move FDRs of 1e-200 by about
1e-4 relative. Loop selection is unaffected in practice (L2, L3 exact; F at loops within 4.3e-7).
