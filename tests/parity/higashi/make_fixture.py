"""Build the shared Higashi parity fixture from the Nagano 2017 1 Mb scool.

usage: python make_fixture.py <out_dir> <nagano_data_dir>

Cell selection replicates t_sc_hic_basics (GEO features table, passed_qc == 1,
scool entry present, total_contacts > 5000) restricted to the three major
phases G1 / early-S / late-S/G2, but keeps *all* such cells (not the 35/phase
tutorial subsample) so the 3-seed statistics are less noisy.

Outputs (DATA dir):
  data.txt            higashi_v1 table of the MASKED contacts (held-out removed)
  label_info.pickle   {'cell type': phase, 'repli_score': ..., 'cell_name': ...}
  chrom_sizes.txt     chrom sizes for chr1..chr19, chrX (mm9, from the scool)
  cells.tsv           cell order (cell_id -> scool name, phase, repli_score)
  full_pixels.npz     ALL intra-chrom pixels (cell, chrom_idx, b1<=b2, count) incl. held-out
  heldout.npz         evaluation pairs: cell, chrom_idx, b1, b2, label (1 = held-out contact,
                      0 = distance-matched pixel with zero count in the full data)
"""
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import cooler

SRC = Path(sys.argv[2])  # folder with nagano_1MB_raw.scool and the two GSE94489 tables
SCOOL = SRC / 'nagano_1MB_raw.scool'
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
CHROMS = [f'chr{i}' for i in range(1, 20)] + ['chrX']
KEEP = ['G1', 'early-S', 'late-S/G2']
HOLDOUT_FRAC = 0.10
SEED = 0

readme = pd.read_csv(SRC / 'GSE94489_README.txt', sep='\t', dtype=str)
readme['cell_nm_us'] = readme['Name'].str.replace('-', '_', regex=False)
features = pd.read_csv(SRC / 'GSE94489_2i_diploids_features_table.txt.gz', sep='\t')
features = features[features['passed_qc'] == 1].copy()
joined = features.merge(readme[['Nature_Reference_ID', 'Five_prime_barcode',
                                'Three_prime_barcode', 'cell_nm_us']],
                        left_on='cell_nm', right_on='cell_nm_us', how='inner')
joined['scool_name'] = (joined['Nature_Reference_ID'] + '_' + joined['Five_prime_barcode']
                        + '_' + joined['Three_prime_barcode'] + '_R1fastqgz')
with h5py.File(SCOOL, 'r') as h:
    avail = set(h.keys())
joined = joined[joined['scool_name'].isin(avail)]
joined = joined[joined['total_contacts'] > 5_000]
joined = joined[joined['group'].isin(KEEP)]
joined = joined.drop_duplicates('scool_name').sort_values('scool_name').reset_index(drop=True)
print('cells:', len(joined)); print(joined['group'].value_counts().to_string())

c0 = cooler.Cooler(f'{SCOOL}::/{joined.scool_name[0]}')
res = int(c0.binsize)
sizes = c0.chromsizes[CHROMS]
(OUT / 'chrom_sizes.txt').write_text(''.join(f'{c}\t{int(s)}\n' for c, s in sizes.items()))
chrom_offsets = {c: int(c0.extent(c)[0]) for c in CHROMS}
nbins = {c: int(np.ceil(sizes[c] / res)) for c in CHROMS}

rng = np.random.default_rng(SEED)
full_rows, mask_rows, eval_rows = [], [], []
for cid, name in enumerate(joined['scool_name']):
    clr = cooler.Cooler(f'{SCOOL}::/{name}')
    for ci, ch in enumerate(CHROMS):
        px = clr.matrix(balance=False, as_pixels=True).fetch(ch)
        if len(px) == 0:
            continue
        b1 = px['bin1_id'].to_numpy() - chrom_offsets[ch]
        b2 = px['bin2_id'].to_numpy() - chrom_offsets[ch]
        cnt = px['count'].to_numpy().astype(np.float64)
        keep = b2 >= b1  # upper triangle (fetch returns both halves for intra)
        b1, b2, cnt = b1[keep], b2[keep], cnt[keep]
        full_rows.append(np.stack([np.full(len(b1), cid), np.full(len(b1), ci), b1, b2, cnt], 1))
        off = np.where(b2 - b1 >= 1)[0]
        n_hold = int(round(HOLDOUT_FRAC * len(off)))
        hold = rng.choice(off, size=n_hold, replace=False) if n_hold > 0 else np.array([], int)
        is_hold = np.zeros(len(b1), bool); is_hold[hold] = True
        mask_rows.append(np.stack([np.full((~is_hold).sum(), cid), np.full((~is_hold).sum(), ci),
                                   b1[~is_hold], b2[~is_hold], cnt[~is_hold]], 1))
        # distance-matched negatives: zero-count pixels (in the FULL data) at the same distance
        occupied = set(zip(b1.tolist(), b2.tolist()))
        n = nbins[ch]
        for h in hold:
            d = int(b2[h] - b1[h])
            if n - d <= 0:
                continue
            for _ in range(50):
                s = int(rng.integers(0, n - d))
                if (s, s + d) not in occupied:
                    eval_rows.append((cid, ci, int(b1[h]), int(b2[h]), 1))
                    eval_rows.append((cid, ci, s, s + d, 0))
                    break
    if cid % 200 == 0:
        print('cell', cid, flush=True)

full = np.concatenate(full_rows); masked = np.concatenate(mask_rows)
ev = np.array(eval_rows, dtype=np.int64)
np.savez_compressed(OUT / 'full_pixels.npz', cell=full[:, 0].astype(np.int32), chrom=full[:, 1].astype(np.int8),
                    b1=full[:, 2].astype(np.int32), b2=full[:, 3].astype(np.int32), count=full[:, 4].astype(np.float32),
                    chroms=np.array(CHROMS), resolution=res)
np.savez_compressed(OUT / 'heldout.npz', cell=ev[:, 0], chrom=ev[:, 1], b1=ev[:, 2], b2=ev[:, 3], label=ev[:, 4],
                    chroms=np.array(CHROMS))
tab = pd.DataFrame({
    'cell_id': masked[:, 0].astype(np.int64),
    'chrom1': np.array(CHROMS)[masked[:, 1].astype(int)],
    'pos1': masked[:, 2].astype(np.int64) * res,
    'chrom2': np.array(CHROMS)[masked[:, 1].astype(int)],
    'pos2': masked[:, 3].astype(np.int64) * res,
    'count': masked[:, 4].astype(np.int64),
})
tab.to_csv(OUT / 'data.txt', sep='\t', index=False)
label_info = {'cell type': np.asarray(joined['group']), 'repli_score': np.asarray(joined['repli_score'], float),
              'cell_name': np.asarray(joined['scool_name'])}
pickle.dump(label_info, open(OUT / 'label_info.pickle', 'wb'))
joined[['scool_name', 'cell_nm', 'group', 'repli_score', 'total_contacts']].rename_axis('cell_id').to_csv(OUT / 'cells.tsv', sep='\t')
print('full pixels', len(full), 'masked', len(masked), 'eval pairs', len(ev), 'positives', int(ev[:, 4].sum()))
