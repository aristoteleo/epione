"""G2: compare 3 reference seeds vs 3 Epione seeds (rule in MANIFEST.md).

usage: python compare.py <results_dir>   (expects ref_s{0,1,2}.json, epi_s{0,1,2}.json)
"""
import json
import os
import sys

import numpy as np

FLOORS = {'ARI': 0.03, 'AMI': 0.03, 'kNN_acc': 0.02, 'AUROC_nbr0': 0.005, 'AUROC_nbr5': 0.005,
          'PB_r_nbr0': 0.01, 'PB_r_nbr5': 0.01}
INFO = ['repli_R2']
d = sys.argv[1]
ref = [json.load(open(os.path.join(d, f'ref_s{s}.json'))) for s in range(3)]
epi = [json.load(open(os.path.join(d, f'epi_s{s}.json'))) for s in range(3)]
rows, n_pass = [], 0
for m, floor in list(FLOORS.items()) + [(k, None) for k in INFO]:
    r = np.array([x[m] for x in ref]); e = np.array([x[m] for x in epi])
    sd = r.std(ddof=1)
    delta = abs(e.mean() - r.mean())
    tol = max(2 * sd, floor) if floor is not None else float('nan')
    ok = (delta <= tol) if floor is not None else None
    n_pass += bool(ok)
    rows.append(dict(metric=m, ref_mean=r.mean(), ref_sd=sd, ref=r.tolist(), epi_mean=e.mean(), epi_sd=e.std(ddof=1),
                     epi=e.tolist(), delta=delta, tol=tol, gated=floor is not None, pass_=ok))
    print(f"{m:12s} ref {r.mean():.4f}±{sd:.4f}  epi {e.mean():.4f}±{e.std(ddof=1):.4f}  |d| {delta:.4f}  tol {tol:.4f}  "
          f"{'' if ok is None else ('PASS' if ok else 'FAIL')}")
print(f'G2: {n_pass}/{len(FLOORS)} gated metrics pass ->', 'PASS' if n_pass == len(FLOORS) else 'FAIL')
json.dump({'rows': rows, 'n_pass': n_pass, 'n_gated': len(FLOORS), 'G2_pass': n_pass == len(FLOORS)},
          open(os.path.join(d, 'g2.json'), 'w'), indent=1, default=float)
