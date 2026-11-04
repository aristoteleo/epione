"""Export RGT's pickled ATAC HMMs to plain arrays (Epione ships these)."""
import pickle, sys, numpy as np, os
src, out = sys.argv[1], sys.argv[2]
for name, fp_state in (("atac_single", 6), ("atac_paired", 8)):
    h = pickle.load(open(os.path.join(src, name + ".pkl"), "rb"), encoding="latin1")
    assert h.covariance_type == "full"
    np.savez(os.path.join(out, name + "_hmm.npz"),
             startprob=np.asarray(h.startprob_, dtype=np.float64),
             transmat=np.asarray(h.transmat_, dtype=np.float64),
             means=np.asarray(h.means_, dtype=np.float64),
             covars=np.asarray(h._covars_, dtype=np.float64),
             fp_state=np.int64(fp_state))
    print(name, h.n_components, h.n_features)
