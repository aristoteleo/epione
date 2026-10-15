"""Reference env only. Official motif_score for six TFs. usage: official_motif.py PROCESSED RUN MOTIF_DIR"""
import sys, numpy as np
from scbasset.utils import make_model, motif_score
from scipy import sparse
m_test = sparse.load_npz(sys.argv[1]+"/m_test.npz"); n=m_test.shape[1]
model = make_model(32, n, show_summary=False); model.load_weights(sys.argv[2]+"/best_model.h5")
out = {t: motif_score(t, model, motif_fasta_folder=sys.argv[3]) for t in ["SPI1","CEBPA","PAX5","EBF1","TBX21","TCF7"]}
np.savez(sys.argv[2]+"/motif_scores.npz", **out); print("ok")
