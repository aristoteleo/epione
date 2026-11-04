"""PRAGA parity: one arm, one seed (GPU). official = upstream main.py code path (Spatial-epigenome-transcriptome)."""
import argparse, os, sys, time, types
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--arm", required=True); ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--out", required=True); a = ap.parse_args()
import scanpy as sc, torch
D = os.path.join(os.environ["SPATIAL_PARITY_DATA"], "praga_zenodo14591305/Mouse_Brain") + "/"
rna = sc.read_h5ad(D + "adata_RNA.h5ad"); atac = sc.read_h5ad(D + "adata_peaks_normalized.h5ad")
rna.var_names_make_unique(); atac.var_names_make_unique()
seed = 2024 + a.seed
t0 = time.perf_counter()
try:
  if a.arm == "official":
      SRC = os.path.join(os.environ["SPATIAL_PARITY_REF"], "praga/src")
      sys.path.insert(0, SRC)
      from PRAGA.preprocess import fix_seed, pca, construct_neighbor_graph, lsi
      from PRAGA.Train_model import Train
      args = types.SimpleNamespace(data_type="Spatial-epigenome-transcriptome", n_clusters=14, init_k=14, KNN_k=20,
                                   alpha=0.9, cl_weight=1.0, RNA_weight=1.0, ADT_weight=10.0, tau=2.0)
      fix_seed(seed)
      a1, a2 = rna, atac
      sc.pp.filter_genes(a1, min_cells=10); sc.pp.filter_cells(a1, min_genes=200)
      sc.pp.highly_variable_genes(a1, flavor="seurat_v3", n_top_genes=3000)
      sc.pp.normalize_total(a1, target_sum=1e4); sc.pp.log1p(a1); sc.pp.scale(a1)
      a1.obsm["feat"] = pca(a1[:, a1.var["highly_variable"]], n_comps=50)
      a2 = a2[a1.obs_names].copy()
      if "X_lsi" not in a2.obsm.keys():
          sc.pp.highly_variable_genes(a2, flavor="seurat_v3", n_top_genes=3000)
          lsi(a2, use_highly_variable=False, n_components=51)
      a2.obsm["feat"] = a2.obsm["X_lsi"].copy()
      data = construct_neighbor_graph(a1, a2, datatype=args.data_type, Arg=args)
      model = Train(data, datatype=args.data_type, device=torch.device("cuda:0"), Arg=args)
      model.clustering.device = torch.device("cuda:0")  # upstream hard-codes cuda:1 (single-GPU node)
      out = model.train()
      emb = out["PRAGA"]; obs = np.asarray(a1.obs_names)
  else:
      import epione as epi
      emb = epi.tl.praga(rna, atac, n_clusters=None, random_seed=seed, device="cuda:0")
      keep = ~np.isnan(rna.obsm["X_praga"][:, 0]); emb = rna.obsm["X_praga"][keep]; obs = np.asarray(rna.obs_names[keep])
except Exception as e:  # MANIFEST A1: divergence / failure is recorded, not hidden
    np.savez(a.out, failed=True, error=repr(e)[:500], seconds=time.perf_counter() - t0); print("FAILED", repr(e)[:300]); raise SystemExit(0)
el = time.perf_counter() - t0
truth = np.loadtxt(D + "MB_cluster.txt", dtype=str)
if not np.isfinite(emb).all():
    np.savez(a.out, failed=True, error="non-finite embedding", seconds=el); raise SystemExit(0)
np.savez(a.out, emb=emb, obs_names=obs, truth=truth, seconds=el)
print(f"{a.arm} seed {a.seed}: {el:.1f}s emb {emb.shape}", flush=True)
