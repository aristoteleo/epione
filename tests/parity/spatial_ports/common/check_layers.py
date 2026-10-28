"""Layer-level exactness: epione.tl._gnn vs torch_geometric (same seed, same input)."""
import os
import sys, numpy as np, torch
sys.path.insert(0, os.path.join(os.environ["SPATIAL_PARITY_REF"], "present/src"))
from torch_geometric.nn import GCNConv as PGCN
from PRESENT.Layers import GATConv as PGAT
from epione.tl._gnn import GCNConv, GATConv
from sklearn.neighbors import kneighbors_graph
rng = np.random.default_rng(0)
xy = rng.random((300, 2)); g = kneighbors_graph(xy, 6).tocoo()
ei = torch.from_numpy(np.vstack([g.row, g.col]).astype(np.int64))
x = torch.randn(300, 40)
res = {}
for name, A, B in [("gcn", PGCN, GCNConv), ("gat", PGAT, GATConv)]:
    torch.manual_seed(1); a = A(40, 16) if name == "gcn" else A(40, 16, dropout=0.1)
    torch.manual_seed(1); b = B(40, 16) if name == "gcn" else B(40, 16, dropout=0.1)
    pa = dict(a.named_parameters()); pb = dict(b.named_parameters())
    wdiff = max((pa[k] - pb[k]).abs().max().item() for k in pb)
    for mode in ("eval", "train"):
        getattr(a, mode)(); getattr(b, mode)()
        torch.manual_seed(5); ya = a(x, ei)
        torch.manual_seed(5); yb = b(x, ei)
        (ya.square().sum()).backward(); (yb.square().sum()).backward()
        gd = max((pa[k].grad - pb[k].grad).abs().max().item() for k in pb)
        res[f"{name}_{mode}"] = (wdiff, (ya - yb).abs().max().item(), gd)
        for p in list(pa.values()) + list(pb.values()): p.grad = None
for k, v in res.items():
    print(f"{k}: max|dW|={v[0]:.2e} max|dy|={v[1]:.2e} max|dgrad|={v[2]:.2e}")
