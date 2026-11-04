"""GCNConv with a torch-sparse normalised adjacency (CANDIES ATAC encoder): epione vs PyG."""
import os
import numpy as np, torch, scipy.sparse as sp, sys
from torch_geometric.nn import GCNConv as PGCN
from epione.tl._gnn import GCNConv
dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
if dev != "cpu":
    torch.use_deterministic_algorithms(True); import os; os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
rng = np.random.default_rng(0); n = 400
from sklearn.neighbors import kneighbors_graph
A = kneighbors_graph(rng.random((n, 2)), 6).toarray(); A = np.where(A + A.T > 1, 1, A + A.T)
A_ = sp.coo_matrix(A) + sp.eye(n); d = sp.diags(np.power(np.asarray(A_.sum(1)), -0.5).ravel())
An = (A_.dot(d).transpose().dot(d)).tocoo().astype(np.float32)
adj = torch.sparse_coo_tensor(torch.from_numpy(np.vstack([An.row, An.col]).astype(np.int64)), torch.from_numpy(An.data), (n, n)).to(dev)
x = torch.randn(n, 30, device=dev)
torch.manual_seed(1); a = PGCN(30, 16).to(dev); torch.manual_seed(1); b = GCNConv(30, 16).to(dev)
ya, yb = a(x, adj), b(x, adj); ya.square().sum().backward(); yb.square().sum().backward()
print(dev, "max|dy|", (ya - yb).abs().max().item(), "max|dgrad W|", (a.lin.weight.grad - b.lin.weight.grad).abs().max().item())
