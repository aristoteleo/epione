# Parity of the spatial-method ports (COSMOS, PRESENT, INSTINCT, CANDIES, PRAGA, MISO)

Each `<method>/MANIFEST.md` was written and sha256-frozen (`MANIFEST.sha256`) before the
port was run; `A<n>` lines record documented amendments (fixture changes made before
the affected results were seen). `<method>/gates.json` holds the gate results;
`test_spatial_ports_parity.py` checks them when `EPIONE_SPATIAL_PARITY_DIR` is set.

Re-running needs the upstream clones, the public fixtures and (except COSMOS) a GPU:

* `SPATIAL_PARITY_REF` - directory with the upstream clones (`cosmos/src`, `present/src`,
  `instinct/src`, `candies/src`, `praga/src`, `miso/clone_pta`) at the commits in the manifests;
* `SPATIAL_PARITY_DATA` - directory with `cosmos_p22/`, `misar_multisp/` and
  `praga_zenodo14591305/` (sources and sha256 in the manifests);
* `SPATIAL_PARITY_WORK` - scratch directory (INSTINCT caches its upstream preprocessing there).

Per seed and arm: `python <method>/run_arm.py --arm official|port --seed s --out runs/<arm>_<s>.npz`
(CANDIES: `run_official.py` / `run_port.py`), then
`python common/parity_gates.py --runs runs --seeds ... --n-domains K [--cluster given] --out gates.json`.
The upstream arms ran with torch 2.6 / PyG 2.6.1 (+torch-sparse 0.6.18) and small
compatibility shims for old episcanpy / BiocPy / scipy APIs (numpy.NaN, anndata.read,
spmatrix.A, numpy-1 `copy=False`); none changes a number.
