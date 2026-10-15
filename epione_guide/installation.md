# Installation

EpiOne requires Python 3.10+.

- From PyPI (recommended once published): `pip install epione`
- From source (this repository):

```bash
git clone https://github.com/aristoteleo/epione.git
cd epione
pip install -U pip
pip install -e .
```

Optional extras for full functionality:

```bash
pip install .[full]
```


GPU acceleration (PyTorch backend of the `device=` arguments):

```bash
pip install .[gpu]
```

See {doc}`gpu` for CUDA builds of torch and the optional rapids-singlecell backend.
