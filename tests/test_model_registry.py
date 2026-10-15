"""epione.models.registry: manifest integrity, checksum / cache / offline
logic (downloads mocked), and loader round trips per family.

Network tests (PRINT, HINT: a few MB) skip when the download fails; the
Borzoi / scooby / ENCODE ones (0.5-0.8 GB each) only run with
``EPIONE_TEST_BIG_MODELS=1``.
"""
import hashlib
import importlib
import io
import json
import os
import re
import tarfile

import numpy as np
import pytest

from epione.models import _manifest
from epione.models import registry as R


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Own cache, online, and registry state restored afterwards."""
    monkeypatch.setenv("EPIONE_MODELS_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("EPIONE_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    R._ensure_builtins()
    reg, ali, fam = dict(R._REGISTRY), dict(R._ALIASES), dict(R._FAMILY_LOADERS)
    yield
    for d, saved in ((R._REGISTRY, reg), (R._ALIASES, ali), (R._FAMILY_LOADERS, fam)):
        d.clear()
        d.update(saved)


def _md5(b):
    return "md5:" + hashlib.md5(b).hexdigest()


def _sha(b):
    return "sha256:" + hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------------------
# manifest integrity
# ---------------------------------------------------------------------------

def test_builtin_entries_validate_and_are_unique():
    names = [e["name"] for e in _manifest.BUILTIN_ENTRIES]
    assert len(names) == len(set(names))
    aliases = [a for e in _manifest.BUILTIN_ENTRIES for a in e.get("aliases", [])]
    assert len(aliases) == len(set(aliases)) and not set(aliases) & set(names)
    for e in _manifest.BUILTIN_ENTRIES:
        entry = R.PretrainedEntry(**e).validate()
        for k in ("task", "organism", "genome", "license", "citation", "description"):
            assert getattr(entry, k), f"{entry.name}: {k} is empty"
        if entry.source["type"] != "package":
            assert entry.size_mb and entry.size_mb > 0
        for f in entry.files:
            if entry.source["type"] in ("huggingface", "encode", "url"):
                assert entry.url(f).startswith("https://")


def test_every_family_has_an_importable_loader():
    fams = {e["family"] for e in _manifest.BUILTIN_ENTRIES}
    assert fams <= set(_manifest.FAMILY_LOADERS)
    for spec in _manifest.FAMILY_LOADERS.values():
        mod, attr = spec.split(":")
        assert callable(getattr(importlib.import_module(mod), attr))


def test_expected_models_are_registered():
    df = R.list_pretrained()
    assert set(df.family) == {"seq2print", "chrombpnet", "print", "hint", "borzoi", "scooby"}
    assert {"chrombpnet_encode_k562", "seq2print_p2s1_fold0_bulk", "print_tn5_bias",
            "hint_atac_bias_tables", "borzoi_replicate_0", "scooby_neurips"} <= set(df.index)


def test_package_files_match_their_checksums():
    for e in _manifest.BUILTIN_ENTRIES:
        if e["source"]["type"] == "package":
            entry = R.get_pretrained(e["name"])
            for f, h in entry.files.items():
                assert R.file_checksum(entry.local_path(f), h.split(":")[0]) == h
                entry.fetch(f)
                assert not R._stamp(entry.local_path(f)).exists()   # package dir untouched


def test_legacy_family_dicts_are_unchanged():
    """The per-family PRETRAINED dicts are now views of the manifest and
    must still hold exactly what they held before the registry."""
    from epione.models.chrombpnet import _pretrained as cbp
    from epione.models.seq2print import _pretrained as s2p

    rev = "e8a36d4ca434e15d1de8002e796a61b8352529b3"
    assert s2p.PRETRAINED == {
        "p2s1_fold0_bulk": {"revision": rev, "files": {
            "model.safetensors": "md5:4b278474646f7e66df0d0f37bb9a0059",
            "manifest.json": "md5:b05f15a3e1dd89e1ac0753e86c2d4b9a"}},
        "p2s1_fold0_lora_bands": {"revision": rev, "files": {
            "model.safetensors": "md5:cb4f77aef16b220e2188746982851a70",
            "manifest.json": "md5:d054d44fddb97ad94e6fa61de1fa94de"}},
    }
    assert cbp.PRETRAINED["encode_k562"] == {
        "accession": "ENCFF008TMU", "md5": "a1b9870fcb52bba03ea5ef71c8acf897",
        "experiment": "ENCSR868FGK", "annotation": "ENCSR893SUD", "biosample": "K562"}
    assert sorted(cbp.PRETRAINED) == ["encode_gm12878", "encode_hepg2", "encode_imr90",
                                      "encode_k562", "encode_tcell"]
    from epione.tl import _hint, _print
    for key, (fname, md5) in _print._MODELS.items():
        assert R.get_pretrained(f"print_{key}").files == {fname: md5}
    assert R.get_pretrained("hint_atac_bias_tables").files == _hint._RGT_FILES


def test_cache_locations_match_the_old_per_family_ones(monkeypatch, tmp_path):
    """Existing caches stay valid: same directories and file names."""
    monkeypatch.delenv("EPIONE_MODELS_DIR")
    monkeypatch.setenv("EPIONE_DATA_DIR", str(tmp_path))
    e = R.get_pretrained("seq2print_p2s1_fold0_bulk")
    assert e.local_path("manifest.json") == (
        tmp_path / "seq2print" / "p2s1_fold0_bulk__e8a36d4ca434__manifest.json")
    assert R.get_pretrained("encode_k562").local_path("ENCFF008TMU.tar.gz") == (
        tmp_path / "chrombpnet" / "ENCFF008TMU.tar.gz")
    # PRINT / HINT used pooch's env= mode: straight into $EPIONE_DATA_DIR.
    assert R.get_pretrained("print_tn5_bias")._legacy_paths("Tn5_NN_model_py_v2.pt") == [
        tmp_path / "Tn5_NN_model_py_v2.pt"]


# ---------------------------------------------------------------------------
# register / list
# ---------------------------------------------------------------------------

def _url_entry(name="toy", payload=b"weights", **kw):
    return dict(dict(name=name, family="toy", task="t", organism="o",
                     source={"type": "url", "base_url": "https://example.org/m/"},
                     files={"w.bin": _sha(payload)},
                     loader=lambda entry, scale=1: entry.fetch().read_bytes() * scale), **kw)


def test_register_validates():
    with pytest.raises(ValueError, match="checksum"):
        R.register(_url_entry(files={"w.bin": "abc"}))
    with pytest.raises(ValueError, match="source type"):
        R.register(_url_entry(source={"type": "ftp"}))
    with pytest.raises(ValueError, match="commit sha"):
        R.register(_url_entry(source={"type": "huggingface", "repo": "a/b", "revision": "main"}))
    with pytest.raises(ValueError, match="no default loader"):
        R.register(_url_entry(loader=None))
    with pytest.raises(ValueError, match="unknown entry field"):
        R.register(dict(_url_entry(), colour="red"))
    with pytest.raises(ValueError, match="module:function"):
        R.register(_url_entry(loader="not a loader"))


def test_register_duplicates_aliases_and_unregister():
    R.register(_url_entry(aliases=["toy_alias"]))
    assert R.get_pretrained("toy_alias").name == "toy"
    with pytest.raises(ValueError, match="already registered"):
        R.register(_url_entry())
    with pytest.raises(ValueError, match="already used"):
        R.register(_url_entry(name="toy2", aliases=["toy_alias"]))
    with pytest.raises(ValueError, match="already registered"):
        R.register(_url_entry(name="encode_k562"))           # a built-in alias
    R.register(_url_entry(task="new"), overwrite=True)
    assert R.get_pretrained("toy").task == "new" and "toy_alias" not in R._ALIASES
    R.unregister("toy")
    with pytest.raises(KeyError, match="unknown pretrained model"):
        R.get_pretrained("toy")


def test_register_family_then_entry_without_loader():
    R.register_family("toy", lambda entry: "built " + entry.name)
    R.register(_url_entry(loader=None))
    assert R.load_pretrained("toy") == "built toy"
    with pytest.raises(ValueError, match="already has a loader"):
        R.register_family("toy", "a.b:c")


def test_list_pretrained_filters():
    df = R.list_pretrained("chrombpnet")
    assert len(df) == 5 and set(df.family) == {"chrombpnet"}
    assert list(R.list_pretrained(biosample="k562").index) == ["chrombpnet_encode_k562"]
    assert set(R.list_pretrained(family=["print", "hint"]).family) == {"print", "hint"}
    big = R.list_pretrained(size_mb=lambda v: v is not None and v > 700)
    assert set(big.family) == {"borzoi", "scooby"}
    assert R.list_pretrained(organism="mouse").family.unique().tolist() == ["seq2print"]
    assert R.list_pretrained("hint", cached=True).index.tolist() == [
        "hint_atac_hmm_paired", "hint_atac_hmm_single"]
    with pytest.raises(ValueError, match="cannot filter"):
        R.list_pretrained(colour="red")
    assert list(R.list_pretrained().columns) == list(R.COLUMNS[1:])


def test_unknown_name_suggests_close_matches():
    with pytest.raises(KeyError, match="chrombpnet_encode_k562"):
        R.get_pretrained("chrombpnet_encode_k56")


# ---------------------------------------------------------------------------
# checksums, cache and offline mode (downloads mocked)
# ---------------------------------------------------------------------------

class FakeNet:
    def __init__(self, payload):
        self.payload, self.calls = payload, []

    def __call__(self, url, target):
        self.calls.append(url)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.payload)
        return target


@pytest.mark.parametrize("hasher", [_md5, _sha])
def test_fetch_downloads_once_and_verifies(monkeypatch, hasher):
    net = FakeNet(b"weights")
    monkeypatch.setattr(R, "_download", net)
    R.register(_url_entry(files={"w.bin": hasher(b"weights")}))
    p = R.fetch_pretrained("toy")
    assert p.read_bytes() == b"weights" and net.calls == ["https://example.org/m/w.bin"]
    assert p.parent.name == "toy" and json.loads(R._stamp(p).read_text())["checksum"]
    assert R.fetch_pretrained("toy") == p and len(net.calls) == 1      # cached
    assert R.load_pretrained("toy", scale=2) == b"weightsweights"
    assert R.list_pretrained("toy").cached.all()


def test_changed_upstream_file_is_refused(monkeypatch):
    monkeypatch.setattr(R, "_download", FakeNet(b"tampered"))
    R.register(_url_entry())
    with pytest.raises(R.ChecksumError, match="refusing"):
        R.fetch_pretrained("toy")
    e = R.get_pretrained("toy")
    assert not e.local_path("w.bin").exists()
    assert not e.local_path("w.bin").with_name("w.bin.part").exists()


def test_corrupt_cache_is_redownloaded(monkeypatch):
    net = FakeNet(b"weights")
    monkeypatch.setattr(R, "_download", net)
    R.register(_url_entry())
    p = R.fetch_pretrained("toy")
    p.write_bytes(b"weightz")                # same size; the stamp's mtime no longer matches
    os.utime(p, ns=(1, 1))
    assert R.fetch_pretrained("toy").read_bytes() == b"weights" and len(net.calls) == 2


def test_stamp_spares_rehashing(monkeypatch):
    monkeypatch.setattr(R, "_download", FakeNet(b"weights"))
    R.register(_url_entry())
    R.fetch_pretrained("toy")
    calls = []
    monkeypatch.setattr(R, "file_checksum", lambda *a, **k: calls.append(a) or "x")
    R.fetch_pretrained("toy")
    assert calls == []


def test_offline_mode(monkeypatch):
    net = FakeNet(b"weights")
    monkeypatch.setattr(R, "_download", net)
    R.register(_url_entry())
    with pytest.raises(FileNotFoundError, match="offline"):
        R.fetch_pretrained("toy", offline=True)
    monkeypatch.setenv("EPIONE_OFFLINE", "1")
    with pytest.raises(FileNotFoundError, match="offline"):
        R.load_pretrained("toy")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("EPIONE_OFFLINE")
    with pytest.raises(FileNotFoundError):
        R.fetch_pretrained("toy")
    assert net.calls == []
    # explicit offline=False wins over the environment, inside loaders too
    assert R.load_pretrained("toy", offline=False) == b"weights"
    assert R.load_pretrained("toy") == b"weights" and len(net.calls) == 1   # cached file used
    p = R.get_pretrained("toy").local_path("w.bin")
    p.write_bytes(b"other!!")
    os.utime(p, ns=(1, 1))
    with pytest.raises(R.ChecksumError, match="offline"):
        R.fetch_pretrained("toy")


def test_legacy_flat_cache_is_reused(monkeypatch, tmp_path):
    """PRINT files cached by the old pooch registry ($EPIONE_DATA_DIR/<file>)
    are picked up without downloading when the checksum matches."""
    from epione.tl import _print

    legacy = tmp_path / "legacy"
    legacy.mkdir()
    monkeypatch.setenv("EPIONE_DATA_DIR", str(legacy))
    monkeypatch.setattr(R, "_download", FakeNet(b""))
    payload = b"tn5 cnn"
    R.register(dict(R.get_pretrained("print_tn5_bias").to_dict(),
                    files={"Tn5_NN_model_py_v2.pt": _md5(payload)}), overwrite=True)
    (legacy / "Tn5_NN_model_py_v2.pt").write_bytes(payload)
    assert _print.fetch_print_model("tn5_bias") == str(legacy / "Tn5_NN_model_py_v2.pt")
    assert R._download.calls == []


def test_local_entries(tmp_path):
    d = tmp_path / "mine"
    d.mkdir()
    (d / "w.bin").write_bytes(b"abc")
    R.register(_url_entry(name="mine", source={"type": "local", "path": str(d)},
                          files={"w.bin": None}))
    assert R.load_pretrained("mine") == b"abc"
    R.register(_url_entry(name="mine", source={"type": "local", "path": str(d)},
                          files={"w.bin": _md5(b"xyz")}), overwrite=True)
    with pytest.raises(R.ChecksumError):
        R.load_pretrained("mine")


def test_hf_urls_and_layouts(monkeypatch, tmp_path):
    e = R.get_pretrained("seq2print_p2s1_fold0_bulk")
    assert e.url("manifest.json") == ("https://huggingface.co/starlitnightly/epione-seq2print/"
                                      "resolve/e8a36d4ca434e15d1de8002e796a61b8352529b3/"
                                      "p2s1_fold0_bulk/manifest.json")
    b = R.get_pretrained("borzoi_replicate_0")
    assert b.url("config.json").endswith("/johahi/borzoi-replicate-0/resolve/"
                                         "521db103372769581b8d1d616e6a3b496da490c5/config.json")
    # transformers needs the files side by side in one directory
    assert b.local_path("config.json").parent == b.local_path("model.safetensors").parent
    assert R.get_pretrained("encode_k562").url("ENCFF008TMU.tar.gz") == (
        "https://www.encodeproject.org/files/ENCFF008TMU/@@download/ENCFF008TMU.tar.gz")


# ---------------------------------------------------------------------------
# family loaders
# ---------------------------------------------------------------------------

def test_hint_hmm_loader_matches_load_hint_hmm():
    from epione.tl import load_hint_hmm

    for model in ("paired", "single"):
        a, b = R.load_pretrained(f"hint_atac_hmm_{model}"), load_hint_hmm(model)
        assert a.name == b.name
        for k in ("startprob", "transmat", "means", "covars"):
            np.testing.assert_array_equal(getattr(a, k), getattr(b, k))


def _try(fn):
    try:
        return fn()
    except (OSError, ValueError) as exc:     # no network on this runner
        if isinstance(exc, R.ChecksumError):
            raise
        pytest.skip(f"download failed: {exc}")


def test_print_loaders_roundtrip():
    from epione.tl import DispersionModel, fetch_print_model

    disp = _try(lambda: R.load_pretrained("print_dispersion"))
    ref = DispersionModel.load(fetch_print_model("dispersion"))
    assert isinstance(disp, DispersionModel) and disp.version == ref.version
    assert sorted(disp.params) == sorted(ref.params)
    torch = pytest.importorskip("torch")
    net = _try(lambda: R.load_pretrained("print_tn5_bias"))
    ref = torch.jit.load(fetch_print_model("tn5_bias"), map_location="cpu").eval()
    x = torch.nn.functional.one_hot(torch.randint(0, 4, (2, 300),
                                    generator=torch.Generator().manual_seed(0)), 4)
    x = x.permute(0, 2, 1).float()                      # (batch, 4, L), as PrintBias feeds it
    with torch.no_grad():
        assert torch.equal(net(x), ref(x))


def test_hint_tables_loader_roundtrip():
    from epione.tl import fetch_hint_file, load_hint_bias_table

    f, r = _try(lambda: R.load_pretrained("hint_atac_bias_tables"))
    f0, r0 = load_hint_bias_table(fetch_hint_file("atac_bias_table_F.txt"),
                                  fetch_hint_file("atac_bias_table_R.txt"))
    assert f == f0 and r == r0 and len(f) == 4 ** 8


def test_seq2print_loader_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from epione.models import seq2print as s2p

    torch.manual_seed(0)
    m = s2p.build_seq2print(n_filters=32, n_layers=3)
    files = s2p.save_pretrained(m, tmp_path / "toy", manifest={
        "architecture": {"n_filters": 32, "n_layers": 3}})
    R.register(dict(name="my_seq2print", family="seq2print", files=files,
                    source={"type": "local", "path": str(tmp_path / "toy")}))
    loaded, manifest = R.load_pretrained("my_seq2print", return_manifest=True)
    assert manifest["architecture"] == {"n_filters": 32, "n_layers": 3}
    x = torch.nn.functional.one_hot(torch.randint(0, 4, (2, m.dna_len)), 4).permute(0, 2, 1).float()
    with torch.no_grad():
        a, b = m.eval()(x), loaded(x)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_seq2print_legacy_fetch_goes_through_registry(monkeypatch):
    from epione.models.seq2print import _pretrained as s2p

    calls = []

    def no_network(url, target):
        calls.append(url)
        raise OSError("no network in this test")

    monkeypatch.setattr(R, "_download", no_network)
    with pytest.raises(OSError):
        s2p._fetch("p2s1_fold0_bulk", "manifest.json")
    assert calls == [R.get_pretrained("seq2print_p2s1_fold0_bulk").url("manifest.json")]


def test_chrombpnet_loader_extracts_from_release_tar(monkeypatch, tmp_path):
    """An ENCODE-shaped tar (mocked download) -> kind/fold member -> BPNet,
    through both the registry and the legacy chrombpnet functions."""
    torch = pytest.importorskip("torch")
    h5py = pytest.importorskip("h5py")
    from test_chrombpnet import _layers, _small, _write_weights

    from epione.models import chrombpnet as cbp

    torch.manual_seed(1)
    net = cbp.glorot_init_(_small())
    h5 = tmp_path / "bias.h5"
    with h5py.File(h5, "w") as h:
        h.attrs["model_config"] = json.dumps({"class_name": "Functional",
                                              "config": {"layers": _layers(net)}})
        _write_weights(h.create_group("model_weights"), net)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        tf.add(h5, arcname="models/fold_1/model.bias_scaled.fold_1.ENCSRTOY.h5")
    payload = buf.getvalue()
    monkeypatch.setattr(R, "_download", FakeNet(payload))
    R.register(dict(name="chrombpnet_toy", family="chrombpnet", aliases=["toy_release"],
                    source={"type": "encode", "accession": "ENCFFTOY"},
                    files={"ENCFFTOY.tar.gz": _md5(payload)},
                    extra={"key": "toy_release", "experiment": "ENCSRTOY"}))
    m = R.load_pretrained("chrombpnet_toy", kind="bias", fold=1)
    x = torch.randint(0, 2, (2, 4, 128)).float()
    with torch.no_grad():
        assert torch.allclose(m(x)[0], net(x)[0], atol=1e-6)
    assert cbp.pretrained_path("toy_release", "bias", 1).exists()     # legacy entry point
    assert isinstance(cbp.load_pretrained("chrombpnet_toy", "bias", 1), cbp.BPNet)
    with pytest.raises(FileNotFoundError, match="not in"):
        R.load_pretrained("chrombpnet_toy", kind="bias", fold=0)
    assert len(R._download.calls) == 1


def test_translate_loader_roundtrip(tmp_path):
    pytest.importorskip("torch")
    from test_translate import _synthetic

    from epione import translate as T

    rna, atac = _synthetic()
    r, a = T.prepare(rna, atac, n_top_genes=80, min_cells_frac=0.01)
    m = T.fit(r[:120], a[:120], method="scbutterfly", device="cpu", seed=3, threads=2,
              rna_pretrain_epochs=1, atac_pretrain_epochs=1, translator_epochs=1)
    m.save(tmp_path / "m.pt")
    blob = (tmp_path / "m.pt").read_bytes()
    R.register(dict(name="my_translator", family="translate", task="RNA <-> ATAC",
                    source={"type": "local", "path": str(tmp_path)},
                    files={"m.pt": _sha(blob)}))
    m2 = R.load_pretrained("my_translator")
    np.testing.assert_array_equal(m.predict(a[120:]).X.toarray(), m2.predict(a[120:]).X.toarray())


# ---------------------------------------------------------------------------
# big downloads (opt-in)
# ---------------------------------------------------------------------------

big = pytest.mark.skipif(os.environ.get("EPIONE_TEST_BIG_MODELS") != "1",
                         reason="set EPIONE_TEST_BIG_MODELS=1 (downloads 0.5-0.8 GB per model)")


@big
def test_real_chrombpnet_release(monkeypatch):
    monkeypatch.delenv("EPIONE_MODELS_DIR")       # use the real cache
    net = R.load_pretrained("chrombpnet_encode_k562", kind="bias", fold=0)
    assert net.input_len == 2114 and net.output_len == 1000


@big
def test_real_borzoi(monkeypatch):
    pytest.importorskip("borzoi_pytorch")
    monkeypatch.delenv("EPIONE_MODELS_DIR")
    m = R.load_pretrained("borzoi_replicate_0")
    assert not any(b.is_meta for b in m.buffers())


@big
def test_real_scooby(monkeypatch):
    src = os.environ.get("SCOOBY_SRC")
    if not src:
        pytest.skip("set SCOOBY_SRC to a clone of gagneurlab/scooby")
    import sys
    sys.path.insert(0, src)
    monkeypatch.delenv("EPIONE_MODELS_DIR")
    m = R.load_pretrained("scooby_neurips")
    w = dict(m.named_parameters())["cell_state_to_conv.0.weight"]
    assert tuple(w.shape) == (128, 14)
