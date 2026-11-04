"""One registry for every pretrained model Epione can load.

Pretrained weights used to be handled per model family, each with its own
dictionary, cache and download code. The registry puts them behind one
manifest and one API, but leaves each family's own functions working as
before:

>>> import epione as epi
>>> epi.models.list_pretrained()                         # every entry
>>> epi.models.list_pretrained("chrombpnet", biosample="K562")
>>> net = epi.models.load_pretrained("chrombpnet_encode_k562", kind="bias")
>>> path = epi.models.fetch_pretrained("print_tn5_bias")  # file only

Each :class:`PretrainedEntry` records what the model is (family, task,
organism, genome, biosample), where it comes from (a Hugging Face repo at a
pinned commit, an ENCODE accession, a URL, a file shipped with Epione or a
local path), every file it needs with an ``md5:`` / ``sha256:`` checksum,
its license and citation, and the loader that turns the files into a model.

Downloads
---------
Files are downloaded on first use with pooch and **verified against the
manifest checksum**; a file whose checksum does not match is never
returned. Pinning both the source revision and the checksum means a later
push to a model repository or portal fails loudly instead of silently
changing what a script loads. Nothing is ever re-hosted: the registry
points at each model's original publisher.

Cache
-----
``EPIONE_MODELS_DIR`` if set, else ``EPIONE_DATA_DIR``, else pooch's OS
cache (``~/.cache/epione`` on Linux); one sub-directory per family. These
are the directories the per-family loaders always used, so existing caches
stay valid. :func:`pretrained_cache_dir` reports the one in use.

Offline mode
------------
With ``EPIONE_OFFLINE=1`` (or ``HF_HUB_OFFLINE=1``), or ``offline=True``
passed to :func:`fetch_pretrained` / :func:`load_pretrained`, nothing is
downloaded: cached files are verified and used, and a missing file raises
:class:`FileNotFoundError` naming where it is expected. Pre-fetch on a
machine with network access with :func:`fetch_pretrained` and copy the
cache directory across.

Adding models
-------------
:func:`register` adds an entry, for instance one from a new model family::

    epi.models.register({
        "name": "aceso_mouse_brain_v1",
        "family": "aceso",
        "task": "spatial RNA -> scATAC",
        "organism": "mouse", "genome": "mm10",
        "source": {"type": "huggingface", "repo": "user/repo",
                   "revision": "<commit sha>", "subfolder": "mouse_brain_v1"},
        "files": {"model.safetensors": "sha256:...", "manifest.json": "md5:..."},
        "license": "GPL-3.0",
        "loader": "epione.models.aceso._pretrained:load_from_registry",
    })

The loader is called as ``loader(entry, **kwargs)`` and fetches what it
needs with ``entry.fetch(filename)``. Built-in entries live in
:mod:`epione.models._manifest`.
"""
from __future__ import annotations

import contextvars
import copy
import hashlib
import importlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

__all__ = [
    "ChecksumError",
    "PretrainedEntry",
    "fetch_pretrained",
    "file_checksum",
    "get_pretrained",
    "list_pretrained",
    "load_pretrained",
    "pretrained_cache_dir",
    "register",
    "register_family",
    "unregister",
]

_log = logging.getLogger("epione.models")

#: Source types understood by :meth:`PretrainedEntry.url`.
SOURCE_TYPES = ("huggingface", "encode", "url", "package", "local")
ENCODE_URL = "https://www.encodeproject.org/files/{acc}/@@download/{acc}.tar.gz"
_HASH_RE = re.compile(r"^(md5:[0-9a-f]{32}|sha256:[0-9a-f]{64})$")
_LOADER_RE = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$")
_NAME_RE = re.compile(r"^[A-Za-z0-9][\w.\-]*$")
_PKG_ROOT = Path(__file__).resolve().parent.parent      # the ``epione`` package

#: Columns of :func:`list_pretrained`, in order.
COLUMNS = ("name", "family", "task", "organism", "genome", "biosample", "source",
           "files", "size_mb", "license", "cached", "description")


class ChecksumError(ValueError):
    """A file does not match the checksum pinned in the registry."""


# ---------------------------------------------------------------------------
# Entries
# ---------------------------------------------------------------------------

@dataclass
class PretrainedEntry:
    """One pretrained model: metadata, pinned files and a loader.

    Parameters
    ----------
    name
        Registry-wide unique name, e.g. ``'chrombpnet_encode_k562'``.
    family
        Model family; selects the default loader when ``loader`` is not
        given (see :func:`register_family`).
    source
        Where the files come from. ``type`` is one of

        * ``'huggingface'`` -- ``repo``, ``revision`` (a commit, never a
          branch), optional ``subfolder``;
        * ``'encode'`` -- ``accession`` of an ENCODE file; the single file
          is ``<accession>.tar.gz``;
        * ``'url'`` -- ``base_url`` the file names are appended to, or a
          ``urls`` mapping ``{filename: url}``;
        * ``'package'`` -- ``path`` relative to the ``epione`` package
          (files shipped in the wheel);
        * ``'local'`` -- ``path``, a directory on this machine (for models
          you trained and want to load by name).
    files
        ``{filename: checksum}``, the checksum as ``'md5:<hex>'`` or
        ``'sha256:<hex>'``. Only ``'local'`` entries may use ``None``
        (not verified).
    loader
        ``'module:function'`` or a callable, called as
        ``loader(entry, **loader_kwargs, **kwargs)``.
    aliases
        Other names that resolve to this entry -- the family-local keys of
        the older per-family registries (``'encode_k562'``).
    """

    name: str
    family: str
    source: Dict[str, Any]
    files: Dict[str, Optional[str]]
    task: str = ""
    organism: str = ""
    genome: str = ""
    biosample: str = ""
    license: str = ""
    citation: str = ""
    description: str = ""
    homepage: str = ""
    size_mb: Optional[float] = None
    loader: Union[str, Callable, None] = None
    loader_kwargs: Dict[str, Any] = field(default_factory=dict)
    aliases: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    # -- validation ---------------------------------------------------------
    def validate(self) -> "PretrainedEntry":
        """Raise ``ValueError`` if the entry is malformed."""
        if not isinstance(self.name, str) or not _NAME_RE.match(self.name):
            raise ValueError(f"invalid model name {self.name!r}")
        if not self.family or not isinstance(self.family, str):
            raise ValueError(f"{self.name}: family is required")
        st = self.source.get("type") if isinstance(self.source, Mapping) else None
        if st not in SOURCE_TYPES:
            raise ValueError(f"{self.name}: source type must be one of {SOURCE_TYPES}, got {st!r}")
        need = {"huggingface": ("repo", "revision"), "encode": ("accession",),
                "package": ("path",), "local": ("path",)}.get(st, ())
        for k in need:
            if not self.source.get(k):
                raise ValueError(f"{self.name}: source {st!r} needs {k!r}")
        if st == "huggingface" and not re.fullmatch(r"[0-9a-f]{40}", str(self.source["revision"])):
            raise ValueError(f"{self.name}: pin a Hugging Face commit sha (40 hex), "
                             f"not {self.source['revision']!r}")
        if st == "url" and not (self.source.get("base_url") or self.source.get("urls")):
            raise ValueError(f"{self.name}: source 'url' needs base_url or urls")
        if not self.files:
            raise ValueError(f"{self.name}: at least one file is required")
        if st == "encode" and list(self.files) != [f"{self.source['accession']}.tar.gz"]:
            raise ValueError(f"{self.name}: an ENCODE entry has the single file "
                             f"'{self.source['accession']}.tar.gz'")
        for f, h in self.files.items():
            if h is None and st == "local":
                continue
            if not isinstance(h, str) or not _HASH_RE.match(h):
                raise ValueError(f"{self.name}: file {f!r} needs an 'md5:'/'sha256:' "
                                 f"checksum, got {h!r}")
        if isinstance(self.loader, str) and not _LOADER_RE.match(self.loader):
            raise ValueError(f"{self.name}: loader must be 'module:function', got {self.loader!r}")
        if self.loader is None and self.family not in _FAMILY_LOADERS:
            raise ValueError(f"{self.name}: family {self.family!r} has no default loader; "
                             "pass loader= or call register_family() first")
        return self

    # -- locations ----------------------------------------------------------
    @property
    def cache_subdir(self) -> str:
        return self.source.get("cache_subdir", self.family)

    def url(self, filename: str) -> Optional[str]:
        """Download URL of one file (``None`` for package / local files)."""
        s, st = self.source, self.source["type"]
        if st == "huggingface":
            sub = s.get("subfolder")
            rel = f"{sub}/{filename}" if sub else filename
            return f"https://huggingface.co/{s['repo']}/resolve/{s['revision']}/{rel}"
        if st == "encode":
            return ENCODE_URL.format(acc=s["accession"])
        if st == "url":
            urls = s.get("urls") or {}
            return urls[filename] if filename in urls else s["base_url"] + filename
        return None

    def local_path(self, filename: str) -> Path:
        """Where ``filename`` lives (or will be cached) on this machine."""
        s, st = self.source, self.source["type"]
        if st == "package":
            return _PKG_ROOT / s["path"] / filename
        if st == "local":
            return Path(s["path"]).expanduser() / filename
        base = pretrained_cache_dir(self.cache_subdir)
        if st == "huggingface":
            rev = s["revision"][:12]
            if s.get("layout") == "flat":   # seq2print's historical cache names
                return base / f"{s.get('subfolder') or self.name}__{rev}__{filename}"
            return base / f"{self.name}__{rev}" / filename
        return base / filename

    def _legacy_paths(self, filename: str) -> List[Path]:
        # PRINT / HINT used ``pooch.create(env='EPIONE_DATA_DIR')``, which
        # caches straight into $EPIONE_DATA_DIR (no family sub-directory).
        d = os.environ.get("EPIONE_DATA_DIR")
        if d and self.source["type"] == "url" and self.source.get("legacy_flat_env"):
            return [Path(d) / filename]
        return []

    def is_cached(self) -> bool:
        """All files present locally (not re-verified)."""
        return all(self.local_path(f).exists() or any(p.exists() for p in self._legacy_paths(f))
                   for f in self.files)

    def source_label(self) -> str:
        s, st = self.source, self.source["type"]
        if st == "huggingface":
            sub = f"/{s['subfolder']}" if s.get("subfolder") else ""
            return f"hf:{s['repo']}{sub}@{s['revision'][:7]}"
        if st == "encode":
            return f"encode:{s['accession']}"
        if st == "url":
            return s.get("label") or "url"
        return st

    # -- use ----------------------------------------------------------------
    def fetch(self, filename: Optional[str] = None, *, offline: Optional[bool] = None) -> Path:
        """Local, checksum-verified path of one file (downloading it if
        needed). ``filename`` may be omitted for single-file entries."""
        if filename is None:
            if len(self.files) != 1:
                raise ValueError(f"{self.name} has several files; pass one of {list(self.files)}")
            filename = next(iter(self.files))
        if filename not in self.files:
            raise KeyError(f"{self.name} has no file {filename!r}; files: {list(self.files)}")
        return _fetch_file(self, filename, _offline(offline))

    def load(self, **kwargs):
        """Build the model with the entry's loader."""
        fn = _resolve_loader(self)
        return fn(self, **{**self.loader_kwargs, **kwargs})

    def to_dict(self) -> dict:
        d = asdict(self)
        if callable(self.loader):
            d["loader"] = f"{self.loader.__module__}:{self.loader.__qualname__}"
        return d


# ---------------------------------------------------------------------------
# Registry state
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, PretrainedEntry] = {}
_ALIASES: Dict[str, str] = {}
_FAMILY_LOADERS: Dict[str, Union[str, Callable]] = {}
_BUILTINS_LOADED = False


def _ensure_builtins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    from ._manifest import BUILTIN_ENTRIES, FAMILY_LOADERS

    _FAMILY_LOADERS.update({k: v for k, v in FAMILY_LOADERS.items() if k not in _FAMILY_LOADERS})
    for e in BUILTIN_ENTRIES:
        if e["name"] not in _REGISTRY:
            register(e)


def register_family(family: str, loader: Union[str, Callable], *, overwrite: bool = False) -> None:
    """Set the default loader of a model family (``'module:function'`` or
    a callable taking ``(entry, **kwargs)``)."""
    _ensure_builtins()
    if family in _FAMILY_LOADERS and not overwrite:
        raise ValueError(f"family {family!r} already has a loader; pass overwrite=True")
    if isinstance(loader, str) and not _LOADER_RE.match(loader):
        raise ValueError(f"loader must be 'module:function', got {loader!r}")
    _FAMILY_LOADERS[family] = loader


def register(entry: Union[PretrainedEntry, Mapping[str, Any]], *, overwrite: bool = False
             ) -> PretrainedEntry:
    """Add a model to the registry.

    Parameters
    ----------
    entry
        A :class:`PretrainedEntry` or a dict of its fields.
    overwrite
        Replace an existing entry of the same name. Off by default so that
        a typo cannot silently shadow a built-in model.

    Returns
    -------
    The validated entry.
    """
    if not isinstance(entry, PretrainedEntry):
        loader = entry.get("loader")
        entry = copy.deepcopy({k: v for k, v in entry.items() if k != "loader"})
        entry["loader"] = loader          # callables are kept as they are
        unknown = set(entry) - set(PretrainedEntry.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown entry field(s) {sorted(unknown)}")
        entry = PretrainedEntry(**entry)
    _ensure_builtins()
    entry.validate()
    taken = entry.name in _REGISTRY or entry.name in _ALIASES
    if taken and not overwrite:
        raise ValueError(f"{entry.name!r} is already registered; pass overwrite=True to replace it")
    for a in entry.aliases:
        owner = _ALIASES.get(a) or (a if a in _REGISTRY else None)
        if owner not in (None, entry.name) and not overwrite:
            raise ValueError(f"alias {a!r} of {entry.name!r} is already used by {owner!r}")
    if entry.name in _REGISTRY:
        unregister(entry.name)
    _REGISTRY[entry.name] = entry
    for a in entry.aliases:
        _ALIASES[a] = entry.name
    return entry


def unregister(name: str) -> PretrainedEntry:
    """Remove an entry (and its aliases); returns it."""
    entry = get_pretrained(name)
    del _REGISTRY[entry.name]
    for a in entry.aliases:
        _ALIASES.pop(a, None)
    return entry


def get_pretrained(name: str) -> PretrainedEntry:
    """The :class:`PretrainedEntry` for ``name`` (or one of its aliases)."""
    _ensure_builtins()
    if isinstance(name, PretrainedEntry):
        return name
    if name in _REGISTRY:
        return _REGISTRY[name]
    if name in _ALIASES:
        return _REGISTRY[_ALIASES[name]]
    import difflib

    close = difflib.get_close_matches(name, list(_REGISTRY) + list(_ALIASES), n=3)
    hint = f"; did you mean {close}?" if close else ""
    raise KeyError(f"unknown pretrained model {name!r}{hint} "
                   "(epi.models.list_pretrained() lists them)")


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------

def list_pretrained(family: Optional[str] = None, **filters):
    """Table of registered pretrained models.

    Parameters
    ----------
    family
        Keep one family (``'chrombpnet'``, ``'seq2print'``, ``'print'``,
        ``'hint'``, ``'scooby'``, ``'borzoi'``...), or a list of them.
    **filters
        Column filters, e.g. ``organism='human'``, ``biosample='K562'``,
        ``cached=True``. A string matches case-insensitively, a list or set
        matches any of its members, a callable is applied to the column
        value.

    Returns
    -------
    :class:`pandas.DataFrame`, one row per model, indexed by ``name``.
    """
    import pandas as pd

    _ensure_builtins()
    if family is not None:
        filters["family"] = family
    bad = set(filters) - set(COLUMNS)
    if bad:
        raise ValueError(f"cannot filter on {sorted(bad)}; columns are {list(COLUMNS)}")
    rows = []
    for e in _REGISTRY.values():
        rows.append({"name": e.name, "family": e.family, "task": e.task,
                     "organism": e.organism, "genome": e.genome, "biosample": e.biosample,
                     "source": e.source_label(), "files": len(e.files), "size_mb": e.size_mb,
                     "license": e.license, "cached": e.is_cached(),
                     "description": e.description})
    df = pd.DataFrame(rows, columns=list(COLUMNS))
    for col, want in filters.items():
        df = df[df[col].map(lambda v, w=want: _match(v, w))]
    return df.set_index("name")


def _match(value, want) -> bool:
    if callable(want):
        return bool(want(value))
    if isinstance(want, (list, tuple, set, frozenset)):
        return any(_match(value, w) for w in want)
    if isinstance(want, str) and isinstance(value, str):
        return value.lower() == want.lower()
    return value == want


# ---------------------------------------------------------------------------
# Fetching and checksums
# ---------------------------------------------------------------------------

def pretrained_cache_dir(family: Optional[str] = None) -> Path:
    """Cache root (``EPIONE_MODELS_DIR`` > ``EPIONE_DATA_DIR`` > pooch's
    OS cache), or the sub-directory of ``family``."""
    d = os.environ.get("EPIONE_MODELS_DIR") or os.environ.get("EPIONE_DATA_DIR")
    if d:
        root = Path(d).expanduser()
    else:
        import pooch
        root = Path(pooch.os_cache("epione"))
    return root / family if family else root


_OFFLINE_OVERRIDE: "contextvars.ContextVar[Optional[bool]]" = contextvars.ContextVar(
    "epione_models_offline", default=None)


def _offline(flag: Optional[bool]) -> bool:
    if flag is None:
        flag = _OFFLINE_OVERRIDE.get()
    if flag is not None:
        return bool(flag)
    on = ("1", "true", "yes", "on")
    return (os.environ.get("EPIONE_OFFLINE", "").lower() in on
            or os.environ.get("HF_HUB_OFFLINE", "").lower() in on)


def file_checksum(path: Union[str, Path], algorithm: str = "sha256") -> str:
    """``'<algorithm>:<hex>'`` of a file, read in 1 MiB chunks."""
    h = hashlib.new(algorithm)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return f"{algorithm}:{h.hexdigest()}"


def _stamp(path: Path) -> Path:
    return path.with_name(path.name + ".verified")


def _verify(path: Path, expected: Optional[str], *, stamp: bool = True) -> bool:
    """Does ``path`` match ``expected``? A ``.verified`` stamp (checksum,
    size, mtime) next to the file spares re-hashing a multi-GB file on
    every load; any change to the file invalidates it."""
    if expected is None:
        return True
    if not stamp:
        return file_checksum(path, expected.split(":", 1)[0]) == expected
    st = path.stat()
    sig = {"checksum": expected, "size": st.st_size, "mtime_ns": st.st_mtime_ns}
    stamp = _stamp(path)
    try:
        if json.loads(stamp.read_text()) == sig:
            return True
    except (OSError, ValueError):
        pass
    ok = file_checksum(path, expected.split(":", 1)[0]) == expected
    if ok:
        try:
            stamp.write_text(json.dumps(sig))
        except OSError:              # read-only cache: verify every time instead
            pass
    return ok


def _download(url: str, target: Path) -> Path:
    """Download ``url`` to ``target`` (no checksum here; the caller
    verifies). Separate so tests can replace it."""
    import logging

    import pooch

    target.parent.mkdir(parents=True, exist_ok=True)
    _log.info("downloading %s", url)
    # pooch would log "use this SHA256 as known_hash" -- misleading here,
    # since the registry pins and checks the checksum itself.
    plog = pooch.get_logger()
    level = plog.level
    plog.setLevel(logging.WARNING)
    try:
        return Path(pooch.retrieve(url, known_hash=None, fname=target.name,
                                   path=target.parent, progressbar=False))
    finally:
        plog.setLevel(level)


def _fetch_file(entry: PretrainedEntry, filename: str, offline: bool) -> Path:
    expected = entry.files[filename]
    target = entry.local_path(filename)
    st = entry.source["type"]

    if st in ("package", "local"):
        if not target.exists():
            raise FileNotFoundError(f"{entry.name}: {target} does not exist")
        # never write stamps into the installed package
        if not _verify(target, expected, stamp=st == "local"):
            raise ChecksumError(f"{entry.name}: {target} does not match {expected}")
        return target

    if target.exists():
        if _verify(target, expected):
            return target
        if offline:
            raise ChecksumError(f"{entry.name}: cached {target} does not match {expected} "
                                "and offline mode forbids re-downloading it")
    for p in entry._legacy_paths(filename):
        if p.exists() and _verify(p, expected):
            return p
    if offline:
        raise FileNotFoundError(
            f"{entry.name}: {filename} is not cached at {target} and offline mode is on "
            "(EPIONE_OFFLINE / HF_HUB_OFFLINE). Fetch it on a machine with network access "
            f"with epi.models.fetch_pretrained({entry.name!r}) and copy "
            f"{pretrained_cache_dir()} across.")

    part = target.with_name(target.name + ".part")
    part.unlink(missing_ok=True)       # a stale partial would be taken as complete
    got = _download(entry.url(filename), part)
    if not _verify(got, expected):
        actual = file_checksum(got, expected.split(":", 1)[0])
        got.unlink()
        _stamp(got).unlink(missing_ok=True)
        raise ChecksumError(f"{entry.name}: download of {filename} from {entry.url(filename)} "
                            f"has {actual}, the registry pins {expected}. The upstream file "
                            "changed; refusing to use it.")
    os.replace(got, target)            # a rename keeps size and mtime, so the stamp stays valid
    if _stamp(got).exists():
        os.replace(_stamp(got), _stamp(target))
    return target


def fetch_pretrained(name: str, filename: Optional[str] = None, *,
                     offline: Optional[bool] = None) -> Union[Path, Dict[str, Path]]:
    """Download (once) and verify a model's files without loading it.

    Returns the path of ``filename``, or ``{filename: path}`` for all files
    when it is omitted and the entry has several.
    """
    e = get_pretrained(name)
    if filename is not None or len(e.files) == 1:
        return e.fetch(filename, offline=offline)
    return {f: e.fetch(f, offline=offline) for f in e.files}


def _resolve_loader(entry: PretrainedEntry) -> Callable:
    spec = entry.loader if entry.loader is not None else _FAMILY_LOADERS.get(entry.family)
    if spec is None:
        raise ValueError(f"{entry.name}: no loader for family {entry.family!r}")
    if callable(spec):
        return spec
    mod, attr = spec.split(":")
    return getattr(importlib.import_module(mod), attr)


def load_pretrained(name: str, *, offline: Optional[bool] = None, **kwargs):
    """Load a registered pretrained model.

    Parameters
    ----------
    name
        A name from :func:`list_pretrained` (or an alias such as
        ChromBPNet's ``'encode_k562'``).
    offline
        Never download (default: ``EPIONE_OFFLINE`` / ``HF_HUB_OFFLINE``).
    **kwargs
        Passed to the family loader, e.g. ``kind='bias', fold=2`` for
        ChromBPNet, ``return_manifest=True`` for seq2PRINT, ``device=`` for
        scooby / Borzoi.

    Returns
    -------
    Whatever the family loader builds -- a ``torch.nn.Module`` for the
    deep models, :class:`~epione.tl.DispersionModel` /
    :class:`~epione.tl.HintHMM` / bias tables for the footprinting ones.
    """
    e = get_pretrained(name)
    if offline is None:
        return e.load(**kwargs)
    token = _OFFLINE_OVERRIDE.set(bool(offline))   # seen by every entry.fetch inside the loader
    try:
        return e.load(**kwargs)
    finally:
        _OFFLINE_OVERRIDE.reset(token)
