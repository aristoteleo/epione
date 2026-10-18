"""Minimal runtime-compiled CUDA kernels for torch device memory.

Some Epione GPU paths need a kernel that torch does not provide, e.g. a
sparse-pattern product that sums every output in a fixed sequential order
(cuSPARSE's default algorithms reduce in a run-dependent order, so they are
neither reproducible nor bit-identical to the numba CPU kernels). This
module compiles CUDA C source with NVRTC and launches it through the CUDA
driver API, both reached with :mod:`ctypes`:

* ``libnvrtc`` ships with every CUDA build of torch (the
  ``nvidia-cuda-nvrtc-cu12`` wheel or the toolkit / conda package), so no
  extra dependency and no host compiler is needed (unlike Triton or
  ``torch.utils.cpp_extension``);
* ``libcuda`` is the driver itself.

Kernels operate on torch tensors (``tensor.data_ptr()``) and are launched on
torch's current stream, so they order correctly with torch operations.
:func:`available` says whether the machinery works on this machine; callers
fall back to their CPU path otherwise.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import glob
import os
import threading
from typing import Dict, Optional, Sequence

__all__ = ["available", "Module", "memcpy_htod_async", "host_register", "host_unregister"]

_LOCK = threading.Lock()
_LIBS: Dict[str, object] = {}
_ERR: Optional[str] = None


def _find_nvrtc() -> Optional[str]:
    cands = []
    try:
        import nvidia.cuda_nvrtc as _n  # pip wheel shipped with torch
        for d in list(getattr(_n, "__path__", [])):
            cands += sorted(glob.glob(os.path.join(d, "lib", "libnvrtc.so*")))
    except Exception:
        pass
    try:
        import torch
        tl = os.path.join(os.path.dirname(torch.__file__), "lib")
        cands += sorted(glob.glob(os.path.join(tl, "libnvrtc*.so*")))
    except Exception:
        pass
    for env in ("CUDA_HOME", "CUDA_PATH", "CONDA_PREFIX"):
        p = os.environ.get(env)
        if p:
            cands += sorted(glob.glob(os.path.join(p, "lib*", "libnvrtc.so*")))
    lib = ctypes.util.find_library("nvrtc")
    if lib:
        cands.append(lib)
    # the main library, not the builtins helper
    cands = [c for c in cands if "builtins" not in os.path.basename(c)]
    return cands[0] if cands else None


def _libs():
    global _ERR
    with _LOCK:
        if _LIBS or _ERR:
            return _LIBS
        try:
            cuda = ctypes.CDLL("libcuda.so.1")
            path = _find_nvrtc()
            if path is None:
                raise OSError("libnvrtc not found")
            nvrtc = ctypes.CDLL(path)
            _LIBS.update(cuda=cuda, nvrtc=nvrtc)
        except OSError as e:
            _ERR = str(e)
        return _LIBS


def _check(res, what):
    if res != 0:
        cuda = _LIBS.get("cuda")
        msg = ctypes.c_char_p()
        if cuda is not None:
            cuda.cuGetErrorString(int(res), ctypes.byref(msg))
        raise RuntimeError(f"{what} failed: CUresult {res} "
                           f"({msg.value.decode() if msg.value else '?'})")


def _ensure_context(dev):
    """Make torch's primary context of ``dev`` current on this thread."""
    import torch
    torch.cuda.set_device(dev)
    torch.cuda.current_stream(dev)
    torch.empty(1, device=dev)                       # initialises the context
    cuda = _LIBS["cuda"]
    ctx = ctypes.c_void_p()
    _check(cuda.cuCtxGetCurrent(ctypes.byref(ctx)), "cuCtxGetCurrent")
    if not ctx.value:
        d = ctypes.c_int()
        _check(cuda.cuDeviceGet(ctypes.byref(d), int(dev.index or 0)), "cuDeviceGet")
        _check(cuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), d), "cuDevicePrimaryCtxRetain")
        _check(cuda.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")


def _compile(src: str, arch: int) -> bytes:
    nvrtc = _LIBS["nvrtc"]
    prog = ctypes.c_void_p()
    r = nvrtc.nvrtcCreateProgram(ctypes.byref(prog), src.encode(), b"epione_kernels.cu", 0, None, None)
    if r != 0:
        raise RuntimeError(f"nvrtcCreateProgram failed ({r})")
    try:
        def attempt(opt_arch, cubin):
            opts = [f"--gpu-architecture={opt_arch}".encode(), b"-default-device", b"--std=c++14"]
            arr = (ctypes.c_char_p * len(opts))(*opts)
            rr = nvrtc.nvrtcCompileProgram(prog, len(opts), arr)
            if rr != 0:
                n = ctypes.c_size_t()
                nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
                log = ctypes.create_string_buffer(n.value)
                nvrtc.nvrtcGetProgramLog(prog, log)
                raise RuntimeError(f"NVRTC compile ({opt_arch}) failed: {log.value.decode()[:2000]}")
            n = ctypes.c_size_t()
            if cubin:
                nvrtc.nvrtcGetCUBINSize(prog, ctypes.byref(n))
                buf = ctypes.create_string_buffer(n.value)
                nvrtc.nvrtcGetCUBIN(prog, buf)
            else:
                nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(n))
                buf = ctypes.create_string_buffer(n.value)
                nvrtc.nvrtcGetPTX(prog, buf)
            return buf.raw
        try:
            return attempt(f"sm_{arch}", True)
        except RuntimeError:
            # NVRTC older than the device: PTX for the newest arch it knows,
            # JIT-compiled by the driver
            cnt = ctypes.c_int()
            nvrtc.nvrtcGetNumSupportedArchs(ctypes.byref(cnt))
            archs = (ctypes.c_int * cnt.value)()
            nvrtc.nvrtcGetSupportedArchs(archs)
            ok = [a for a in archs if a <= arch] or list(archs)
            return attempt(f"compute_{max(ok)}", False)
    finally:
        nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))


class Module:
    """Compiled CUDA module with ``extern "C"`` kernels ``names``."""

    _cache: Dict[tuple, "Module"] = {}

    def __init__(self, src: str, names: Sequence[str], dev):
        import torch
        if not _libs():
            raise RuntimeError(f"CUDA runtime compilation unavailable: {_ERR}")
        _ensure_context(dev)
        cap = torch.cuda.get_device_capability(dev)
        image = _compile(src, cap[0] * 10 + cap[1])
        cuda = _LIBS["cuda"]
        self._mod = ctypes.c_void_p()
        _check(cuda.cuModuleLoadData(ctypes.byref(self._mod), ctypes.c_char_p(image)), "cuModuleLoadData")
        self._fn = {}
        for nm in names:
            f = ctypes.c_void_p()
            _check(cuda.cuModuleGetFunction(ctypes.byref(f), self._mod, nm.encode()),
                   f"cuModuleGetFunction({nm})")
            self._fn[nm] = f
        self.dev = dev

    @classmethod
    def get(cls, src: str, names: Sequence[str], dev) -> "Module":
        key = (hash(src), tuple(names), str(dev))
        m = cls._cache.get(key)
        if m is None:
            m = cls._cache[key] = cls(src, names, dev)
        return m

    def launch(self, name: str, grid: int, block: int, args: Sequence[ctypes._SimpleCData]):
        """Launch ``name`` on torch's current stream. ``args`` are ctypes
        scalars (``c_void_p`` for device pointers)."""
        import torch
        stream = ctypes.c_void_p(torch.cuda.current_stream(self.dev).cuda_stream)
        params = (ctypes.c_void_p * len(args))(
            *[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
        _check(_LIBS["cuda"].cuLaunchKernel(self._fn[name], int(grid), 1, 1, int(block), 1, 1, 0,
                                            stream, params, None), f"cuLaunchKernel({name})")


def available(dev) -> bool:
    """Whether kernels can be compiled and launched on ``dev``."""
    try:
        Module.get('extern "C" __global__ void epi_probe(double* x) { x[0] = 1.0; }',
                   ["epi_probe"], dev)
        return True
    except Exception as e:  # pragma: no cover - depends on the machine
        global _ERR
        _ERR = _ERR or repr(e)
        return False


def last_error() -> Optional[str]:
    return _ERR


def memcpy_htod_async(dst_ptr: int, src_ptr: int, nbytes: int, stream) -> None:
    """Asynchronous host -> device copy on ``stream`` (a torch stream). Truly
    asynchronous only for page-locked (registered) host memory."""
    _check(_LIBS["cuda"].cuMemcpyHtoDAsync_v2(ctypes.c_uint64(dst_ptr), ctypes.c_void_p(src_ptr),
                                               ctypes.c_size_t(nbytes),
                                               ctypes.c_void_p(stream.cuda_stream)),
           "cuMemcpyHtoDAsync")


def memcpy_dtoh_async(dst_ptr: int, src_ptr: int, nbytes: int, stream) -> None:
    """Asynchronous device -> host copy on ``stream`` (a torch stream)."""
    _check(_LIBS["cuda"].cuMemcpyDtoHAsync_v2(ctypes.c_void_p(dst_ptr), ctypes.c_uint64(src_ptr),
                                               ctypes.c_size_t(nbytes),
                                               ctypes.c_void_p(stream.cuda_stream)),
           "cuMemcpyDtoHAsync")


def host_register(ptr: int, nbytes: int) -> bool:
    """Page-lock an existing host buffer in place (no copy); False if the
    driver refuses (then copies are staged by the driver, ~4x slower)."""
    if nbytes <= 0:
        return False
    r = _LIBS["cuda"].cuMemHostRegister_v2(ctypes.c_void_p(ptr), ctypes.c_size_t(nbytes), 0)
    return r == 0


def host_unregister(ptr: int) -> None:
    _LIBS["cuda"].cuMemHostUnregister(ctypes.c_void_p(ptr))
