"""Load official ChromBPNet ``.h5`` files into the PyTorch modules.

Reads the Keras HDF5 format with :mod:`h5py` only -- TensorFlow is never
imported. The file's ``model_config`` attribute (JSON) describes the layer
graph; the architecture (filters, number of dilated layers, input and
output length) is read from it, and each Conv1D/Dense kernel is mapped to
its PyTorch counterpart:

* Keras ``Conv1D`` kernel ``(k, in, out)`` -> torch ``(out, in, k)``
  (both frameworks compute cross-correlation, so no flip);
* Keras ``Dense`` kernel ``(in, out)`` -> torch ``(out, in)``.

Three official file kinds are supported, as released by ENCODE and written
by ``chrombpnet train``:

* a single BPNet (``model.bias_scaled.*.h5``, ``model.chrombpnet_nobias.*.h5``)
  -> :class:`~._model.BPNet`;
* the full model (``model.chrombpnet.*.h5``), whose bias and no-bias parts
  are nested Keras models -> :class:`~._model.ChromBPNet`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Union

import numpy as np

__all__ = ["load_keras_h5", "bpnet_layout"]


def _model_config(f) -> dict:
    raw = f.attrs["model_config"]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _inbound(layer: dict) -> List[str]:
    """Names of the layers feeding ``layer`` (Keras 2 functional config)."""
    out = []
    for node in layer.get("inbound_nodes", []):
        for item in node:
            if isinstance(item, list):
                out.append(item[0])
            elif isinstance(item, dict):  # Keras 3 style
                for arg in item.get("args", []):
                    hist = arg.get("config", {}).get("keras_history")
                    if hist:
                        out.append(hist[0])
    return out


def bpnet_layout(layers: List[dict]) -> Dict:
    """Read a BPNet architecture from the layer list of a Keras config.

    Returns the conv/dense layer names in network order and the
    hyper-parameters needed to build :class:`~._model.BPNet`.
    """
    by_name = {L["config"]["name"]: L for L in layers}
    convs = [L for L in layers if L["class_name"] == "Conv1D"]
    dense = [L for L in layers if L["class_name"] == "Dense"]
    if len(dense) != 1:
        raise ValueError(f"expected one Dense layer, found {len(dense)}")
    first = [L for L in convs if L["config"]["kernel_size"][0] != 3
             and L["config"]["filters"] != 1]
    prof = [L for L in convs if L["config"]["filters"] == 1]
    dil = [L for L in convs if L["config"]["kernel_size"][0] == 3
           and L["config"]["filters"] != 1]
    if len(first) != 1 or len(prof) != 1:
        raise ValueError("not a BPNet: expected one stem and one profile Conv1D")
    dil.sort(key=lambda L: L["config"]["dilation_rate"][0])
    rates = [L["config"]["dilation_rate"][0] for L in dil]
    if rates != [2 ** i for i in range(1, len(dil) + 1)]:
        raise ValueError(f"unexpected dilation rates {rates}")

    inputs = [L for L in layers if L["class_name"] == "InputLayer"]
    shape = inputs[0]["config"].get("batch_input_shape") or inputs[0]["config"].get("batch_shape")
    input_len = int(shape[1])

    # The profile crop: the Cropping1D fed by the profile convolution.
    prof_name = prof[0]["config"]["name"]
    crops = [L for L in layers if L["class_name"] == "Cropping1D"
             and prof_name in _inbound(L)]
    stem_k = first[0]["config"]["kernel_size"][0]
    prof_k = prof[0]["config"]["kernel_size"][0]
    L_ = input_len - (stem_k - 1) - sum(2 * r for r in rates) - (prof_k - 1)
    if crops:
        c = crops[0]["config"]["cropping"]
        c = c[0] if isinstance(c, (list, tuple)) else c
        output_len = L_ - 2 * int(c)
    else:
        output_len = L_
    return {
        "stem": first[0]["config"]["name"],
        "dilated": [L["config"]["name"] for L in dil],
        "profile": prof_name,
        "dense": dense[0]["config"]["name"],
        "kwargs": dict(
            filters=int(first[0]["config"]["filters"]),
            n_dil_layers=len(dil),
            input_len=input_len,
            output_len=int(output_len),
            conv1_kernel_size=int(stem_k),
            profile_kernel_size=int(prof_k),
        ),
        "_by_name": by_name,
    }


def _find_weights(group, layer_name: str) -> Dict[str, np.ndarray]:
    """kernel/bias arrays for ``layer_name`` anywhere under ``group``."""
    found: Dict[str, np.ndarray] = {}

    def visit(name, obj):
        parts = name.split("/")
        if hasattr(obj, "shape") and layer_name in parts[:-1]:
            key = parts[-1].split(":")[0]
            if key in found:
                raise ValueError(f"layer {layer_name!r} is ambiguous in the h5 file")
            found[key] = obj[()]

    group.visititems(visit)
    if "kernel" not in found:
        raise KeyError(f"no weights for layer {layer_name!r}")
    return found


def _fill_bpnet(layout: Dict, weights_group):
    import torch

    from ._model import BPNet

    net = BPNet(**layout["kwargs"])

    def conv(mod, name):
        w = _find_weights(weights_group, name)
        mod.weight.data = torch.from_numpy(np.ascontiguousarray(
            np.transpose(w["kernel"], (2, 1, 0)))).float()
        mod.bias.data = torch.from_numpy(w["bias"]).float()

    conv(net.conv1, layout["stem"])
    for mod, name in zip(net.dil_convs, layout["dilated"]):
        conv(mod, name)
    conv(net.prof_conv, layout["profile"])
    w = _find_weights(weights_group, layout["dense"])
    net.count_dense.weight.data = torch.from_numpy(np.ascontiguousarray(w["kernel"].T)).float()
    net.count_dense.bias.data = torch.from_numpy(w["bias"]).float()
    return net.eval()


def load_keras_h5(path: Union[str, Path]):
    """Load an official ChromBPNet/BPNet ``.h5`` into PyTorch.

    Returns a :class:`~._model.BPNet` for single models (bias, no-bias) and
    a :class:`~._model.ChromBPNet` for the full ``model.chrombpnet`` file.
    No TensorFlow is needed.
    """
    import h5py

    from ._model import ChromBPNet

    with h5py.File(path, "r") as f:
        cfg = _model_config(f)
        layers = cfg["config"]["layers"]
        wg = f["model_weights"] if "model_weights" in f else f
        nested = [L for L in layers if L["class_name"] in ("Functional", "Model")]
        if not nested:
            return _fill_bpnet(bpnet_layout(layers), wg)

        if len(nested) != 2:
            raise ValueError(f"expected 2 nested models, found {len(nested)}")
        parts = {}
        for sub in nested:
            name = sub["config"]["name"]
            layout = bpnet_layout(sub["config"]["layers"])
            grp = wg[name] if name in wg else wg
            parts[name] = _fill_bpnet(layout, grp)
        # Upstream names the no-bias sub-model "model_wo_bias".
        acc_name = next((n for n in parts if "wo_bias" in n), None)
        if acc_name is None:
            raise ValueError(f"cannot tell bias from accessibility model in {list(parts)}")
        bias_name = next(n for n in parts if n != acc_name)
        return ChromBPNet(parts[bias_name], parts[acc_name]).eval()
