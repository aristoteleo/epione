"""seq2PRINT training targets: multi-scale footprint z-scores in torch.

seq2PRINT is trained to predict, from DNA alone, the PRINT footprint
z-score at every (scale, position) of a 1 kb window -- computed from the
observed insertions and the Tn5 bias with PRINT's dispersion model.
:func:`epione.tl.print_footprint_score` computes the same statistic in
numpy for analysis; this is scPrinter's batched torch version, which is
what training uses and differs on purpose:

* it returns z-scores, not -log10 p, and does not smooth;
* z is clipped to [-10, 10] and positions where it is undefined are NaN
  (the loss masks them);
* the first and last 100 bp are dropped, leaving the 800 bp the model
  predicts.

``dispModel`` and ``multiscaleFoot`` below are vendored verbatim from
scPrinter (``scprinter/seq/minimum_footprint.py`` at commit ``ed7cafb``,
MIT licence, Copyright (c) 2023 Broad Institute): window sums are
convolutions and the MLP runs batched across scales, so a re-derivation
would round differently from the targets scPrinter trains on. Use
:func:`footprint_target_model` to build one from Epione's
:class:`~epione.tl.DispersionModel` instead of scPrinter's loader.
"""
# fmt: off
import numpy as np
import torch
import torch.nn as nn

__all__ = ["dispModel", "footprint_target_model", "multiscaleFoot"]


def footprint_target_model(dispersion=None) -> "dispModel":
    """Torch footprint-target module from a :class:`~epione.tl.DispersionModel`.

    Converts Epione's loaded model into the dict layout scPrinter's
    ``loadDispModel`` returns -- the same arrays, weights as float32
    tensors -- and wraps it.
    """
    from epione.tl._print import DispersionModel

    if dispersion is None:
        dispersion = DispersionModel.load()
    legacy = {}
    for radius, p in dispersion.params.items():
        legacy[str(radius)] = {
            "featureMean": p["featureMean"], "featureSD": p["featureSD"],
            "targetMean": p["targetMean"], "targetSD": p["targetSD"],
            "modelWeights": [torch.from_numpy(np.asarray(w)).float() for w in p["weights"]],
        }
    return dispModel(legacy)


class dispModel(nn.Module):
    """
    A torch module that wraps the dispersion model in the old format into a torch module
    such that it would takes the atac and bias and directly do the footprints

    Parameters
    ----------
    dispmodels:
        The pretrained dispersion models loaded in scPrinter.
    """

    def __init__(self, dispmodels=None):
        super().__init__()
        if dispmodels is not None:
            self.initialize(dispmodels)

    def initialize(self, dispmodels):
        self.models = []

        self.scale1_weight_all = []
        self.scale1_bias_all = []
        self.linear1_weight_all = []
        self.linear1_bias_all = []
        self.linear2_weight_all = []
        self.linear2_bias_all = []
        self.scale2_weight_all = []
        self.scale2_bias_all = []

        for i in range(2, 101, 1):
            key = str(i)
            # The standard way of normalization is to have the feature minus the mean divided by the standard deviation
            # The nn.Linear would do the feature transformation by the formula y = x * weight + bias
            # So the equivalent transformation is to have y = x * (1/std) + (-mean/std) = (x - mean) / std
            # divided by the standard deviation, and then minus the mean / deviation
            weight = torch.diag(1 / torch.from_numpy(dispmodels[key]["featureSD"])).float()
            bias = (
                -torch.from_numpy(dispmodels[key]["featureMean"]).float()
                / torch.from_numpy(dispmodels[key]["featureSD"]).float()
            )
            scale1 = nn.Linear(weight.shape[1], weight.shape[0])
            scale1.weight.data = weight
            scale1.bias.data = bias

            # The target normalization is to have the target multiply by the standard deviation and then add the mean
            weight = torch.diag(torch.from_numpy(dispmodels[key]["targetSD"])).float()
            bias = torch.from_numpy(dispmodels[key]["targetMean"]).float()
            scale2 = nn.Linear(weight.shape[1], weight.shape[0])
            scale2.weight.data = weight
            scale2.bias.data = bias

            # Now tansform the linear layer in the dispersion model into a torch module
            linear1 = nn.Linear(
                dispmodels[key]["modelWeights"][0].shape[1],
                dispmodels[key]["modelWeights"][0].shape[0],
            )
            linear1.weight.data = dispmodels[key]["modelWeights"][0].float()
            linear1.bias.data = dispmodels[key]["modelWeights"][1].float()

            linear2 = nn.Linear(
                dispmodels[key]["modelWeights"][2].shape[1],
                dispmodels[key]["modelWeights"][2].shape[0],
            )
            linear2.weight.data = dispmodels[key]["modelWeights"][2].float()
            linear2.bias.data = dispmodels[key]["modelWeights"][3].float()

            self.scale1_weight_all.append(scale1.weight.data.T)
            self.scale1_bias_all.append(scale1.bias.data[None])
            self.linear1_weight_all.append(linear1.weight.data.T)
            self.linear1_bias_all.append(linear1.bias.data[None])
            self.linear2_weight_all.append(linear2.weight.data.T)
            self.linear2_bias_all.append(linear2.bias.data[None])
            self.scale2_weight_all.append(scale2.weight.data.T)
            self.scale2_bias_all.append(scale2.bias.data[None])

            self.models.append(nn.Sequential(scale1, linear1, nn.ReLU(), linear2, scale2))
        self.models = nn.ModuleList(self.models)

        # The weights and bias are also stacked here for the usage in the footprint function
        # The advantages are that this way one can specify the scales to be used in the footprint function
        # And the computation would still happen in parallel, not in a for loop for modes
        self.scale1_weight_all = nn.Parameter(torch.stack(self.scale1_weight_all, dim=0))
        self.scale1_bias_all = nn.Parameter(torch.stack(self.scale1_bias_all, dim=0))
        self.linear1_weight_all = nn.Parameter(torch.stack(self.linear1_weight_all, dim=0))
        self.linear1_bias_all = nn.Parameter(torch.stack(self.linear1_bias_all, dim=0))
        self.linear2_weight_all = nn.Parameter(torch.stack(self.linear2_weight_all, dim=0))
        self.linear2_bias_all = nn.Parameter(torch.stack(self.linear2_bias_all, dim=0))
        self.scale2_weight_all = nn.Parameter(torch.stack(self.scale2_weight_all, dim=0))
        self.scale2_bias_all = nn.Parameter(torch.stack(self.scale2_bias_all, dim=0))

        # pretrained, no need to be optimized
        for param in self.parameters():
            param.requires_grad = False

        # The smooth kernels are used for the windowsum
        # smooth_kernels are the ones for left and right window sum
        # smooth_kernels_center are the ones for the center window sum
        self.smooth_kernels = torch.zeros((99, 1, 100)).float()
        for i in range(99):
            size = int((i + 2) / 2)
            self.smooth_kernels[i, :, 50 - size : 50 + size] = 1
        self.smooth_kernels = nn.Parameter(self.smooth_kernels, requires_grad=False)

        self.smooth_kernels_center = torch.zeros((99, 1, 100 * 2)).float()
        for i in range(99):
            size = i + 2
            self.smooth_kernels_center[i, :, 100 - size : 100 + size] = 1
        self.smooth_kernels = nn.Parameter(self.smooth_kernels, requires_grad=False)
        self.smooth_kernels_center = nn.Parameter(self.smooth_kernels_center, requires_grad=False)

    def windowsum(self, x, modes):
        shapes = list(x.shape)
        width = shapes[-1]

        if len(shapes) == 1:
            x = x[None, None, :]
        else:
            x = x.reshape((-1, 1, shapes[-1]))

        smooth_kernels = self.smooth_kernels[modes - 2]
        smooth_kernels_center = self.smooth_kernels_center[modes - 2]

        # This replaces the rz_conv_torch
        a = torch.nn.functional.conv1d(
            x,
            smooth_kernels,
            torch.zeros((len(modes)), device=x.device, dtype=x.dtype),
            padding=200,
        )[..., 1:]

        # Now we need to slice the tensor to get the left and right window sum
        l = []
        r = []
        for i, mode in enumerate(modes):
            ori_padding = int(mode / 2) + mode
            extra = 150 - ori_padding
            l.append(a[..., i, extra : extra + width])
            if extra == 0:
                r.append(a[..., i, -extra - width :])
            else:
                r.append(a[..., i, -extra - width : -extra])
        l = torch.stack(l, dim=-2)
        r = torch.stack(r, dim=-2)

        # This replaces the rz_conv_torch to get the center window sum
        c = torch.nn.functional.conv1d(
            x,
            smooth_kernels_center,
            torch.zeros((len(modes)), device=x.device, dtype=x.dtype),
            padding=100,
        )[..., :, 1:]

        # If the input is a single insertion track, return without adding first dimension
        if len(shapes) == 1:
            l, c, r = l[0], c[0], r[0]

        else:
            shapes = shapes[:-1] + [len(modes), width]
            l, c, r = l.reshape(shapes), c.reshape(shapes), r.reshape(shapes)

        return l, c, r

    def footprint(self, atac, bias, modes, clip_min=-10, clip_max=10):
        """
        The footprint function that takes the atac and bias and return the footprint scores

        Parameters
        ----------
        atac: torch.Tensor
            The atac insertion signal
        bias: torch.Tensor
            The bias signal. Has to be the same shape as atac
        modes: torch.Tensor
            The scales to be used in the footprint function
        clip_min: float
            The minimum value to be clipped
        clip_max: float
            The maximum value to be clipped

        Returns
        -------
        zs: torch.Tensor
            The footprint scores as z-scores

        """
        # Bias and atac go through the same window sum process, might as well stack them for simplicity
        bias_and_atac = torch.cat([bias, atac], dim=0)
        bulk_l, bulk_c, bulk_r = self.windowsum(bias_and_atac, modes)
        biasWindowSums_l_bulk, biasWindowSums_c_bulk, biasWindowSums_r_bulk = (
            bulk_l[: len(bias)],
            bulk_c[: len(bias)],
            bulk_r[: len(bias)],
        )
        (
            insertionWindowSums_l_bulk,
            insertionWindowSums_c_bulk,
            insertionWindowSums_r_bulk,
        ) = (bulk_l[len(bias) :], bulk_c[len(bias) :], bulk_r[len(bias) :])
        leftTotalInsertion_bulk = insertionWindowSums_c_bulk + insertionWindowSums_l_bulk
        rightTotalInsertion_bulk = insertionWindowSums_c_bulk + insertionWindowSums_r_bulk
        fgFeatures_bulk = torch.stack(
            [
                biasWindowSums_l_bulk,
                biasWindowSums_r_bulk,
                biasWindowSums_c_bulk,
                torch.log10(leftTotalInsertion_bulk),
                torch.log10(rightTotalInsertion_bulk),
            ],
            dim=-1,
        )  # shape of [..., scales, length, 5]
        shapes = list(fgFeatures_bulk.shape)
        num_dims = len(shapes)
        # Create a tuple representing the new order of dimensions
        # The first (num_dims - 3) dimensions remain unchanged
        new_order = tuple(range(num_dims - 3)) + (
            num_dims - 2,
            num_dims - 3,
            num_dims - 1,
        )
        fgFeatures_bulk = fgFeatures_bulk.permute(
            *new_order
        )  # shapes of [...., length, scales, input]

        # Now find all the corresponding scales and do batch matmul
        scale1_weights = self.scale1_weight_all[modes - 2]  # shape of [scales, input, output]
        scale1_bias = self.scale1_bias_all[modes - 2]  # shape of [scales, output]
        linear1_weights = self.linear1_weight_all[modes - 2]
        linear1_bias = self.linear1_bias_all[modes - 2]
        linear2_weights = self.linear2_weight_all[modes - 2]
        linear2_bias = self.linear2_bias_all[modes - 2]
        scale2_weights = self.scale2_weight_all[modes - 2]
        scale2_bias = self.scale2_bias_all[modes - 2]

        # This is batched multiplication, basically for matrix A,B of shape(...,n,m), (...,m,p) -> (...,n,p)
        fgFeatures_bulk = fgFeatures_bulk[..., None, :]  # [..., length, scales, 1, input]
        missing_dim = int(len(fgFeatures_bulk.shape) - len(scale1_weights.shape))
        for _ in range(missing_dim):
            scale1_weights = scale1_weights[None]  # [..., 1, scales, input, output]
            linear1_weights = linear1_weights[None]
            linear2_weights = linear2_weights[None]
            scale2_weights = scale2_weights[None]

        # output = (
        #     torch.matmul(fgFeatures_bulk[..., None, :], scale1_weights[None, None]) + scale1_bias
        # )
        # output = torch.matmul(output, linear1_weights[None, None]) + linear1_bias
        # output = torch.nn.functional.relu(output)
        # output = torch.matmul(output, linear2_weights[None, None]) + linear2_bias
        # output = (
        #         torch.matmul(output, scale2_weights[None, None]) + scale2_bias
        # )  # shape of [..., length, scales, 1, 4]
        output = (
            torch.matmul(fgFeatures_bulk, scale1_weights) + scale1_bias
        )  # [..., length, scales, 1, output]
        output = torch.matmul(output, linear1_weights) + linear1_bias
        output = torch.nn.functional.relu(output)
        output = torch.matmul(output, linear2_weights) + linear2_bias
        output = (
            torch.matmul(output, scale2_weights) + scale2_bias
        )  # shape of [..., length, scales, 1, 4]

        predDispersion = output[..., 0, :].permute(
            *new_order
        )  # First to [..., length, scales, 4] then [..., scales, length, 4]

        leftPredRatioMean = predDispersion[..., 0]  # shape of [..., scales, length]
        leftPredRatioSD = predDispersion[..., 1]
        rightPredRatioMean = predDispersion[..., 2]
        rightPredRatioSD = predDispersion[..., 3]

        leftPredRatioSD[leftPredRatioSD < 0] = 0
        rightPredRatioSD[rightPredRatioSD < 0] = 0

        fgLeftRatio = (
            insertionWindowSums_c_bulk / leftTotalInsertion_bulk
        )  # shape of [..., scales, length]
        fgRightRatio = insertionWindowSums_c_bulk / rightTotalInsertion_bulk

        leftZ = (fgLeftRatio - leftPredRatioMean) / leftPredRatioSD
        rightZ = (fgRightRatio - rightPredRatioMean) / (rightPredRatioSD)
        p = torch.maximum(leftZ, rightZ)

        # Mask positions with zero coverage on either flanking side
        p[
            (leftTotalInsertion_bulk < 1)
            | (rightTotalInsertion_bulk < 1)
            | (leftPredRatioSD == 0)
            | (rightPredRatioSD == 0)
        ] = float("nan")
        zs = p
        mask = torch.isinf(zs)
        zs[mask] = float("nan")
        if clip_min is not None and clip_max is not None:
            zs.clamp_(clip_min, clip_max)
        zs = zs[..., 100:-100]  # The first 100 and last 100 are not used as they are not stable
        return zs.float()

    def forward(self, x: torch.Tensor, i: int):
        # The forward function is not used in the footprint function
        x = self.models[i - 2](x)
        return x


@torch.no_grad()
def multiscaleFoot(atac, bias, modes, dispmodel, clip_min=-10, clip_max=10):
    """
    The multiscaleFoot function that takes the atac and bias and return the footprint scores

    Parameters
    ----------
    atac: torch.Tensor
        The atac insertion signal
    bias: torch.Tensor
        The bias signal. Has to be the same shape as atac
    modes: torch.Tensor
        The scales to be used in the footprint function
    dispmodel: torch.nn.Module
        The trained dispersion model
    clip_min: float
        The minimum value to be clipped
    clip_max: float
        The maximum value to be clipped

    Returns
    -------
    zs: torch.Tensor
        The footprint scores as z-scores


    """
    atac = atac.float()
    bias = bias.float()
    return dispmodel.footprint(atac, bias, modes, clip_min=clip_min, clip_max=clip_max)
