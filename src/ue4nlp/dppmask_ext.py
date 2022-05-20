from alpaca.uncertainty_estimator.masks import DPPMask
from alpaca.uncertainty_estimator.masks import build_masks

from dppy.finite_dpps import FiniteDPP

import numpy as np
import torch

from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics.pairwise import cosine_similarity

import logging

log = logging.getLogger()


class DPPMaskExt(DPPMask):
    def __init__(self, ht_norm=False, covariance=False, kernel_type="corr"):
        super().__init__(ht_norm, covariance)
        self._kernel_type = kernel_type

    def _setup_dpp(self, x_matrix, layer_num):
        log.info("Custom dpp mask")
        # if len(x_matrix.shape) == 4:
        #    x_matrix = x_matrix[:, 0, 0, :]
        # if len(x_matrix.shape) == 3:
        #    x_matrix = x_matrix[:, 0, :]
        # print(f'x_matrix.shape is {x_matrix.shape}')
        self.x_matrix = x_matrix
        micro = 1e-12
        x_matrix += (
            np.random.random(x_matrix.shape) * micro
        )  # for computational stability

        if self._kernel_type == "cov":
            L = np.cov(x_matrix.T)

        elif self._kernel_type == "corr":
            L = np.corrcoef(x_matrix.T)

        elif self._kernel_type == "rbf":
            rbf = RBF(length_scale=16)
            L = rbf(x_matrix.T, x_matrix.T)

        elif self._kernel_type == "cosine":
            L = cosine_similarity(x_matrix.T, dense_output=True)  # TODO: does not work

        else:
            raise ValueError(f"Wrong kernel type: {self._kernel_type}")

        self.dpps[layer_num] = FiniteDPP("likelihood", **{"L": L})
        self.layer_correlations[layer_num] = L

        if self.ht_norm:
            L = torch.DoubleTensor(L).cuda()
            I = torch.eye(len(L)).double().cuda()
            K = torch.mm(L, torch.inverse(L + I))

            self.norm[layer_num] = torch.reciprocal(
                torch.diag(K)
            )  # / len(correlations)
            self.L = L
            self.K = K


def build_mask_ext(mask_name):
    if mask_name == "ht_cosine":
        return DPPMaskExt(ht_norm=True, kernel_type="cosine")
    elif mask_name == "ht_dpp":
        return DPPMaskExt(ht_norm=True, kernel_type="corr")
    elif mask_name == "ht_rbf":
        return DPPMaskExt(ht_norm=True, kernel_type="rbf")
    elif mask_name == "rbf":
        return DPPMaskExt(ht_norm=False, kernel_type="rbf")
    elif mask_name == "cosine":
        return DPPMaskExt(ht_norm=False, kernel_type="cosine")
    elif mask_name == "corr":
        return DPPMaskExt(ht_norm=False, kernel_type="corr")
    elif mask_name == "cov":
        return DPPMaskExt(ht_norm=False, kernel_type="cov")
    else:
        return build_masks([mask_name])
