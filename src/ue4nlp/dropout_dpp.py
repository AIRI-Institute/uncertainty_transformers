import torch
from alpaca.uncertainty_estimator.masks import build_mask
from .dppmask_ext import build_mask_ext

from .dropout_mc import DropoutMC
import numpy as np

import time
import datetime
import random
import os

import logging

log = logging.getLogger(__name__)


class DropoutDPP_v3(DropoutMC):
    dropout_id = -1

    @classmethod
    def update(cls):
        cls.dropout_id += 1
        return cls.dropout_id
    
    def __init__(
        self,
        p: float,
        activate=False,
        mask_name="ht_dpp",
        max_n=100,
        max_frac=0.4,
        coef=1.0,
        is_reused_mask=False,
        inference_step=0,
        mask_name_for_mask="rbf",
        calib_temp=1.
    ):
        super().__init__(p=p, activate=activate)
        self.curr_dropout_id = DropoutDPP_v3.update()

        self.mask = (
            build_mask_ext(mask_name)
            if mask_name != "dpp"
            else build_mask_ext(mask_name)["dpp"]
        )
        self.max_n = max_n
        self.max_frac = max_frac
        self.coef = coef
        self.calib_temp = calib_temp
        self.init_change_mask = 0

        self.is_reused_mask = is_reused_mask
        if self.is_reused_mask:
            self.saved_masks = []
            self.calib_temps = []
            self.dpp_masks = (
                build_mask_ext(mask_name_for_mask)
                if mask_name_for_mask != "dpp"
                else build_mask_ext(mask_name_for_mask)["dpp"]
            )
            self.inference_step = inference_step
            self.used_mask_id = 0
            self.diverse_masks = None

        log.debug(f"Dropout id: {self.curr_dropout_id}")

    def _get_mask(self, x: torch.Tensor):
        if x.dim() == 2:
            return self.mask(
                x, dropout_rate=self.p, layer_num=self.curr_dropout_id
            ).float()

        return self.mask(
            x.view(x.shape[0] * x.shape[1], -1),
            dropout_rate=self.p,
            layer_num=self.curr_dropout_id,
        ).float()  # [None, None, :]

    def _calc_non_zero_neurons(self, sum_mask):
        frac_nonzero = (sum_mask != 0).sum(axis=-1).item() / sum_mask.shape[-1]
        return frac_nonzero

    def _predict_with_sampled_mask(self, x: torch.Tensor):
        sum_mask = self._get_mask(x)
        norm = 1.0
        i = 1
        frac_nonzero = self._calc_non_zero_neurons(sum_mask)
        while i < self.max_n and frac_nonzero < self.max_frac:
            mask = self._get_mask(x)

            # sum_mask = self.coef * sum_mask + mask
            sum_mask += mask
            i += 1
            # norm = self.coef * norm + 1

            frac_nonzero = self._calc_non_zero_neurons(sum_mask)
            log.debug(
                f"==========Non zero neurons: {frac_nonzero} iter: {i}*****************"
            )

        log.debug(f"Number of averaged DPP masks: {i}")

        sum_mask /= i
        # sum_mask /= norm
        res = x * sum_mask

        if self.is_reused_mask:
            self.saved_masks.append(sum_mask.cpu())

        return res
    
    def construct_pool_of_masks(self, sampling=True):
        self.saved_masks = torch.stack(self.saved_masks).T
        self.saved_masks_clean = self.saved_masks.clone()

        if sampling:
            n = 7  # TODO:
            mask_indices = torch.zeros(self.saved_masks.shape[1])
            for i in range(n):
                msk_idx = self.dpp_masks(
                    self.saved_masks,
                    dropout_rate=self.p,
                    layer_num=self.curr_dropout_id,
                ).float()

            self.diverse_masks = self.saved_masks_clean[:, msk_idx > 0]
        else:
            self.diverse_masks = self.saved_masks_clean

        max_n = 200
        self.diverse_masks = self.diverse_masks[:, :max_n]
        if not len(self.calib_temps):
            self.calib_temps = [1.] * self.diverse_masks.shape[1]

        log.debug(f"\n\nself.diverse_masks: {self.diverse_masks.shape}")

        self.used_mask_id = 0
    
    def get_calib_temp(self):
        return self.calib_temps[self.used_mask_id] if self.is_reused_mask and self.inference_step else self.calib_temp
    
    def change_mask(self, mask_id=None, on_calibration=False):
        if mask_id is not None:
            assert self._used_mask_id < self.diverse_masks.shape[1]
            
            self.used_mask_id = mask_id
            return mask_id

        if on_calibration:
            self.init_change_mask = 1
        else:
            self.used_mask_id += 1 
            self.used_mask_id %= self.diverse_masks.shape[1]
        return self.used_mask_id
    
    def _predict_with_reused_mask(self, x: torch.Tensor):
        if self.diverse_masks is None:
            self.construct_pool_of_masks()
            
        mask = self.diverse_masks[:, self.used_mask_id].to(device=x.device)
        if self.init_change_mask:
            self.change_mask(on_calibration=False)
            self.init_change_mask = 0
        return x * mask
        
    def forward(self, x: torch.Tensor):
        if self.training:
            return torch.nn.functional.dropout(x, self.p, training=True)
        
        else:
            if not self.activate:
                return x
            
            if self.is_reused_mask and self.inference_step:
                return self._predict_with_reused_mask(x)

            else:
                return self._predict_with_sampled_mask(x)