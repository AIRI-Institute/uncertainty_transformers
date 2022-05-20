import torch
from alpaca.uncertainty_estimator.masks import build_mask
from .dppmask_ext import build_mask_ext

from .dropout_mc import DropoutMC
import numpy as np

import time
import datetime
import random

import logging

log = logging.getLogger(__name__)
import os


class DropoutDPP(DropoutMC):
    dropout_id = -1

    def __init__(
        self,
        p: float,
        activate=False,
        mask_name="dpp",
        max_n=100,
        max_frac=0.4,
        coef=1.0,
    ):
        super().__init__(p=p, activate=activate)

        self.mask = (
            build_mask_ext(mask_name)
            if mask_name != "dpp"
            else build_mask_ext(mask_name)["dpp"]
        )
        self.reset_mask = False
        self.max_n = max_n
        self.max_frac = max_frac
        self.coef = coef

        self.curr_dropout_id = DropoutDPP.update()

        log.debug(f"Dropout id: {self.curr_dropout_id}")

        # print(self.mask)

    @classmethod
    def update(cls):
        cls.dropout_id += 1
        return cls.dropout_id

    def calc_mask(self, x: torch.Tensor):
        return self.mask(x, dropout_rate=self.p, layer_num=self.curr_dropout_id).float()

    def get_mask(self, x: torch.Tensor):
        return self.mask(x, dropout_rate=self.p, layer_num=self.curr_dropout_id).float()

    #         if not self.reset_mask:
    #             if len(x.shape) == 2:
    #                 return self.calc_mask(x)
    #             else:
    #                 mask = self.calc_mask(x.reshape(-1, x.shape[-1]))
    #                 while len(mask.shape) < len(x.shape):
    #                     mask = mask.unsqueeze(dim=0)

    #                 return mask
    #         else:
    #             self.mask.reset()

    #             self.calc_mask(x) # dry run
    #             for _ in range(random.randint(0, 100)):
    #                 self.calc_mask(x)

    #             return self.calc_mask(x)

    def calc_non_zero_neurons(self, sum_mask):
        # print('Trash=========================')
        # print(sum_mask)
        frac_nonzero = (sum_mask != 0).sum(axis=-1).item() / sum_mask.shape[-1]
        return frac_nonzero

    def forward(self, x: torch.Tensor):
        if self.training:
            return torch.nn.functional.dropout(x, self.p, training=True)
        else:
            if not self.activate:
                return x

            sum_mask = self.get_mask(x)
            # print('Mask')
            # print(sum_mask)

            norm = 1.0
            i = 1
            frac_nonzero = self.calc_non_zero_neurons(sum_mask)
            # print('==========Non zero neurons:', frac_nonzero, 'iter:', i, 'id:', self.curr_dropout_id, '******************')
            # while i < 30:
            while i < self.max_n and frac_nonzero < self.max_frac:
                # while frac_nonzero < self.max_frac:
                mask = self.get_mask(x)

                # sum_mask = self.coef * sum_mask + mask
                sum_mask += mask
                i += 1
                # norm = self.coef * norm + 1

                frac_nonzero = self.calc_non_zero_neurons(sum_mask)
                # print('==========Non zero neurons:', frac_nonzero, 'iter:', i, '******************')

            # res = x * sum_mask / norm
            # print(sum_mask / i)
            # print('Number of masks:', i)
            res = x * sum_mask / i
            return res


class DropoutDPP_v2(DropoutMC):
    dropout_id = -1

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
    ):
        super().__init__(p=p, activate=activate)

        self.mask = (
            build_mask_ext(mask_name)
            if mask_name != "dpp"
            else build_mask_ext(mask_name)["dpp"]
        )
        self.reset_mask = False
        self.max_n = max_n
        self.max_frac = max_frac
        self.coef = coef

        self.curr_dropout_id = DropoutDPP_v2.update()

        self.is_reused_mask = is_reused_mask
        self.change_mask = 1

        if self.is_reused_mask:
            self.saved_masks = []
            self.dpp_masks = (
                build_mask_ext(mask_name_for_mask)
                if mask_name_for_mask != "dpp"
                else build_mask_ext(mask_name_for_mask)["dpp"]
            )
            self.inference_step = inference_step
            self.used_mask_id = 0
            self.diverse_masks = None

        log.debug(f"Dropout id: {self.curr_dropout_id}")

    @classmethod
    def update(cls):
        cls.dropout_id += 1
        return cls.dropout_id

    def calc_mask(self, x: torch.Tensor):
        return self.mask(x, dropout_rate=self.p, layer_num=self.curr_dropout_id).float()

    def get_mask(self, x: torch.Tensor):
        if x.dim() == 2:
            return self.mask(
                x, dropout_rate=self.p, layer_num=self.curr_dropout_id
            ).float()

        return self.mask(
            x.view(x.shape[0] * x.shape[1], -1),
            dropout_rate=self.p,
            layer_num=self.curr_dropout_id,
        ).float()  # [None, None, :]

    def calc_non_zero_neurons(self, sum_mask):
        frac_nonzero = (sum_mask != 0).sum(axis=-1).item() / sum_mask.shape[-1]
        return frac_nonzero

    def dry_run(self, sampling=True):
        self.saved_masks = torch.stack(self.saved_masks).T
        self.saved_masks_clean = self.saved_masks.clone()

        n = 7  # TODO:
        if sampling:
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

        max_n = 200  # TODO:
        self.diverse_masks = self.diverse_masks[:, :max_n]

        log.debug(f"\n\nself.diverse_masks: {self.diverse_masks.shape}")
        self.count_diverse_masks = self.diverse_masks.shape[1] - 1
        self.used_mask_id = 0


    def forward(self, x: torch.Tensor):
        if self.training:
            return torch.nn.functional.dropout(x, self.p, training=True)

        elif self.is_reused_mask and self.inference_step:
            if not self.activate:
                return x

            if self.diverse_masks is None:
                self.dry_run()

            mask = self.diverse_masks[:, self.used_mask_id].to(device=x.device)

            if self.change_mask:
                self.used_mask_id += 1
                if self.used_mask_id > self.diverse_masks.shape[1]:
                    self.used_mask_id = np.random.randint(
                        1, self.diverse_masks.shape[1]
                    )
                self.change_mask = 1 - self.change_mask

            return x * mask

        else:
            if not self.activate:
                return x

            sum_mask = self.get_mask(x)

            norm = 1.0
            i = 1
            frac_nonzero = self.calc_non_zero_neurons(sum_mask)
            while i < self.max_n and frac_nonzero < self.max_frac:
                mask = self.get_mask(x)

                # sum_mask = self.coef * sum_mask + mask
                sum_mask += mask
                i += 1
                # norm = self.coef * norm + 1

                frac_nonzero = self.calc_non_zero_neurons(sum_mask)
                log.debug(
                    f"==========Non zero neurons: {frac_nonzero} iter: {i}*****************"
                )

            # res = x * sum_mask / norm
            log.debug(f"Number of masks: {i}")

            res = x * sum_mask / i

            if self.is_reused_mask:
                mask_i = sum_mask / i
                mask_i = mask_i.to(device="cuda:0")
                self.saved_masks.append(mask_i)

            return res
