from typing import Dict

import torch

# from flair.nn import LockedDropout, WordDropout
from torch.nn.functional import dropout

from collections import Counter

from typing import Iterable, Union


class LinearDropConnectMC(torch.nn.Module):
    def __init__(self, linear, p_dropconnect=0.0, activate=False):
        super().__init__()
        self.linear = linear
        self.p_dropconnect = p_dropconnect
        self.activate = activate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.linear.weight
        bias = self.linear.bias
        p_dc = self.p_dropconnect
        return torch.nn.functional.linear(
            x,
            torch.nn.functional.dropout(
                weight, p=p_dc, training=(self.training or self.activate)
            ),
            bias,
        )


def convert_to_mc_dropconnect(
    model: torch.nn.Module, substitution_dict: Dict[str, torch.nn.Module] = None
):
    for i, layer in enumerate(list(model.children())):
        module_name = list(model._modules.items())[i][0]
        layer_name = layer._get_name()
        if layer_name in substitution_dict.keys():
            model._modules[module_name] = substitution_dict[layer_name](
                linear=layer, activate=False
            )
        else:
            convert_to_mc_dropconnect(model=layer, substitution_dict=substitution_dict)


def hide_dropout(model: torch.nn.Module, p: float = 0.0, verbose: bool = True):
    for layer in model.children():
        if isinstance(layer, torch.nn.Dropout):
            if verbose:
                print(layer)
                print(f"Current DO state: {layer.p}")
                print(f"Switching state to: {p}")
            layer.p = p
        else:
            hide_dropout(model=layer, p=p, verbose=verbose)


def activate_mc_dropconnect(
    model: torch.nn.Module, activate: bool, random: float = 0.0, verbose: bool = False
):
    for layer in model.children():
        if isinstance(layer, LinearDropConnectMC):
            if verbose:
                print(layer)
                print(f"Current DO state: {layer.activate}")
                print(f"Switching state to: {activate}")
            layer.activate = activate
            if activate and random:
                layer.p_dropconnect = random
        else:
            activate_mc_dropconnect(
                model=layer, activate=activate, random=random, verbose=verbose
            )
