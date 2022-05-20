import torch.nn as nn
from .mixout_linear import MixLinear
import re


def replace_dropout_with_mixout(model, lmbd=0.9, prefix=""):
    for i, (layer_name, layer) in enumerate(list(model.named_children())):
        if isinstance(layer, nn.Dropout) and "encoder" in prefix:
            name = list(model._modules.items())[i][0]
            model._modules[name] = nn.Dropout(p=0)
        elif isinstance(layer, nn.Linear):
            name = list(model._modules.items())[i][0]

            mod_name = prefix + "." + name
            if re.match(
                "\.bert\.encoder\.layer\.\d+\..*\.(dense|query|key|value)", mod_name
            ):
                # if re.match('\.bert\.encoder\.layer\.\d+\.(output|attention\..*)\.(dense|query|key|value)', mod_name): # TODO: Fix
                target_state_dict = layer.state_dict()
                bias = True if layer.bias is not None else False
                new_module = MixLinear(
                    layer.in_features,
                    layer.out_features,
                    bias,
                    target_state_dict["weight"],
                    lmbd,
                )
                new_module.load_state_dict(target_state_dict)
                model._modules[name] = new_module
        else:
            replace_dropout_with_mixout(
                layer, lmbd=lmbd, prefix=prefix + "." + layer_name
            )
