# modification of https://gist.github.com/rosinality/a96c559d84ef2b138e486acf27b5a56e#file-spect_norm-py

from torch.autograd import Variable
from torch import nn
import torch
from torch.nn.functional import normalize
from typing import Any, Optional, TypeVar
from torch.nn.modules import Module
from torch.nn.utils.spectral_norm import SpectralNorm, SpectralNormStateDictHook, SpectralNormLoadStateDictPreHook


# uncomment for version with more than 1 iteration
'''
# modify SpectralNorm class from pytorch - add scaling to new spectral norm
class ScaledSpectralNorm(SpectralNorm):
    def __init__(self, name: str = 'weight', n_power_iterations: int = 1, dim: int = 0, eps: float = 1e-12, sn_value: float = 1.0) -> None:
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps = eps
        self.sn_value = sn_value

    def compute_weight(self, module: Module, do_power_iteration: bool) -> torch.Tensor:
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        v = getattr(module, self.name + '_v')
        weight_mat = self.reshape_weight_to_matrix(weight)

        if do_power_iteration:
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    v = normalize(torch.mv(weight_mat.t(), u), dim=0, eps=self.eps, out=v)
                    u = normalize(torch.mv(weight_mat, v), dim=0, eps=self.eps, out=u)
                if self.n_power_iterations > 0:
                    # See above on why we need to clone
                    u = u.clone(memory_format=torch.contiguous_format)
                    v = v.clone(memory_format=torch.contiguous_format)

        sigma = torch.dot(u, torch.mv(weight_mat, v))
        if self.sn_value < sigma:
            weight = self.sn_value * weight / sigma
        return weight
    
    def apply(module: Module, name: str, n_power_iterations: int, dim: int, eps: float, sn_value: float) -> 'SpectralNorm':
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, SpectralNorm) and hook.name == name:
                raise RuntimeError("Cannot register two spectral_norm hooks on "
                                   "the same parameter {}".format(name))

        fn = ScaledSpectralNorm(name, n_power_iterations, dim, eps, sn_value)
        weight = module._parameters[name]

        with torch.no_grad():
            weight_mat = fn.reshape_weight_to_matrix(weight)

            h, w = weight_mat.size()
            # randomly initialize `u` and `v`
            u = normalize(weight.new_empty(h).normal_(0, 1), dim=0, eps=fn.eps)
            v = normalize(weight.new_empty(w).normal_(0, 1), dim=0, eps=fn.eps)

        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        # We still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an nn.Parameter and
        # gets added as a parameter. Instead, we register weight.data as a plain
        # attribute.
        setattr(module, fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_buffer(fn.name + "_v", v)

        module.register_forward_pre_hook(fn)
        module._register_state_dict_hook(SpectralNormStateDictHook(fn))
        module._register_load_state_dict_pre_hook(SpectralNormLoadStateDictPreHook(fn))
        return fn

T_module = TypeVar('T_module', bound=Module)

def spectral_norm(module: T_module,
                  name: str = 'weight',
                  n_power_iterations: int = 100,
                  eps: float = 1e-12,
                  sn_value: float = 1.0,
                  dim: Optional[int] = None) -> T_module:
    if dim is None:
        if isinstance(module, (torch.nn.ConvTranspose1d,
                               torch.nn.ConvTranspose2d,
                               torch.nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    ScaledSpectralNorm.apply(module, name, n_power_iterations, dim, eps, sn_value)
    return module
'''

# previous version
class SpectralNorm:
    def __init__(self, name, sn_value=1):
        self.name = name
        self.c = sn_value
        print(f'set spectral_norm = {self.c} \n')

    def compute_weight(self, module):
        weight = getattr(module, self.name + "_orig")
        u = getattr(module, self.name + "_u")
        size = weight.size()
        weight_mat = weight.contiguous().view(size[0], -1)
        if weight_mat.is_cuda:
            u = u.cuda()
        v = weight_mat.t() @ u
        v = v / v.norm()
        u = weight_mat @ v
        u = u / u.norm()
        spec_norm = u.t() @ weight_mat @ v
        if self.c < spec_norm:
            weight_sn = self.c * weight_mat / spec_norm
        else:
            weight_sn = weight_mat
        weight_sn = weight_sn.view(*size)

        return weight_sn, Variable(u.data)

    @staticmethod
    def apply(module, sn_value, name):
        fn = SpectralNorm(name, sn_value)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + "_orig", nn.Parameter(weight.data))
        input_size = weight.size(0)
        u = Variable(torch.randn(input_size, 1) * 0.1, requires_grad=False)
        setattr(module, name + "_u", u)
        setattr(module, name, fn.compute_weight(module)[0])

        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight_sn, u = self.compute_weight(module)
        setattr(module, self.name, weight_sn)
        setattr(module, self.name + "_u", u)


def spectral_norm(module, sn_value=1, name="weight"):
    print('\nuse custom spectral norm')
    SpectralNorm.apply(module, sn_value, name)

    return module
