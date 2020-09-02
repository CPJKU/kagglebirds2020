# -*- coding: utf-8 -*-

"""
Parameter initialization implementations applicable to different models.

Author: Jan Schlüter
"""

import contextlib
import fnmatch
import functools
import math

import torch.nn as nn


def conv_nn_resize_(tensor_or_module, subinit, stride=None):
    """
    Nearest neighbor resize initialization for strided convolution.
    https://arxiv.org/abs/1707.02937
    """
    if isinstance(tensor_or_module, nn.Module):
        tensor = tensor_or_module.weight.data
        stride = stride or tensor_or_module.stride
    else:
        tensor = tensor_or_module
    try:
        stride = tuple(stride)
    except TypeError:
        stride = (stride,) * (tensor.dim() - 2)
    if all(s == 1 for s in stride):
        return subinit(tensor)
    subtensor = tensor[(slice(None), slice(None)) +
                       tuple(slice(None, None, s) for s in stride)]
    subinit(subtensor)
    for d, s in enumerate(stride):
        subtensor = subtensor.repeat_interleave(s, 2 + d)
    return tensor.copy_(subtensor[tuple(slice(None, l) for l in tensor.shape)])


@contextlib.contextmanager
def manual_seeds(seed):
    """
    Context manager that sets the numpy and torch global random generator
    seeds to the given seed (only if it is true-ish) and restores the state
    afterwards.
    """
    if seed:
        np_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        np.random.seed(seed)
        torch.manual_seed(seed)
    try:
        yield
    finally:
        if seed:
            np.random.set_state(np_state)
            torch.set_rng_state(torch_state)


def apply(model, function, modules=None, params=None, **kwargs):
    """
    Apply an initialization function to parameters of a model, filtered by the
    given module and parameter specifications. Additional keyword arguments are
    passed on to the initialization function. The initialization function will
    first be called with the module, then if this fails, with its parameters.
    """
    def matches(thing, spec):
        if spec is None:
            return True
        elif isinstance(spec, (tuple, list)):
            return any(matches(thing, s) for s in spec)
        elif isinstance(spec, type):
            return isinstance(thing, spec)
        elif isinstance(spec, str):
            return fnmatch.fnmatchcase(str(thing), spec)
        else:
            return spec(thing)

    for module in model.modules():
        if matches(module, modules):
            try:
                function(module, **kwargs)
            except Exception:
                pass
            else:
                continue
            for name, param in module.named_parameters(recurse=False):
                if matches(name, params):
                    function(param, **kwargs)


def get_init_function(specification):
    """
    Returns an initialization function given a specification string.
    """
    try:
        kind, params = specification.split(':', 1)
    except ValueError:
        kind, params = specification, ''
    if kind in ('kaiming', 'he'):
        a = float(params) if params else math.sqrt(5)  # Pytorch for conv
        return functools.partial(nn.init.kaiming_uniform_, a=a)
    elif kind in ('xavier', 'glorot'):
        a = float(params) if params else math.sqrt(5)  # Pytorch for conv
        gain = nn.init.calculate_gain('leaky_relu', a)
        return functools.partial(nn.init.glorot_uniform_, gain=gain)
    elif kind in ('constant', 'const'):
        val = float(params) if params else 0
        return functools.partial(nn.init.constant_, val=val)
    elif kind == 'icnr':
        subinit = get_init_function(params)
        return functools.partial(conv_nn_resize_, subinit=subinit)
    else:
        raise ValueError("unknown init function %r" % kind)


def init_model(model, cfg):
    """
    Initializes parameters of a model given a configuration dictionary.
    """
    with manual_seeds(cfg.get('model.init_seed')):
        for key, modules, params in (
                ('model.init.conv_weight', 'Conv*', 'weight'),
                ('model.init.conv_bias', 'Conv*', 'bias'),
                ('model.init.conv_transposed_weight', 'ConvTranspose*',
                 'weight'),
                ('model.init.conv_strided_weight', 'Conv*stride=(2,*',
                 'weight'),
        ):
            if cfg.get(key, None):
                apply(model, get_init_function(cfg[key]), modules, params)
