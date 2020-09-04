#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Backend options shared by the different architectures.

Author: Jan Schlüter
"""

import numpy as np
import torch
import torch.nn as nn

from .. import ReceptiveField
from ..layers import SpatialLogMeanExp


class SpatialMaxPool(nn.Module):
    """
    Performs max pooling over spatial dimensions; keeps only the first `ndim`
    dimensions of the input.
    """
    def __init__(self, ndim=2):
        super(SpatialMaxPool, self).__init__()
        self.ndim = ndim

    def forward(self, x):
        max, argmax = x.flatten(self.ndim).max(dim=-1)
        return max


class SpatialMeanPool(nn.Module):
    """
    Performs mean pooling over spatial dimensions; keeps only the first `ndim`
    dimensions of the input.
    """
    def __init__(self, ndim=2):
        super(SpatialMeanPool, self).__init__()
        self.ndim = ndim

    def forward(self, x):
        return x.mean(tuple(range(self.ndim, x.ndim)))


class LocalPool(nn.Module):
    """
    Applies a given global pooling module over the given dimension (the last by
    default) with the given `size` and `stride`.
    """
    def __init__(self, pool, size, stride=None, dim=-1):
        super(LocalPool, self).__init__()
        if pool.dim != -1:
            raise ValueError("The embedded global pooling module must pool "
                             "over the last dimension.")
        self.pool = pool
        self.size = size
        self.stride = stride or size
        self.dim = dim

    def forward(self, x):
        # add strided view (will be appended as a new dimension)
        x = x.unfold(self.dim, self.size, self.stride)
        # pass through pooling of super class
        return self.pool(x)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        # store attributes of self.pool directly, so global pooling and
        # local pooling modules are directly exchangeable
        result = super(LocalPool, self).state_dict(destination=destination,
                                                   prefix=prefix,
                                                   keep_vars=keep_vars)
        keys = list(k for k in result.keys()
                    if k.startswith('%spool.' % prefix))
        for k in keys:
            result[k[:len(prefix)] + k[len(prefix) + len('pool.'):]] = (
                    result.pop(k)
            )
        return result

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        keys = list(k for k in state_dict.keys()
                    if k.startswith(prefix))
        for k in keys:
            state_dict[k[:len(prefix)] + 'pool.' + k[len(prefix):]] = (
                    state_dict.pop(k)
            )
        super(LocalPool, self)._load_from_state_dict(state_dict, prefix,
                                                     *args, **kwargs)


def create(cfg, shapes, dtypes, num_classes):
    """
    Instantiates a backend for the given data shapes and dtypes.
    """
    if cfg.get('model.global_pool', 'max') == 'max':
        pool = SpatialMaxPool()
    elif cfg['model.global_pool'] == 'mean':
        pool = SpatialMeanPool()
    elif cfg['model.global_pool'].startswith('lme'):
        pool_args = cfg['model.global_pool'].split(':', 1)
        sharpness = float(pool_args[1]) if len(pool_args) > 1 else 1
        trainable = (pool_args[0].startswith('lmex'))
        exp_sharpness = (pool_args[0].startswith('lmexx'))
        per_channel = (pool_args[0][-1] == 'c')
        pool = SpatialLogMeanExp(sharpness=sharpness, trainable=trainable,
                                 exp=exp_sharpness, per_channel=per_channel,
                                 in_channels=num_classes)
    elif cfg['model.global_pool'] == 'none':
        return None
    else:
        raise ValueError("Unknown model.global_pool='%s'" %
                         cfg['model.global_pool'])

    if cfg.get('model.global_pool_size', 0):
        size = cfg['model.global_pool_size']
        overlap = cfg.get('model.global_pool_overlap', 0)
        pool = LocalPool(pool=pool, size=size, stride=(size - overlap))
        pool.receptive_field = ReceptiveField(size, size - overlap, 0)
    else:
        pool.receptive_field = ReceptiveField()  # no useful representation

    return pool
