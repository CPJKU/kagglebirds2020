# -*- coding: utf-8 -*-

"""
Optimizer definitions.

Author: Jan Schlüter
"""
from fnmatch import fnmatchcase

import torch
import torch.nn.functional as F

from .. import config
from .radam import RAdam
from .lookahead import Lookahead


def get_optimizer(cfg, params):
    """
    Return an Optimizer instance for the given parameters.
    """
    config.add_defaults(cfg, pyfile=__file__)
    optimizer_name = cfg['optimizer']
    optimizer_class = (globals().get(optimizer_name) or
                       getattr(torch.optim, optimizer_name))
    kwargs = dict(lr=cfg['train.eta'])
    for k, v in cfg.items():
        if k.startswith('optimizer.') and fnmatchcase(optimizer_name,
                                                      k.split('.', 2)[1]):
            if hasattr(v, 'startswith') and v.startswith('$'):
                v = cfg[v[1:]]
            kwargs[k.rsplit('.', 1)[1]] = v
    # special case: meta-optimizers like Lookahead wrap another optimizer
    if 'optimizer' in kwargs:
        cfg_ = dict(cfg)
        cfg_['optimizer'] = kwargs['optimizer']
        kwargs['optimizer'] = get_optimizer(cfg_, params)
        # do not pass a learning rate, and do not pass parameters
        del kwargs['lr']
        return optimizer_class(**kwargs)
    return optimizer_class(params, **kwargs)
