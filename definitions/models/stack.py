# -*- coding: utf-8 -*-

"""
Meta model that stacks other model definitions.
Provides create().

Author: Jan Schlüter
"""
import importlib

import torch

from .. import config


def create(cfg, shapes, dtypes, num_classes):
    submodels = cfg['model.stack'].split(',')
    stack = torch.nn.Sequential()
    for idx, submodel in enumerate(submodels):
        # extract submodel name, if any
        try:
            name, submodel = submodel.split(':')
        except ValueError:
            name = str(idx)
        submodel = importlib.import_module('.' + submodel, __package__)
        # for a named submodel, move 'model.name.*' to 'model.*' in cfg
        if name:
            subcfg = config.renamed_prefix(cfg, 'model.%s' % name, 'model')
        else:
            subcfg = cfg
        # add submodel to stack
        submodel = submodel.create(subcfg, shapes, dtypes, num_classes)
        stack.add_module(name, submodel)
    return stack
