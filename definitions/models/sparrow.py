# -*- coding: utf-8 -*-

"""
CNN 'sparrow' from Grill/Schlüter, EUSIPCO 2017.
Provides create(). Can be used as a local predictor in audioclass/__init__.py.

Author: Jan Schlüter
"""

import torch.nn as nn
import torch.nn.functional as F

from . import ReceptiveField


class Sparrow(nn.Module):
    """
    CNN from Grill/Schlüter, EUSIPCO 2017.
    """
    def __init__(self, num_channels, num_bands, num_outputs=1,
                 output_bias=False, global_pool=False):
        super(Sparrow, self).__init__()
        self.global_pool = global_pool
        lrelu = nn.LeakyReLU(0.01)
        self.conv_stage = nn.ModuleList([
                nn.Conv2d(num_channels, 32, 3, bias=False),
                nn.BatchNorm2d(32),
                lrelu,
                nn.Conv2d(32, 32, 3, bias=False),
                nn.BatchNorm2d(32),
                nn.MaxPool2d(3),
                lrelu,
                nn.Conv2d(32, 32, 3, bias=False),
                nn.BatchNorm2d(32),
                lrelu,
                nn.Conv2d(32, 32, 3, bias=False),
                nn.BatchNorm2d(32),
                lrelu,
                nn.Conv2d(32, 64, ((num_bands-4)//3-6, 3), bias=False),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(3),
                lrelu,
                nn.Dropout(),
                nn.Conv2d(64, 256, (1, 9), bias=False),
                nn.BatchNorm2d(256),
                lrelu,
                nn.Dropout(),
                nn.Conv2d(256, 64, 1, bias=False),
                nn.BatchNorm2d(64),
                lrelu,
                nn.Dropout(),
                nn.Conv2d(64, num_outputs, 1, bias=output_bias),
        ])
        self.receptive_field = ReceptiveField(103, 9, 0)

    def forward(self, x):
        for layer in self.conv_stage:
            x = layer(x)
        if x.shape[1] == 1:
            x = x.flatten(1)
        else:
            x = x.flatten(2)
        if self.global_pool:
            x, _ = x.max(dim=-1)
        return x


def create(cfg, shapes, dtypes, num_classes):
    """
    Instantiates a Model for the given data shapes and dtypes.
    """
    num_channels = shapes['input'][0]
    num_bands = cfg['filterbank.num_bands']
    num_outputs = 1 if num_classes == 2 else num_classes
    output_bias = cfg.get('model.output_bias', 0)
    return Sparrow(num_channels, num_bands,
                   num_outputs, output_bias,
                   global_pool=False)
