# -*- coding: utf-8 -*-

"""
PANN from https://github.com/qiuqiangkong/audioset_tagging_cnn.
Provides create(). Includes a frontend and backend from audioclass/__init__.py.

Author: Jan Schlüter
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ... import config
from .. import ReceptiveField, stack_receptive_fields
from ..audioclass import AudioClassifier
from ..audioclass.frontend import (STFT, MelFilter, Log1p, TemporalBatchNorm,
                                   SubtractMedian)
from ..audioclass import backend
from ..custom_cnn import custom_cnn


class ScaleShift(nn.Module):
    def __init__(self, scale=1, shift=0):
        super(ScaleShift, self).__init__()
        self.scale = scale
        self.shift = shift

    def forward(self, x):
        if self.scale != 1:
            x = x * self.scale
        if self.shift != 0:
            x = x + self.shift
        return x


def conv_block(in_channels, out_channels, pool_size=1):
    block = nn.Sequential()
    receptive_field = ReceptiveField(3, 1, 1)
    block.add_module('conv1', nn.Conv2d(in_channels, out_channels, 3,
                                        padding=1, bias=False))
    block.add_module('bn1', nn.BatchNorm2d(out_channels))
    block.add_module('relu1', nn.ReLU(inplace=True))
    receptive_field *= ReceptiveField(3, 1, 1)
    block.add_module('conv2', nn.Conv2d(out_channels, out_channels, 3,
                                        padding=1, bias=False))
    block.add_module('bn2', nn.BatchNorm2d(out_channels))
    block.add_module('relu2', nn.ReLU(inplace=True))
    if pool_size > 1:
        receptive_field *= ReceptiveField(pool_size, pool_size, 0)
        block.add_module('pool', nn.AvgPool2d(pool_size))
    block.receptive_field = receptive_field
    return block


def cnn14_16k(mel_bins=64):
    """
    Creates a Cnn14_16k-compatible Sequential model (without spectrogramming).
    """
    model = nn.Sequential()
    model.add_module('bn0', nn.BatchNorm2d(mel_bins))
    model.add_module('conv_block1', conv_block(1, 64, 2))
    model.add_module('dropout1', nn.Dropout(0.2))
    model.add_module('conv_block2', conv_block(64, 128, 2))
    model.add_module('dropout2', nn.Dropout(0.2))
    model.add_module('conv_block3', conv_block(128, 256, 2))
    model.add_module('dropout3', nn.Dropout(0.2))
    model.add_module('conv_block4', conv_block(256, 512, 2))
    model.add_module('dropout4', nn.Dropout(0.2))
    model.add_module('conv_block5', conv_block(512, 1024, 2))
    model.add_module('dropout5', nn.Dropout(0.2))
    model.add_module('conv_block6', conv_block(1024, 2048, 1))
    return model


def create(cfg, shapes, dtypes, num_classes):
    """
    Instantiates a Model for the given data shapes and dtypes.
    """
    config.add_defaults(cfg, pyfile=__file__)
    num_channels = shapes['input'][0]
    num_outputs = 1 if num_classes == 2 else num_classes

    # create a frontend matching what Cnn14_16k expects
    sample_rate = cfg['data.sample_rate']
    winsize = 512 * sample_rate // 16000
    hopsize = 160 * sample_rate // 16000
    mel_bands = 64
    frontend = nn.Sequential()
    filterbank = nn.Sequential()
    filterbank.add_module('stft', STFT(winsize, hopsize))
    filterbank.add_module('melfilter',
                          MelFilter(sample_rate, winsize,
                                    num_bands=mel_bands,
                                    min_freq=50, max_freq=8000,
                                    random_shift=cfg['filterbank.random_shift']))
    filterbank.receptive_field = ReceptiveField(winsize, hopsize, 0)
    frontend.add_module('filterbank', filterbank)
    frontend.add_module('magscale', Log1p(a=5,
                                          trainable=cfg['magscale.trainable']))
    frontend.add_module('adjustment', ScaleShift(9.2, -126))
    if cfg['spect.denoise'] == 'submedian':
        frontend.add_module('denoise', SubtractMedian())
    frontend.add_module('norm', TemporalBatchNorm(mel_bands))
    frontend.receptive_field = filterbank.receptive_field

    # load the pretrained predictor
    pretrained = cnn14_16k()
    if cfg['model.pretrained_weights']:
        state = torch.load(os.path.join(os.path.dirname(__file__),
                                        cfg['model.pretrained_weights']),
                           map_location='cpu')
        pretrained.load_state_dict(state['model'], strict=False)
        del state

    # transplant the temporal batch normalization
    frontend.norm.bn.load_state_dict(pretrained.bn0.state_dict())

    # create a predictor with the remaining pretrained layers...
    predictor = nn.Sequential()
    pretrained = pretrained[1:2 * cfg['model.num_blocks']]
    for p in pretrained.parameters():
        if p.data.ndim == 4:
            # need to swap time and frequency, PANN has it the other way
            p.data[:] = p.data.transpose(-1, -2).clone()
    pretrained.receptive_field = stack_receptive_fields(pretrained.modules())
    predictor.add_module('pretrained', pretrained)
    # ... and a custom cnn definition
    predictor.add_module('novel', custom_cnn(
            input_channels=int(2**(5 + cfg['model.num_blocks'])),
            specification=cfg['model.predictor.arch'].replace(
                    'C', str(num_outputs)),
            input_name='', output_name=''))
    predictor.receptive_field = (predictor.pretrained.receptive_field *
                                 predictor.novel.receptive_field)

    # bundle everything
    return AudioClassifier(
            frontend, predictor,
            backend.create(cfg, shapes, dtypes, num_classes),
            chunk_size=cfg.get('model.chunk_size', None),
            chunk_overlap=cfg.get('model.chunk_overlap', None),
            chunk_axis=cfg.get('model.chunk_axis', -1),
            extra_outputs=cfg.get('model.extra_outputs', '').split(','),
    )
