#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CNN completely definable via command line arguments.
Provides create().

Author: Jan Schlüter
"""

import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import PickDictKey, PutDictKey, ReceptiveField
from .layers import (nonlinearity, SpatialLogMeanExp, Shift, Crop, Squeeze,
                     ShakeShake)


class Cat(nn.ModuleList):
    """
    Modules applied to the same input and concatenated along the channels.
    """
    def forward(self, x):
        return torch.cat([module(x) for module in self], dim=1)


class Add(nn.ModuleList):
    """
    Modules applied to the same input and then added up.
    """
    def forward(self, x):
        modules = iter(self)
        first = next(modules)
        return sum((module(x) for module in modules), first(x))


class Mul(nn.ModuleList):
    """
    Modules applied to the same input and then multiplied.
    """
    def forward(self, x):
        modules = iter(self)
        y = next(modules)(x)
        for module in modules:
            y = y * module(x)
        return y


def custom_cnn(input_channels, specification, input_name='input',
               output_name='output', default_nonlin='relu', batch_norm=False):
    """
    Creates a CNN for the given number of input channels, with an architecture
    defined as a comma-separated string of layer definitions. Supported layer
    definitions are (with variables in <>, and optional parts in []):
    - pad1d:<method>@<size>
    - pad2d:<method>@<size>
    - crop1d:<size>
    - crop2d:<size>
    - conv1d:<channels>@<size>[s<stride>][p<pad>][d<dilation>][g<groups>]
    - conv2d:<channels>@<size0>x<size1>[s<stride>][p<pad>][d<dilation>][g<groups>]
    - pool1d:<method>@<size>[s<stride>][p<pad>][d<dilation]
    - pool2d:<method>@<size0>x<size1>[s<stride>][p<pad>][d<dilation>]
    - globalpool1d:<method>
    - globalpool2d:<method>
    - globallmepool:<alpha>[t<trainable>][c<channelwise>][e<exponentiated>]
    - bn1d
    - bn2d
    - groupnorm:<groups>
    - dropout:<drop_probability>
    - relu
    - lrelu
    - sigm
    - swish
    - mish
    - bipol:<nonlin>
    - shift:<amount>
    - bypass (does nothing)
    - squeeze:<dim>
    - cat[layers1|layers2|...] (apply stacks to same input, then concat)
    - add[layers1|layers2|...] (apply stacks to same input, then add)
    - shake[layers1|layers2|...] (apply stacks to same input, then shake-shake)
    If there is a batch normalization one or two layers after a convolution,
    the convolution will not have a bias term.
    """
    def read_layers(s):
        """
        Yields all layer definitions (as separated by , | [ or ]) as tuples
        of the definition string and the following delimiter.
        """
        pos = 0
        for match in re.finditer(r'[,|[\]]', s):
            yield s[pos:match.start()], s[match.start():match.end()]
            pos = match.end()
        yield s[pos:], None


    def read_size(s, t=int, expect_remainder=True):
        """
        Read and parse a size (e.g., 1, 1x1, 1x1x1) at the beginning of `s`,
        with elements of type `t`. If `expect_remainder`, returns the
        remainder, otherwise tries to parse the complete `s` as a size.
        """
        if expect_remainder:
            # yes, we could use a precompiled regular expression...
            p = next((i for i, c in enumerate(s) if c not in '0123456789x'),
                        len(s))
            remainder = s[p:]
            s = s[:p]
        size = tuple(map(t, s.split('x')))
        if len(size) == 1:
            size = size[0]
        if expect_remainder:
            return size, remainder
        else:
            return size


    def size_string(size):
        """
        Convert a size integer or tuple back into its string form.
        """
        try:
            return 'x'.join(map(str, size))
        except TypeError:
            return str(size)


    def read_extra_sizes(s, prefixes, t=int):
        """
        Read and parse any extra size definitions prefixed by any of the
        allowed prefixes, and returns them as a dictionary. If `prefixes` is
        a dictionary, the prefixes (keys) will be translated to the expanded
        names (values) in the returned dictionary. Values will be converted
        from strings to `t`.
        """
        if not isinstance(prefixes, dict):
            prefixes = {prefix: prefix for prefix in prefixes}
        result = {}
        while s:
            for prefix, return_key in prefixes.items():
                if s.startswith(prefix):
                    size, s = read_size(s[len(prefix):], t)
                    result[return_key] = size
                    break
            else:
                raise ValueError("unrecognized part in layer definition: "
                                 "%r" % s)
        return result


    stack = []
    layers = []
    if input_name:
        layers = [PickDictKey(input_name)]
    # track receptive field for the full network
    receptive_field = ReceptiveField()
    # split specification string into definition, delimiter tuples
    specification = list(read_layers(specification))
    # iterate over it (in a way that allows us to expand macro definitions)
    while specification:
        layer_def, delim = specification.pop(0)
        layer_def = layer_def.split(':')
        kind = layer_def[0]
        if kind in ('pad1d', 'pad2d'):
            method, size = layer_def[1].split('@')
            size = read_size(size, expect_remainder=False)
            cls = {'reflectpad1d': nn.ReflectionPad1d,
                   'reflectpad2d': nn.ReflectionPad2d}[method + kind]
            layers.append(cls(size))
            receptive_field *= ReceptiveField(padding=size)
        elif kind in ('crop1d', 'crop2d'):
            size = int(layer_def[1])
            dimensionality = int(kind[-2])
            layers.append(Crop(dimensionality, size))
            receptive_field *= ReceptiveField(padding=-size)
        elif kind in ('conv1d', 'conv2d'):
            channels, remainder = layer_def[1].split('@')
            channels = int(channels)
            size, remainder = read_size(remainder)
            params = dict(stride=1, padding=0, dilation=1, groups=1)
            params.update(read_extra_sizes(
                    remainder, dict(s='stride', p='padding', d='dilation',
                                    g='groups')))
            cls = {'conv1d': nn.Conv1d, 'conv2d': nn.Conv2d}[kind]
            layers.append(cls(input_channels, channels, size, **params))
            input_channels = channels
            # effective kernel size:
            size = (np.array(size) - 1) * params['dilation'] + 1
            receptive_field *= ReceptiveField(size, params['stride'],
                                              params['padding'])
        elif kind in ('pool1d', 'pool2d'):
            method, size = layer_def[1].split('@')
            size, remainder = read_size(size)
            params = dict(stride=None, padding=0, dilation=1)
            params.update(read_extra_sizes(
                    remainder, dict(s='stride', p='padding', d='dilation')))
            cls = {'maxpool1d': nn.MaxPool1d, 'meanpool1d': nn.AvgPool1d,
                   'maxpool2d': nn.MaxPool2d, 'meanpool2d': nn.AvgPool2d}[method + kind]
            layers.append(cls(size, **params))
            # effective kernel size:
            size = (np.array(size) - 1) * params['dilation'] + 1
            if params['stride'] is None:
                params['stride'] = size
            receptive_field *= ReceptiveField(size, params['stride'],
                                              params['padding'])
        elif kind in ('globalpool1d', 'globalpool2d'):
            method = layer_def[1]
            cls = {'maxglobalpool1d': nn.AdaptiveMaxPool1d,
                   'meanglobalpool1d': nn.AdaptiveAvgPool1d,
                   'maxglobalpool2d': nn.AdaptiveMaxPool2d,
                   'meanglobalpool2d': nn.AdaptiveAvgPool2d}[method + kind]
            layers.append(cls(output_size=1))
            # we do not adjust the receptive field; it spans the whole input
        elif kind == 'globallmepool':
            alpha, remainder = read_size(layer_def[1], float)
            params = read_extra_sizes(
                remainder, dict(t='trainable', c='per_channel', e='exp'),
                t=lambda s: bool(int(s)))
            layers.append(SpatialLogMeanExp(alpha, in_channels=input_channels,
                                            keepdim=True, **params))
            # we do not adjust the receptive field; it spans the whole input
        elif kind == 'bn1d':
            if len(layers) >= 1 and hasattr(layers[-1], 'bias'):
                layers[-1].register_parameter('bias', None)
            elif len(layers) >=2 and hasattr(layers[-2], 'bias'):
                layers[-2].register_parameter('bias', None)
            layers.append(nn.BatchNorm1d(input_channels))
        elif kind == 'bn2d':
            if len(layers) >= 1 and hasattr(layers[-1], 'bias'):
                layers[-1].register_parameter('bias', None)
            elif len(layers) >= 2 and hasattr(layers[-2], 'bias'):
                layers[-2].register_parameter('bias', None)
            layers.append(nn.BatchNorm2d(input_channels))
        elif kind == 'groupnorm':
            groups = int(layer_def[1])
            layers.append(nn.GroupNorm(groups, input_channels))
        elif kind == 'dropout':
            p = float(layer_def[1])
            layers.append(nn.Dropout(p))
        elif kind == 'squeeze':
            dim = int(layer_def[1])
            layers.append(Squeeze(dim))
        elif kind == 'shift':
            amount = float(layer_def[1])
            layers.append(Shift(amount))
        elif kind == 'bypass':
            layers.append(nn.Identity())
        elif kind == 'cat':
            stack.append((layers, input_channels, receptive_field))
            stack.append((Cat(), input_channels, receptive_field))
            layers = []
            receptive_field = ReceptiveField()
        elif kind == 'add':
            stack.append((layers, input_channels, receptive_field))
            stack.append((Add(), input_channels, receptive_field))
            layers = []
            receptive_field = ReceptiveField()
        elif kind == 'mul':
            stack.append((layers, input_channels, receptive_field))
            stack.append((Mul(), input_channels, receptive_field))
            layers = []
            receptive_field = ReceptiveField()
        elif kind == 'shake':
            stack.append((layers, input_channels, receptive_field))
            stack.append((ShakeShake(), input_channels, receptive_field))
            layers = []
            receptive_field = ReceptiveField()
        elif kind == '':
            pass
        elif kind == 'mbconv2d':
            # mobile inverted bottleneck convolution layer from MobileNetV2
            channels, remainder = layer_def[1].split('@')
            channels = int(channels)
            size, remainder = read_size(remainder)
            params = dict(stride=1, dilation=1, groups=1, expansion=1,
                          size=size, channels=channels)
            params.update(read_extra_sizes(
                    remainder, dict(s="stride", d="dilation", g="groups",
                                    e="expansion")))
            hidden_channels = int(input_channels * params['expansion'])
            # define layers
            macro = []
            # 1x1 channel expansion
            if hidden_channels != input_channels:
                macro.append('conv2d:%d@1x1g%d' %
                             (hidden_channels, params['groups']))
                if batch_norm:
                    macro.append('bn2d')
                macro.append(default_nonlin)
            # channelwise convolution
            macro.append('conv2d:%d@%ss%sd%sg%d' %
                         (hidden_channels, size_string(size),
                          size_string(params['stride']),
                          size_string(params['dilation']),
                          hidden_channels))
            if batch_norm:
                macro.append('bn2d')
            macro.append(default_nonlin)
            # linear projection
            macro.append('conv2d:%d@1x1g%d' % (channels, params['groups']))
            # residual shortcut, if applicable
            macro = ','.join(macro)
            if params['stride'] == 1 and channels == input_channels:
                crop = ((np.array(size) - 1) * params['dilation'] + 1) // 2
                macro = 'add[%s|%s]' % ('crop2d:%d' % crop[0], macro)
            # push to beginning of remaining layer specifications
            specification[:0] = read_layers(macro)
        elif kind == 'bipol':
            layers.append(nonlinearity('bipol:' + layer_def[1]))
        else:
            try:
                layers.append(nonlinearity(kind))
            except KeyError:
                raise ValueError('Unknown layer type "%s"' % kind)
        if delim is not None and delim in '|]':
            if isinstance(layers, list):
                layers = nn.Sequential(*layers) if len(layers) > 1 else layers[0]
            layers.receptive_field = receptive_field
            layers.out_channels = input_channels
            # append layers to Cat() or Add()
            stack[-1][0].append(layers)
            if delim == '|':
                # reset input_channels to match input of Cat() or Add()
                input_channels = stack[-1][1]
                # we expect another set of layers
                layers = []
                receptive_field = ReceptiveField()
            elif delim == ']':
                # take the Cat() or Add() from the stack
                layers, _, receptive_field = stack.pop()
                # append it to what we were building before
                stack[-1][0].append(layers)
                # and continue there
                if isinstance(layers, Cat):
                    input_channels = sum(path.out_channels for path in layers)
                receptive_field *= sum(path.receptive_field for path in layers)
                layers, _, _ = stack.pop()
    if stack:
        raise ValueError('There seems to be a missing "]" bracket.')
    if output_name:
        layers.append(PutDictKey(output_name))
    if isinstance(layers, list):
        layers = nn.Sequential(*layers)
    layers.receptive_field = receptive_field
    layers.out_channels = input_channels
    return layers


def create(cfg, shapes, dtypes, num_classes):
    """
    Instantiates a Model for the given data shapes and dtypes.
    """
    input_channels = shapes['input'][0]
    specification = cfg['model.arch']
    num_outputs = 1 if num_classes == 2 else num_classes
    specification = specification.replace('C', str(num_outputs))
    input_name = cfg.get('model.input_name', 'input')
    output_name = cfg.get('model.output_name', 'output')
    return custom_cnn(input_channels, specification, input_name, output_name,
                      default_nonlin=cfg.get('model.nonlin', 'relu'),
                      batch_norm=cfg.get('model.batch_norm', False))
