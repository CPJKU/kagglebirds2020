# -*- coding: utf-8 -*-

"""
Model definitions.

Author: Jan Schlüter
"""
import importlib
from functools import reduce
import operator

import torch
import numpy as np

from .init import manual_seeds, init_model


def get_model(cfg, shapes, dtypes, num_classes=None, device_ids=None):
    """
    Return a Model for the given data shapes, dtypes, number of classes and
    CUDA devices. If `device_ids` is empty or None, leave it on the CPU.
    """
    model = importlib.import_module('.' + cfg['model'], __package__)
    if num_classes is None:
        num_classes = cfg.get('model.num_classes')
    with manual_seeds(cfg.get('model.init_seed')):
        model = model.create(cfg, shapes, dtypes, num_classes)
    if isinstance(device_ids, int):
        device_ids = [device_ids]
    if device_ids:
        model.to('cuda:%d' % device_ids[0])
    if len(device_ids) > 1:
        model = in_parallel(model, device_ids)
    return model


def print_model_info(model):
    """
    Print information on the model's receptive field, if it provides any.
    """
    if hasattr(model, 'receptive_field'):
        receptive_field = model.receptive_field
        print("- receptive field: %s" % receptive_field.size)
        print("- stride: %s" % receptive_field.stride)
        print("- padding: %s" % receptive_field.padding)


class DictToKwargs(torch.nn.Module):
    """
    Module wrapper that unpacks a dictionary input into keyword arguments.
    """
    def __init__(self, module):
        super(DictToKwargs, self).__init__()
        self.module = module

    def forward(self, dictionary):
        return self.module.forward(**dictionary)


class KwargsToDict(torch.nn.Module):
    """
    Module wrapper that packs keyword argument inputs into a dictionary.
    """
    def __init__(self, module):
        super(KwargsToDict, self).__init__()
        self.module = module

    def forward(self, **kwargs):
        return self.module.forward(kwargs)


class ArgsToDict(torch.nn.Module):
    """
    Module wrapper that packs argument list inputs into a dictionary.
    """
    def __init__(self, module, keys):
        super(ArgsToDict, self).__init__()
        self.module = module

    def forward(self, *args):
        return self.module.forward(dict(zip(self.keys, args)))


class RemoveStatePrefix(torch.nn.Module):
    """
    Module wrapper that removes a given prefix from all state_dict entries.
    Note that it automatically removes its own 'module.' prefix.
    """
    def __init__(self, module, prefix):
        super(RemoveStatePrefix, self).__init__()
        self.module = module
        self.prefix = 'module.' + prefix

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        result = super(RemoveStatePrefix, self).state_dict(
                destination, prefix, keep_vars)
        # remove prefix
        expected_prefix = prefix + self.prefix
        p1 = len(prefix)
        p2 = len(expected_prefix)
        for k in list(result.keys()):
            if not k.startswith(expected_prefix):
                raise ValueError("state_dict key %r not starting "
                                 "with prefix %r" % (k, expected_prefix))
            result[k[:p1] + k[p2:]] = result.pop(k)
        return result

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # add prefix
        new_prefix = prefix + self.prefix
        p1 = len(prefix)
        # (we must modify state_dict in place, not replace it with a new one)
        for k in list(state_dict.keys()):
            state_dict[new_prefix + k[p1:]] = state_dict.pop(k)
        return super(RemoveStatePrefix, self)._load_from_state_dict(
                state_dict, prefix, *args, **kwargs)


def in_parallel(model, device_ids):
    """
    Wraps a model in a DataParallel module such that it supports
    an input dictionary for multiple inputs rather than kwargs.
    """
    return RemoveStatePrefix(DictToKwargs(torch.nn.DataParallel(
            KwargsToDict(model), device_ids)), 'module.module.module.')


class PickDictKey(torch.nn.Module):
    """
    Module that returns a particular dictionary key from its input.
    """
    def __init__(self, key):
        super(PickDictKey, self).__init__()
        self.key = key

    def extra_repr(self):
        return 'key={!r}'.format(self.key)

    def forward(self, input):
        return input[self.key]


class PutDictKey(torch.nn.Module):
    """
    Module that wraps its output with a particular dictionary key.
    """
    def __init__(self, key):
        super(PutDictKey, self).__init__()
        self.key = key

    def extra_repr(self):
        return 'key={!r}'.format(self.key)

    def forward(self, input):
        return {self.key: input}


class ReceptiveField(object):
    """
    Represents the dimensions of a receptive field. Multiplication of receptive
    fields evaluates to the receptive field of two layers applied in sequence.
    Addition of receptive fields evaluates to the receptive field of two
    layers added up or concatenated.
    """
    def __init__(self, size=1, stride=1, padding=0):
        super(ReceptiveField, self).__init__()
        self.size = np.asarray(size, np.int)
        self.stride = np.asarray(stride)
        self.padding = np.asarray(padding)

    def __mul__(self, other):
        size = self.size + (other.size - 1) * self.stride
        padding = self.padding + other.padding * self.stride
        stride = self.stride * other.stride
        return ReceptiveField(size, stride, padding)

    def __add__(self, other):
        if not np.all(self.stride == other.stride):
            raise ValueError("Cannot combine receptive fields of different "
                             "strides (%s and %s)" % (self.stride,
                                                      other.stride))
        size = np.maximum(self.size, other.size)
        padding = np.maximum(self.padding, other.padding)
        return ReceptiveField(size, self.stride, padding)

    def __radd__(self, other):
        if other == 0:
            return self

    def __repr__(self):
        return 'ReceptiveField(size=%s, stride=%s, padding=%s)' % (
                self.size, self.stride, self.padding)


def stack_receptive_fields(modules):
    """
    Computes the receptive field of all `modules` applied in sequence. If a
    module does not have a `receptive_field` attribute, it is assumed to not
    modify the receptive field.
    """
    return reduce(operator.mul,
                  (module.receptive_field
                   for module in modules
                   if hasattr(module, 'receptive_field')),
                  ReceptiveField())
