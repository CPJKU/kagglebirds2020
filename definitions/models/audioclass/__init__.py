#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generic model combining a frontend, a local predictor and a backend for
predicting global or local labels of audio recordings.
Provides create().

Author: Jan Schlüter
"""

import importlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import frontend
from . import backend
from ... import config


class AudioClassifier(nn.Module):
    """
    An audio classifier that consists of a frontend computing a spectrogram,
    a predictor computing local predictions, and an optional backend pooling
    the predictions over time. The frontend and predictions can be told to
    process the input in overlapping chunks that are assembled and passed
    through the backend. If the chunk overlap is not specified, it will be
    deduced automatically such that the results match an unchunked processing,
    adjusting the chunk size if necessary.
    """
    def __init__(self, frontend, predictor, backend=None,
                 input_name='input', output_name='output',
                 chunk_size=None, chunk_overlap=None, chunk_axis=-1,
                 extra_outputs=()):
        super(AudioClassifier, self).__init__()
        self.frontend = frontend
        self.predictor = predictor
        self.backend = backend
        self.input_name = input_name
        self.output_name = output_name
        if chunk_size and not chunk_overlap:
            rf = self.frontend.receptive_field * self.predictor.receptive_field
            win_size = rf.size[chunk_axis]
            hop_size = rf.stride[chunk_axis]
            padding = rf.padding[chunk_axis]
            # chunk size must fit an even number of prediction windows
            num_windows = (chunk_size - win_size) // hop_size + 1
            chunk_size = (num_windows - 1) * hop_size + win_size
            # chunk overlap must match the prediction window overlap
            chunk_overlap = win_size - hop_size - 2 * padding
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_axis = chunk_axis
        self.extra_outputs = set(extra_outputs)

    def iter_chunks(self, x):
        """
        Iterate over the input in chunks according to self.chunk_size,
        self.chunk_overlap and self.chunk_axis.
        """
        x_size = x.shape[self.chunk_axis]
        if (not self.chunk_size or x_size <= self.chunk_size):
            yield x
        else:
            rf = self.frontend.receptive_field * self.predictor.receptive_field
            win_size = rf.size[self.chunk_axis]
            axis = self.chunk_axis
            if axis < 0:
                axis += len(x.shape)
            colons = (slice(None),) * axis
            for pos in range(0, x_size - win_size + 1,
                             self.chunk_size - self.chunk_overlap):
                yield x[colons + (slice(pos, pos + self.chunk_size),)]

    @property
    def receptive_field(self):
        rf = self.frontend.receptive_field * self.predictor.receptive_field
        if self.backend is not None:
            rf = rf * self.backend.receptive_field
        return rf

    def forward(self, inputs, extra_outputs=()):
        """
        Passes the inputs through the frontend, predictor and backend.
        `inputs` are expected to be a dictionary with a key matching the
        `input_name` given to the constructor. The result will be a dictionary
        with the `output_name` given to the constructor mapped to the output
        of the backend (unless no backend was given, in which case it's the
        output of the predictor). If `extra_outputs` contains `'frontend'`
        or `'predictor'`, their outputs will be returned under those names.
        If `extra_outputs` contains `'filterbank'`, will return the filterbank
        under this name. If processing data in chunks, only the last chunk
        will be included in the `'frontend'` output; we assume we cannot keep
        all chunks in memory at once.
        """
        y = {}
        extra_outputs = self.extra_outputs | set(extra_outputs)
        if 'filterbank' in extra_outputs:
            # this will recompute the filterbank, but everything else would
            # require a large refactoring or some relatively smart caching.
            y['filterbank'] = self.frontend.filterbank.filterbank()
        x = inputs[self.input_name]
        xs = []
        for x in self.iter_chunks(x):
            x = self.frontend(x)
            if 'frontend' in extra_outputs:
                y['frontend'] = x
            x = self.predictor(x)
            xs.append(x)
        if len(xs) > 1:
            x = torch.cat(xs, self.chunk_axis)
        else:
            x = xs[0]
        if 'predictor' in extra_outputs:
            y['predictor'] = x
        if self.backend is not None:
            x = self.backend(x)
        y[self.output_name] = x
        return y


def create(cfg, shapes, dtypes, num_classes):
    """
    Instantiates a Model for the given data shapes and dtypes.
    """
    config.add_defaults(cfg, pyfile=__file__)
    # instantiate predictor, passing model.predictor.* as model.*
    predictor = importlib.import_module('..' + cfg['model.predictor'],
                                        __package__)
    predictor_cfg = config.renamed_prefix(cfg, 'model.predictor', 'model')
    return AudioClassifier(
            frontend.create(cfg, shapes, dtypes, num_classes),
            predictor.create(predictor_cfg, shapes, dtypes, num_classes),
            backend.create(cfg, shapes, dtypes, num_classes),
            chunk_size=cfg.get('model.chunk_size', None),
            chunk_overlap=cfg.get('model.chunk_overlap', None),
            chunk_axis=cfg.get('model.chunk_axis', -1),
            extra_outputs=cfg.get('model.extra_outputs', '').split(','),
    )
