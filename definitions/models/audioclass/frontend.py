#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Frontend options shared by the different architectures.

Author: Jan Schlüter
"""

from functools import reduce
import operator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import ReceptiveField


if torch.__version__ < '1.6':
    class Module(nn.Module):
        """
        A torch.nn.Module subclass that allows to register non-persistent buffers.
        """
        def __init__(self):
            super(Module, self).__init__()
            self._nonpersistent_buffers = set()

        def register_buffer(self, name, tensor, persistent=True):
            super(Module, self).register_buffer(name, tensor)
            if not persistent:
                self._nonpersistent_buffers.add(name)

        def state_dict(self, destination=None, prefix='', keep_vars=False):
            result = super(Module, self).state_dict(destination, prefix, keep_vars)
            # remove non-persistent buffers
            for k in self._nonpersistent_buffers:
                del result[prefix + k]
            return result

        def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
            # temporarily hide the non-persistent buffers
            persistent_buffers = {k: v for k, v in self._buffers.items()
                                if k not in self._nonpersistent_buffers}
            all_buffers = self._buffers
            self._buffers = persistent_buffers
            result = super(Module, self)._load_from_state_dict(state_dict, prefix, *args, **kwargs)
            self._buffers = all_buffers
            return result
else:
    # PyTorch 1.6+ supports non-persistent buffers out of the box
    Module = nn.Module


class TemporalBatchNorm(nn.Module):
    """
    Batch normalization of a (batch, channels, bands, time) tensor over all but
    the previous to last dimension (the frequency bands).
    """
    def __init__(self, num_bands, affine=True):
        super(TemporalBatchNorm, self).__init__()
        self.bn = nn.BatchNorm1d(num_bands, affine=affine)

    def forward(self, x):
        shape = x.shape
        # squash channels into the batch dimension
        x = x.reshape((-1,) + x.shape[-2:])
        # pass through 1D batch normalization
        x = self.bn(x)
        # restore squashed dimensions
        return x.reshape(shape)


class Log1p(nn.Module):
    """
    Applies `log(1 + 10**a * x)`, with `a` fixed or trainable.
    """
    def __init__(self, a=0, trainable=False):
        super(Log1p, self).__init__()
        if trainable:
            a = nn.Parameter(torch.tensor(a, dtype=torch.get_default_dtype()))
        self.a = a
        self.trainable = trainable

    def forward(self, x):
        if self.trainable or self.a != 0:
            x = 10 ** self.a * x
        return torch.log1p(x)

    def extra_repr(self):
        return 'trainable={}'.format(repr(self.trainable))


class Pow(nn.Module):
    """
    Applies `x ** sigmoid(a)`, with `a` fixed or trainable.
    """
    def __init__(self, a=0, trainable=False):
        super(Pow, self).__init__()
        if trainable:
            a = nn.Parameter(torch.tensor(a, dtype=torch.get_default_dtype()))
        self.a = a
        self.trainable = trainable

    def forward(self, x):
        if self.trainable or self.a != 0:
            x = torch.pow(x, torch.sigmoid(self.a))
        else:
            x = torch.sqrt(x)
        return x

    def extra_repr(self):
        return 'trainable={}'.format(repr(self.trainable))


class PCEN(nn.Module):
    """
    Trainable PCEN (Per-Channel Energy Normalization) layer:
    .. math::
        Y = (\\frac{X}{(\\epsilon + M)^\\alpha} + \\delta)^r - \\delta^r
        M_t = (1 - s) M_{t - 1} + s X_t
    Assumes spectrogram input of shape ``(batchsize, channels, bands, time)``.
    Implements an automatic gain control through the division by :math:`M`, an
    IIR filter estimating the local magnitude, followed by root compression.
    As proposed in https://arxiv.org/abs/1607.05666, all parameters are
    trainable, and learned separately per frequency band. In contrast to the
    paper, the smoother :math:`M` is learned by backpropagating through the
    recurrence relation to tune :math:`s`, not by mixing a set of predefined
    smoothers.
    """
    def __init__(self, num_bands,
                 s=0.025,
                 alpha=1.,
                 delta=1.,
                 r=1.,
                 eps=1e-6,
                 init_smoother_from_data=True):
        super(PCEN, self).__init__()
        self.log_s = nn.Parameter(torch.full((num_bands,), np.log(s)))
        self.log_alpha = nn.Parameter(torch.full((num_bands,), np.log(alpha)))
        self.log_delta = nn.Parameter(torch.full((num_bands,), np.log(delta)))
        self.log_r = nn.Parameter(torch.full((num_bands,), np.log(r)))
        self.eps = torch.as_tensor(eps)
        self.init_smoother_from_data = init_smoother_from_data

    def forward(self, x):
        init = x[..., 0]  # initialize the filter with the first frame
        if not self.init_smoother_from_data:
            init = torch.zeros_like(init)  # initialize with zeros instead
        s = self.log_s.exp()
        smoother = [init]
        for frame in range(1, x.shape[-1]):
            smoother.append((1 - s) * smoother[-1] + s * x[..., frame])
        smoother = torch.stack(smoother, -1)
        alpha = self.log_alpha.exp()[:, np.newaxis]
        delta = self.log_delta.exp()[:, np.newaxis]
        r = self.log_r.exp()[:, np.newaxis]
        # stable reformulation due to Vincent Lostanlen; original formula was:
        # return (input / (self.eps + smoother)**alpha + delta)**r - delta**r
        smoother = torch.exp(-alpha * (torch.log(self.eps) +
                                       torch.log1p(smoother / self.eps)))
        return (x * smoother + delta)**r - delta**r


class STFT(Module):
    def __init__(self, winsize, hopsize, complex=False):
        super(STFT, self).__init__()
        self.winsize = winsize
        self.hopsize = hopsize
        self.register_buffer('window',
                             torch.hann_window(winsize, periodic=False),
                             persistent=False)
        self.complex = complex

    def forward(self, x):
        # we want each channel to be treated separately, so we mash
        # up the channels and batch size and split them up afterwards
        batchsize, channels = x.shape[:2]
        x = x.reshape((-1,) + x.shape[2:])
        # we apply the STFT
        x = torch.stft(x, self.winsize, self.hopsize, window=self.window,
                       center=False)
        # we compute magnitudes, if requested
        if not self.complex:
            x = x.norm(p=2, dim=-1)
        # restore original batchsize and channels in case we mashed them
        x = x.reshape((batchsize, channels, -1) + x.shape[2:])
        return x

    def extra_repr(self):
        return 'winsize={}, hopsize={}, complex={}'.format(self.winsize,
                                                           self.hopsize,
                                                           repr(self.complex))


def create_mel_filterbank(sample_rate, frame_len, num_bands, min_freq, max_freq,
                          norm=True, crop=False):
    """
    Creates a mel filterbank of `num_bands` triangular filters, with the first
    filter starting at `min_freq` and the last one stopping at `max_freq`.
    Returns the filterbank as a matrix suitable for a dot product against
    magnitude spectra created from samples at a sample rate of `sample_rate`
    with a window length of `frame_len` samples. If `norm`, will normalize
    each filter by its area. If `crop`, will exclude rows that exceed the
    maximum frequency and are therefore zero.
    """
    # mel-spaced peak frequencies
    min_mel = 1127 * np.log1p(min_freq / 700.0)
    max_mel = 1127 * np.log1p(max_freq / 700.0)
    peaks_mel = torch.linspace(min_mel, max_mel, num_bands + 2)
    peaks_hz = 700 * (torch.expm1(peaks_mel / 1127))
    peaks_bin = peaks_hz * frame_len / sample_rate

    # create filterbank
    input_bins = (frame_len // 2) + 1
    if crop:
        input_bins = min(input_bins,
                         int(np.ceil(max_freq * frame_len /
                                     float(sample_rate))))
    x = torch.arange(input_bins, dtype=peaks_bin.dtype)[:, np.newaxis]
    l, c, r = peaks_bin[0:-2], peaks_bin[1:-1], peaks_bin[2:]
    # triangles are the minimum of two linear functions f(x) = a*x + b
    # left side of triangles: f(l) = 0, f(c) = 1 -> a=1/(c-l), b=-a*l
    tri_left = (x - l) / (c - l)
    # right side of triangles: f(c) = 1, f(r) = 0 -> a=1/(c-r), b=-a*r
    tri_right = (x - r) / (c - r)
    # combine by taking the minimum of the left and right sides
    tri = torch.min(tri_left, tri_right)
    # and clip to only keep positive values
    filterbank = torch.clamp(tri, min=0)

    # normalize by area
    if norm:
        filterbank /= filterbank.sum(0)

    return filterbank


class MelFilter(Module):
    def __init__(self, sample_rate, winsize, num_bands, min_freq, max_freq):
        super(MelFilter, self).__init__()
        melbank = create_mel_filterbank(sample_rate, winsize, num_bands,
                                        min_freq, max_freq, crop=True)
        self.register_buffer('bank', melbank, persistent=False)
        self._extra_repr = 'num_bands={}, min_freq={}, max_freq={}'.format(
                num_bands, min_freq, max_freq)

    def forward(self, x):
        x = x.transpose(-1, -2)  # put fft bands last
        x = x[..., :self.bank.shape[0]]  # remove unneeded fft bands
        x = x.matmul(self.bank)  # turn fft bands into mel bands
        x = x.transpose(-1, -2)  # put time last
        return x

    def extra_repr(self):
        return self._extra_repr


class Slice(nn.Module):
    def __init__(self, slice_obj, dim=-1):
        super(Slice, self).__init__()
        self.slice = slice_obj
        self.dim = dim

    def forward(self, x):
        if self.dim >= 0:
            return x[(slice(None),) * self.dim + (self.slice,)]
        else:
            return x[(Ellipsis, self.slice) + (slice(None),) * (-self.dim - 1)]


def mel_filterbank(num_channels, num_bands, min_freq, max_freq, hopsize,
                   cfg, **kwargs):
    sample_rate = cfg['data.sample_rate']
    winsize = int(cfg['filterbank.winsize'] * sample_rate + .5)
    nyquist = sample_rate / 2
    result = nn.Sequential()
    result.add_module('stft', STFT(winsize, hopsize))
    result.add_module('melfilter', MelFilter(cfg['data.sample_rate'], winsize,
                                             num_bands, min_freq * nyquist,
                                             max_freq * nyquist))
    result.receptive_field = ReceptiveField(winsize, hopsize, 0)
    return result


def stft_filterbank(num_channels, num_bands, min_freq, max_freq, hopsize,
                    cfg, **kwargs):
    sample_rate = cfg['data.sample_rate']
    winsize = int(cfg['filterbank.winsize'] * sample_rate + .5)
    result = nn.Sequential()
    result.add_module('stft', STFT(winsize, hopsize))
    slice_obj = slice(int(winsize * min_freq / 2.),
                      int(winsize * max_freq / 2.))
    result.add_module('crop', Slice(slice_obj, dim=-2))
    # infer and set the number of output bands
    cfg['filterbank.num_bands'] = slice_obj.stop - slice_obj.start
    result.receptive_field = ReceptiveField(winsize, hopsize, 0)
    return result


def create(cfg, shapes, dtypes, num_classes):
    """
    Instantiates a frontend for the given data shapes and dtypes.
    """
    num_channels = shapes['input'][0]
    sample_rate = cfg['data.sample_rate']
    nyquist = .5 * sample_rate
    fps = cfg['spect.fps']
    supersample = cfg.get('spect.supersample', 1)
    smooth = cfg.get('spect.smooth', 0)
    hopsize = int(sample_rate / float(fps)) // supersample
    network = nn.Sequential()
    fbargs = {}
    if cfg['filterbank'] == 'mel':
        filterbank = mel_filterbank
    elif cfg['filterbank'] == 'stft':
        filterbank = stft_filterbank
    elif cfg['filterbank'] == 'none':
        filterbank = None
    else:
        raise ValueError("unknown filterbank '%s'" % cfg['filterbank'])
    if filterbank is not None:
        num_bands = cfg['filterbank.num_bands']
        min_freq = cfg['filterbank.min_freq'] / nyquist
        max_freq = cfg['filterbank.max_freq'] / nyquist
        network.add_module('filterbank',
                           filterbank(num_channels, num_bands, min_freq,
                                      max_freq, hopsize, cfg=cfg, **fbargs))
        num_bands = cfg['filterbank.num_bands']  # may have been updated
    else:
        num_bands = 1
    if supersample > 1 or smooth > 0:
        network.add_module('smoothing',
                           nn.AvgPool2d((1, supersample+smooth),
                                        stride=(1, supersample),
                                        padding=(0, (supersample+smooth) // 2)))
    if cfg['spect.magscale'] == 'linear':
        pass
    elif cfg['spect.magscale'].startswith('log1p'):
        if cfg['spect.magscale'] in ('log1p', 'log1px'):
            a = 1
        else:
            a = float(cfg['spect.magscale'].split(':', 1)[1])
        network.add_module('magscale', Log1p(
                a=np.log10(a),
                trainable=cfg['spect.magscale'].startswith('log1px')))
    elif cfg['spect.magscale'].startswith('pow'):
        if cfg['spect.magscale'] in ('pow', 'powx'):
            a = 0.5
        else:
            a = float(cfg['spect.magscale'].split(':', 1)[1])
        network.add_module('magscale', Pow(
                a=np.log(a / (1 - a)),
                trainable=cfg['spect.magscale'].startswith('powx')))
    elif cfg['spect.magscale'] == 'pcen':
        network.add_module('magscale', PCEN(num_bands))
    else:
        raise ValueError("unknown spect.magscale '%s'" % cfg['spect.magscale'])
    if cfg['spect.norm'] == 'batchnorm':
        network.add_module('norm', TemporalBatchNorm(num_bands))
    elif cfg['spect.norm'] == 'batchnorm_plain':
        network.add_module('norm', TemporalBatchNorm(num_bands, affine=False))
    elif cfg['spect.norm'] == 'none':
        pass
    else:
        raise ValueError("unknown spect.norm '%s'" % cfg['spect.norm'])

    network.receptive_field = reduce(operator.mul,
                                     (module.receptive_field
                                      for module in network.modules()
                                      if hasattr(module, 'receptive_field')),
                                     ReceptiveField())

    return network
