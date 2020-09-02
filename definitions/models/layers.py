# -*- coding: utf-8 -*-

"""
Layer implementations shared by different models.

Author: Jan Schlüter
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class Bipolar(nn.Module):
    def __init__(self, nonlinearity, flip_input=True, flip_output=True,
                 inplace=True):
        super(Bipolar, self).__init__()
        self.nonlinearity = nonlinearity
        self.flip_input = flip_input
        self.flip_output = flip_output
        self.inplace = inplace
        if not inplace:
            self.flipped_signs = None

    def forward(self, x):
        if (self.flip_input or self.flip_output) and not self.inplace:
            feats = x.shape[1]
            flipped_signs = self.flipped_signs
            if (flipped_signs is None or
                    len(flipped_signs) != feats or
                    flipped_signs.dtype != x.dtype or
                    flipped_signs.device != x.device):
                flipped_signs = torch.ones((feats,), dtype=x.dtype,
                                           device=x.device)
                flipped_signs[::2] = -1
                self.flipped_signs = flipped_signs
            flipped_signs = flipped_signs[(Ellipsis,) +
                                          (None,) * (x.dim() - 2)]
        if self.flip_input:
            if self.inplace:
                x[:, ::2] *= -1
            else:
                x = x * flipped_signs
        x = self.nonlinearity(x)
        if self.flip_output:
            if self.inplace:
                x[:, ::2] *= -1
            else:
                x = x * flipped_signs
        return x


def nonlinearity(name):
    """
    Creates a nonlinearity Module. Supports `"relu"`, `"lrelu"`, `"sigm"`,
    `"swish"`, `"mish"`, and bipolar versions of each, prefixed with
    `"bipol:"`.
    """
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name.startswith('bipol:'):
        base = nonlinearity(name[len('bipol:'):])
        return Bipolar(base)
    else:
        cls = {
            'lrelu': nn.LeakyReLU,
            'sigm': nn.Sigmoid,
            'swish': Swish,
            'mish': Mish,
        }[name]
        return cls()


def _lme(x, alpha, dim=-1, keepdim=False):
    """
    Apply log-mean-exp pooling with sharpness `alpha` across dimension `dim`.
    """
    # shortcut if there is nothing to pool over
    if x.shape[dim] <= 1:
        return x if keepdim else x.squeeze(dim)
    # stable version of log(mean(exp(alpha * x), dim, keepdim)) / alpha
    if not torch.is_tensor(alpha) and alpha == 0:
        return x.mean(dim, keepdim=keepdim)
    if torch.is_tensor(alpha) or alpha != 1:
        x = x * alpha
    xmax, _ = x.max(dim=dim, keepdim=True)
    x = (x - xmax)
    x = torch.log(torch.mean(torch.exp(x), dim, keepdim=keepdim))
    if not keepdim:
        xmax = xmax.squeeze(dim)
        if torch.is_tensor(alpha) and abs(dim) <= alpha.dim():
            alpha = alpha.squeeze(dim)
    x = x + xmax
    if torch.is_tensor(alpha) or alpha != 1:
        x = x / alpha
    return x


class SpatialLogMeanExp(nn.Module):
    """
    Performs global log-mean-exp pooling over all spatial dimensions. If
    `trainable`, then the `sharpness` becomes a trainable parameter. If
    `per_channel`, then separate parameters are learned for the feature
    dimension (requires `in_channels`). If `exp`, the exponential of the
    trainable parameter is taken in the forward pass (i.e., the logarithm of
    the sharpness is learned). `per_channel`, `in_channels`, and `exp` are
    ignored if not `trainable`. If `keepdim`, will keep singleton spatial dims.
    See https://arxiv.org/abs/1411.6228, Eq. 6.
    """
    def __init__(self, sharpness=1, trainable=False, per_channel=False,
                 in_channels=None, exp=False, keepdim=False):
        super(SpatialLogMeanExp, self).__init__()
        self.trainable = trainable
        if trainable:
            if exp:
                sharpness = np.log(sharpness)
            self.exp = exp
            if per_channel:
                if in_channels is None:
                    raise ValueError("per_channel requires in_channels")
                sharpness = torch.full((in_channels,), sharpness)
            else:
                sharpness = torch.tensor(sharpness)
            self.per_channel = per_channel
            sharpness = nn.Parameter(sharpness)
        self.sharpness = sharpness
        self.keepdim = keepdim

    def extra_repr(self):
        if not self.trainable:
            return 'sharpness={:.3g}, trainable=False'.format(self.sharpness)
        else:
            return 'trainable=True, per_channel={!r}, exp={!r}'.format(
                    self.per_channel, self.exp)

    def forward(self, x):
        sharpness = self.sharpness
        # reshape input to flatten the trailing spatial dimensions
        if self.keepdim:
            spatial_dims = x.dim() - 2
        x = x.reshape(x.shape[:2] + (-1,))
        # skip pooling if there is nothing to pool over
        if x.shape[-1] <= 1:
            x = x.squeeze(-1)
        else:
            # if requested, exponentiate sharpness
            if self.trainable and self.exp:
                sharpness = torch.exp(sharpness)
            # if sharpness is a vector, broadcast over flattened spatial dims
            if self.trainable and self.per_channel:
                sharpness = sharpness.view(sharpness.shape + (1,))
            # apply pooling
            x = _lme(x, sharpness)
        # restore dimensions, if needed
        if self.keepdim:
            x = x[(Ellipsis,) + (None,) * spatial_dims]
        return x


class Shift(nn.Module):
    def __init__(self, amount, inplace=False):
        super(Shift, self).__init__()
        self.amount = amount
        self.inplace = inplace

    def extra_repr(self):
        return 'amount={}'.format(self.amount)

    def forward(self, x):
        if self.inplace:
            x += self.amount
        else:
            x = x + self.amount
        return x


class Crop(nn.Module):
    """
    Crop the same number of pixels on each side of each spatial dimension.
    """
    def __init__(self, dimensionality, size):
        super(Crop, self).__init__()
        self.dimensionality = dimensionality
        self.size = size

    def extra_repr(self):
        return 'size={}'.format(self.size)

    def forward(self, x):
        crop = slice(self.size, -self.size)
        return x[(Ellipsis,) + (crop,) * self.dimensionality]


class Squeeze(nn.Module):
    def __init__(self, dim):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)


class ShakeShakeFunction(torch.autograd.Function):
    """
    Computes a shake-shake weighted sum of several tensors, with randomly drawn
    weights for each item and for the forward and backward pass.
    """
    @staticmethod
    def sample_weights(k, n, dtype, device):
        if k == 2:
            u = torch.rand((n,), dtype=dtype, device=device)
            return u, 1 - u
        else:
            d = torch.distributions.Dirichlet(torch.ones((k,),
                                                         dtype=dtype,
                                                         device=device))
            return d.sample((n,)).t()

    @staticmethod
    def weighted(x, w):
        # extend `w` as needed to align with the batch dimension of `x`
        return x * w[(Ellipsis,) + (None,) * (x.dim() - 1)]

    @staticmethod
    def forward(ctx, *xs):
        num_inputs = len(xs)
        batchsize = len(xs[0])
        alphas = ShakeShakeFunction.sample_weights(num_inputs, batchsize,
                                                   xs[0].dtype, xs[0].device)
        return sum((ShakeShakeFunction.weighted(x, alpha)
                    for x, alpha in zip(xs[1:], alphas[1:])),
                   ShakeShakeFunction.weighted(xs[0], alphas[0]))

    @staticmethod
    def backward(ctx, grad_output):
        num_inputs = len(ctx.needs_input_grad)
        batchsize = len(grad_output)
        betas = ShakeShakeFunction.sample_weights(num_inputs, batchsize,
                                                  grad_output.dtype,
                                                  grad_output.device)
        return tuple(ShakeShakeFunction.weighted(grad_output, beta)
                     for beta in betas)


class ShakeShake(nn.ModuleList):
    """
    Modules applied to the same input and then added with random weights,
    drawn separately for each item and for the forward and backward pass
    during training. At test time, inputs are averaged instead.
    """
    def forward(self, x):
        num_modules = len(self)
        modules = iter(self)
        if self.train:
            # training mode: use shake-shake
            y = ShakeShakeFunction.apply(*(module(x) for module in modules))
        else:
            # testing mode: use average
            y = next(modules)(x)
            for module in modules:
                y = y + module(x)
            if num_modules > 1:
                y = y / num_modules
        return y
