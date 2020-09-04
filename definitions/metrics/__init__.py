# -*- coding: utf-8 -*-

"""
Metric function definitions.

Author: Jan Schlüter
"""
import functools
from collections import OrderedDict, defaultdict
from fnmatch import fnmatchcase

import torch
import torch.nn.functional as F

from .. import config


def get_metrics(cfg, designation):
    """
    Return a dictionary of metric functions for a given designation
    ('train', 'valid' or 'test'). Each function can be called with a
    batch of predictions and targets of a model to return a tensor of
    one result per item or two tensors for updating the numerators
    and denominators.
    """
    config.add_defaults(cfg, pyfile=__file__)
    metrics = OrderedDict()
    for metric in cfg['metrics'].split(','):
        config_name, function_name = metric.split(':')
        metrics[config_name] = get_single_metric(cfg, function_name,
                                                 config_name)
    return metrics


def get_single_metric(cfg, function_name, config_name):
    """
    Return a metric function for the given function name.
    """
    metric_fn = globals()[function_name]
    kwargs = {}
    for k in sorted(cfg.keys()):
        if k.startswith('metrics.') and (fnmatchcase(function_name,
                                                     k.split('.', 2)[1]) or
                                         fnmatchcase(config_name,
                                                     k.split('.', 2)[1])):
            v = cfg[k]
            if hasattr(v, 'startswith') and v.startswith('$'):
                v = cfg[v[1:]]
            kw = k.rsplit('.', 1)[1]
            if kw == 'class_weights':
                v = torch.tensor(list(map(float, v.split(','))))
            if v == '':
                kwargs.pop(kw, None)
            else:
                kwargs[kw] = v
    if not kwargs:
        return metric_fn
    else:
        return functools.partial(metric_fn, **kwargs)


def get_loss_from_metrics(cfg):
    """
    Return a loss function. It will be called with the result of all
    metric functions and should return a single scalar value.
    """
    if cfg['loss']:
        losses = []
        for lossdef in cfg['loss'].split('+'):
            lossdef = lossdef.split('*', 1)
            if len(lossdef) == 2:
                losses.append((float(lossdef[0]), lossdef[1]))
            elif lossdef[0].startswith('-'):
                losses.append((-1, lossdef[0][1:]))
            else:
                losses.append((1, lossdef[0]))
        return lambda metrics: sum(factor * metrics[key].mean()
                                   for factor, key in losses)
    else:
        return lambda _: float('nan')


def print_metrics(prefix, values):
    """
    Print a dictionary of values with a common prefix, line by line.
    """
    for k, v in values.items():
        if not k.startswith('_'):
            if hasattr(v, 'shape') and len(v.shape) == 1:
                print(('%s %s: ' % (prefix, k)) +
                      (' '.join('%.3g' % x for x in v)) +
                      ' (avg %.3g)' % v.mean())
            else:
                print('%s %s: %.3g' % (prefix, k, v))


def binary_crossentropy(preds, targets, pred_name='output', target_name='mask',
                        ignore=None, weight_name=None):
    """
    Computes the binary cross-entropy against the predictions (as logits).
    Will ignore labels of value `ignore` if given and nonnegative. Optionally
    uses different weights per pixel.
    """
    gt = targets[target_name]
    if ignore is not None and ignore >= 0:
        valid = (gt != 255).float()
    gt = gt.float()
    weight = weight_name and targets.get(weight_name)
    if weight is not None and weight.ndim < gt.ndim:
        # append singleton dimensions to make sure the batch dimensions line up
        weight = weight.view(weight.shape + (1,) * (gt.ndim - weight.ndim))
    not_batch = tuple(range(1, gt.dim()))
    if ignore is not None and ignore >= 0:
        return F.binary_cross_entropy_with_logits(
                preds[pred_name], gt * valid,
                valid * weight if weight is not None else valid,
                reduction='none').sum(not_batch) / valid.sum(not_batch)
    else:
        return F.binary_cross_entropy_with_logits(
                preds[pred_name], gt, weight,
                reduction='none').mean(not_batch)

multilabel_crossentropy = binary_crossentropy


def categorical_crossentropy(preds, targets, pred_name='output',
                             target_name='map', ignore=None, weight_name=None,
                             focal=0, class_weights=None, label_smoothing=0):
    """
    Computes the categorical cross-entropy against the predictions (as logits).
    Will ignore labels of value `ignore` if given and nonnegative. Optionally
    uses different weights per pixel, a focal loss, a given vector of class
    weights, and label smoothing.
    """
    gt = targets[target_name]
    weight = weight_name and targets.get(weight_name)
    if weight is not None and weight.ndim < gt.ndim:
        # append singleton dimensions to make sure the batch dimensions line up
        weight = weight.view(weight.shape + (1,) * (gt.ndim - weight.ndim))
    not_batch = tuple(range(1, gt.dim()))
    preds = F.log_softmax(preds[pred_name], dim=1)
    if class_weights is not None:
        class_weights = class_weights.to(preds)
        if label_smoothing:
            raise NotImplementedError("label_smoothing currently cannot "
                                      "be combined with class_weights")
    if focal != 0:
        preds = preds * (1 - preds.exp())**focal
    kwargs = {}
    if ignore is not None and ignore >= 0:
        valid = (gt != ignore).float()
        kwargs['ignore_index'] = ignore
    else:
        valid = 1
    loss = F.nll_loss(preds, gt.long(), weight=class_weights, reduction='none',
                      **kwargs)
    if label_smoothing:
        # maximize the log probs for all classes, and a bit less for the gt
        loss = ((1 - label_smoothing) * loss -
                label_smoothing * valid * preds.sum(1))
    if weight is not None:
        loss = loss * weight
    if ignore is not None and ignore >= 0:
        return loss.sum(not_batch) / valid.sum(not_batch)
    else:
        return loss.mean(not_batch)


def spatial_sum(tensor, ndim=1):
    """
    Sum `tensor` over the last dimensions; keep only the first `ndim` ones.
    """
    if tensor.ndim > ndim:
        return tensor.sum(tuple(range(ndim, tensor.ndim)))
    return tensor


def spatial_mean(tensor, ndim=1):
    """
    Average `tensor` over the last dimensions; keep only the first `ndim` ones.
    """
    if tensor.ndim > ndim:
        return tensor.mean(tuple(range(ndim, tensor.ndim)))
    return tensor


def binary_accuracy(preds, targets, pred_name='output', target_name='mask',
                    ignore=None):
    """
    Computes binary classification accuracy from predictions given as logits,
    for a binary segmentation task, assuming a neutral threshold of 0.5.
    Will ignore labels of value `ignore` if given and nonnegative.
    """
    gt = targets[target_name]
    if ignore is not None and ignore >= 0:
        valid = (gt != ignore)
    gt = gt.bool()
    preds = preds[pred_name] > 0  # corresponds to a sigmoid > 0.5
    correct = (gt == preds)
    if ignore is not None and ignore >= 0:
        correct = spatial_sum((correct & valid).float())
        valid = spatial_sum(valid.float())
        return correct, valid
    else:
        return spatial_mean(correct.float())

multilabel_accuracy = binary_accuracy


def categorical_accuracy(preds, targets, pred_name='output', target_name='map',
                         classwise=True, ignore=None):
    """
    Computes classification accuracy from predictions given as logits, for a
    categorical segmentation task. Will report accuracy separately per class if
    `classwise` is true-ish, otherwise the overall accuracy. Will ignore labels
    of value `ignore` if given and nonnegative.
    """
    gt = targets[target_name].long()
    if ignore is not None and ignore >= 0:
        valid = (gt != ignore)
    batchsize, num_classes = preds[pred_name].shape[:2]
    preds = preds[pred_name].argmax(1)
    correct = (gt == preds)
    if classwise:
        # collapse all spatial dimensions or invent a singleton dimension
        gt = gt.flatten(1) if gt.ndim > 1 else gt[:, None]
        correct = correct.flatten(1) if correct.ndim > 1 else correct[:, None]
        if ignore is not None and ignore >= 0:
            valid = valid.flatten(1) if valid.ndim > 1 else valid[:, None]
            correct = (correct & valid)
            gt = torch.clamp(gt, max=num_classes - 1)
            valid_per_class = torch.zeros(batchsize, num_classes,
                                          device=gt.device).scatter_add_(
                    1, gt, valid.float())
        else:
            valid_per_class = torch.zeros(batchsize, num_classes,
                                          device=gt.device).scatter_add_(
                    1, gt, torch.ones_like(gt, dtype=torch.float))
        correct_per_class = torch.zeros(batchsize, num_classes,
                                        device=gt.device).scatter_add_(
                1, gt, correct.float())
        return correct_per_class, valid_per_class
    else:
        if ignore is not None and ignore >= 0:
            correct = spatial_sum((correct & valid).float())
            valid = spatial_sum(valid.float())
            return correct, valid
        else:
            return spatial_mean(correct.float())


class AggregatedFraction(object):
    """
    Helper class to compute a weighted average.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.num = 0  # aggregated numerator
        self.denom = 0  # aggregated denominator

    def update(self, value, denom=None):
        """
        Adds the given value(s) to the aggregated numerator, and the given
        denominator(s) to the aggregated denuminator. If the latter is omitted,
        adds the number of values to the aggregated denominator.
        """
        try:
            len(value)
        except TypeError:
            # handle a single value
            self.num += value
            self.denom += denom if denom is not None else 1
        else:
            # handle a whole batch
            self.num += value.sum(0)
            self.denom += denom.sum(0) if denom is not None else len(value)

    def __iadd__(self, value):
        self.update(value)

    def result(self):
        return self.num / self.denom


class Welford(AggregatedFraction):
    """
    Helper class to compute a weighted average.
    """
    def reset(self):
        self.n = 0  # item count
        self.m = 0  # aggregated mean

    @staticmethod
    def _safe_div(a, b):
        """
        Returns a / b, but defining 0 / 0 = 0.
        """
        return a / (b + 1 * (a == 0))

    def update(self, value, denom=None):
        """
        Adds the given value(s) to the aggregated numerator, and the given
        denominator(s) to the aggregated denuminator. If the latter is omitted,
        adds the number of values to the aggregated denominator.
        """
        try:
            len(value)
        except TypeError:
            # handle a single value
            self.n += denom if denom is not None else 1
            m = self.m * denom if denom is not None else self.m
            delta = value - m
        else:
            # handle a whole batch
            self.n += denom.sum(0) if denom is not None else len(value)
            m = self.m * denom if denom is not None else self.m
            delta = (value - m).sum(0)
        self.m += self._safe_div(delta, self.n)

    def result(self):
        return self.m


class AverageMetrics(object):
    """
    Helper class that aggregates dictionaries of metrics vectors to
    compute metric-wise averages.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.aggregates = defaultdict(Welford)

    def update(self, metrics):
        for key, value in metrics.items():
            # extract denominator if any
            if isinstance(value, tuple):
                value, denom = value
            else:
                denom = None
            # move to plain numpy
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            if isinstance(denom, torch.Tensor):
                denom = denom.detach().cpu().numpy()
            # append item counts and values
            self.aggregates[key].update(value, denom)

    def aggregate(self):
        return {k: a.result() for k, a in self.aggregates.items()}

    def __iadd__(self, other):
        self.update(other)
        return self
