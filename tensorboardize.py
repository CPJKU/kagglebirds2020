#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Saves statistics and graphics for tensorboard.
Can be called as a script for a trained model, or imported and used during
training.

For usage information, call with --help.

Authors: Jan Schlüter
"""

from __future__ import print_function

import os
import io
from argparse import ArgumentParser
import itertools
import colorsys

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from definitions import config
from definitions import (get_dataset,
                         get_dataloader,
                         get_model)
from definitions.datasets import (Dataset,
                                  iterate_data,
                                  print_data_info)
from definitions.models import print_model_info, PutDictKey
from compute_erf import compute_erf


def opts_parser():
    descr = ("Saves statistics and graphics for tensorboard.")
    parser = ArgumentParser(description=descr)
    parser.add_argument('modelfile', metavar='MODELFILE',
            type=str, default=None,
            help='File to load learned weights from. Also expects a '
                 '*.hist.npz file of the same name, with a different '
                 'extension.')
    parser.add_argument('logdir', metavar='LOGDIR',
            type=str, default=None,
            help='Directory to write the logs to. Will be created if needed.')
    parser.add_argument('--cuda-device',
            type=int, action='append', default=[],
            help='If given, run on the given CUDA device (starting with 0). '
                 'Can be given multiple times to parallelize over GPUs.')
    config.prepare_argument_parser(parser)
    return parser


class TensorboardLogger(object):
    """
    Initializes a logger for writing tensorboard artifacts. The target log
    directory is mandatory, the configuration, model, dataloader and optimizer
    enable different artifacts to be logged, but are optional. The model graph
    is only logged if requested with `include_graph`; it's not very useful.
    """
    def __init__(self, logdir, cfg=None, dataloader=None, model=None,
                 optimizer=None, include_graph=False):
        self.writer = SummaryWriter(logdir)
        self.cfg = cfg
        self.dataloader = dataloader
        self.model = model
        self.optimizer = optimizer
        self.include_graph = include_graph

    def log_start(self, ignore_dataloader=False, ignore_model=False,
                  ignore_optimizer=False):
        if self.cfg is not None:
            self.log_config()
        use_model = not ignore_model and self.model is not None
        use_dataloader = not ignore_dataloader and self.dataloader is not None
        if use_model and use_dataloader:
            self.log_model_graph()
            self.log_erf(initial=True)
        if use_dataloader:
            self.log_input()

    def log_epoch(self, epoch, metrics, ignore_dataloader=False,
                  ignore_model=False, ignore_optimizer=False):
        self.log_metrics(epoch, metrics)
        use_model = not ignore_model and self.model is not None
        use_dataloader = not ignore_dataloader and self.dataloader is not None
        use_optimizer = not ignore_optimizer and self.optimizer is not None
        if use_model and use_dataloader:
            self.log_erf(epoch)
            self.log_output(epoch)
        if use_optimizer:
            self.log_learning_rate(epoch)
        self.log_memory_usage(epoch)

    def log_end(self, history, ignore_dataloader=False, ignore_model=False,
                ignore_optimizer=False):
        final_results = []
        for k, v in history.items():
            if '__' in k:
                continue
            v = v[-1]
            if hasattr(v, 'shape') and len(v.shape) == 1:
                final_results.append('*%s*: ' % k +
                                     ' '.join('%.3f' % x for x in v) +
                                     ' (%.3f)' % v.mean())
            else:
                final_results.append('*%s*: %.3f' % (k, v))
        final_results.sort()
        epochs = len(next(iter(history.values())))
        duration = '%d epochs' % epochs
        if self.cfg is not None:
            epochsize = self.cfg['train.epochsize']
            batchsize = self.cfg['batchsize']
            duration += ' of %d batches of %d items (= %d total examples)' % (
                    epochsize, batchsize, epochs * epochsize * batchsize)
        final_results.insert(0, 'Trained %s.' % duration)
        self.writer.add_text('final_results', '<br/>\n'.join(final_results),
                             epochs)
        self.writer.flush()

    def get_batches(self, num_batches=None):
        batches = iter(self.dataloader)
        if num_batches:
            batches = itertools.islice(batches, num_batches)
        device = next(self.model.parameters()).device
        return iterate_data(batches, device)

    def log_config(self):
        cfg = "<br/>\n".join("%s=%s" % (k, self.cfg[k])
                             for k in sorted(self.cfg.keys()))
        self.writer.add_text('vars', cfg)

    def log_model_graph(self):
        if not self.include_graph:
            return
        batch = next(self.get_batches(1))
        # add_graph only allows (nested collections of) Tensors
        if isinstance(batch, dict):
            for k in tuple(batch.keys()):
                v = batch[k]
                if not(isinstance(v, torch.Tensor) or
                       isinstance(v[0], torch.Tensor)):
                    del batch[k]
        # add_graph disallows dictionary outputs
        model = self.model
        if (isinstance(model, torch.nn.Sequential) and
                isinstance(model[-1], PutDictKey)):
            model = torch.nn.Sequential(model._modules)
            del model[-1]
        # call add_graph
        try:
            self.writer.add_graph(model, batch)
        except Exception:
            print("Warning: could not log model graph")

    def log_data_dict(self, data, prefix='', epoch=None):
        for k, v in data.items():
            if hasattr(v, 'shape'):
                data_format = 'NCHW'
                if len(v.shape) == 3:
                    if v.shape[1] < 5000 and v.shape[2] < 5000:
                        # add a singleton channel dimension to handle it below
                        v = v[:, np.newaxis]
                    elif v.shape[1] <= 2 and 'data.sample_rate' in self.cfg:
                        # could be a (batchsize, channels, time) waveform
                        sample_rate = self.cfg['data.sample_rate']
                        if v.shape[2] / sample_rate > 0.1:  # if at least 0.1s
                            self.writer.add_audio(prefix + k, v[0], epoch,
                                                  sample_rate=sample_rate)
                if len(v.shape) == 4:
                    if k == 'frontend':
                        v = v[:, :1]
                        v = (v - v.min()) / (v.max() - v.min())
                    elif (v.shape[1] == 1 and
                            v.dtype.is_floating_point and
                            (v.max() > 1 or v.min() < 0)):
                        v = v.sigmoid()
                    if v.shape[1] not in (1, 3):
                        v = v.argmax(1, keepdim=True)
                    if (v.shape[1] == 1 and
                            not v.dtype.is_floating_point and
                            not v.dtype == torch.bool):
                        # generate unique colors for each class index
                        classes = self.cfg.get('data.num_classes', 30)
                        colors = [colorsys.hls_to_rgb(hue, 0.6, 1)
                                  for hue in np.linspace(0, 1, classes,
                                                         endpoint=False)]
                        # ensure 255 is mapped to black
                        colors = torch.cat(
                                (torch.tensor(colors),
                                 torch.zeros((256 - len(colors), 3))))
                        # drop channel dimension and replace ints by colors
                        v = colors[v[:, 0].to(torch.long)]
                        data_format = 'NHWC'
                    self.writer.add_images(prefix + k, v, epoch,
                                           dataformats=data_format)

    def log_input(self):
        batch = next(iter(self.dataloader))
        self.log_data_dict(batch, 'input/')

    def log_output(self, epoch=None):
        batch = next(self.get_batches(1))
        kwargs = {}
        if hasattr(self.model, 'frontend'):
            kwargs['extra_outputs'] = ['frontend']
        preds = self.model(batch, **kwargs)
        self.log_data_dict(preds, 'output/', epoch)

    def log_erf(self, epoch=None, initial=False, num_batches=10):
        batches = self.get_batches(num_batches)
        erf = compute_erf(self.model, batches)
        erf -= erf.min()
        erf /= erf.max()
        tag = 'erf/initial' if initial else 'erf/trained'
        self.writer.add_image(tag, erf, epoch)

    def log_metrics(self, epoch, metrics):
        for k, v in metrics.items():
            datasplit, kind = k.split('_', 1)
            if kind.startswith('_'):
                continue
            if hasattr(v, 'mean'):
                v = v.mean()
            self.writer.add_scalar('%s/%s' % (kind, datasplit), v, epoch)

    def log_learning_rate(self, epoch):
        for i, pg in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar('lr' if i == 0 else 'lr.%d' % i, pg['lr'],
                                   epoch)

    def log_memory_usage(self, epoch):
        if torch.cuda.is_available():
            self.writer.add_scalar('memory/cuda_alloc',
                                   torch.cuda.memory_allocated(), epoch)
            self.writer.add_scalar('memory/cuda_cached',
                                   torch.cuda.memory_reserved(), epoch)
            self.writer.add_scalar('memory/cuda_alloc_max',
                                   torch.cuda.max_memory_allocated(), epoch)
            self.writer.add_scalar('memory/cuda_cached_max',
                                   torch.cuda.max_memory_reserved(), epoch)


def main():
    # parse command line
    parser = opts_parser()
    options = parser.parse_args()
    modelfile = options.modelfile
    if os.path.exists(os.path.splitext(modelfile)[0] + '.vars'):
        options.vars.insert(1, os.path.splitext(modelfile)[0] + '.vars')
    cfg = config.from_parsed_arguments(options)
    if not options.cuda_device:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:%d' % options.cuda_device[0])
        torch.cuda.set_device(options.cuda_device[0])

    # prepare validation data generator
    print("Preparing data reading...")
    valid_data = get_dataset(cfg, 'valid')
    print_data_info(valid_data)
    valid_loader = get_dataloader(cfg, valid_data, 'valid')

    # prepare model
    print("Preparing network...")
    # instantiate neural network
    model = get_model(cfg, valid_data.shapes, valid_data.dtypes,
                      valid_data.num_classes, options.cuda_device)
    print_model_info(model)

    # create logger
    logger = TensorboardLogger(options.logdir, cfg=cfg,
                               dataloader=valid_loader, model=model)

    # populate pre-training artifacts
    print("Logging pre-training artifacts...")
    logger.log_start()

    # populate training artifacts
    histfile = os.path.splitext(modelfile)[0] + '.hist.npz'
    hist = np.load(histfile)
    metric_keys = hist.files
    print("Logging training artifacts...")
    epochs = len(hist[metric_keys[0]])
    for epoch, metric_values in enumerate(zip(*(hist[k]
                                                for k in metric_keys))):
        metrics = dict(zip(metric_keys, metric_values))
        if epoch < epochs - 1:
            logger.log_epoch(epoch, metrics, ignore_model=True)
        else:
            # last epoch: load final model weights and allow using the model
            model.load_state_dict(torch.load(options.modelfile,
                                             map_location=device))
            logger.log_epoch(epoch, metrics)

    # populate post-training artifacts
    print("Logging post-training artifacts...")
    logger.log_end(hist)


if __name__ == "__main__":
    main()
