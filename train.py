#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trains a neural network.

For usage information, call with --help.

Author: Jan Schlüter
"""

from __future__ import print_function

import os
import sys
import time
from argparse import ArgumentParser
from collections import OrderedDict

import tqdm
import numpy as np
import torch

from definitions import config
from definitions import (get_dataset,
                         get_dataloader,
                         get_model,
                         get_metrics,
                         get_loss_from_metrics,
                         get_optimizer)
from definitions.datasets import (iterate_infinitely,
                                  iterate_data,
                                  print_data_info)
from definitions.models import print_model_info, init_model
from definitions.metrics import AverageMetrics, print_metrics


def opts_parser():
    descr = "Trains a neural network."
    parser = ArgumentParser(description=descr)
    parser.add_argument('modelfile', metavar='MODELFILE',
            type=str,
            help='File to save the learned weights to')
    parser.add_argument('--logdir',
            type=str, default=None,
            help='Directory to write tensorboard log files to (optional). '
                 'Will be created if needed.')
    parser.add_argument('--cuda-device',
            type=int, action='append', default=[],
            help='If given, run on the given CUDA device (starting with 0). '
                 'Can be given multiple times to parallelize over GPUs.')
    parser.add_argument('--cuda-sync-mode',
            type=str, choices=('auto', 'spin', 'yield', 'block'),
            default='block',
            help='If running on GPU, how to synchronize the host to the GPU: '
                 'auto, spin, yield or block (default: %(default)s).')
    parser.add_argument('--no-resume',
            action='store_false', dest='resume', default=True,
            help='If given, does not continue training from a MODELFILE.resume '
                 'file if it exists, but always starts from scratch.')
    config.prepare_argument_parser(parser)
    return parser


def save_model(filename, model, cfg):
    """
    Saves the learned weights to `filename`, and the corresponding
    configuration to ``os.path.splitext(filename)[0] + '.vars'``.
    """
    config.write_config_file(os.path.splitext(filename)[0] + '.vars', cfg)
    torch.save(model.state_dict(), filename)


def log_metrics(train_values, valid_values, history, modelfile):
    """
    Save all metrics into a history (a dict of lists), and as a file.
    """
    values = {'train_' + k: v for k, v in train_values.items()}
    values.update(('valid_' + k, v) for k, v in valid_values.items())
    for k, v in values.items():
        try:
            history[k].append(v)
        except KeyError:
            history[k] = [v]
    np.savez(modelfile.rsplit('.', 1)[0] + '.hist.npz', **history)


def set_cuda_sync_mode(mode):
    """
    Set the CUDA device synchronization mode: auto, spin, yield or block.
    auto: Chooses spin or yield depending on the number of available CPU cores.
    spin: Runs one CPU core per GPU at 100% to poll for completed operations.
    yield: Gives control to other threads between polling, if any are waiting.
    block: Lets the thread sleep until the GPU driver signals completion.
    """
    import ctypes
    try:
        ctypes.CDLL('libcudart.so').cudaSetDeviceFlags(
                {'auto': 0, 'spin': 1, 'yield': 2, 'block': 4}[mode])
    except Exception:
        pass


def add_optimizer_params(optimizer, scheduler, params, eta_scale=1):
    """
    Add a parameter group to the optimizer and scheduler, optionally with a
    scaled learning rate (relative to the first existing parameter group).
    """
    optimizer.add_param_group(dict(
            params=params, lr=(optimizer.param_groups[0]['lr'] * eta_scale)))
    scheduler.min_lrs.append(scheduler.min_lrs[0] * eta_scale)


def main():
    # parse command line
    parser = opts_parser()
    options = parser.parse_args()
    modelfile = options.modelfile
    cfg = config.from_parsed_arguments(options)
    if not options.cuda_device:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:%d' % options.cuda_device[0])
        torch.cuda.set_device(options.cuda_device[0])
        if options.cuda_sync_mode != 'auto':
            set_cuda_sync_mode(options.cuda_sync_mode)

    # prepare training data generator
    print("Preparing training data feed...")
    train_data = get_dataset(cfg, 'train')
    print_data_info(train_data)
    train_loader = get_dataloader(cfg, train_data, 'train')

    # start training data generation in background
    train_batches = iterate_infinitely(train_loader)
    train_batches = iterate_data(train_batches, device, cfg)

    # if told so, benchmark the creation of a given number of minibatches
    if cfg.get('benchmark_datafeed'):
        print("Benchmark: %d minibatches of %d items..." %
              (cfg['benchmark_datafeed'], cfg['batchsize']))
        import itertools
        t0 = time.time()
        next(itertools.islice(train_batches,
                              cfg['benchmark_datafeed'],
                              cfg['benchmark_datafeed']), None)
        t1 = time.time()
        print("%.3gs per minibatch." % ((t1 - t0) / cfg['benchmark_datafeed']))
        return

    # prepare validation data generator
    print("Preparing validation data feed...")
    val_data  = get_dataset(cfg, 'valid')
    print_data_info(val_data)
    val_loader = get_dataloader(cfg, val_data, 'valid')

    # enable cuDNN auto-tuning if on GPU and all data sizes are constant
    if options.cuda_device and not any(s is None
                                       for data in (train_data, val_data)
                                       for shape in data.shapes.values()
                                       for s in shape):
        torch.backends.cudnn.benchmark = True

    # prepare model
    print("Preparing network...")
    # instantiate neural network
    model = get_model(cfg, train_data.shapes, train_data.dtypes,
                      train_data.num_classes, options.cuda_device)
    print(model)
    print_model_info(model)

    # obtain cost functions
    train_metrics = get_metrics(cfg, 'train')
    val_metrics = get_metrics(cfg, 'valid')
    extract_loss = get_loss_from_metrics(cfg)

    # initialize optimizer
    params = model.parameters()
    if cfg['train.first_params']:
        first_params_count = cfg['train.first_params']
        # if a string, treat as a submodule name, figure out its param count
        if isinstance(first_params_count, str):
            first_params_count = len(list(getattr(
                    model, first_params_count).parameters()))
        # advance the `params` iterator, keep the first parameters separately
        params = iter(params)
        first_params = [next(params) for _ in range(first_params_count)]
    optimizer = get_optimizer(cfg, params)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=cfg['train.eta_decay'],
            patience=cfg['train.patience'], cooldown=cfg['train.cooldown'],
            verbose=True)

    # initialize mixed-precision training
    if cfg['float16']:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=cfg['float16.opt_level'])

    # initialize tensorboard logger, if requested
    if options.logdir:
        from tensorboardize import TensorboardLogger
        logger = TensorboardLogger(options.logdir, cfg=cfg,
                                   dataloader=val_loader, model=model,
                                   optimizer=optimizer)
    else:
        logger = None

    # resume training state if possible
    if options.resume and os.path.exists(options.modelfile + '.resume'):
        state = torch.load(options.modelfile + '.resume', map_location=device)
        model.load_state_dict(state.pop('model'))
        optimizer.load_state_dict(state.pop('optimizer'))
        scheduler.load_state_dict(state.pop('scheduler'))
        history = state.pop('history')
        epoch = state.pop('epoch')
        if cfg['float16']:
            amp.load_state_dict(state.pop('amp'))
        if (cfg['train.first_params'] and
                epoch > cfg['train.first_params.delay']):
            add_optimizer_params(optimizer, scheduler, first_params,
                                 cfg['train.first_params.eta_scale'])
    else:
        history = {}
        epoch = 0
        # load pretrained weights if requested
        if cfg['model.init_from']:
            model.load_state_dict(torch.load(
                    os.path.join(os.path.dirname(__file__),
                                 cfg['model.init_from'])),
                    map_location=device)
        else:
            # run custom initializations
            init_model(model, cfg)
        # log initial state
        if logger is not None:
            logger.log_start()

    # warn about unused configuration keys
    config.warn_unused_variables(
            cfg, ('train.epochs', 'train.epochsize', 'train.min_eta',
                  'train.patience_reference','loss'))

    # run training loop
    print("Training:")
    for epoch in range(epoch, cfg['train.epochs']):
        # add first_params to optimizer when the delay has passed
        if (cfg['train.first_params'] and
                cfg['train.first_params.delay'] == epoch):
            add_optimizer_params(optimizer, scheduler, first_params,
                                 cfg['train.first_params.eta_scale'])
            if cfg['debug']:
                print('Training first %d parameters with learning rate '
                      'scaled by %f.' % (first_params_count,
                                         cfg['train.first_params.eta_scale']))
        # training pass
        model.train(True)
        if cfg['debug']:
            torch.autograd.set_detect_anomaly(True)
        train_errors = AverageMetrics()
        nans_in_a_row = 0
        for _ in tqdm.trange(
                cfg['train.epochsize'],
                desc='Epoch %d/%d' % (epoch + 1, cfg['train.epochs']),
                ascii=bool(cfg['tqdm.ascii'])):
            # grab the next minibatch
            batch = next(train_batches)
            # reset gradients
            optimizer.zero_grad()
            # compute output and error
            preds = model(batch)
            metrics = OrderedDict((k, fn(preds, batch))
                                  for k, fn in train_metrics.items())
            loss = extract_loss(metrics)
            # bail out if Not a Number
            if not np.isfinite(loss.item()):
                if cfg['debug']:
                    raise RuntimeError('Training error is NaN!')
                nans_in_a_row += 1
                if nans_in_a_row < 5:
                    print('Training error is NaN! Skipping step.')
                    continue
                else:
                    print('Training error is NaN! Stopping training.')
                    return 1
            else:
                nans_in_a_row = 0
            train_errors += metrics
            train_errors += {'loss': loss.item()}
            # backprop and update
            if cfg['float16']:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
        print_metrics('Train', train_errors.aggregate())
        del batch, preds, loss

        # validation pass
        model.train(False)
        val_errors = AverageMetrics()
        for batch in iterate_data(iter(val_loader), device, cfg):
            with torch.no_grad():
                preds = model(batch)
                metrics = {k: fn(preds, batch) for k, fn in val_metrics.items()}
            val_loss = float(extract_loss(metrics).item())
            val_errors += metrics
            val_errors += {'loss': val_loss}
        print_metrics('Valid', val_errors.aggregate())
        del batch, preds, val_loss

        log_metrics(train_errors.aggregate(), val_errors.aggregate(),
                    history, modelfile)
        if logger is not None:
            logger.log_epoch(epoch, {k: v[-1] for k, v in history.items()})

        # learning rate update
        reference = history[cfg['train.patience_reference'].lstrip('-')][-1]
        if hasattr(reference, 'mean'):
            reference = reference.mean()
        if cfg['train.patience_reference'].startswith('-'):
            reference *= -1
        scheduler.step(reference)
        if optimizer.param_groups[0]['lr'] < cfg['train.min_eta']:
            print('Learning rate fell below threshold. Stopping training.')
            break

        # save training state to resume file
        resume_state = dict(model=model.state_dict(),
                            optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict(),
                            epoch=epoch + 1, history=history)
        if cfg['float16']:
            resume_state['amp'] = amp.state_dict()
        torch.save(resume_state, options.modelfile + '.resume')
        del resume_state


    # save final network and the configuration used
    print("Saving final model")
    save_model(modelfile, model, cfg)

    # delete resume file if any
    if os.path.exists(options.modelfile + '.resume'):
        os.remove(options.modelfile + '.resume')

    # log the final state
    if logger is not None:
        logger.log_end(history)


if __name__ == "__main__":
    sys.exit(main() or 0)
