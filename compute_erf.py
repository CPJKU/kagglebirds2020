#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plots the effective receptive field of a neural network.

For usage information, call with --help.

Authors: Jan Schlüter
"""

from __future__ import print_function

import os
import io
from argparse import ArgumentParser
try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np
import tqdm
import torch

from definitions import config
from definitions import (get_dataset,
                         get_dataloader,
                         get_model)
from definitions.datasets import (Dataset,
                                  iterate_data,
                                  print_data_info)
from definitions.models import print_model_info, init_model


def opts_parser():
    descr = ("Plots the effective receptive field of a neural network.")
    parser = ArgumentParser(description=descr)
    parser.add_argument('modelfile', metavar='MODELFILE', nargs='?',
            type=str, default=None,
            help='File to load learned weights from (optional).')
    parser.add_argument('--num-items',
            type=int, default=None,
            help='If given, compute for a limited number of data points.')
    parser.add_argument('--random-data', metavar='CHANNELSxHEIGHTxWIDTH',
            type=sizestr, default=None,
            help='If given, use random inputs instead of a real dataset.')
    parser.add_argument('--save', metavar='FILENAME',
            type=str, default=None,
            help='If given, save the effective receptive field to the given '
                 '.npy or graphics file.')
    parser.add_argument('--plot',
            action='store_true', default=False,
            help='If given, plot the effective receptive field on screen.')
    parser.add_argument('--cuda-device',
            type=int, action='append', default=[],
            help='If given, run on the given CUDA device (starting with 0). '
                 'Can be given multiple times to parallelize over GPUs.')
    config.prepare_argument_parser(parser)
    return parser


def sizestr(s):
    """
    Parses a CHANNELSxHEIGHTxWIDTH string into a tuple of int.
    """
    return tuple(map(int, s.split('x')))


class RandomDataset(Dataset):
    def __init__(self, size, num_classes, num_items):
        super(RandomDataset, self).__init__(
                shapes=dict(input=size),
                dtypes=dict(input=torch.float32),
                num_classes=num_classes,
                num_items=num_items)
        self.size = size

    def __getitem__(self, index):
        return {'input': torch.randn(self.size)}


class Subset(torch.utils.data.Subset):
    def __getattr__(self, attr):
        return getattr(self.dataset, attr)


def crop_center(img, size):
    """
    Crops out the center of `img` according to the given `size` tuple.
    """
    return img[(Ellipsis,) + tuple(slice((x - s) // 2, (x + s) // 2)
                                   for x, s in zip(img.shape[-len(size):],
                                                   size))]


def compute_erf(model, batches, input_name='input'):
    """
    Computes the effective receptive field for the given model using the
    given iterable batches. If `model` has a `predictor` and `frontend`
    submodule, computes the ERF of the predictor with respect to the frontend.
    """
    model.train(False)
    if hasattr(model, 'predictor') and hasattr(model, 'frontend'):
        frontend = model.frontend
        model = model.predictor
    else:
        frontend = None
    total_erf = 0
    count = 0
    for batch in batches:
        data = batch[input_name]
        if frontend is not None:
            with torch.no_grad():
                data = frontend(data)
        if hasattr(model, 'receptive_field'):
            # if the model provides its analytic receptive field size, we can
            # crop the inputs accordingly to save some computation
            data = crop_center(data, model.receptive_field.size)
        data.requires_grad = True
        # pass the batch to the network
        if frontend is None:
            batch[input_name] = data
        else:
            batch = data
        preds = model(batch)
        # collect all outputs with more than 2 dimensions
        if isinstance(preds, dict):
            outputs = list(preds.values())
        elif isinstance(preds, (list, tuple)):
            outputs = preds
        else:
            outputs = [preds]
        outputs = [v for v in outputs if v.dim() > 2]
        # set the gradients for the central pixel of each output to 1.0
        grads = [torch.zeros_like(v) for v in outputs]
        for g in grads:
            non_spatial_dims = min(g.dim() - 2, 1)
            center = ((Ellipsis,) * non_spatial_dims +
                      tuple(s // 2 for s in g.shape[non_spatial_dims:]))
            g[center] = 1
        # backpropagate to the input
        erf, = torch.autograd.grad(outputs, [data], grads)
        total_erf += erf.abs().sum(0)
        count += len(erf)
    # return accumulated effective receptive field
    return total_erf / count


def imsave(fn, img):
    """
    Save numpy array `img` as a grayscale image under file name `fn`.
    """
    from PIL import Image
    if len(img.shape) == 3:
        img = img[0]
    img -= img.min()
    img /= img.max()
    img = np.multiply(img, 255, np.empty(img.shape, np.uint8),
                      casting='unsafe')
    Image.fromarray(img).save(fn)


def main():
    # parse command line
    parser = opts_parser()
    options = parser.parse_args()
    if not options.save and not options.plot:
        parser.error('Either pass --save or --plot (or both).')
    modelfile = options.modelfile
    if modelfile and os.path.exists(os.path.splitext(modelfile)[0] + '.vars'):
        options.vars.insert(1, os.path.splitext(modelfile)[0] + '.vars')
    cfg = config.from_parsed_arguments(options)
    if not options.cuda_device:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:%d' % options.cuda_device[0])
        torch.cuda.set_device(options.cuda_device[0])

    # prepare test data generator
    print("Preparing data reading...")
    if options.random_data:
        test_data = RandomDataset(options.random_data,
                                  num_classes=cfg['data.num_classes'],
                                  num_items=options.num_items or 32)
    else:
        test_data = get_dataset(cfg, 'test')
        print_data_info(test_data)
    # limit to the requested number of items
    if options.num_items:
        test_data = Subset(test_data, range(options.num_items))
    test_loader = get_dataloader(cfg, test_data, 'test')
    # start the generator in a background thread
    test_batches = iterate_data(iter(test_loader), device, cfg)

    # prepare model
    print("Preparing network...")
    # instantiate neural network
    model = get_model(cfg, test_data.shapes, test_data.dtypes,
                      test_data.num_classes, options.cuda_device)
    print_model_info(model)
    if options.modelfile:
        # restore state dict
        state_dict = torch.load(options.modelfile,
                                map_location=device)
        model.load_state_dict(state_dict)
        del state_dict
    else:
        # run custom initializations
        init_model(model, cfg)

    # compute ERF
    print("Computing:")
    try:
        num_batches = len(test_data) // cfg['batchsize']
    except TypeError:
        num_batches = None
    erf = compute_erf(model, tqdm.tqdm(test_batches, 'Batch',
                                       total=num_batches))

    # save ERF if needed
    if options.save:
        if options.save.endswith('.npy'):
            np.save(options.save, erf)
        else:
            imsave(options.save, erf.cpu().numpy())

    # show ERF if needed
    if options.plot:
        from matplotlib import pyplot as plt
        if len(erf.shape) == 3:
            erf = erf[0]
        plt.imshow(erf, cmap='gray')
        plt.show()

if __name__ == "__main__":
    main()
