# -*- coding: utf-8 -*-

"""
Dataset definitions.

Author: Jan Schlüter
"""
from __future__ import print_function

import importlib
import functools
import itertools

import torch
from torch.utils.data.dataloader import default_collate


class Dataset(torch.utils.data.Dataset):
    def __init__(self, shapes, dtypes, num_classes, num_items=None):
        super(Dataset, self).__init__()
        self.shapes = shapes
        self.dtypes = dtypes
        self.num_classes = num_classes
        self.num_items = num_items

    def __len__(self):
        if self.num_items is None:
            raise TypeError('len() not implemented for this Dataset')
        return self.num_items

    def collate(self, data):
        # override in subclasses if needed
        return default_collate(data)


class ClassWeightedRandomSampler(torch.utils.data.Sampler):
    """
    Provides a sampler with fixed class probabilities. Requires an iterable
    giving the class index for each item of the dataset. Optionally takes a
    tensor of weights per class or a string; if `'equal'`, will weight all
    classes equally, if `'roundrobin'`, will sample classes strictly in a
    round-robin manner. This sampler will run infinitely long, drawing from
    each class without replacement.
    """
    def __init__(self, item_classes, class_weights='equal'):
        super(ClassWeightedRandomSampler, self).__init__(None)
        item_classes = torch.as_tensor(item_classes, dtype=torch.uint8)
        if not isinstance(class_weights, str):
            class_weights = torch.as_tensor(class_weights, dtype=torch.double)
            num_classes = len(class_weights)
        else:
            num_classes = item_classes.max() + 1
        self.class_samplers = [
                torch.utils.data.SubsetRandomSampler(
                        torch.nonzero(item_classes == c)[:, 0])
                for c in range(num_classes)]
        self.num_classes = num_classes
        self.class_weights = class_weights

    def __iter__(self):
        class_iters = [iterate_infinitely(sampler)
                       for sampler in self.class_samplers]
        if torch.is_tensor(self.class_weights):
            class_sampler = functools.partial(torch.multinomial,
                                              self.class_weights, 1)
        elif self.class_weights == 'equal':
            class_sampler = functools.partial(torch.randint,
                                              self.num_classes, (1,))
        elif self.class_weights == 'roundrobin':
            classes = itertools.cycle(range(self.num_classes))
            class_sampler = lambda: [next(classes)]
        else:
            raise ValueError("Unsupported class_weights=%r" %
                             self.class_weights)
        while True:
            yield next(class_iters[class_sampler()[0]])


def get_dataset(cfg, designation):
    """
    Return a Dataset for the given designation ('train', 'valid', 'test').
    """
    dataset = importlib.import_module('.' + cfg['dataset'], __package__)
    return dataset.create(cfg, designation)


def get_dataloader(cfg, dataset, designation):
    """
    Return a DataLoader for the given Dataset and designation ('train',
    'valid', or 'test').
    """
    kwargs = {}
    sampler = getattr(dataset, 'sampler', None)
    if sampler is not None:
        kwargs['sampler'] = sampler
    else:
        kwargs['shuffle'] = (designation == 'train')
    collate_fn = getattr(dataset, 'collate', None)
    return torch.utils.data.DataLoader(dataset, cfg['batchsize'],
                                       drop_last=(designation == 'train'),
                                       collate_fn=collate_fn, **kwargs)


def iterate_infinitely(iterable):
    """
    Yields items from an iterable until exhausted, then restarts, forever.
    """
    while True:
        for item in iter(iterable):
            yield item


def iterate_in_background(iterable, num_cached=10):
    """
    Runs an iterator in a background thread, caching up to `num_cached` items.
    """
    try:
        from Queue import Queue
    except ImportError:
        from queue import Queue
    queue = Queue(maxsize=num_cached)
    sentinel = object()  # guaranteed unique reference

    # define producer (putting items into queue)
    def producer():
        try:
            for item in iterable:
                queue.put(item)
            queue.put(sentinel)
        except Exception as e:
            import traceback
            queue.put(type(e)('%s caught in background.\nOriginal %s' %
                              (e.args[0], traceback.format_exc())))

    # start producer (in a background thread)
    import threading
    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()

    # run as consumer (read items from queue, in current thread)
    item = queue.get()
    while item is not sentinel:
        if isinstance(item, Exception):
            raise item
        yield item
        item = queue.get()


def apply_to_collection(data, fn):
    """
    Applies a given function recursively to a tuple, list or dictionary of
    things with arbitrary nesting (including none). Returns the new collection.
    """
    if isinstance(data, (tuple, list)):
        return type(data)(apply_to_collection(item, fn) for item in data)
    elif isinstance(data, dict):
        return type(data)((k, apply_to_collection(v, fn))
                          for k, v in data.items())
    else:
        return fn(data)


def iterate_to_device(iterable, device):
    """
    Moves all PyTorch tensors in an iterable to the specified device.
    """
    for item in iterable:
        yield apply_to_collection(
                item, lambda x: x.to(device) if hasattr(x, 'to') else x)


def iterate_data(iterable, device, cfg=None):
    """
    Convenience method calling iterate_in_background(), unless `cfg` is a
    dictionary with `'singlethreaded'` mapped to true-ish, and
    iterate_to_device().
    """
    if not cfg or not cfg.get('singlethreaded', False):
        iterable = iterate_in_background(iterable)
    return iterate_to_device(iterable, device)


def print_data_info(dataset):
    """
    Prints the shapes and dtypes of a given dataset.
    """
    print("Got %d items with fields:" % len(dataset))
    for name, shape in dataset.shapes.items():
        print("- %s: %r, %s" % (name, shape, dataset.dtypes[name].__name__))
