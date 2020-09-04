# -*- coding: utf-8 -*-

"""
Kaggle birdcall recognition dataset.

Author: Jan Schlüter
"""

import os
import re
import glob

import numpy as np
import pandas as pd
import tqdm

from ... import config
from .. import Dataset, ClassWeightedRandomSampler
from .. import audio
from .. import splitting


def common_shape(arrays):
    """
    Infers the common shape of an iterable of array-likes (assuming all are of
    the same dimensionality). Inconsistent dimensions are replaced with `None`.
    """
    arrays = iter(arrays)
    shape = next(arrays).shape
    for array in arrays:
        shape = tuple(a if a == b else None
                      for a, b in zip(shape, array.shape))
    return shape


class BirdcallDataset(Dataset):
    def __init__(self, itemids, wavs, labelset, annotations=None):
        shapes = dict(input=common_shape(wavs), itemid=())
        dtypes = dict(input=wavs[0].dtype, itemid=str)
        num_classes = len(labelset)
        if annotations is not None:
            if 'label_fg' in annotations:
                shapes['label_fg'] = ()
                dtypes['label_fg'] = np.uint8
            if 'label_bg' in annotations:
                shapes['label_bg'] = (num_classes,)
                dtypes['label_bg'] = np.uint8
            if 'label_all' in annotations:
                shapes['label_all'] = (num_classes,)
                dtypes['label_all'] = np.float32
            if 'rating' in annotations:
                shapes['rating'] = ()
                dtypes['rating'] = np.float32
        super(BirdcallDataset, self).__init__(
            shapes=shapes,
            dtypes=dtypes,
            num_classes=num_classes,
            num_items=len(itemids),
        )
        self.itemids = itemids
        self.wavs = wavs
        self.labelset = labelset
        self.annotations = annotations

    def __getitem__(self, idx):
        # get audio
        item = dict(itemid=self.itemids[idx], input=self.wavs[idx])
        # get targets, if any
        for key in self.shapes:
            if key not in item:
                item[key] = self.annotations[key][idx]
        # return
        return item


def loop(array, length):
    """
    Loops a given `array` along its first axis to reach a length of `length`.
    """
    if len(array) < length:
        array = np.asanyarray(array)
        if len(array) == 0:
            return np.zeros((length,) + array.shape[1:], dtype=array.dtype)
        factor = length // len(array)
        if factor > 1:
            array = np.tile(array, (factor,) + (1,) * (array.ndim - 1))
        missing = length - len(array)
        if missing:
            array = np.concatenate((array, array[:missing:]))
    return array


def crop(array, length, deterministic=False):
    """
    Crops a random excerpt of `length` along the first axis of `array`. If
    `deterministic`, perform a center crop instead.
    """
    if len(array) > length:
        if not deterministic:
            pos = np.random.randint(len(array) - length + 1)
            array = array[pos:pos + length:]
        else:
            l = len(array)
            array = array[(l - length) // 2:(l + length) // 2]
    return array


class FixedSizeExcerpts(Dataset):
    """
    Dataset wrapper that returns batches of random excerpts of the same length,
    cropping or looping inputs along the first axis as needed. If
    `deterministic`, will always do a center crop for too long inputs.
    """
    def __init__(self, dataset, length, deterministic=False, key='input'):
        shapes = dict(dataset.shapes)
        shapes[key] = (length,) + shapes[key][1:]
        super(FixedSizeExcerpts, self).__init__(
                shapes=shapes, dtypes=dataset.dtypes,
                num_classes=dataset.num_classes, num_items=len(dataset))
        self.dataset = dataset
        self.length = length
        self.deterministic = deterministic
        self.key = key

    def __getattr__(self, attr):
        return getattr(self.dataset, attr)

    def __getitem__(self, idx):
        item = dict(self.dataset[idx])
        data = item[self.key]
        if len(data) < self.length:
            data = loop(data, self.length)
        elif len(data) > self.length:
            data = crop(data, self.length, deterministic=self.deterministic)
        item[self.key] = data
        return item


class Floatify(Dataset):
    """
    Dataset wrapper that converts audio samples to float32 with proper scaling,
    possibly transposing the data on the way to swap time and channels.
    """
    def __init__(self, dataset, transpose=False, key='input'):
        dtypes = dict(dataset.dtypes)
        dtypes[key] = np.float32
        shapes = dict(dataset.shapes)
        if transpose:
            shapes[key] = shapes[key][::-1]
        super(Floatify, self).__init__(
                shapes=shapes, dtypes=dtypes,
                num_classes=dataset.num_classes, num_items=len(dataset))
        self.dataset = dataset
        self.transpose = transpose
        self.key = key

    def __getattr(self, attr):
        return getattr(self.dataset, attr)

    def __getitem__(self, idx):
        item = dict(self.dataset[idx])
        data = item[self.key]
        if self.transpose:
            data = np.asanyarray(data).T
        item[self.key] = audio.to_float(data)
        return item


class DownmixChannels(Dataset):
    """
    Dataset wrapper that downmixes multichannel audio to mono, either
    deterministically (method='average') or randomly (method='random_uniform').
    """
    def __init__(self, dataset, key='input', axis=0, method='average'):
        shapes = dict(dataset.shapes)
        shape = list(shapes[key])
        shape[axis] = 1
        shapes[key] = tuple(shape)
        super(DownmixChannels, self).__init__(
                shapes=shapes, dtypes=dataset.dtypes,
                num_classes=dataset.num_classes, num_items=len(dataset))
        self.dataset = dataset
        self.key = key
        self.axis = axis
        self.method = method

    def __getattr(self, attr):
        return getattr(self.dataset, attr)

    def __getitem__(self, idx):
        item = dict(self.dataset[idx])
        wav = item[self.key]
        num_channels = wav.shape[self.axis]
        if num_channels > 1:
            if self.method == 'average':
                wav = np.mean(wav, axis=self.axis, keepdims=True)
            elif self.method == 'random_uniform':
                weights = np.random.dirichlet(np.ones(num_channels))
                weights = weights.astype(wav.dtype)
                if self.axis == -1 or self.axis == len(wav.shape) - 1:
                    wav = np.dot(wav, weights)[..., np.newaxis]
                else:
                    weights = weights[(Ellipsis,) +
                                      (np.newaxis,) *
                                      (len(wav.shape[self.axis:] - 1))]
                    wav = (wav * weights).sum(self.axis, keepdims=True)
        item[self.key] = wav
        return item


def get_itemid(filename):
    """
    Returns the file name without path and without file extension.
    """
    return os.path.splitext(os.path.basename(filename))[0]


def find_files(basedir, regexp):
    """
    Finds all files below `basedir` that match `regexp`, sorted alphabetically.
    """
    regexp = re.compile(regexp)
    return sorted(fn for fn in glob.glob(os.path.join(basedir, '**'),
                                         recursive=True)
                  if regexp.match(fn))


def derive_labelset(train_csv):
    """
    Returns the set of used ebird codes, sorted by latin names.
    """
    labelset_latin = sorted(set(train_csv.primary_label))
    latin_to_ebird = dict(zip(train_csv.primary_label, train_csv.ebird_code))
    labelset_ebird = [latin_to_ebird[latin] for latin in labelset_latin]
    if len(set(labelset_ebird)) != len(labelset_ebird):
        raise RuntimeError("Inconsistent latin names in train.csv!")
    return labelset_ebird


def make_multilabel_target(num_classes, classes):
    """
    Creates a k-hot vector of length `num_classes` with 1.0 at every index in
    `classes`.
    """
    target = np.zeros(num_classes, dtype=np.uint8)
    target[classes] = 1
    return target


def create(cfg, designation):
    config.add_defaults(cfg, pyfile=__file__)
    here = os.path.dirname(__file__)

    # browse for audio files
    basedir = os.path.join(here, cfg['data.audio_dir'])
    audio_files = find_files(basedir, cfg['data.audio_regexp'])
    if cfg['debug']:
        print("Found %d audio files in %s matching %s." %
            (len(audio_files), basedir, cfg['data.audio_regexp']))
    if not audio_files:
        raise RuntimeError("Did not find any audio files in %s matching %s." %
                           (basedir, cfg['data.audio_regexp']))

    # read official train.csv file
    train_csv = pd.read_csv(os.path.join(here, cfg['data.train_csv']),
                                         index_col='filename')

    # derive set of labels, ordered by latin names
    labelset_ebird = derive_labelset(train_csv)
    ebird_to_idx = {ebird: idx for idx, ebird in enumerate(labelset_ebird)}
    num_classes = len(labelset_ebird)

    # for training and validation, read and convert all required labels
    if designation in ('train', 'valid'):
        # combine with additional .csv files
        for fn in cfg['data.extra_csvs'].split(':'):
            train_csv = train_csv.append(pd.read_csv(os.path.join(here, fn),
                                                     index_col='filename'))
        if cfg['debug']:
            print("Found %d entries in .csv files." % len(train_csv))

        # remove file extensions from .csv index column
        train_csv.rename(index=lambda fn: os.path.splitext(fn)[0],
                         inplace=True)

        # add additional ebird codes for inconsistent latin names
        latin_to_ebird = dict(zip(train_csv.primary_label,
                                  train_csv.ebird_code))


        # constrain .csv rows to selected audio files and vice versa
        csv_ids = set(train_csv.index)
        audio_ids = {get_itemid(fn): fn for fn in audio_files}
        audio_ids = {k: fn for k, fn in audio_ids.items() if k in csv_ids}
        train_csv = train_csv.loc[[i in audio_ids for i in train_csv.index]]
        train_csv['audiofile'] = [audio_ids[i] for i in train_csv.index]
        if cfg['debug']:
            print("Found %d entries matching the audio files." %
                len(train_csv))

        # convert foreground and background labels to numbers
        latin_to_idx = {latin: ebird_to_idx[ebird]
                        for latin, ebird in latin_to_ebird.items()}
        train_csv['label_fg'] = [latin_to_idx[latin]
                                 for latin in train_csv.primary_label]
        train_csv['label_bg'] = [
                make_multilabel_target(num_classes,
                                       [latin_to_idx[latin]
                                        for latin in eval(labels)
                                        if latin in latin_to_idx])
                for labels in train_csv.secondary_labels]
        weight_fg = cfg['data.label_fg_weight']
        weight_bg = cfg['data.label_bg_weight']
        label_fg_onehot = np.eye(num_classes,
                                 dtype=np.float32)[train_csv.label_fg]
        label_bg = np.stack(train_csv.label_bg.values)
        train_csv['label_all'] = list(weight_fg * label_fg_onehot +
                                      weight_bg * label_bg)

        # train/valid split
        if cfg['data.split_mode'] == 'byrecordist':
            train_idxs, valid_idxs = splitting.grouped_split(
                    train_csv.index,
                    groups=pd.factorize(train_csv.recordist, sort=True)[0],
                    test_size=(cfg['data.valid_size'] / len(train_csv)
                               if cfg['data.valid_size'] >= 1 else
                               cfg['data.valid_size']),
                    seed=cfg['data.split_seed'])
        elif cfg['data.split_mode'] == 'stratified':
            train_idxs, valid_idxs = splitting.stratified_split(
                    train_csv.index,
                    y=train_csv.label_fg,
                    test_size=cfg['data.valid_size'],
                    seed=cfg['data.split_seed'])
        else:
            raise ValueError("Unknown data.split_mode=%s" % cfg['data.split_mode'])
        if designation == 'train':
            train_csv = train_csv.iloc[train_idxs]
        elif designation == 'valid':
            train_csv = train_csv.iloc[valid_idxs]
        if cfg['debug']:
            print("Kept %d items for this split." % len(train_csv))

        # update audio_files list to match train_csv
        audio_files = train_csv.audiofile
        itemids = train_csv.index
    elif designation == 'test':
        itemids = audio_files

    # prepare the audio files, assume a consistent sample rate
    if not cfg.get('data.sample_rate'):
        cfg['data.sample_rate'] = audio.get_sample_rate(audio_files[0])
    sample_rate = cfg['data.sample_rate']
    # TODO: support .mp3?
    audios = [audio.WavFile(fn, sample_rate=sample_rate)
              for fn in tqdm.tqdm(audio_files, 'Reading audio',
                                  ascii=bool(cfg['tqdm.ascii']))]

    # prepare annotations
    train_csv.rating = train_csv.rating.astype(np.float32)

    # create the dataset
    dataset = BirdcallDataset(itemids, audios, labelset_ebird,
                              annotations=train_csv)

    # unified length, if needed
    if cfg['data.len_min'] < cfg['data.len_max']:
        raise NotImplementedError("data.len_min < data.len_max not allowed yet")
    elif cfg['data.len_max'] > 0:
        dataset = FixedSizeExcerpts(dataset,
                                    int(sample_rate * cfg['data.len_min']),
                                    deterministic=designation != 'train')

    # convert to float and move channel dimension to the front
    dataset = Floatify(dataset, transpose=True)

    # downmixing, if needed
    if cfg['data.downmix'] != 'none':
        dataset = DownmixChannels(dataset,
                                  method=(cfg['data.downmix']
                                          if designation == 'train'
                                          else 'average'))

    # custom sampling
    if cfg['data.class_sample_weights'] and designation == 'train':
        class_weights = cfg['data.class_sample_weights']
        if class_weights not in ('equal', 'roundrobin'):
            class_weights = list(map(float, class_weights.split(',')))
        dataset.sampler = ClassWeightedRandomSampler(train_csv.label_fg,
                                                     class_weights)

    return dataset
