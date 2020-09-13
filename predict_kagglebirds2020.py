#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute bird classification predictions for submitting to Kaggle.

For usage information, call with --help.

Author: Jan Schlüter
"""

from __future__ import print_function

import os
import io
from argparse import ArgumentParser
import warnings

import numpy as np
import tqdm
import torch
import pandas as pd

from definitions import config, get_model
from definitions.models import init_model
from definitions.datasets import audio, iterate_data
from definitions.datasets.kagglebirds2020 import derive_labelset


def opts_parser():
    descr = ("Compute bird classification predictions for submitting to "
             "Kaggle.")
    parser = ArgumentParser(description=descr)
    parser.add_argument('modelfile', metavar='MODELFILE', nargs='*',
            type=str,
            help='File to load the learned weights from (.npz format), can be '
                 'given multiple times to average predictions.')
    parser.add_argument('outfile', metavar='OUTFILE',
            type=str,
            help='File to save the predictions to (.csv format)')
    parser.add_argument('--train-csv',
            type=str,
            help='File to read the set of class labels from.')
    parser.add_argument('--test-csv',
            type=str,
            help='File to read the test segment definitions from (.csv format)')
    parser.add_argument('--test-audio',
            type=str,
            help='Directory to find the test audio files in (.mp3 format)')
    parser.add_argument('--local-then-global',
            action='store_true', default=False,
            help='If given, to compute global predictions, first apply the '
                 'backend to chunks of data.len_max, then apply global max '
                 'pooling over these chunks.')
    parser.add_argument('--local-then-global-overlap',
            type=float, default=0,
            help='Proportion of overlap of local chunks for --local-then-'
                 'global, in range [0, 1) (default: %(default)s).')
    parser.add_argument('--threshold',
            type=float, default=0,
            help='Binarization threshold (default: %(default)s)')
    parser.add_argument('--threshold-convert',
            type=str, choices=('none', 'logit'), default='none',
            help='Optional function to apply to --threshold: none or logit '
                 '(to convert probabilities to logits) (default: %(default)s)')
    parser.add_argument('--postprocess',
            type=str, action='append', choices=('sigmoid',), default=[],
            help='Postprocessing operation to apply (before bagging), can be '
                 'given multiple times. Choices are: sigmoid')
    parser.add_argument('--cuda-device',
            type=int, action='append', default=[],
            help='If given, run on the given CUDA device (starting with 0). '
                 'Can be given multiple times to parallelize over GPUs.')
    config.prepare_argument_parser(parser)
    return parser


def pool_chunkwise(preds, times, backend, chunk_ends, chunk_lens=5):
    chunks = []
    for start, end in zip(chunk_ends - chunk_lens, chunk_ends):
        a, b = np.searchsorted(times, [start, end])
        chunk_preds = preds[..., a:b]
        with torch.no_grad():
            pooled = backend(chunk_preds).detach().cpu().numpy().ravel()
        chunks.append(pooled)
    return np.stack(chunks)


def pool_local_then_global(preds, times, backend, chunk_len, chunk_overlap=0):
    # compute hop size in seconds from proportion of overlap
    chunk_hop = chunk_len * (1 - chunk_overlap)
    # figure out how many windows fully fit into the input
    num_full_chunks = int((times[-1] - chunk_len) / chunk_hop) + 1
    # compute their endings, and add one that ends at the end of the input
    chunk_ends = np.arange(1, num_full_chunks + 2) * chunk_hop
    chunk_ends[-1] = times[-1]
    # compute local predictions
    predictions = pool_chunkwise(preds, times, backend, chunk_ends, chunk_len)
    # apply max pooling over chunkwise predictions
    return predictions.max(0, keepdims=True)


def pool_global(preds, backend):
    with torch.no_grad():
        pooled = backend(preds).detach().cpu().numpy().ravel()
    return pooled[np.newaxis]


def derive_labels(preds, row_ids, labelset, threshold=0):
    chunks = []
    for row_id, chunk_preds in zip(row_ids, preds):
        birds = np.where(chunk_preds > threshold)[0]
        birds = ' '.join(labelset[bird] for bird in birds) or 'nocall'
        chunks.append({'row_id': row_id, 'birds': birds})
    return pd.DataFrame(chunks)


def modelconfigfile(modelfile):
    """Derive the file name of a model-specific config file"""
    return os.path.splitext(modelfile)[0] + '.vars'


def postprocess(preds, methods):
    """
    Apply postprocessing `methods` to `preds` of shape (items, classes, time).
    """
    for method in methods:
        if method == 'sigmoid':
            preds = torch.as_tensor(preds).sigmoid().numpy()
        else:
            raise ValueError("Unknown postprocessing method %s" % method)
    return preds


def main():
    # parse command line
    parser = opts_parser()
    options = parser.parse_args()
    if not (options.train_csv and options.test_csv and options.test_audio):
        parser.error('requires --train-csv, --test-csv, --test-audio')
    modelfiles = options.modelfile
    outfile = options.outfile
    if modelfiles:
        if os.path.exists(modelconfigfile(modelfiles[0])):
            options.vars.insert(1, modelconfigfile(modelfiles[0]))
    cfg = config.from_parsed_arguments(options)
    extra_cfg = config.parse_variable_assignments(options.var)
    if not options.cuda_device:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:%d' % options.cuda_device[0])
        torch.cuda.set_device(options.cuda_device[0])

    # prepare dataset
    print("Preparing data reading...")
    test_dir = options.test_audio
    test_csv = pd.read_csv(options.test_csv)
    filelist = test_csv.audio_id.unique()

    # - mp3 loading generator
    sample_rate = cfg['data.sample_rate']
    # librosa is slow and gives slightly different results
    #warnings.filterwarnings('ignore', 'PySoundFile failed')
    #test_audio = (([librosa.load(os.path.join(test_dir, fn + '.mp3'),
    #                             sr=sample_rate)[0] * 2.**15],
    #               {})
    #              for fn in progress(filelist, 'File '))
    test_audio = ((fn, torch.as_tensor(audio.to_float(audio.read_ffmpeg(
                    os.path.join(test_dir, fn + '.mp3'),
                    sample_rate, dtype=np.int16))[np.newaxis]))
                  for fn in filelist)
    shapes = dict(input=(1, None))  # 1 channel, arbitrary length
    dtypes = dict(input=np.float32)

    # - we start the generator in a background thread
    batches = iterate_data(test_audio, device, cfg)

    # figure out the set of labels
    labelset_ebird = derive_labelset(pd.read_csv(options.train_csv))

    # prepare model
    print("Preparing %d network(s)..." % max(1, len(modelfiles)))
    # instantiate neural network
    num_classes = len(labelset_ebird)
    models = []
    backends = []
    if not modelfiles:
        # no weights given: just instantiate a model and initialize it
        model = get_model(cfg, shapes, dtypes, num_classes,
                          options.cuda_device)
        init_model(model, cfg)
        # take out the backend
        model.train(False)
        backends.append(model.backend)
        model.backend = None
        models.append(model)
    else:
        # read all the models
        for modelfile in modelfiles:
            # instantiate model according to its configuration file
            model_cfg = dict(cfg)
            model_cfg.update(config.parse_config_file(
                    modelconfigfile(modelfile)))
            model_cfg.update(extra_cfg)
            model = get_model(model_cfg, shapes, dtypes, num_classes,
                              options.cuda_device)
            # restore state dict
            state_dict = torch.load(modelfile, map_location=device)
            model.load_state_dict(state_dict)
            del state_dict
            model.train(False)
            # take out the backend
            backends.append(model.backend)
            model.backend = None
            models.append(model)

    # configure binarization threshold
    threshold = options.threshold
    if options.threshold_convert == 'logit':
        threshold = np.log(threshold / (1 - threshold))

    # run prediction loop
    print("Predicting:")
    predictions = []
    for audio_id, wav in tqdm.tqdm(batches, 'File ', len(filelist)):
        wav = wav[np.newaxis]  # add batch dimension
        preds_per_model = []
        clip_csv = test_csv.query("audio_id == '%s'" % audio_id)
        for model, backend in zip(models, backends):
            # pass the batch to the network
            with torch.no_grad():
                preds = model(dict(input=wav))
            # grab the main output
            if isinstance(preds, dict):
                if len(preds) == 1:
                    _, preds = preds.popitem()
                else:
                    preds = preds['output']
            # infer time step for every prediction frame
            rf = model.receptive_field
            offset = (np.atleast_1d(rf.size)[-1] // 2 -
                      np.atleast_1d(rf.padding)[-1])
            stride = np.atleast_1d(rf.stride)[-1]
            times = (np.arange(offset, wav.shape[-1] - offset, stride) /
                     sample_rate)
            # we now have preds and matching times for a full audio recording,
            # we need to evaluate it in chunks as specified in test.csv
            if (clip_csv.site.values[0] == 'site_3' or
                    not np.isfinite(clip_csv.seconds.values[0])):
                if options.local_then_global:
                    preds = pool_local_then_global(
                            preds, times, backend, cfg['data.len_max'],
                            options.local_then_global_overlap)
                else:
                    preds = pool_global(preds, backend)
            else:
                preds = pool_chunkwise(preds, times, backend, clip_csv.seconds)
            # apply postprocessing
            preds = postprocess(preds, options.postprocess)
            preds_per_model.append(preds)
        # average predictions for all the models
        preds = sum(preds_per_model[1:], preds_per_model[0])
        if len(preds_per_model) > 1:
            preds /= len(preds_per_model)
        # turn probabilities to label lists
        preds = derive_labels(preds, clip_csv.row_id, labelset_ebird,
                              threshold)
        predictions.append(preds)

    # save predictions
    print("Saving predictions")
    predictions = pd.concat(predictions, axis=0, sort=False).reset_index(drop=True)
    if outfile.endswith('.pkl'):
        try:
            import cPickle as pickle
        except ImportError:
            import pickle
        with io.open(outfile, 'wb') as f:
            pickle.dump(predictions, f, protocol=4)
    else:
        predictions.to_csv(outfile, index=False)


if __name__ == "__main__":
    main()
