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
    parser.add_argument('modelfile', metavar='MODELFILE',
            type=str,
            help='File to load the learned weights from (.npz format)')
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
    parser.add_argument('--threshold',
            type=float, default=0,
            help='Binarization threshold (default: %(default)s)')
    parser.add_argument('--threshold-convert',
            type=str, choices=('none', 'logit'),
            help='Optional function to apply to --threshold: none or logit '
                 '(to convert probabilities to logits) (default: %(default)s)')
    parser.add_argument('--cuda-device',
            type=int, action='append', default=[],
            help='If given, run on the given CUDA device (starting with 0). '
                 'Can be given multiple times to parallelize over GPUs.')
    config.prepare_argument_parser(parser)
    return parser


def pool_chunkwise(preds, times, clip_csv, model, labelset, threshold=0):
    chunks = []
    for _, row in clip_csv.iterrows():
        if row.site != 'site_3' and np.isfinite(row.seconds):
            start = row.seconds - 5  # hard-coded for the challenge
            end = row.seconds
            a, b = np.searchsorted(times, [start, end])
            chunk_preds = preds[..., a:b]
        else:
            chunk_preds = preds
        with torch.no_grad():
            chunk_preds = model(chunk_preds).detach().cpu().numpy().ravel()
        birds = np.where(chunk_preds > threshold)[0]
        birds = ' '.join(labelset[bird] for bird in birds) or 'nocall'
        chunks.append({'row_id': row.row_id, 'birds': birds})
    chunks = pd.DataFrame(chunks)
    return chunks


def main():
    # parse command line
    parser = opts_parser()
    options = parser.parse_args()
    if not (options.train_csv and options.test_csv and options.test_audio):
        parser.error('requires --train-csv, --test-csv, --test-audio')
    modelfile = options.modelfile
    outfile = options.outfile
    if os.path.exists(os.path.splitext(modelfile)[0] + '.vars'):
        options.vars.insert(1, os.path.splitext(modelfile)[0] + '.vars')
    cfg = config.from_parsed_arguments(options)
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
    print("Preparing network...")
    # instantiate neural network
    num_classes = len(labelset_ebird)
    model = get_model(cfg, shapes, dtypes, num_classes, options.cuda_device)
    # restore state dict
    if options.modelfile:
        state_dict = torch.load(options.modelfile,
                                map_location=device)
        model.load_state_dict(state_dict)
        del state_dict
    else:
        init_model(model, cfg)
    # take out the backend
    model_backend = model.backend
    model.backend = None

    # configure binarization threshold
    threshold = options.threshold
    if options.threshold_convert == 'logit':
        threshold = np.log(threshold / (1 - threshold))

    # run prediction loop
    print("Predicting:")
    model.train(False)
    model_backend.train(False)
    predictions = []
    for audio_id, wav in tqdm.tqdm(batches, 'File ', len(filelist)):
        wav = wav[np.newaxis]  # add batch dimension
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
        offset = (rf.size[-1] // 2 - rf.padding[-1])
        stride = rf.stride[-1]
        times = np.arange(offset, wav.shape[-1] - offset, stride) / sample_rate
        # we now have preds and matching times for a full audio recording, we
        # need to evaluate it in chunks as specified in test.csv
        clip_csv = test_csv.query("audio_id == '%s'" % audio_id)
        pooled_preds = pool_chunkwise(preds, times, clip_csv, model_backend,
                                      labelset_ebird, threshold)
        predictions.append(pooled_preds)

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
