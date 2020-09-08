#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluates bird classification predictions in the Kaggle submission format.

For usage information, call with --help.

Author: Jan Schlüter
"""

from __future__ import print_function

from argparse import ArgumentParser

import numpy as np
import pandas as pd
import sklearn.metrics

from definitions.datasets.kagglebirds2020 import (derive_labelset,
                                                  make_multilabel_target)


def opts_parser():
    descr = ("Evaluates bird classification predictions in the Kaggle "
             "submission format.")
    parser = ArgumentParser(description=descr)
    parser.add_argument('gt', metavar='GTFILE',
            type=str,
            help='Ground truth .csv file (perfect_submission.csv)')
    parser.add_argument('preds', metavar='PREDFILE',
            type=str,
            help='Prediction .csv file (submission.csv)')
    parser.add_argument('--train-csv',
            type=str, required=True,
            help='File to read the set of class labels from.')
    return parser


def main():
    # parse command line
    parser = opts_parser()
    options = parser.parse_args()
    gtfile = options.gt
    predfile = options.preds

    # figure out the set of labels
    labelset = derive_labelset(pd.read_csv(options.train_csv))
    label_to_idx = dict((label, idx) for idx, label in enumerate(labelset))

    # read ground truth
    gt = pd.read_csv(gtfile)
    gt = np.stack([make_multilabel_target(len(labelset),
                                          [label_to_idx[label]
                                           for label in birds.split(' ')
                                           if label in label_to_idx])
                   for birds in gt.birds])

    # read predictions
    pr = pd.read_csv(predfile)
    pr = np.stack([make_multilabel_target(len(labelset),
                                          [label_to_idx[label]
                                           for label in birds.split(' ')
                                           if label in label_to_idx])
                   for birds in pr.birds])

    # evaluate
    p, r, f, _ = sklearn.metrics.precision_recall_fscore_support(
                gt, pr, average='micro')
    print('micro-prec', p)
    print('micro-rec', r)
    print('micro-f1', f)


if __name__ == "__main__":
    main()
