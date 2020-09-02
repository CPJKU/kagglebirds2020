# -*- coding: utf-8 -*-

"""
Dataset splitting routines.

Author: Jan Schlüter
"""

import sklearn.model_selection


def grouped_split(X, groups, test_size, seed):
    """
    Random train/test split of array `X` by information in array `groups`.
    Items of the same group will not be spread across the training and test
    set. Test set size can be given as an int (referring to the number of
    groups) or a float (referring to the fraction of groups). Returns arrays
    of training indices and test indices.
    """
    s = sklearn.model_selection.GroupShuffleSplit(1, test_size=test_size,
                                                  random_state=seed)
    return next(s.split(X, groups=groups))


def stratified_split(X, y, test_size, seed):
    """
    Random train/test split of array `X` by information in array `labels`.
    Items will be split to have the same distribution of labels in the training
    and test set. Test set size can be given as an int (referring to the number
    of items) or a float (referring to the fraction of items). Returns arrays
    of training indices and test indices.
    """
    s = sklearn.model_selection.StratifiedShuffleSplit(1, test_size=test_size,
                                                       random_state=seed)
    return next(s.split(X, y=y))
