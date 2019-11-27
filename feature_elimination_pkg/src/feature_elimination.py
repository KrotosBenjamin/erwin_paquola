#!/usr/bin/env python
""" 
feature_elimination for random forest classifier.

...

"""

__author__ = 'Kynon Jade Benjamin and Apua Paquola'

import numpy as np
import pandas as pd
from itertools import chain
from sklearn.metrics import accuracy_score, normalized_mutual_info_score


def features_rank_fnc(features, rank, n_features_to_keep, fold, out_dir):
    """
    Args:
    features: A vector of feature names
    rank: A vector with feature ranks based on absolute value of 
          feature importance
    n_features_to_keep: Number of features to keep. (Int)
    fold: Fold to analyzed. (Int)
    out_dir: Output directory for text file. Default '.'

    returns:
    Text file: Ranked features by fold tab-delimitated text file
    """
    jj = n_features_to_keep + 1
    eliminated = rank[n_features_to_keep: ]
    #print("Keep: %d\tEliminate: %d\tRank: %d" % (n_features_to_keep, len(eliminated), len(rank)))
    if len(eliminated) == 1:
        rank_df = pd.DataFrame({'Geneid': features[eliminated],
                                'Fold': fold,
                                'Rank': n_features_to_keep+1})
    elif len(eliminated) == 0:
        rank_df = pd.DataFrame({'Geneid': features[rank], 
                                'Fold': fold, 
                                'Rank': 1})
    else:
        rank_df = pd.DataFrame({'Geneid': features[eliminated],
                                'Fold': fold,
                                'Rank': np.arange(jj, jj+len(eliminated))})
                               
    rank_df.sort_values('Rank', ascending=False)\
           .to_csv(out_dir+'rank_features.txt', sep='\t', mode='a', index=False, header=False)


def n_features_iter(nf, keep_rate):
    """
    Determines the features to keep.

    Args:
    nf: current number of features
    keep_rate: percentage of features to keep

    Yields:
    int: number of features to keep
    """    
    while nf != 1:
        nf = max(1, int(nf * keep_rate))
        yield nf


def oob_predictions(estimator):
    """
    Extracts out-of-bag (OOB) predictions from random forest 
    classifier classes.

    Args:
    estimator: Random forest classifier object

    Yields:
    vector: OOB predicted labels
    """
    return estimator.classes_[(estimator.oob_decision_function_[:, 1]
                               > 0.5).astype(int)]


def oob_score_nmi(estimator, Y):
    """
    Calculates the normalized mutual information score
    from the OOB predictions.

    Args:
    estimator: Random forest classifier object
    Y: a vector of sample labels from training data set

    Yields:
    float: normalized mutual information score
    """
    labels_pred = oob_predictions(estimator)
    return normalized_mutual_info_score(Y, labels_pred)


def oob_score_accuracy(estimator, Y):
    """
    Calculates the accuracy score from the OOB predictions.

    Args:
    estimator: Random forest classifier object
    Y: a vector of sample labels from training data set

    Yields:
    float: accuracy score
    """
    labels_pred = oob_predictions(estimator)
    return accuracy_score(Y, labels_pred)


def rf_fe_step(estimator, X, Y, n_features_to_keep, features, fold, out_dir):
    """
    Apply random forest to training data, rank features, conduct feature
    elimination.

    Args:
    estimator: Random forest classifier object
    X: a data frame of training data
    Y: a vector of sample labels from training data set
    n_features_to_keep: number of features to keep
    features: a vector of feature names
    fold: current fold
    out_dir: output directory. default '.'

    Yields:
    dict: a dictionary with number of features, normalized mutual
          informatino score, accuracy score, and selected features
    """
    assert n_features_to_keep <= X.shape[1]
    estimator.fit(X, Y)
    rank = np.argsort(-estimator.feature_importances_)
    selected = rank[0:n_features_to_keep]
    features_rank_fnc(features, rank, n_features_to_keep, fold, out_dir)
    return {'n_features': X.shape[1],
            'nmi_score': oob_score_nmi(estimator, Y),
            'accuracy_score': oob_score_accuracy(estimator, Y),
            'selected': selected}


def rf_fe(estimator, X, Y, n_features_iter, features, fold, out_dir):
    """
    Iterate over features to by eliminated by step.

    Args:
    estimator: Random forest classifier object
    X: a data frame of training data
    Y: a vector of sample labels from training data set
    n_features_iter: iterator for number of features to keep loop
    features: a vector of feature names
    fold: current fold
    out_dir: output directory. default '.'

    Yields:
    list: a list with number of features, normalized mutual
          information score, accuracy score, and array of the indices
          for features to keep
    """
    indices = np.array(range(X.shape[1]))
    for nf in chain(n_features_iter, [1]):
        p = rf_fe_step(estimator, X, Y, nf, features, fold, out_dir)
        yield p['n_features'], p['nmi_score'], p['accuracy_score'], indices
        indices = indices[p['selected']]
        X = X[:, p['selected']]
        features = features[p['selected']]


def feature_elimination(estimator, X, Y, features, fold, out_dir='.',
                        elimination_rate=0.2):
    """
    Runs random forest feature elimination step over iterator process.

    Args:
    estimator: Random forest classifier object
    X: a data frame of training data
    Y: a vector of sample labels from training data set
    features: a vector of feature names
    fold: current fold
    out_dir: output directory. default '.'
    eliminatino_rate: percent rate to reduce feature list. default .2

    Yields:
    dict: a dictionary with number of features, normalized mutual
          information score, accuracy score, and array of the indexes
          for features to keep
    """
    d = dict()
    pfirst = None
    keep_rate = 1-elimiantion_rate
    for p in rf_fe(estimator, X, Y, n_features_iter(X.shape[1], keep_rate), 
                   features, fold, out_dir):
        if pfirst is None:
            pfirst = p
        d[p[0]] = p
