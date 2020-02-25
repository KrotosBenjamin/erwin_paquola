#!/usr/bin/env python
""" 
This package has several function to run feature elimination for random forest classifier. Specifically,
Out-of-Bag (OOB) must be set to True. Three measurements are calculated for feature selection.

1. Normalized mutual information
2. Accuracy
3. Area under the curve (AUC) ROC curve

Original author Apua Paquola.
Edits: Kynon Jade Benjamin
Feature ranking modified from Tarun Katipalli ranking function.
"""

__author__ = 'Kynon Jade Benjamin'

import numpy as np
import pandas as pd
from itertools import chain
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import normalized_mutual_info_score

from plotnine import *
from warnings import filterwarnings
from matplotlib.cbook import mplDeprecation
filterwarnings("ignore", category=mplDeprecation)
filterwarnings('ignore', category=UserWarning, module='plotnine.*')
filterwarnings('ignore', category=DeprecationWarning, module='plotnine.*')


def features_rank_fnc(features, rank, n_features_to_keep, fold, out_dir, RANK):
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
    if RANK:
        jj = n_features_to_keep + 1
        eliminated = rank[n_features_to_keep: ]
        #print("Keep: %d\tEliminate: %d\tRank: %d" % (n_features_to_keep, 
        #                                             len(eliminated), len(rank)))
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
               .to_csv(out_dir+'/rank_features.txt', sep='\t', mode='a', 
                       index=False, header=False)


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


def oob_score_roc(estimator, Y):
    """
    Calculates the area under the ROC curve score
    for the OOB predictions.

    Args:
    estimator: Random forest classifier object
    Y: a vector of sample labels from training data set

    Yields:
    float: AUC ROC score    
    """
    labels_pred = estimator.oob_decision_function_
    kwargs = {'multi_class': 'ovr'} if len(np.unique(Y)) > 2 else {}
    return roc_auc_score(Y, labels_pred, **kwargs)


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
    return normalized_mutual_info_score(Y, labels_pred,
                                        average_method='arithmetic')


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


def rf_fe_step(estimator, X, Y, n_features_to_keep, features, fold, out_dir, RANK):
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
          information score, accuracy score, auc roc score and selected features
    """
    assert n_features_to_keep <= X.shape[1]
    estimator.fit(X, Y)
    rank = np.argsort(-estimator.feature_importances_)
    selected = rank[0:n_features_to_keep]
    features_rank_fnc(features, rank, n_features_to_keep, fold, out_dir, RANK)
    return {'n_features': X.shape[1],
            'nmi_score': oob_score_nmi(estimator, Y),
            'accuracy_score': oob_score_accuracy(estimator, Y),
            'roc_auc_score': oob_score_roc(estimator, Y),
            'selected': selected}


def rf_fe(estimator, X, Y, n_features_iter, features, fold, out_dir, RANK):
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
          information score, accuracy score, auc roc curve and array of the indices
          for features to keep
    """
    indices = np.array(range(X.shape[1]))
    for nf in chain(n_features_iter, [1]):
        p = rf_fe_step(estimator, X, Y, nf, features, fold, out_dir, RANK)
        yield p['n_features'], p['nmi_score'], p['accuracy_score'], p['roc_auc_score'], indices
        indices = indices[p['selected']]
        X = X[:, p['selected']]
        features = features[p['selected']]


def feature_elimination(estimator, X, Y, features, fold, out_dir='.',
                        elimination_rate=0.2, RANK=True):
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
          information score, accuracy score, auc roc curve and array of the indexes
          for features to keep
    """
    d = dict()
    pfirst = None
    keep_rate = 1-elimination_rate
    for p in rf_fe(estimator, X, Y, n_features_iter(X.shape[1], keep_rate), 
                   features, fold, out_dir, RANK):
        if pfirst is None:
            pfirst = p
        d[p[0]] = p

    return d, pfirst


def save_plot(p, fn, width=7, height=7):
    '''Save plot as svg, png, and pdf with specific label and dimension.'''
    for ext in ['.svg', '.png', '.pdf']:
        p.save(fn+ext, width=width, height=height)
        

def plot_nmi(d, fold, output_dir):
    """
    Plot feature elimination results for normalized mutual information.

    Args:
    d: feature elimination class dictionary
    fold: current fold
    out_dir: output directory. default '.'

    Yields:
    graph: plot of feature by NMI, automatically saves files as png and svg
    """
    df_elim = pd.DataFrame([{'n features':k,
                             'normalized mutual information':d[k][1]} for k in d.keys()])
    gg = ggplot(df_elim, aes(x='n features', y='normalized mutual information'))\
        + geom_point() + scale_x_log10() + theme_light()
    save_plot(gg, output_dir+"/nmi_fold_%d" % (fold))
    print(gg)


def plot_roc(d, fold, output_dir):
    """
    Plot feature elimination results for AUC ROC curve.

    Args:
    d: feature elimination class dictionary
    fold: current fold
    out_dir: output directory. default '.'

    Yields:
    graph: plot of feature by AUC, automatically saves files as png and svg
    """
    df_elim = pd.DataFrame([{'n features':k,
                             'ROC AUC':d[k][3]} for k in d.keys()])
    gg = ggplot(df_elim, aes(x='n features', y='ROC AUC'))\
        + geom_point() + scale_x_log10() + theme_light()
    save_plot(gg, output_dir+"/roc_fold_%d" % (fold))
    print(gg)


def plot_acc(d, fold, output_dir):
    """
    Plot feature elimination results for accuracy.

    Args:
    d: feature elimination class dictionary
    fold: current fold
    out_dir: output directory. default '.'

    Yields:
    graph: plot of feature by accuracy, automatically saves files as png and svg
    """
    df_elim = pd.DataFrame([{'n features':k,
                             'Accuracy':d[k][3]} for k in d.keys()])
    gg = ggplot(df_elim, aes(x='n features', y='Accuracy'))\
        + geom_point() + scale_x_log10() + theme_light()
    save_plot(gg, output_dir+"/acc_fold_%d" % (fold))
    print(gg)


# def plot_scores(d, alpha, output_dir):
#     df_nmi = pd.DataFrame([{'n features':k, 'Score':d[k][1]} for k in d.keys()])
#     df_nmi['Type'] = 'NMI'
#     df_acc = pd.DataFrame([{'n features':k, 'Score':d[k][2]} for k in d.keys()])
#     df_acc['Type'] = 'Acc'
#     df_roc = pd.DataFrame([{'n features':k, 'Score':d[k][3]} for k in d.keys()])
#     df_roc['Type'] = 'ROC'
#     df_elim = pd.concat([df_nmi, df_acc, df_roc], axis=0)
#     gg = ggplot(df_elim, aes(x='n features', y='Score', color='Type'))\
#         + geom_point() + scale_x_log10() + theme_light()
#     gg.save(output_dir+"/scores_wgt_%.2f.png" % (alpha))
#     gg.save(output_dir+"/scores_wgt_%.2f.svg" % (alpha))
#     print(gg)
