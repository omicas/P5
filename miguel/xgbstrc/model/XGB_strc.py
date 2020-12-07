#!/usr/bin/env python
# coding: utf-8

# Imported Libraries
import time
import numpy as np
import igraph as ig
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from tqdm import tqdm
from matplotlib import rc

from .plots import *

# Classifier Libraries
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import average_precision_score

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import NearMiss
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline

import xgboost as xgb
import multiprocessing

# Other libraries
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore',category=DeprecationWarning)
warnings.filterwarnings(action='ignore',category=FutureWarning)

rc('font', family='serif', size=18)
rc('text', usetex=False)

# Default colors
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#7f7f7f', '#bcbd22', '#17becf']
EPS = 1e-6



class XGB_strc:

  # Default constructor
  def __init__(self):
    self.clf = None # classifier
    self.g = None # graph
    self.df = None # dataframe with node features
    self.ylabel = None # feature to predict (binary class), column of df
    self.y = None # feature to predict (binary class), column of df
    self.load_classifier() # load XGB classifier


  # Load data
  def load_data(self, g, ylabel, df):
    self.g = g # graph
    self.df = df # dataframe
    self.ylabel = ylabel # attribute to predict
    self.y = df[self.ylabel] # attribute to predict
    self.df = self.df.drop([self.ylabel], axis=1)


  # Load graph
  def load_graph(self, g):
    self.g = g # adyacency matrix


  # Load dataframe from csv
  def load_cvs_file(self, filename):
    self.df = pd.read_csv(filename)
    self.y = df[self.ylabel] # attribute to predict
    self.df = self.df.drop([self.ylabel], axis=1)


  # load xgboost classifier
  def load_classifier(self, nthread=4, n_iter=2, cv=2, n_jobs=None, seed=None):
    if n_jobs == None: n_jobs = multiprocessing.cpu_count() // 2

    param_grid = {
        'max_depth': [6, 10, 15, 20],
        'learning_rate': [0.001, 0.01, 0.1, 0.2, 0,3],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bylevel': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
        'gamma': [0, 0.25, 0.5, 1.0],
        'reg_lambda': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]}

    clf = xgb.XGBClassifier(nthread=nthread, booster='gbtree',
                            n_jobs=n_jobs, random_state=seed)

    self.clf = RandomizedSearchCV(clf, param_grid, n_iter=n_iter, n_jobs=n_jobs,
                                  cv=cv, random_state=seed)


  # compute topological properties of graph and include them in the df
  def compute_strc_prop(self, df):
    # igraph
    g = ig.Graph.Adjacency((self.g > 0).tolist())
    g.to_undirected()

    # get node properties form graph
    clust = np.array(g.transitivity_local_undirected(mode="zero"))
    deg = np.array(g.degree())
    neigh_deg = np.array(g.knn()[0])
    centr_betw = np.array(g.betweenness(directed=False))
    centr_clos = np.array(g.closeness())
    # new measures
    eccec = np.array(g.eccentricity())
    pager = np.array(g.personalized_pagerank(directed=False))
    const = np.array(g.constraint())
    hubs = np.array(g.hub_score())
    auths = np.array(g.authority_score())
    coren = np.array(g.coreness())
    diver = np.array(g.diversity())

    # add node properties to new df
    df['clust'] = pd.Series(clust) # clustering
    df['deg'] = pd.Series(deg) # degree
    df['neigh_deg'] = pd.Series(neigh_deg) # average_neighbor_degree
    df['betw'] = pd.Series(centr_betw) # betweenness_centrality
    df['clos'] = pd.Series(centr_clos) # closeness_centrality
    df['eccec'] = pd.Series(eccec) # eccentricity
    df['pager'] = pd.Series(pager) # page rank
    df['const'] = pd.Series(const) # constraint
    df['hubs'] = pd.Series(hubs) # hub score
    df['auths'] = pd.Series(auths) # authority score
    df['coren'] = pd.Series(coren) # coreness
    df['diver'] = pd.Series(diver) # diversity

    return df


  # Scale data
  def scale_data(self, df):
    # MinMaxScaler does not modify the distribution of data
    minmax_scaler = MinMaxScaler() # Must be first option

    new_df = pd.DataFrame()
    for fn in df.columns:
      scaled_feature = minmax_scaler.fit_transform(df[fn].values.reshape(-1,1))
      new_df[fn] = scaled_feature[:,0].tolist()

    return new_df.copy()


  # Splitting the Data
  def split_data(self, X, y, n_splits=5, seed=None):
    sss = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=False)

    for train_index, test_index in sss.split(X, y):
      Xtrain, Xtest = X.iloc[train_index], X.iloc[test_index]
      ytrain, ytest = y.iloc[train_index], y.iloc[test_index]

    # Turn into an array
    Xtrain = Xtrain.values
    Xtest = Xtest.values
    ytrain = ytrain.values
    ytest = ytest.values

    return Xtrain, Xtest, ytrain, ytest


  # measures the prediction performance
  def get_performance(self, yorig, ypred, ypred_prb, final=True):
    mts = dict()

    mts['acc'] = accuracy_score(yorig, ypred) # accuracy
    mts['bacc'] = balanced_accuracy_score(yorig, ypred) # balanced accuracy
    mts['f1s'] = f1_score(yorig, ypred) # f1 score
    mts['loss'] = log_loss(yorig, ypred_prb) # log loss
    mts['avp'] = average_precision_score(yorig, ypred_prb) # average precision
    mts['auc'] = roc_auc_score(yorig, ypred_prb) # auc roc score
    if final:
      mts['prc'], mts['rec'], _ = precision_recall_curve(yorig, ypred_prb) # precision and recall
      mts['fpr'], mts['tpr'], _ = roc_curve(yorig, ypred_prb) # false and true positive rate
      mts['cm'] = confusion_matrix(yorig, ypred) # confusion matrix

    return mts


  # training with xgboost and smote (oversampling)
  def train(self, X, y, n_splits=5, seed=None):
    Xtrain, Xtest, ytrain, ytest = self.split_data(X, y, n_splits, seed)
    while np.sum(ytrain) == 0 or np.sum(ytest) == 0:
      Xtrain, Xtest, ytrain, ytest = self.split_data(X, y, n_splits, seed)

    # Prediction performance for every fold in cross-validation
    acc, bacc, loss = list(), list(), list()
    f1s, auc, avp = list(), list(), list()

    # Implementing SMOTE Technique
    # Cross Validating the right way
    sss = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=False)
    for train, test in sss.split(Xtrain, ytrain):
      if np.sum(ytrain[test]) == 0:# or np.sum(ytrain) == len(ytrain):
        continue

      try:
        # SMOTE happens during Cross Validation not before..
        pipeline = imbalanced_make_pipeline(SMOTE(sampling_strategy='minority'), self.clf)
        model = pipeline.fit(Xtrain[train], ytrain[train])
      except:
        pipeline = imbalanced_make_pipeline(RandomOverSampler(sampling_strategy='minority'), self.clf)
        model = pipeline.fit(Xtrain[train], ytrain[train])

      best_est = self.clf.best_estimator_
      ypred = best_est.predict(Xtrain[test])
      ypred_prb = best_est.predict_proba(Xtrain[test])[:,1]

      perf = self.get_performance(ytrain[test], ypred, ypred_prb, False)
      acc.append(perf['acc'])
      auc.append(perf['auc'])
      avp.append(perf['avp'])
      bacc.append(perf['bacc'])
      f1s.append(perf['f1s'])
      loss.append(perf['loss'])

    # prediction performance on test set
    test_mts = dict()
    test_mts['acc'] = (np.mean(acc), np.std(acc))
    test_mts['auc'] = (np.mean(auc), np.std(auc))
    test_mts['avp'] = (np.mean(avp), np.std(avp))
    test_mts['bacc'] = (np.mean(bacc), np.std(bacc))
    test_mts['f1s'] = (np.mean(f1s), np.std(f1s))
    test_mts['loss'] = (np.mean(loss), np.std(loss))

    # prediction performance on validation set
    yval = best_est.predict(Xtest)
    yval_prb = best_est.predict_proba(Xtest)[:,1]
    val_mts = self.get_performance(ytest, yval, yval_prb)

    return test_mts, val_mts


  # compare performance with and w/o topological properties of graph
  def structural_test(self, path=None, normalize=True, n_splits=5, seed=None):
    # use self.df to compute structural properties and train a model with them,
    # and another one without. Then compare the performance of both models in
    # order to check whether the structural properties are useful for the
    # prediction
    print('')
    print('**Original data**')
    df = self.scale_data(self.df)
    test, val = self.train(df, self.y, n_splits=n_splits, seed=seed)
    self.print_performance(test, val, train=False)
    self.plot_cm(val, 'orig', path=path, normalize=normalize)

    print('')
    print('**Structural data**')
    strc_df = self.df.copy()
    strc_df = self.compute_strc_prop(strc_df)
    strc_df = self.scale_data(strc_df)
    strc_test, strc_val = self.train(strc_df, self.y, n_splits=n_splits, seed=seed)
    self.print_performance(strc_test, strc_val, train=False)
    self.plot_cm(strc_val, 'strc', path=path, normalize=normalize)

    self.plot_auc_roc_comp(val, strc_val, ['orig', 'strc'], 'auc_roc_comp', path)

    return self.get_best_params()


  # print metrics
  def print_metrics(self, mts):
    print('Accuracy: {0:.3f}'.format(mts['acc']))
    print('Balanced accuracy: {0:.3f}'.format(mts['bacc']))
    print('f1 score: {0:.3f}'.format(mts['f1s']))
    print('AUC ROC: {0:.3f}'.format(mts['auc']))
    print('Average precision: {0:.3f}'.format(mts['avp']))
    print('Log Loss: {0:.3f}'.format(mts['loss']))


  # print metrics
  def print_metrics_std(self, mts):
    print('Accuracy: {0:.3f} ({1:.3f})'.format(mts['acc'][0], mts['acc'][1]))
    print('Balanced accuracy: {0:.3f} ({1:.3f})'.format(mts['bacc'][0], mts['bacc'][1]))
    print('f1 score: {0:.3f} ({1:.3f})'.format(mts['f1s'][0], mts['f1s'][1]))
    print('AUC ROC: {0:.3f} ({1:.3f})'.format(mts['auc'][0], mts['auc'][1]))
    print('Average precision: {0:.3f} ({1:.3f})'.format(mts['avp'][0], mts['avp'][1]))
    print('Log Loss: {0:.3f} ({1:.3f})'.format(mts['loss'][0], mts['loss'][1]))


  # print prediction performance for test (cv) and validation
  def print_performance(self, test_mts, val_mts, train=True):
    if train:
      print('')
      print('# Training')
      self.print_metrics_std(test_mts)

    print('')
    print('# Validation')
    self.print_metrics(val_mts)


  # get parameters that gave the best results
  def get_best_params(self):
    return self.clf.best_params_


  # plot auc roc curve
  def plot_auc_roc(self, mts, filename, path=None):
    fpr = mts['fpr']
    tpr = mts['tpr']
    auc = mts['auc']
    plot_roc(fpr, tpr, auc, filename, path)


  # plot auc roc curve
  def plot_auc_roc_comp(self, mts_a, mts_b, labels, filename, path=None):
    fprl = [mts_a['fpr'], mts_b['fpr']]
    tprl = [mts_a['tpr'], mts_b['tpr']]
    aucl = [mts_a['auc'], mts_b['auc']]
    plot_mroc(fprl, tprl, aucl, labels, filename, path)


  # plot precision recall curve
  def plot_averagep(self, mts, filename, path=None):
    rec = mts['rec']
    prc = mts['prc']
    avp = mts['avp']
    plot_ap(rec, prc, avp, filename, path)


  # plot confusion matrix
  def plot_cm(self, mts, filename, path=None, normalize=True):
    cm = mts['cm']
    plot_conf_matrix(cm, filename, path=path, normalize=normalize)
