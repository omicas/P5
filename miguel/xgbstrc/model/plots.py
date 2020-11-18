#!/usr/bin/env python
# coding: utf-8

# Imported Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Default colors
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#7f7f7f', '#bcbd22', '#17becf']


#!/usr/bin/env python
# coding: utf-8

# Gene function prediction - Plot function
# Miguel Romero, nov 3rd

import os
import sys
import numpy as np
import pandas as pd

# Ploting
import seaborn as sns
from matplotlib import rc
from matplotlib import pyplot as plt

rc('font', family='serif', size=18)
rc('text', usetex=False)

# Default colors
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
          '#7f7f7f', '#bcbd22', '#17becf']



# plot auc roc curve
def plot_roc(fpr, tpr, auc, filename, path=None):
  fig, ax = plt.subplots(figsize=(6.5,6.5))
  plt.plot(fpr, tpr, lw=2, label='AUC = {0:.2f}'.format(auc))
  plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.0])
  plt.xlabel('1 - Specificity')
  plt.ylabel('Sensitivity')
  plt.legend(loc='lower right')
  figname = ('{0}/'.format(path) if path != None else '') + filename
  plt.savefig(figname, format='pdf', dpi=600)
  plt.close()


# plot multiple auc roc curves in one figure
def plot_mroc(fprl, tprl, aucl, labels, filename, path=None):
  fig, ax = plt.subplots(figsize=(6.5,6.5))
  for fpr, tpr, auc, l in zip(fprl, tprl, aucl, labels):
    plt.plot(fpr, tpr, lw=2, label='{0} AUC = {1:.2f}'.format(l, auc))
  plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.0])
  plt.xlabel('1 - Specificity')
  plt.ylabel('Sensitivity')
  plt.legend(loc='lower right')
  figname = ('{0}/'.format(path) if path != None else '') + filename
  plt.savefig(figname, format='pdf', dpi=600)
  plt.close()


# plot average precision curve
def plot_ap(rec, prc, ap, filename, path=None):
  fig, ax = plt.subplots(figsize=(6.5,6.5))
  plt.plot(rec, prc, lw=2, label='AP = {0:.2f}'.format(ap))
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.0])
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.legend(loc='lower right')
  figname = ('{0}/'.format(path) if path != None else '') + filename
  plt.savefig(figname, format='pdf', dpi=600)
  plt.close()


# plot multiple average precision curves in one figure
def plot_map(recl, prcl, apl, labels, filename, path=None):
  fig, ax = plt.subplots(figsize=(6.5,6.5))
  for rec, prc, ap, l in zip(recl, prcl, apl, labels):
    plt.plot(rec, prc, lw=2, label='{0} AP = {1:.2f}'.format(l, ap))
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.0])
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.legend(loc='lower right')
  figname = ('{0}/'.format(path) if path != None else '') + filename
  plt.savefig(figname, format='pdf', dpi=600)
  plt.close()


# plot confusion matrix
def plot_conf_matrix(cm, filename, path=None, axis=[0,1], normalize=True):
  if normalize: cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  df_cm = pd.DataFrame(cm, index=axis, columns=axis)

  fig, ax = plt.subplots(figsize=(5,5))
  figname = '{0}/'.format(path) if path != None else ''
  if normalize:
    sns.heatmap(df_cm, annot=True, cbar=False, linewidths=.5, center=0, cmap=plt.cm.Blues)
    figname = '{0}{1}_cmat_norm.pdf'.format(figname,filename)
  else:
    sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, linewidths=.5, center=0, cmap=plt.cm.Blues)
    figname = '{0}{1}_cmat.pdf'.format(figname,filename)
  plt.xlabel('Predicted label')
  plt.ylabel('True label')
  plt.tight_layout()
  plt.savefig(figname, format='pdf', dpi=600)
  plt.close()


#
#
# # plot line with std dev
# def plot_mts_hist(x,y,e,folder,name):
#   fig, ax = plt.subplots(figsize=(6.5,6.5))
#   plt.plot(x, y, '--')
#   lowerb = [yi-ei for yi,ei in zip(y,e)]
#   upperb = [yi+ei for yi,ei in zip(y,e)]
#   plt.fill_between(x, lowerb, upperb, alpha=.3)
#   plt.ylabel(name)
#   plt.xticks(rotation=90)
#   plt.tight_layout()
#   plt.savefig('{0}/{1}_hist.pdf'.format(folder,name), format='pdf', dpi=600)
#   plt.close()
#
# # plot pie
# def plot_feat_imp(l,x,folder,name):
#   fig, ax = plt.subplots(figsize=(8,5))
#   plt.pie(x, autopct='%1.1f%%', textprops=dict(size=14, color='w', weight='bold'))
#   plt.legend(l, loc='best')
#   plt.axis('equal')
#   plt.tight_layout()
#   plt.savefig('{0}/{1}_fimp.pdf'.format(folder,name), format='pdf', dpi=600)
#   plt.close()
#
# # plot confusion matrix
# def plot_conf_matrix(cm, folder, name, axis=[0,1], normalize=True):
#   if normalize:
#     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#   df_cm = pd.DataFrame(cm, index=axis, columns=axis)
#   fig, ax = plt.subplots(figsize=(5,5))
#   if normalize:
#     sns.heatmap(df_cm, annot=True, cbar=False, linewidths=.5, center=0, cmap=plt.cm.Blues)
#     plt.tight_layout()
#     plt.savefig('{0}/{1}_cmat_norm.pdf'.format(folder,name), format='pdf', dpi=600)
#   else:
#     sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, linewidths=.5, center=0, cmap=plt.cm.Blues)
#     plt.tight_layout()
#     plt.savefig('{0}/{1}_cmat.pdf'.format(folder,name), format='pdf', dpi=600)
#   plt.xlabel('Predicted label')
#   plt.ylabel('True label')
#   plt.close()
#
# # plot auc roc vs height in hierarchy
# def plot_auc_height(data,folder,name):
#   x, y = [x for x,y in data], [y for x,y in data]
#   fig, ax = plt.subplots(figsize=(5,5))
#   plt.plot(x,y,'.')
#   plt.xlabel('Height')
#   plt.ylabel('AUC ROC')
#   plt.tight_layout()
#   plt.savefig('{0}/{1}_auc_height.pdf'.format(folder,name), format='pdf', dpi=600)
#   plt.close()
#
#
# # plot metric and score for validation
# def plot_metric_val(x, y, ystd, yval, label, metric_name, file):
#   fig, ax = plt.subplots(figsize=(10,4))
#   x, y, ystd, yval = np.array(x), np.array(y), np.array(ystd), np.array(yval)
#   plt.plot(x, y, 'o--', color=colors[0], label=label[0], linewidth=1)
#   plt.fill_between(x, y - ystd, y + ystd, color=colors[0], alpha=0.5)
#   plt.plot(x, yval, 'o--', color=colors[1], label=label[1], linewidth=1)
#
#   plt.legend(loc='best', shadow=True, fontsize='medium')
#   plt.xticks(rotation='vertical')
#   plt.margins(0.015)
#   plt.xlabel('Target')
#   plt.ylabel(metric_name)
#   plt.ylim(0,1)
#   plt.tight_layout()
#   plt.savefig('{0}.pdf'.format(file), format='pdf', dpi=600)
#   plt.close()
#
#
# # plot metric with std
# def plot_metric_std(X, Y, Ystd, label, metric_name, file):
#   fig, ax = plt.subplots(figsize=(10,4))
#   for idx in range(len(Y)):
#     plt.plot(X, Y[idx], 'o--', color=colors[idx], label=label[idx], linewidth=1)
#     # plt.fill_between(X, Y[idx] - Ystd[idx], Y[idx] + Ystd[idx], color=colors[idx], alpha=0.5)
#
#   plt.legend(loc='best', shadow=True, fontsize='medium')
#   plt.xticks(rotation='vertical')
#   plt.margins(0.015)
#   plt.xlabel('Target')
#   plt.ylabel(metric_name)
#   plt.ylim(0,1)
#   plt.tight_layout()
#   plt.savefig('{0}.pdf'.format(file), format='pdf', dpi=600)
#   plt.close()
