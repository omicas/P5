#!/usr/bin/env python
# coding: utf-8

# Gene function prediction for Oryza sativa Japonica
# Miguel Romero 18/11/20

import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
from time import time
from tqdm import tqdm
from collections import deque

from HBN import mst
from HBN import direct_pa

from context import model
from model import XGB_strc


##########################
# 1. Import and clean data
##########################

data_isa = pd.read_csv('data/data_isa.csv', dtype='object')
data_ppi = pd.read_csv('data/data_ppi.csv', dtype='object')
data_term_def = pd.read_csv('data/data_term_def.csv', dtype='object')
data_gene_term = pd.read_csv('data/data_gene_term.csv', dtype='object')

data_isa = data_isa[data_isa['Child']!='GO:0008150']
data_isa = data_isa[data_isa['Ancestor']!='GO:0008150']
data_term_def = data_term_def[data_term_def['Term']!='GO:0008150']
data_gene_term = data_gene_term[data_gene_term['Term']!='GO:0008150']

P = np.array(sorted(list(set(data_ppi['Source'].tolist()+data_ppi['Target'].tolist()))))
nP, idxP = len(P), dict([(p,i) for i,p in enumerate(P)])

G = np.array(sorted(list(set(data_term_def['Term'].tolist()+data_gene_term['Term'].tolist()))))
nG, idxG = len(G), dict([(g,i) for i,g in enumerate(G)])

print('**Initial data**')
print('Genes: \t\t{0}'.format(len(P)))
print('Gene annot.: \t{0}'.format(len(data_gene_term)))
print('Co-expression: \t{0:.0f}'.format(len(data_ppi)))
print('GO terms: \t{0}'.format(len(data_term_def)))
print('GO hier.: \t{0}'.format(len(data_isa)))
print()


##############################
# 2. Create matrices from data
##############################

# PPI matrix
# nP:number of genes, idxP:gene index map
ppi = np.zeros((nP,nP))
for edge in tqdm([tuple(x) for x in data_ppi.to_numpy()]):
  u, v = idxP[edge[0]], idxP[edge[1]]
  ppi[u][v] = ppi[v][u] = 1

# go by go matrix
# nG:number of terms, idxG:term index map
go_by_go = np.zeros((nG,nG))
for edge in tqdm([tuple(x) for x in data_isa.to_numpy()]):
  u, v = idxG[edge[0]], idxG[edge[1]]
  go_by_go[u,v] = 1

# compute the transitive closure of the ancestor of a term (idx)
def ancestor_terms(term):
  tmp = np.nonzero(go_by_go[term,:])[0].tolist()
  ancs = list()
  while len(tmp) > 0:
    tmp1 = list()
    for i in tmp:
      ancs.append(i)
      tmp1 += np.nonzero(go_by_go[i,:])[0].tolist()
    tmp = list(set(tmp1))
  return ancs

# gene by go matrix
gene_by_go = np.zeros((nP,nG))
for edge in tqdm([tuple(x) for x in data_gene_term.to_numpy()]):
  u, v = idxP[edge[0]], idxG[edge[1]]
  gene_by_go[u,v] = 1
  gene_by_go[u,ancestor_terms(v)] = 1

print()
print('**Final matrices**')
print('Proteins: \t{0:6}'.format(len(ppi)))
print('Prot. annot.: \t{0:6}'.format(np.count_nonzero(gene_by_go)))
print('Interactions: \t{0:6.0f}'.format(np.sum(ppi)/2))
print('GO terms: \t{0:6}'.format(len(go_by_go)))
print('GO hier.: \t{0:6.0f}'.format(np.sum(go_by_go)))


#####################################
# 3. Prepare term data for prediction
#####################################

print()
print('**Term selection**')
# Prune terms according to paper, very specific and extremes with little to
# no information terms are avoided. Select genes used for prediction
# Accoding to restriction 5 <= genes annotated <= 300
filt_terms_idx = list()
for i in range(nG):
  if 5 <= np.count_nonzero(gene_by_go[:,i]) <= 300:
    filt_terms_idx.append(i)

# Including the ancestor of the selected terms
pred_terms_idx = list(filt_terms_idx)
for tidx in filt_terms_idx:
  pred_terms_idx += np.nonzero(go_by_go[tidx,:])[0].tolist()
pred_terms_idx = np.array(sorted(list(set(pred_terms_idx))))
print('Number of terms to predict: {0}'.format(len(pred_terms_idx)))

# Graph for subhiearchies creation
go_by_go_edgelist = np.transpose(np.nonzero(np.transpose(go_by_go))).tolist()
nxG = nx.DiGraph()
nxG.add_nodes_from(np.arange(nG))
nxG.add_edges_from(go_by_go_edgelist)

# Subgraph from terms to predict
sub_go_by_go = go_by_go[np.ix_(pred_terms_idx,pred_terms_idx)].copy()
sub_go_by_go_edgelist = np.transpose(np.nonzero(np.transpose(sub_go_by_go))).tolist()
sub_nxG = nx.DiGraph()
sub_nxG.add_nodes_from(np.arange(len(pred_terms_idx)))
sub_nxG.add_edges_from(sub_go_by_go_edgelist)

# find possible root terms in go subgraph
subh_root_idx = list()
for tidx, term in enumerate(pred_terms_idx):
  if np.count_nonzero(sub_go_by_go[tidx,:]) == 0: # terms wo ancestors
    subh_root_idx.append(tidx)
subh_root_idx = np.array(subh_root_idx)
print('Number of roots in GO subgraph: {0}'.format(len(subh_root_idx)))

# convert a bfs object to a list
def nodes_in_bfs(bfs, root):
  nodes = sorted(list(set([u for u,v in bfs] + [v for u,v in bfs])))
  nodes = np.setdiff1d(nodes, [root]).tolist()
  nodes = [root] + nodes
  return nodes

# detect isolated terms and create sub-hierarchies
subh_pred_terms, subh_all_terms = list(), list()
nodes_in_subh = list()
_subh_root_idx = list()
for root in subh_root_idx:
  bfs = nx.bfs_tree(sub_nxG, root).edges()
  bfs_all = nx.bfs_tree(nxG, pred_terms_idx[root]).edges()

  if len(bfs) > 0: # if no isolated term
    _subh_root_idx.append(pred_terms_idx[root])
    bfs_nodes = pred_terms_idx[nodes_in_bfs(bfs, root)]
    subh_pred_terms.append(bfs_nodes)
    nodes_in_subh += list(bfs_nodes)
    subh_all_terms.append(nodes_in_bfs(bfs_all, pred_terms_idx[root]))

subh_root_idx = np.array(_subh_root_idx)
print('Number of sub-hierarchies: {0}'.format(len(subh_root_idx)))
len_subh_pred_terms = [len(x) for x in subh_pred_terms]

# list sub-hierarchies
print()
print('**Sub-hierarchies**')
df_subh = pd.DataFrame(columns=['Root_idx', 'Root','Terms','Pred','Genes','Desc'])
for i, tidx in enumerate(subh_root_idx):
  term = G[tidx]
  data = [tidx, term]
  data += [len(subh_all_terms[i])]  # total number of terms in sub-hier.
  data += [len(subh_pred_terms[i])] # number of terms to predict in sub-hier.
  data += [np.count_nonzero(gene_by_go[:,idxG[term]])] # number of genes in sub.
  data += [data_term_def[data_term_def['Term']==term]['Name'].tolist()[0]]
  df_subh.loc[i] = data

df_subh = df_subh.sort_values(by=['Terms','Pred','Genes'], ascending=False).reset_index(drop=True)

# sub-hierarchies used for prediction
df_test_subh = df_subh.sort_values(by=['Terms','Pred','Genes'], ascending=True).tail(15).copy().reset_index(drop=True)
test_subh_root_idx = df_test_subh['Root_idx'].tolist()
print(df_test_subh)

test_subh_terms = list()
for i, root in enumerate(test_subh_root_idx):
  idx = np.where(subh_root_idx==root)[0][0]
  test_subh_terms.append(subh_pred_terms[idx])


###################
# 4. Design dataset
###################

i, root = 13, test_subh_root_idx[13]
term = G[root]

# get term info
print()
print('#####################')
print('Root term: {0}'.format(term))
terms_pred_idx = np.array(test_subh_terms[i])
genes_pred_idx = np.nonzero(gene_by_go[:,root])[0]

genes = P[genes_pred_idx]
terms = G[terms_pred_idx]

# create sub matrix terms_hier_idx hierarchy
sm_ppi = ppi[np.ix_(genes_pred_idx,genes_pred_idx)].copy()
sm_gene_by_go = gene_by_go[np.ix_(genes_pred_idx,terms_pred_idx)].copy()

# Conver DAG to tree, will be used for prediction
tree = mst(genes_pred_idx, terms_pred_idx, gene_by_go.copy(), go_by_go.copy())
sm_go_by_go = np.zeros((len(terms_pred_idx),len(terms_pred_idx)))

for i, idx in enumerate(terms_pred_idx):
  parents = direct_pa(idx, terms_pred_idx, tree)
  parents = [np.where(terms_pred_idx == p)[0][0] for p in parents]
  assert len(parents) <= 1
  sm_go_by_go[i, parents] = 1

df = pd.DataFrame()
df['Gene'] = pd.Series(genes)

term_idx = 3 # term to predict
term = terms[term_idx] # term to predict

# BFS variables
queue = deque()
queue.append((0,[],0))

while len(queue) != 0:
  pos, hist, d = queue.popleft()

  for idx in np.nonzero(sm_go_by_go[:,pos])[0]:
    queue.append((idx, hist+[idx],d+1))

  if pos == term_idx:
    queue = deque()
    for i, trm in enumerate(hist):
      df[G[terms_pred_idx[trm]]] = pd.Series(sm_gene_by_go[:,i])

df[term] = pd.Series(sm_gene_by_go[:,term_idx])
# df.to_csv('test/data_resume.csv', index=False)


###############
# 5. Prediction
###############

print()
print('**Prediction**')
print('Term: {0}'.format(term))
test = XGB_strc()
test.load_data(sm_ppi, term, df)
params = test.structural_test(path='test_a', normalize=False, seed=2020)

print('')
print('Training parameters')
print(params)
