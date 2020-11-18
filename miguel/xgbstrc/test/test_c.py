#!/usr/bin/env python
# coding: utf-8

# Gene function prediction for Oryza sativa Japonica
# Miguel Romero 18/11/20

import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm

from context import model
from model import XGB_strc

##########################
# 1. Import and clean data
##########################

data_ppi = pd.read_csv('data/data_ppi.csv', dtype='object')
data_isa = pd.read_csv('data/data_isa.csv', dtype='object')
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
terms_pred_idx = list()
for i in range(nG):
  count = np.count_nonzero(gene_by_go[:,i])
  if 10 <= count <= 300:
    terms_pred_idx.append(i)
print('Number of filtered terms: {0}'.format(len(terms_pred_idx)))


###################
# 4. Design dataset
###################

term_idx = np.random.choice(terms_pred_idx, 1)[0]
term = G[term_idx]
df = pd.DataFrame()
df['Gene'] = pd.Series(P)
for idx, trm in zip(terms_pred_idx, G[terms_pred_idx]):
  df[trm] = pd.Series(gene_by_go[:,idx])
# df.to_csv('test/data_resume.csv', index=False)


###############
# 5. Prediction
###############

print()
print('**Prediction**')
print('Term: {0}'.format(term))
test = XGB_strc()
test.load_data(ppi, term, df)
params = test.structural_test(path='test_c', normalize=False, seed=2020)

print('')
print('Training parameters')
print(params)
