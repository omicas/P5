# Node type prediction algorithm
Algorithm to predict node types using supervised machine learning. The algorithm use an implementation of XGBoost and the SMOTE sampling technique.

The **parameters** required for the algorithm are:

| parameter | description |
|---|---|
| data_file | csv file containing data |
| type_file | csv file with node type definition (type id - type description) |
| label_col | name of column in data_file with node id |
| features | list of features in data_file use only for training |
| obj_features | list of features in data_file use for training and prediction (list for prediction is pruned) |
| threshold | minimum number of nodes annotated required for prediction (features above this threshold aren't use for prediction) |
| top_features | list of features in data_file use for training and targets for prediction (optional, in case there is no prune). Subset of obj_features. |
| folds | number of folds used in training |
| fp_folds | number of folds used in false positive analysis |
| seed | random seed (used for replication of results) |

The **outputs** are:

| file | description |
|---|---|
| acc_train | Accuracy score plot in training |
| avp_train | Average precision plot score in training |
| bac_train | Balanced accuracy plot score in training |
| f1s_train | f1-score plot in training |
| roc_train | ROC AUC score plot in training |
| roc_val | ROC AUC score plot in false positive analysis |
| fp-genes | list of false positive for each predicted feature (top_features), csv file |
| fp-resume | resume of false positive analysis for each predicted feature, csv file |

The example illustrate how the algorithm can be used to predict gene annotations from a gene co-expression network of rice (*Oryza sativa Japonica*) and some of its topological properties.
