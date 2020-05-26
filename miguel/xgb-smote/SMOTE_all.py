#!/usr/bin/env python
# coding: utf-8

# Imported Libraries
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Classifier Libraries
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.under_sampling import NearMiss
import xgboost as xgb

# Other libraries
import warnings
warnings.filterwarnings("ignore")


# Default colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#7f7f7f', '#bcbd22', '#17becf']
EPS = 1e-6


# Load dataset
def load_dataset(file, label_col): # load_dataset('OSA-GeneAnnotation(BP).csv', 'entrez')
  df = pd.read_csv(file)
  label_df = df[label_col]
  df.drop([label_col], axis=1, inplace=True)

  return df, label_df


# Scale data
def scale_data(df, features): # scale_data(['ClosenessCentrality','Eccentricity','Degree','ClusteringCoefficient','TopologicalCoefficient','BetweennessCentrality','NeighborhoodConnectivity'])
  # RobustScaler is less prone to outliers
  rob_scaler = RobustScaler()

  i = 0
  for fn in features:
    scaled_feature = rob_scaler.fit_transform(df[fn].values.reshape(-1,1))
    df.drop([fn], axis=1, inplace=True)
    df.insert(i, fn, scaled_feature)
    i += 1

  return df


# Load type names
def load_type_name(file): # load_type_file('OSA-GeneAnnotation_names(BP).csv')
  tnames = pd.read_csv(file, header=None, names=['value', 'name'], dtype={'value': object, 'name': object})
  return tnames


# Select feature obj
def select_obj_feature(df, obj_features, threshold): # select_obj_feature(['0019509','0006457','0016226','0006508','0006810','0055085','0006096','0051205','0006418','0006355','0007165','0006396','0006189','0006099','0015977','0015940','0055114','0016070','0006833','0000105','0008152','0000272','0005975','0006265','0015979','0015995','0006779','0030001','0006605','0006886','0017038','0006629','0015937','0045454','0006437','0042549','0006400','0006470','0006098','0006573','0016117','0009765','0006412','0015976','0006364','0006367','0006468','0009234','0006855','0009058','0071951','0006520','0009088','0071805','0006821','0000917','0006436','0007205','0048544','0016075','0006535','0006139','0015031','0006631','0006259','0006281','0003333','0010024','0000160','0006614','0002098','0008033','0006662','0006221','0044237','0006424','0043039','0000154','0008652','0009165','0009116','0009156','0044249','0006505','0018160','0033014','0015992','0008219','0006351','0031123','0043631','0006979','0006777','0009813','0042398','0009231','0006814','0009082','0006730','0006435','0009785','0006817','0006487','0006414','0006430','0006807','0009107','0030163','0008299','0071722','0006465','0051188','0015986','0006427','0009987','0051252','0009308','0006796','0006428','0043086','0006289','0006413','0009408','0006790','0006069','0006729','0016559','0009089','0006006','0009416','0045038','0006857','0001522','0009451','0006749','0006754','0006812','0006353','0006431','0006801','0019538','0006415','0006352','0042254','0009094','0009584','0009585','0017006','0018106','0018298','0006163','0015969','0006788','0016485','0070588','0006260','0009306','0006450','0022900','0009228','0006164','0032957','0005978','0031167','0006544','0006563','0006108','0044262','0009052','0032259','0032968','0044238','0006334','0006426','0006633','0006568','0006464','0006597','0008295','0009081','0006546','0006542','0016480','0006298','0045005','0006694','0009396','0006000','0006003','0000079','0051726','0008643','0009168','0030494','0006783','0006887','0006824','0009236','0016114','0006223','0006811','0006541','0006537','0042026','0044267','0006388','0006014','0006073','0006419','0006952','0016310','0006595','0006564','0006571','0042218','0006438','0006433','0009435','0006402','0006072','0046168','0019752','0008610','0006168','0006421','0010027','0042558','0019430','0009252','0015684','0046939','0006184','0009073','0015746','0006950','0006012','0006423','0006760','0042823','0008615','0009439','0007050','0008654','0019464','0008360','0009273','0051301','0006310','0006974','0006869','0043043','0048268','0019836','0007067','0043412','0006434','0006750','0006813','0006432','0006165','0006183','0006228','0006241','0034755','0006506','0006200','0006308','0007018','0007264','0042545','0009725','0006913','0030244','0006284','0007017','0051258','0009664','0016567','0046274','0006725','0032312','0006486','0006081','0019953','0005985','0000103','0035556','0006071','0008283','0006644','0016042','0006536','0016043','0030036','0009607','0030833','0006559','0000077','0000226','0071577','0051013','0010215','0016049','0016311','0016192','0006102','0007275','0006354','0006357','0032784','0005992','0046488','0045892','0048193','0006511','0006066','0009247','0030259','0006032','0016998','0006909','0006278','0010338','0009086','0016125','0015991','0045116','0006820','0044070','0006888','0009611','0006499','0019856','0006879','0051603','0018279','0006561','0006422','0046417','0043085','0006635','0006461','0032012','0006596','0072488','0046836','0006166','0030042','0006626','0045039','0006591','0006207','0006222','0042546','0006275','0007021','0006566','0008272','0015671','0006526','0042450','0030418','0022904','0006106','0017183','0007186','0000902','0007010','0006302','0006452','0008612','0045901','0045905','0006122','0007030','0006865','0019318','0045900','0007047','0006744','0019307','0006808','0006425','0006621','0046034','0006333','0030071','0031145','0019673','0006090','0015780','0010315','0001510','0007155','0042256','0042255','0009966','0006665','0006680','0016575','0043161','0009102','0051186','0042176','0045980','0006479','0006825','0035434','0006897','0009405','0031120','0042742','0050832','0034968','0007034','0006839','0006397','0009072','0009443','0009909','0048573','0015743','0005986','0009690','0006471','0015770','0046470','0017148','0006306','0090116','0006527','0015074','0006904','0010044','0009790','0006094','0016068','0009873','0009269','0019419','0032313','0006021','0009415','0016458','0006606','0051103','0009250','0006208','0044205','0032065','0010029','0016973','0043666','0006370','0019370','0000162','0046835','0019358','0007585','0006446','0030488','0032324','0045893','0046907','0006379','0009103','0007154','0000398','0007049','0018342','0009245','0043461','0009067','0007031','0006013','0008616','0010038','0046373','0000245','0071267','0006231','0006545','0016568','0032955','0071705','0051276','0006075','0006914','0006891','0046854','0048015','0046938','0045944','0045010','0009733','0031929','0048280','0006366','1904668','0030041','0006282','0010508','0000184','0008380','0007131','0016570','0051090','0015693','0001932','0043401','0006338','0034227','0015914','0019288','0019363','0070481','0070966','0071025','0016925','0031668','0043044','0006420','0016598','0006885','0009059','0017004','0006476','0006659','0032502','0048478','0006270','0045226','0046080','0006826','0006261','0006269','0009186','0030261','0015936','0006348','0016573','0007020','0006884','0031047','0000724','1990426','0006556','0043066','0042128','0009312','0006672','0006467','0010264','0009446','0006772','0009229','0007166','0035435','0000042','0006562','0048278','0006007','0000413','0006188','0009152','0009452','0009098','0009097','0019725','0006383','0009113','0071266','0000723','0006303','0006177','0006529','0006429','0045040','0007062','0071704','0042773','0006555','0006890','0000045','0008202','0030245','0015696','0015851','0016036','0009910','0046949','0006097','0006570','0019310','0019295','0005991','2000028','0001678','0006525','0016579','0042752','0009688','0006873','0006481','0010223','1901601','0046654','0006637','0006002','0060003','0019556','0017003','0019478','0000918','0071918'])
  # Top most frequent features
  ans = [(df[objf].sum(), objf) for objf in obj_features]
  ans.sort(reverse=True)

  top_feature, i = list(), 0
  a, n = ans[i]
  while a >= threshold:
    top_feature.append(n)
    i += 1
    a, n = ans[i]

  print('Annotations: {0} -> {1}'.format(len(obj_features), len(top_feature)))
  return top_feature


# Splitting the Data
def split_data(df, target, n_splits, seed=None):
  X = df.drop(target, axis=1)
  y = df[target]
  sss = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=False)
  for train_index, test_index in sss.split(X, y):
      original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
      original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]
  # Turn into an array
  original_Xtrain = original_Xtrain.values
  original_Xtest = original_Xtest.values
  original_ytrain = original_ytrain.values
  original_ytest = original_ytest.values

  return original_Xtrain, original_ytrain, original_Xtest, original_ytest, sss


################
# Under-Sampling
################
def undersampling(rand_xgb, original_Xtrain, original_ytrain, original_Xtest, original_ytest, sss):
  # List to append the score and then find the average
  undersample_accuracy, undersample_balancedacc = list(), list()
  undersample_f1, undersample_auc, undersample_average_precision = list(), list(), list()

  # Cross Validating the right way
  for train, test in sss.split(original_Xtrain, original_ytrain):
    undersample_pipeline = imbalanced_make_pipeline(NearMiss(sampling_strategy='majority'), rand_xgb)
    undersample_model = undersample_pipeline.fit(original_Xtrain[train], original_ytrain[train])
    undersample_prediction = undersample_model.predict(original_Xtrain[test])

    undersample_accuracy.append(undersample_pipeline.score(original_Xtrain[test], original_ytrain[test]))
    undersample_balancedacc.append(balanced_accuracy_score(original_ytrain[test], undersample_prediction))
    undersample_f1.append(f1_score(original_ytrain[test], undersample_prediction))
    undersample_average_precision.append(average_precision_score(original_ytrain[test], undersample_prediction))
    undersample_auc.append(roc_auc_score(original_ytrain[test], undersample_prediction))

  # mean and std of metrics
  acc = (np.mean(undersample_accuracy), np.std(undersample_accuracy))
  bac = (np.mean(undersample_balancedacc), np.std(undersample_balancedacc))
  f1s = (np.mean(undersample_f1), np.std(undersample_f1))
  auc = (np.mean(undersample_auc), np.std(undersample_auc))
  avp = (np.mean(undersample_average_precision), np.std(undersample_average_precision))
  return acc, bac, f1s, auc, avp


################
# SMOTE Technique (Over-Sampling)
################
def oversampling_smote(rand_xgb, original_Xtrain, original_ytrain, original_Xtest, original_ytest, sss):
  # List to append the score and then find the average
  oversample_accuracy, oversample_balancedacc = list(), list()
  oversample_f1, oversample_auc, oversample_average_precision = list(), list(), list()

  # Implementing SMOTE Technique
  # Cross Validating the right way
  for train, test in sss.split(original_Xtrain, original_ytrain):
    # SMOTE happens during Cross Validation not before..
    pipeline = imbalanced_make_pipeline(SMOTE(sampling_strategy='minority'), rand_xgb)
    model = pipeline.fit(original_Xtrain[train], original_ytrain[train])
    best_est = rand_xgb.best_estimator_
    prediction = best_est.predict(original_Xtrain[test])

    oversample_accuracy.append(pipeline.score(original_Xtrain[test], original_ytrain[test]))
    oversample_balancedacc.append(balanced_accuracy_score(original_ytrain[test], prediction))
    oversample_f1.append(f1_score(original_ytrain[test], prediction))
    oversample_average_precision.append(average_precision_score(original_ytrain[test], prediction))
    oversample_auc.append(roc_auc_score(original_ytrain[test], prediction))

  # mean of metrics
  acc = (np.mean(oversample_accuracy), np.std(oversample_accuracy))
  bac = (np.mean(oversample_balancedacc), np.std(oversample_balancedacc))
  f1s = (np.mean(oversample_f1), np.std(oversample_f1))
  auc = (np.mean(oversample_auc), np.std(oversample_auc))
  avp = (np.mean(oversample_average_precision), np.std(oversample_average_precision))

  smote_prediction = best_est.predict(original_Xtest)
  auc_val = (roc_auc_score(original_ytest, smote_prediction), 0)

  return acc, bac, f1s, auc, avp#, auc_val


# Candidates to carry out further studies (i.e., in-vivo experiments in gene analysis)
def false_positive_smote(rand_xgb, original_Xtrain, original_ytrain, original_Xtest, original_ytest, sss):
  fp = list()

  # Implementing SMOTE Technique
  # Cross Validating the right way
  # accuracy_lst, balancedacc_lst, average_precision_lst, f1_lst, auc_lst = list(), list(), list(), list(), list()
  for train, test in sss.split(original_Xtrain, original_ytrain):
    pipeline = imbalanced_make_pipeline(SMOTE(sampling_strategy='minority'), rand_xgb) # SMOTE happens during Cross Validation not before..
    model = pipeline.fit(original_Xtrain[train], original_ytrain[train])
    best_est = rand_xgb.best_estimator_
    prediction = best_est.predict(original_Xtrain)
    _tmp = ((original_ytrain == 0) & (prediction == 1))
    _tmp = np.where(_tmp==True)[0].tolist()
    fp += _tmp

    # accuracy_lst.append(pipeline.score(original_Xtrain, original_ytrain))
    # balancedacc_lst.append(balanced_accuracy_score(original_ytrain, prediction))
    # f1_lst.append(f1_score(original_ytrain, prediction))
    # average_precision_lst.append(average_precision_score(original_ytrain, prediction))
    # auc_lst.append(roc_auc_score(original_ytrain, prediction))

  # acc = (np.mean(accuracy_lst), np.std(accuracy_lst))
  # bac = (np.mean(balancedacc_lst), np.std(balancedacc_lst))
  # f1s = (np.mean(f1_lst), np.std(f1_lst))
  # auc = (np.mean(average_precision_lst), np.std(average_precision_lst))
  # avp = (np.mean(auc_lst), np.std(auc_lst))

  smote_prediction = best_est.predict(original_Xtest)
  auc_val = roc_auc_score(original_ytest, smote_prediction)

  # return fp, acc, bac, f1s, auc, avp, auc_val
  return fp, auc_val


# Comparing metrics for the models
def compare_metrics(model, mname, top_features, mobj=0, mcom=1, mcri=['roc']):
  measures = ['acc','bac','f1s','roc','avp']
  mename = ['Accuracy','Balanced acc.','f1-score','AUC ROC','Avg. precision']
  best_target, target_auc, target_auc_std = list(), list(), list()
  for midx in range(len(measures)):
    ms, msn = measures[midx], mename[midx]
    y_diff, y_1, y_2 = list(), list(), list()
    c = 0
    for i in range(len(model[0][ms])):
      target = top_features[i]
      yarr = [model[m][ms][i] for m in range(len(model))]
      y_diff.append((yarr[mobj][0] - yarr[mcom][0], yarr, target))

      # Selection criteria
      if yarr[mobj][0] - yarr[mcom][0] > EPS:
        c += 1
        if ms in mcri:
          best_target.append(target)
          target_auc.append(yarr[mobj][0])
          target_auc_std.append(yarr[mobj][1])

    y_diff.sort(reverse=True)

    # print('{0}: oversampling is better than undersampling {1} times ({2:.2f}%)'.format(ms, c,(c*100)/len(xgb_under[ms])))

    X, yarr = np.array([y[2] for y in y_diff]), [y[1] for y in y_diff]
    Y, Ystd = list(), list()
    for yidx in range(len(model)):
      y, yd = np.array([t[yidx][0] for t in yarr]), np.array([t[yidx][1] for t in yarr])
      Y.append(y)
      Ystd.append(yd)
    plot_metric_std(X, Y, Ystd, mname, msn, '{0}_train'.format(ms))

  return best_target, target_auc, target_auc_std


# plot metric and score for validation
def plot_metric_val(x, y, ystd, yval, label, metric_name, file):
  fig, ax = plt.subplots(figsize=(10,4))
  x, y, ystd, yval = np.array(x), np.array(y), np.array(ystd), np.array(yval)
  plt.plot(x, y, 'o--', color=colors[0], label=label[0], linewidth=1)
  plt.fill_between(x, y - ystd, y + ystd, color=colors[0], alpha=0.5)
  plt.plot(x, yval, 'o--', color=colors[1], label=label[1], linewidth=1)

  plt.legend(loc='best', shadow=True, fontsize='medium')
  plt.xticks(rotation='vertical')
  plt.margins(0.015)
  plt.xlabel('Target')
  plt.ylabel(metric_name)
  plt.ylim(0,1)
  plt.tight_layout()
  # plt.show()
  plt.savefig('{0}.pdf'.format(file), format='pdf', dpi=600)
  plt.close()


# plot metric with std
def plot_metric_std(X, Y, Ystd, label, metric_name, file):
  fig, ax = plt.subplots(figsize=(10,4))
  for idx in range(len(Y)):
    plt.plot(X, Y[idx], 'o--', color=colors[idx], label=label[idx], linewidth=1)
    # plt.fill_between(X, Y[idx] - Ystd[idx], Y[idx] + Ystd[idx], color=colors[idx], alpha=0.5)

  plt.legend(loc='best', shadow=True, fontsize='medium')
  plt.xticks(rotation='vertical')
  plt.margins(0.015)
  plt.xlabel('Target')
  plt.ylabel(metric_name)
  plt.ylim(0,1)
  plt.tight_layout()
  # plt.show()
  plt.savefig('{0}.pdf'.format(file), format='pdf', dpi=600)
  plt.close()


# false positive to csv (through dataframe)
def false_positive_export(false_positive, best_target, label_df, tnames, auc_val):
  dtmp = dict()
  for k, v in zip(best_target, false_positive):
    dtmp[k] = pd.Series(v)
  fp_df = pd.DataFrame(dtmp)
  fp_df.to_csv(r'fp-genes.csv', index=False)
  data = list()

  for target, auct in zip(best_target, auc_val):
    # Count fp genes frequency
    fp_vc = pd.DataFrame(fp_df[target].value_counts(dropna=True, sort=True).reset_index())
    fp_count = fp_vc.shape[0]
    fp_vc.columns = ['value', 'counts']
    fp_vc['value'] = pd.to_numeric(fp_vc['value'], downcast='integer')
    mfp_target = fp_vc['counts'].max()

    # Replace target idx with name
    vc = fp_vc[fp_vc['counts'] == mfp_target]
    fp_names = list()
    for v in vc['value'].tolist():
      name = label_df.loc[v]
      fp_names.append(name)
    vc['value'] = fp_names

    data.append((target, tnames.loc[tnames['value']==target].values[0][1].strip(), fp_df[target].count(), len(fp_df[target].unique()), mfp_target, vc.shape[0], auct))

  res_df = pd.DataFrame(data, columns=['id','desc','totalfp','uniquefp','maxtries','freq','aucval'])
  res_df.sort_values(by=['freq'])
  res_df.sort_values(by=['aucval', 'maxtries'], ascending=False)
  res_df.to_csv(r'fp-resume.csv', index=False)


# training under vs over sampling
def training(df, obj_features, top_features, n_splits, seed=None):
  # Use RandomizedSearchCV to find the best parameters.
  # GradientBoosting Classifier
  xgb_params = {"max_depth": list(range(2,5,1)), "n_estimators": list(range(1,5,1)),
                "min_samples_leaf": list(range(5,7,1)), 'colsample_bytree': list(np.arange(0.1, 1.1, 0.1))}
  rand_xgb = RandomizedSearchCV(xgb.XGBClassifier(nthread=-1, random_state=seed), xgb_params, n_iter=4)

  xgb_under = {'acc':list(), 'bac':list(), 'f1s':list(), 'roc':list(), 'avp':list()}
  xgb_over = {'acc':list(), 'bac':list(), 'f1s':list(), 'roc':list(), 'avp':list()}
  xgb_over_wo = {'acc':list(), 'bac':list(), 'f1s':list(), 'roc':list(), 'avp':list()}

  for target in tqdm(top_features):
    t0 = time.time()

    Xtrain, ytrain, Xtest, ytest, sss = split_data(df, target, n_splits, seed)

    acc, bac, f1s, auc, avp = undersampling(rand_xgb, Xtrain, ytrain, Xtest, ytest, sss)
    xgb_under['acc'].append(acc)
    xgb_under['bac'].append(bac)
    xgb_under['f1s'].append(f1s)
    xgb_under['roc'].append(auc)
    xgb_under['avp'].append(avp)

    acc, bac, f1s, auc, avp = oversampling_smote(rand_xgb, Xtrain, ytrain, Xtest, ytest, sss)
    xgb_over['acc'].append(acc)
    xgb_over['bac'].append(bac)
    xgb_over['f1s'].append(f1s)
    xgb_over['roc'].append(auc)
    xgb_over['avp'].append(avp)

    # training only with obj features
    _df = df[obj_features]
    Xtrain, ytrain, Xtest, ytest, sss = split_data(_df, target, n_splits, seed)

    acc, bac, f1s, auc, avp = oversampling_smote(rand_xgb, Xtrain, ytrain, Xtest, ytest, sss)
    xgb_over_wo['acc'].append(acc)
    xgb_over_wo['bac'].append(bac)
    xgb_over_wo['f1s'].append(f1s)
    xgb_over_wo['roc'].append(auc)
    xgb_over_wo['avp'].append(avp)

    t1 = time.time()
    # print("Took {:.2}s".format(t1 - t0))

  best_target, bt_auc, bt_auc_std = compare_metrics([xgb_over, xgb_over_wo, xgb_under], ['Over sampling', 'Over sampling (wo)', 'Under sampling'], top_features)

  return best_target, bt_auc, bt_auc_std


def false_positive_analysis(df, best_target, t_auc, t_auc_std, n_splits=10, seed=None):
  # Use RandomizedSearchCV to find the best parameters.
  # GradientBoosting Classifier
  xgb_params = {"max_depth": list(range(2,5,1)), "n_estimators": list(range(1,5,1)),
                "min_samples_leaf": list(range(5,7,1)), 'colsample_bytree': list(np.arange(0.1, 1.1, 0.1))}
  rand_xgb = RandomizedSearchCV(xgb.XGBClassifier(nthread=-1, random_state=seed), xgb_params, n_iter=4)

  # xgb_trg = {'acc':list(), 'bac':list(), 'f1s':list(), 'roc':list(), 'avp':list(), 'auc_val':list()}
  xgb_trg = list()
  false_positive = list()

  for target in tqdm(best_target):
    t0 = time.time()

    # Splitting the Data (Original DataFrame)
    Xtrain, ytrain, Xtest, ytest, sss = split_data(df, target, 5, seed)
    sss = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=False)

    # fp, acc, bac, f1s, auc, avp, auc_val = false_positive_smote(rand_xgb, Xtrain, ytrain, Xtest, ytest, sss)
    fp, auc_val = false_positive_smote(rand_xgb, Xtrain, ytrain, Xtest, ytest, sss)
    false_positive.append(fp)
    # xgb_trg['acc'].append(acc)
    # xgb_trg['bac'].append(bac)
    # xgb_trg['f1s'].append(f1s)
    # xgb_trg['roc'].append(auc)
    # xgb_trg['avp'].append(avp)
    xgb_trg.append(auc_val)

  # measures = ['acc','bac','f1s','roc','avp', 'auc_val']
  # mename = ['Accuracy','Balanced acc.','f1-score','AUC ROC','Avg. precision', 'AUC ROC (Validation)']
  # for idx in range(len(measures)):
  #   X = np.array(best_target)
  #   Y, Ystd = np.array([y[0] for y in xgb_trg[measures[idx]]]), np.array([y[1] for y in xgb_trg[measures[idx]]])
  #   plot_metric_std(X, [Y], [Ystd], ['Over-Sampling'], mename[idx], '{0}_fp'.format(measures[idx]))
  x = np.array(best_target)
  plot_metric_val(x, t_auc, t_auc_std, xgb_trg, ['Training', 'Validation'], 'AUC ROC', '{0}_val'.format('roc'))

  return false_positive, xgb_trg


# Main
'''
Input:
    data_file       csv file containing data
    type_file       csv file with node type definition (type id - type description)
    label_col       name of column in data_file with node id
    features        list of features in data_file use only for training
    obj_features    list of features in data_file use for training and prediction (list for prediction is pruned)
    threshold       minimum number of nodes annotated required for prediction (features above this threshold aren't use for prediction)
    top_features    list of features in data_file use for training and targets for prediction (optional, in case there is no prune). Subset of obj_features.
    folds           number of folds used in training
    fp_folds        number of folds used in false positive analysis
    seed            random seed (used for replication of results)

Output:
    acc_train       Accuracy score plot in training
    avp_train       Average precision plot score in training
    bac_train       Balanced accuracy plot score in training
    f1s_train       f1-score plot in training
    roc_train       ROC AUC score plot in training
    roc_val         ROC AUC score plot in false positive analysis
    fp-genes        list of false positive for each predicted feature (top_features), csv file
    fp-resume       resume of false positive analysis for each predicted feature, csv file
'''
def main():
  data_file = 'PCC.csv'
  type_file = 'go_name.csv'
  label_col = 'gene'
  features = ['BetweennessCentrality', 'ClosenessCentrality', 'ClusteringCoefficient', 'Degree', 'NeighborhoodConnectivity', 'TopologicalCoefficient']
  obj_features = ['0000012','0000019','0000023','0000025','0000027','0000028','0000045','0000050']
  top_features = ['0000025', '0000045']
  threshold = 30
  folds = 10
  fp_folds = 50
  seed = 32020

  df, label_df = load_dataset(data_file, label_col)
  df = scale_data(df, features)
  tnames = load_type_name(type_file)
  top_features = select_obj_feature(df, obj_features, threshold)

  # Annotations which the topological models improve the prediction model using ATTED data
  # atted = ['0006397', '0006281', '0055114', '0006886', '0006888', '0030244', '0045454', '0007165', '0006357', '0006457', '0006952', '0006096']

  targets, t_auc, t_auc_std = training(df, obj_features, top_features, n_splits=folds, seed=seed)
  false_p, auc_val = false_positive_analysis(df, targets, t_auc, t_auc_std, n_splits=fp_folds, seed=seed)
  false_positive_export(false_p, targets, label_df, tnames, auc_val)

main()
