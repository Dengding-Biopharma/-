import csv

import pandas as pd


table = pd.read_csv('hmdb_metabolites.csv',quoting=csv.QUOTE_NONE,error_bad_lines=False)


print(table.columns.values)
print(table['inchi'])
quit()
import math
from math import nan

import numpy as np
from scipy.stats import ttest_ind
from sklearn.impute import SimpleImputer

data = pd.read_excel('files/ad files/peaktablePOSout_POS_noid_replace_variable_ours.xlsx')
print(data)
for column in data.columns.values:
    if '16' in column:
        del data[column]

color_exist = []
targets = data.columns.values[1:]

print(targets)

saved_label = data['dataMatrix'].values
print(saved_label)
del data['dataMatrix']
# 分别插值,根据column mean（所有sample这个variable的mean）插值
imputer_mean_ad = SimpleImputer(missing_values=np.nan,strategy='mean')
data_impute = imputer_mean_ad.fit_transform(data)
# imputer_mean_hc = SimpleImputer(missing_values=np.nan,strategy='mean')
# data_impute_hc = imputer_mean_ad.fit_transform(df_hc)
print(data_impute)
sum_baseline = 13800
for i in range(data_impute.shape[1]):
    coe = sum_baseline/np.sum(data_impute[:,i])
    data_impute[:, i] = (data_impute[:, i]*coe)/sum_baseline

normalized_data_impute = data_impute
print(normalized_data_impute.shape)
ad_index=[]
hc_index=[]
for i in range(len(targets)):
    if "AD" in targets[i]:
        ad_index.append(i)
    else:
        hc_index.append(i)


normalized_data_impute_ad = []
for index in ad_index:
    normalized_data_impute_ad.append(normalized_data_impute[:,index].T)
normalized_data_impute_ad = np.array(normalized_data_impute_ad)

normalized_data_impute_hc =[]
for index in hc_index:
    normalized_data_impute_hc.append(normalized_data_impute[:,index].T)
normalized_data_impute_hc = np.array(normalized_data_impute_hc)


top_k = 20
p_list =[]
for i in range(normalized_data_impute_ad.shape[1]):
    t,p = ttest_ind(normalized_data_impute_ad[:,i:i+1],normalized_data_impute_hc[:,i:i+1],equal_var=True)
    p_list.append(p[0])
p_list = np.array(p_list)
count = 0
for p in p_list:
    if p < 0.05:
        count +=1

top_k_index = p_list.argsort()[::-1][len(p_list)-count:]
#top_k_index = p_list.argsort()[::-1][:]
print(top_k_index)
print(len(top_k_index))
ids=[]
for k in top_k_index:
    for i in range(len(table['smiles'].values)):
        if saved_label[k] == table['smiles'].values[i]:
            ids.append(table['accession'].values[i])

print(ids)
print(len(ids))