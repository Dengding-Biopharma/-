import csv
from rdkit import Chem
import pandas as pd


table = pd.read_csv('hmdb_metabolites.csv',sep='\t')


print(table.columns.values)

print(table['inchi'])

import math
from math import nan

import numpy as np
from scipy.stats import ttest_ind
from sklearn.impute import SimpleImputer

data = pd.read_excel('files/ad files/peaktablePOSout_POS_noid_more_puring.xlsx')
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
sum_baseline = 30000
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
names = []
for k in top_k_index:
    # smile = saved_label[k]
    # m = Chem.MolFromSmiles(smile)
    # inchikey = Chem.MolToInchiKey(m)
    # print(inchikey)
    name = saved_label[k]

    for i in range(len(table['name'].values)):
        if name == table['name'].values[i][2:-1]:
            try:
                ids.append(str(table['name'].values[i][2:-1]))

            except:
                continue

            print(ids[-1])
# df = pd.DataFrame()
# df['name']=names
# df['id'] = ids
# df.to_excel('POS_pubchem_compound_id.xlsx',index=False,na_rep=np.nan)
print(ids)
print(len(ids))
for id in ids:
    print(id)