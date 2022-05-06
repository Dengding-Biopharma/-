import matplotlib
matplotlib.rc('font',family='Microsoft YaHei')
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.impute import SimpleImputer
import csv
from rdkit import Chem
import math
from math import nan

table = pd.read_csv('../hmdb_metabolites.csv',sep='\t')


data = pd.read_excel('../files/pollen files/results/process_output_quantid_pos_camera_noid/peaktablePOSout_POS_noid_replace.xlsx')

print(data)

sample_labels = []


color_exist = []
targets = data.columns.values[1:]


for i in range(len(targets)):
    if 'XYCH_WX_' not in targets[i] and 'GYCH_WX_' not in targets[i]:
        del data[targets[i]]
targets = data.columns.values[1:]
print(targets)



print(targets)


saved_label = data['dataMatrix'].values
print(saved_label)
del data['dataMatrix']

imputer_mean_XYCH_WX = SimpleImputer(missing_values=np.nan,strategy='mean')
data_impute = imputer_mean_XYCH_WX.fit_transform(data)


sum_baseline = 10000
for i in range(data_impute.shape[1]):
    coe = sum_baseline/np.sum(data_impute[:,i])
    data_impute[:, i] = (data_impute[:, i]*coe)/sum_baseline

normalized_data_impute = data_impute
print(normalized_data_impute)

XYCH_WX_index=[]
GYCH_WX_index=[]
for i in range(len(targets)):
    if "XYCH_WX_" in targets[i]:
        XYCH_WX_index.append(i)
    else:
        GYCH_WX_index.append(i)
print(XYCH_WX_index)
print(GYCH_WX_index)


normalized_data_impute_XYCH_WX = []
for index in XYCH_WX_index:
    normalized_data_impute_XYCH_WX.append(normalized_data_impute[:,index].T)
normalized_data_impute_XYCH_WX = np.array(normalized_data_impute_XYCH_WX)

normalized_data_impute_GYCH_WX =[]
for index in GYCH_WX_index:
    normalized_data_impute_GYCH_WX.append(normalized_data_impute[:,index].T)
normalized_data_impute_GYCH_WX = np.array(normalized_data_impute_GYCH_WX)


top_k = 20
p_list =[]
for i in range(normalized_data_impute_XYCH_WX.shape[1]):
    t,p = ttest_ind(normalized_data_impute_XYCH_WX[:,i:i+1],normalized_data_impute_GYCH_WX[:,i:i+1],equal_var=True)
    p_list.append(p[0])
p_list = np.array(p_list)
count = 0
for p in p_list:
    if p < 0.05:
        count +=1

top_k_index = p_list.argsort()[::-1][len(p_list)-count:]
print(top_k_index)

ids=[]
names = []
for k in top_k_index:
    # smile = saved_label[k]
    # m = Chem.MolFromSmiles(smile)
    # inchikey = Chem.MolToInchiKey(m)
    # print(inchikey)
    name = saved_label[k]
    print(name)
    for i in range(len(table['name'].values)):
        if name == table['name'].values[i][2:-1]:
            try:
                ids.append(str(table['accession'].values[i]))
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