import csv

import numpy as np
from rdkit import Chem
import pandas as pd

table = pd.read_csv('hmdb_metabolites.csv', sep='\t')
# count = 0
print(table.columns.values)
# for i in range(len(table)):
#     if type(table['kegg_id'][i]) != float:
#         count += 1
# print(count)
# print(table['inchi'])

# import math
# from math import nan
#
# import numpy as np
# from scipy.stats import ttest_ind
# from sklearn.impute import SimpleImputer
#
# data = pd.read_excel('files/ad files/peaktablePOSout_POS_noid_more_puring_mean_full.xlsx')
#
# targets = data.columns.values[1:]
#
# print(targets)
#
# saved_label = data['dataMatrix'].values
# print(saved_label)
# del data['dataMatrix']
# data_impute = data.values
# for i in range(data_impute.shape[1]):
#     data_impute[:, i] = data_impute[:, i] / np.sum(data_impute[:, i])
# print(data_impute)
#
# normalized_data_impute = data_impute
# print(normalized_data_impute.shape)
# ad_index=[]
# hc_index=[]
# for i in range(len(targets)):
#     if "AD" in targets[i]:
#         ad_index.append(i)
#     else:
#         hc_index.append(i)
#
#
# normalized_data_impute_ad = []
# for index in ad_index:
#     normalized_data_impute_ad.append(normalized_data_impute[:,index].T)
# normalized_data_impute_ad = np.array(normalized_data_impute_ad)
#
# normalized_data_impute_hc =[]
# for index in hc_index:
#     normalized_data_impute_hc.append(normalized_data_impute[:,index].T)
# normalized_data_impute_hc = np.array(normalized_data_impute_hc)
#
#
# top_k = 20
# p_list =[]
# for i in range(normalized_data_impute_ad.shape[1]):
#     t,p = ttest_ind(normalized_data_impute_ad[:,i:i+1],normalized_data_impute_hc[:,i:i+1],equal_var=True)
#     p_list.append(p[0])
# p_list = np.array(p_list)
# count = 0
# for p in p_list:
#     if p < 0.05:
#         count +=1
#
# top_k_index = p_list.argsort()[::-1][len(p_list)-count:]
# #top_k_index = p_list.argsort()[::-1][:]
# print(top_k_index)
# print(count)
# df = pd.DataFrame()
# df['P']=p_list[top_k_index]
# df['name']=saved_label[top_k_index]
# df.to_excel('pos_significant.xlsx', index=False)
# data = pd.read_excel('pos_significant.xlsx')
# print(data)
#
# dup_row = data.duplicated(subset=['name'],keep=False)
# print(dup_row)
#
# dup_df = data[dup_row]
# for i in range(len(data)):
#     if dup_row[i] == True:
#         data = data.drop(i)
# print(data)
# dup_df=dup_df.sort_values(by='name')
#
# print(dup_df)
#
# dup_df.to_excel('temp.xlsx',index=False)
# dup_df = pd.read_excel('temp.xlsx')
# print(dup_df)
# temp = []
# for i in range(len(dup_df)):
#     if len(temp) == 0 or dup_df['name'][i] != temp[0]:
#         temp = [dup_df['name'][i], dup_df['P'][i],i]
#     elif dup_df['name'][i] == temp[0] and dup_df['P'][i] < temp[1]:
#         dup_df = dup_df.drop(temp[2])
#         temp =[dup_df['name'][i],dup_df['P'][i],i]
#     elif dup_df['name'][i] == temp[0] and dup_df['P'][i] > temp[1]:
#         dup_df = dup_df.drop(i)
#
# print(len(dup_df))
finalDF = pd.read_excel('pos_significant.xlsx')


finalDF['accession'] = np.nan
finalDF['inchikey']=np.nan
print(finalDF)
ids = []
inchiKeys = []
for k in range(len(finalDF)):
    smile = finalDF['smile'][k]
    name = finalDF['name'][k]
    try:
        m = Chem.MolFromSmiles(smile)
        inchikey = Chem.MolToInchiKey(m)
    except:
        inchikey = None
    # print(name,inchiKeys)
    for i in range(len(table['name'].values)):
        if name == table['name'].values[i][2:-1]:
            print('match!!!!!!!',table['name'].values[i][2:-1])
            ids.append(table['accession'].values[i])
            finalDF['inchikey'][k] = table['inchikey'].values[i]
            finalDF['accession'][k] = table['accession'].values[i]
            print(ids[-1])
            break
        elif inchikey == str(table['inchikey'][i]):
            print('match!!!!!!!',str(table['inchikey'][i]))
            ids.append(table['accession'].values[i])
            finalDF['inchikey'][k] = table['inchikey'].values[i]
            finalDF['accession'][k] = table['accession'].values[i]
            print(ids[-1])
            break

print(len(ids))
for id in ids:
    print(id)
print(finalDF)
finalDF.to_excel('pos_hmdb_id_smile_inchikey.xlsx',index=False,na_rep=np.nan)
