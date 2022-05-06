import math
import random

import numpy as np
import pandas as pd

data =pd.read_csv('ad files/hmdb_metabolites.csv',delimiter='\t')
data_dic = pd.read_excel('ad files/varsPOSout_pos_noid_more_from_gnps.xlsx')
table = pd.read_excel('ad files/peaktablePOSout_POS_noid.xlsx')
print(data.columns.values)
print(data_dic.columns.values)
data['dataMatrix'] = data_dic['max_name']

for i in range(len(data_dic)):
    mz = data_dic['xcmsCamera_mz'][i]-1.0032 # POS mode
    # mz = data_dic['xcmsCamera_mz'][i]+1.0032 # NEG mode
    mzmin = mz-((15/1000000) * mz)
    mzmax = mz+((15/1000000) * mz)
    bool_mz = data['monisotopic_molecular_weight'].between(mzmin,mzmax, inclusive=True)
    temp = data[bool_mz]
    if temp.empty:
        continue
    try:
        data_dic['max_name'][i] = str(random.choice(temp['name'].values))

    except:
        continue

table['dataMatrix'] = data_dic['max_name']

for i in range(len(table)):
    try:
        if np.isnan(table['dataMatrix'][i]) or 'no_match' in table['dataMatrix'][i]:
            table = table.drop(i)
    except:
        continue
table.to_excel('ad files/temp.xlsx',index=False,na_rep=np.nan)
print(table)
data = pd.read_excel('ad files/temp.xlsx')
targets = data.columns.values[1:]
for i in range(len(data)):
    temp = []
    for j in targets:
        temp.append(data[j][i])
    for k in range(len(temp)):
        temp[k] = math.isnan(temp[k])
    if temp.count(True) > len(temp) /2:
        data = data.drop(i)

print(data)

data.to_excel('ad files/peaktablePOSout_POS_noid_more.xlsx',index=False,na_rep=np.nan)
table = pd.read_excel('ad files/peaktablePOSout_POS_noid_more.xlsx')
for i in range(len(table)):
    if 'no_match' in table['dataMatrix'][i]:
        table = table.drop(i)
    elif 'Massbank' in table['dataMatrix'][i]:
        temp = table['dataMatrix'][i].split(' ',1)[1].split('|')[0]
        table['dataMatrix'][i] = temp
    elif ';' in table['dataMatrix'][i]:
        temp = table['dataMatrix'][i].split(';')[0]
        table['dataMatrix'][i] = temp
    elif 'ReSpect' in table['dataMatrix'][i]:
        temp = table['dataMatrix'][i].split(' ',1)[1].split('|')[0]
        table['dataMatrix'][i] = temp
    elif 'HMDB' in table['dataMatrix'][i]:
        temp = table['dataMatrix'][i].split(' ',1)[1].split('|')[0]
        table['dataMatrix'][i] = temp
    elif 'Spectral Match to' in table['dataMatrix'][i]:
        temp = table['dataMatrix'][i].split('Spectral Match to ',1)[1]
        table['dataMatrix'][i] = temp
    elif table['dataMatrix'][i] == ' ':
        table = table.drop(i)

print(table)
table.to_excel('ad files/peaktablePOSout_POS_noid_more_puring.xlsx',index=False,na_rep=np.nan)