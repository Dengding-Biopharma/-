import math

import numpy as np
import pandas as pd

pos_data = pd.read_excel('ad files/peaktablePOSout_POS_noid_replace.xlsx')
neg_data = pd.read_excel('ad files/peaktableNEGout_NEG_noid_replace.xlsx')

pos_columns = pos_data.columns.values
neg_columns = neg_data.columns.values

for i in range(len(pos_columns)):
    if pos_columns[i] not in neg_columns:
        neg_data.insert(i,pos_columns[i],np.nan)

neg_columns = neg_data.columns.values

for i in range(len(neg_columns)):
    if neg_columns[i] not in pos_columns:
        pos_data.insert(i,neg_columns[i],np.nan)
print(neg_data.columns.values)
print(pos_data.columns.values)

data = pd.concat([pos_data, neg_data], ignore_index=True)
targets = data.columns.values[2:]

for i in range(len(data)):
    temp = []
    for j in targets:
        temp.append(data[j][i])
    for k in range(len(temp)):
        temp[k] = math.isnan(temp[k])
    if temp.count(True) > len(temp) /2:
        data = data.drop(i)

print(data)

data.to_excel('ad files/peaktableBOTHout_BOTH_noid_replace.xlsx',index=False,na_rep=np.nan)