import numpy as np
import pandas as pd

pos_data = pd.read_excel('ad files/peaktablePOSout_POS_noid_source.xlsx')
neg_data = pd.read_excel('ad files/peaktableNEGout_NEG_noid_source.xlsx')

pos_columns = pos_data.columns.values
neg_columns = neg_data.columns.values

for i in range(len(pos_columns)):
    if pos_columns[i] not in neg_columns:
        neg_data.insert(i, pos_columns[i], np.nan)

neg_columns = neg_data.columns.values

for i in range(len(neg_columns)):
    if neg_columns[i] not in pos_columns:
        pos_data.insert(i, neg_columns[i], np.nan)
pos_columns = pos_data.columns.values
neg_columns = neg_data.columns.values
print(pos_columns)
print(neg_columns)
print(pos_columns == neg_columns)

pos_data.to_excel('ad files/peaktablePOSout_POS_noid.xlsx',index=False,na_rep=np.nan)
neg_data.to_excel('ad files/peaktableNEGout_NEG_noid.xlsx',index=False,na_rep=np.nan)