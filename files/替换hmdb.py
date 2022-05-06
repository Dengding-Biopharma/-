import numpy as np
import pandas as pd

data = pd.read_excel('ad files/Pos-summary-0313-16.xlsx')

print(data.columns.values)

for i in range(len(data)):
    if data['HMDB_Score'][i] >=0.7:
        data['Max_NAME'][i] = data['HMDB_Name'][i]
        data['Max_pepmass'][i] = data['HMDB_pepmass'][i]
        data['Max_SMILES'][i] = data['HMDB_SMILES'][i]
        data['Max_Score'][i] = data['HMDB_Score' ][i]
        data['Max_Source'][i] = 'HMDB'
        data['Max_INCHI'][i] = data['HMDB_INCHI'][i]

print(data)
data.to_excel('ad files/Pos-summary-0313-16_hmdb.xlsx',index=False,na_rep=np.nan)

