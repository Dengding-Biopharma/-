import csv

import numpy as np
from rdkit import Chem
import pandas as pd

table = pd.read_csv('hmdb_metabolites.csv', sep='\t')
filename = 'NEG_significant.xlsx'
finalDF = pd.read_excel(filename)


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

    for i in range(len(table['name'].values)):
        if name == table['name'].values[i][2:-1]:
            print('match name!!!!!!!',table['name'].values[i][2:-1])
            ids.append(table['accession'].values[i])
            finalDF['inchikey'][k] = table['inchikey'].values[i]
            finalDF['accession'][k] = table['accession'].values[i]
            print(ids[-1])
            break
        elif inchikey == str(table['inchikey'][i]):
            print('match inchikey!!!!!!!',str(table['inchikey'][i]))
            ids.append(table['accession'].values[i])
            finalDF['inchikey'][k] = table['inchikey'].values[i]
            finalDF['accession'][k] = table['accession'].values[i]
            print(ids[-1])
            break

print(len(ids))
for id in ids:
    print(id)
print(finalDF)
finalDF.to_excel('{}_hmdbId_smile_inchikey.xlsx'.format(filename[:-5]),index=False,na_rep=np.nan)
