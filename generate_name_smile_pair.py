import pandas as pd

data1 = pd.read_excel('files/ad files/varsPOSout_pos_noid_more_from_gnps.xlsx')
data2 = pd.read_excel('files/ad files/varsNEGout_neg_noid_replace.xlsx')

df = pd.DataFrame()
names=[]
smiles=[]
for i in range(len(data1)):
    if 'no_match' in data1['max_name'][i]:
        continue
    names.append(data1['max_name'][i])
    smiles.append(data1['max_smile'][i])

for i in range(len(data2)):
    if 'no_match' in data2['max_name'][i]:
        continue
    names.append(data2['max_name'][i])
    smiles.append(data2['max_smile'][i])

df['name'] = names
df['smile'] = smiles
print(df)