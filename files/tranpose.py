import numpy as np
import pandas as pd

data = pd.read_excel('ad files/peaktableNEGout_NEG_noid_replace_mean_full.xlsx')


df = pd.DataFrame()
df['Patient ID'] = data.columns.values[2:]
print(df)
group = []
for i in range(len(df)):
    if 'AD' in df['Patient ID'][i]:
        group.append('AD')
    if 'HC' in df['Patient ID'][i]:
        group.append('Control')
df['Group'] = group
print(df)
for i, row in data.iterrows():
    temp = pd.DataFrame(columns=[row.values[0]],data=row.values[2:])
    df = pd.concat([df,temp],axis=1)

df.to_excel('transpose_peaktable_neg_mean_full.xlsx',index=False,na_rep=np.nan)





