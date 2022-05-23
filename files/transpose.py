import numpy as np
import pandas as pd

data = pd.read_excel('pollen files/results/peaktableBOTHout_BOTH_noid_replace_mean_full.xlsx')


df = pd.DataFrame()
df['Sample ID'] = data.columns.values[2:]

group = []
for i in range(len(df)):
    group.append(df['Sample ID'][i][:-2])
df['Group'] = group

for i, row in data.iterrows():
    temp = pd.DataFrame(columns=[row.values[0]],data=row.values[2:])
    df = pd.concat([df,temp],axis=1)

df.to_excel('pollen files/results/transpose_peaktable_both_mean_full.xlsx',index=False,na_rep=np.nan)
