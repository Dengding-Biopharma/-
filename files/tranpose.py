import numpy as np
import pandas as pd

data = pd.read_excel('ad files/peaktablePOSout_POS_noid_more_puring.xlsx')
print(data.index.values)
print(data.columns.values)


print(data.T)
df = pd.DataFrame()
df['Patient ID'] = data.columns.values[1:]
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
    temp = pd.DataFrame(columns=[row.values[0]],data=row.values[1:])

    df = pd.concat([df,temp],axis=1)

print(df)

df.to_excel('transpose_peaktable_pos.xlsx',index=False,na_rep=np.nan)
