import numpy as np
import pandas as pd

# data = pd.read_excel('ad files/peaktablePOSout_POS_noid_more_puring.xlsx')
# print(data.index.values)
# print(data.columns.values)
# print(data.T)
# for column in data.columns.values:
#     if '0316' in column:
#         del data[column]
# df = pd.DataFrame()
# df['Patient ID'] = data.columns.values[1:]
# print(df)
# group = []
# for i in range(len(df)):
#     if 'AD' in df['Patient ID'][i]:
#         group.append('AD')
#     if 'HC' in df['Patient ID'][i]:
#         group.append('Control')
# df['Group'] = group
# print(df)
# for i, row in data.iterrows():
#     temp = pd.DataFrame(columns=[row.values[0]],data=row.values[1:])
#     df = pd.concat([df,temp],axis=1)
#
# df.to_excel('transpose_peaktable_pos.xlsx',index=False,na_rep=np.nan)

data1= pd.read_excel('transpose_peaktable_pos_mean_full.xlsx')
data2 = pd.read_excel('transpose_peaktable_pos.xlsx')
print(data1.columns.values)
print(data2.columns.values)
print(data1.columns.values==data2.columns.values)
for i in data1.columns.values==data2.columns.values:
    if i != True:
        print(i)



