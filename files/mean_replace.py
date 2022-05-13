import numpy as np
import pandas as pd

data = pd.read_excel('../files/ad files/peaktablePOSout_POS_noid_more_puring.xlsx')
for column in data.columns.values:
    if '0316' in column:
        del data[column]
print(data.values)
targets = data.columns.values
print(targets)

saved_label = data['dataMatrix'].values
print(saved_label)
ad_index = []
hc_index = []
for i in range(len(targets)):
    if "AD" in targets[i]:
        ad_index.append(i)
    elif 'HC' in targets[i]:
        hc_index.append(i)
print(ad_index)
print(hc_index)

data_array = []
for i in range(len(data)):
    ad = []
    for j in ad_index:
        ad.append(data.values[i][j])
    temp = []
    for num in ad:
        if np.isnan(num):
            continue
        temp.append(num)
    ad_mean = np.mean(temp)
    for m in range(len(ad)):
        if np.isnan(ad[m]):
            ad[m] = ad_mean


    hc = []
    for k in hc_index:
        hc.append(data.values[i][k])
    temp = []
    for num in hc:
        if np.isnan(num):
            continue
        temp.append(num)
    hc_mean = np.mean(temp)
    for n in range(len(hc)):
        if np.isnan(hc[n]):
            hc[n] = hc_mean
    row = [saved_label[i]]+ad+hc
    data_array.append(row)


df = pd.DataFrame(data_array,columns=targets)

df.to_excel('../files/ad files/peaktablePOSout_POS_noid_more_puring_mean_full.xlsx',index=False,na_rep=np.nan)





