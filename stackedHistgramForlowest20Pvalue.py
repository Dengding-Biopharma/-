import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import ttest_ind
from sklearn.impute import SimpleImputer

data = pd.read_excel('files/peaktablePOSout_POS_noid_replace_variable.xlsx')

for column in data.columns.values:
    if '16' in column:
        del data[column]

color_exist = []
targets = data.columns.values[1:]


print(data)
print(targets)

for i in range(len(data)):
    temp = []
    for j in targets:
        temp.append(data[j][i])
    for k in range(len(temp)):
        temp[k] = math.isnan(temp[k])
    if temp.count(True) >= len(temp) /2:
        data = data.drop(i)

saved_label = data['dataMatrix'].values
print(saved_label)
del data['dataMatrix']
# 分别插值,根据column mean（所有sample这个variable的mean）插值
imputer_mean_ad = SimpleImputer(missing_values=np.nan,strategy='mean')
data_impute = imputer_mean_ad.fit_transform(data)
# imputer_mean_hc = SimpleImputer(missing_values=np.nan,strategy='mean')
# data_impute_hc = imputer_mean_ad.fit_transform(df_hc)
print(data_impute)
sum_baseline = 30000
for i in range(data_impute.shape[1]):
    coe = sum_baseline/np.sum(data_impute[:,i])
    data_impute[:, i] = (data_impute[:, i]*coe)/sum_baseline

normalized_data_impute = data_impute
print(normalized_data_impute.shape)

ad_index=[]
hc_index=[]
for i in range(len(targets)):
    if "AD" in targets[i]:
        ad_index.append(i)
    else:
        hc_index.append(i)
print(ad_index)
print(hc_index)

normalized_data_impute_ad = []
for index in ad_index:
    normalized_data_impute_ad.append(normalized_data_impute[:,index].T)
normalized_data_impute_ad = np.array(normalized_data_impute_ad)

normalized_data_impute_hc =[]
for index in hc_index:
    normalized_data_impute_hc.append(normalized_data_impute[:,index].T)
normalized_data_impute_hc = np.array(normalized_data_impute_hc)

data_impute_ad = normalized_data_impute_ad
data_impute_hc = normalized_data_impute_hc

top_k = 20
p_list =[]
for i in range(data_impute_ad.shape[1]):
    t,p = ttest_ind(data_impute_ad[:,i:i+1],data_impute_hc[:,i:i+1],equal_var=True)
    p_list.append(p[0])
p_list = np.array(p_list)

top_k_index = p_list.argsort()[::-1][len(p_list)-top_k:]
print(top_k_index)

X_ad = np.array(data_impute_ad)
X_hc = np.array(data_impute_hc)
X = np.vstack((X_ad,X_hc))



X_top = []
sum_list = []

for row in range(X.shape[0]):
    sum = np.sum(X[row,:])
    temp = []
    for k in top_k_index:
        percentage = (X[row, k:k + 1]/sum) * 100
        temp.append(percentage[0])

    X_top.append(temp)



X_top = np.array(X_top)

X_top = X_top.reshape(top_k,X_top.shape[0])

labels = []
for i in top_k_index:
    labels.append(saved_label[i])




#
# targets = list(targets)
# targets.append('others')
# targets = np.array(targets)

fig,ax = plt.subplots()

plt.xticks(rotation = 90)
ax.bar(targets,X_top[0],0.2,label=labels[0])
for i in range(1,len(X_top)):
    ax.bar(targets,X_top[i],0.2,bottom=X_top[i-1],label=labels[i])

plt.title('Histogram of the 20 most different metabolic distribution between groups')
ax.legend(bbox_to_anchor=(1, 1),prop={'size': 8},loc='best')
plt.show()