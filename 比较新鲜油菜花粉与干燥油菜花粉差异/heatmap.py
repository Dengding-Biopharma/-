import math

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import ttest_ind
from sklearn.impute import SimpleImputer
import dash_bio

data = pd.read_excel('../files/pollen files/results/process_output_quantid_neg_camera_noid/peaktableNEGout_NEG_noid_replace.xlsx')


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
print(len(saved_label))

del data['dataMatrix']
# 分别插值,根据column mean（所有sample这个variable的mean）插值
imputer_mean_ad = SimpleImputer(missing_values=np.nan,strategy='mean')
data_impute = imputer_mean_ad.fit_transform(data)
# imputer_mean_hc = SimpleImputer(missing_values=np.nan,strategy='mean')
# data_impute_hc = imputer_mean_ad.fit_transform(df_hc)
print(data_impute)
sum_baseline = 10000
for i in range(data_impute.shape[1]):
    coe = sum_baseline/np.sum(data_impute[:,i])
    data_impute[:, i] = (data_impute[:, i]*coe)/sum_baseline

normalized_data_impute = data_impute
print(normalized_data_impute.shape)
print(targets)

xych_index=[]
gych_index=[]
for i in range(len(targets)):
    if "XYCH" in targets[i]:
        xych_index.append(i)
    elif 'GYCH' in targets[i]:
        gych_index.append(i)
print(xych_index)
print(gych_index)
quit()


normalized_data_impute_xych = []
for index in xych_index:
    normalized_data_impute_xych.append(normalized_data_impute[:,index].T)
normalized_data_impute_xych = np.array(normalized_data_impute_xych)

normalized_data_impute_gych =[]
for index in gych_index:
    normalized_data_impute_gych.append(normalized_data_impute[:,index].T)
normalized_data_impute_gych = np.array(normalized_data_impute_gych)

data_impute_xych = normalized_data_impute_xych
data_impute_gych = normalized_data_impute_gych



top_k = 20
sum_list =[]
for i in range(normalized_data_impute_xych.shape[1]):
    sum = np.sum(data_impute_xych[:,i:i+1])
    sum_list.append(sum)

sum_list = np.array(sum_list)
top_k_index = sum_list.argsort()[::-1][0:top_k]
print(top_k_index)

X_xych = np.array(normalized_data_impute_xych)
X_gych = np.array(normalized_data_impute_gych)
X = np.vstack((X_xych,X_gych))
X = X.transpose()
print(X)
print(X.shape)

X_top = []
for k in top_k_index:
    X_top.append(X[k])
X_top = np.array(X_top)
print(X_top)
print(len(X_top))
print(len(X_top[0]))


df = pd.DataFrame()

for i in range(len(targets)):
    print(X_top[:,i:i+1])
    temp = []
    for j in X_top[:,i:i+1]:
        temp.append(j[0])

    df[targets[i]] = temp
print(saved_label)
labels = []
for k in top_k_index:
    labels.append(saved_label[k])
df['label'] = labels
print(df)
df = df.set_index('label')
print(df.index.values)
print(df)
import dash_bio as dashbio
from dash import dcc


clustergram = dashbio.Clustergram(
    data=df,
    column_labels=list(df.columns.values),
    row_labels=list(df.index),
    height=1000,
    width=3000,
)

clustergram.show()
dcc.Graph(figure=clustergram)
# df.to_excel('data_new.xlsx')