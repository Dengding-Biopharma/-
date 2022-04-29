import math
import random
import matplotlib
matplotlib.rc('font',family='Microsoft YaHei')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.preprocessing
from matplotlib.patches import Ellipse
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from skimage.measure import EllipseModel

data = pd.read_excel('../files/pollen files/results/process_output_quantid_pos_camera_noid/peaktablePOSout_POS_noid_replace.xlsx')
# data = pd.read_excel('../files/pollen files/results/process_output_quantid_neg_camera_noid/peaktableNEGout_NEG_noid_replace.xlsx')
print(data)

sample_labels = []


color_exist = []
targets = data.columns.values[1:]


for i in range(len(targets)):
    if 'XYCH_WX_' not in targets[i] and 'GYCH_WX_' not in targets[i]:
        del data[targets[i]]
targets = data.columns.values[1:]
print(targets)






saved_label = data['dataMatrix'].values
print(saved_label)
del data['dataMatrix']

imputer_mean_XYCH_WX = SimpleImputer(missing_values=np.nan,strategy='mean')
data_impute = imputer_mean_XYCH_WX.fit_transform(data)


sum_baseline = 10000
for i in range(data_impute.shape[1]):
    coe = sum_baseline/np.sum(data_impute[:,i])
    data_impute[:, i] = (data_impute[:, i]*coe)/sum_baseline

normalized_data_impute = data_impute
print(normalized_data_impute)

XYCH_WX_index=[]
GYCH_WX_index=[]
for i in range(len(targets)):
    if "XYCH_WX_" in targets[i]:
        XYCH_WX_index.append(i)
    else:
        GYCH_WX_index.append(i)


normalized_data_impute_XYCH_WX = []
for index in XYCH_WX_index:
    normalized_data_impute_XYCH_WX.append(normalized_data_impute[:,index].T)
normalized_data_impute_XYCH_WX = np.array(normalized_data_impute_XYCH_WX)

normalized_data_impute_GYCH_WX =[]
for index in GYCH_WX_index:
    normalized_data_impute_GYCH_WX.append(normalized_data_impute[:,index].T)
normalized_data_impute_GYCH_WX = np.array(normalized_data_impute_GYCH_WX)

data_impute_XYCH_WX = normalized_data_impute_XYCH_WX
data_impute_GYCH_WX = normalized_data_impute_GYCH_WX



top_k = 20
sum_list =[]
for i in range(normalized_data_impute_XYCH_WX.shape[1]):
    sum = np.sum(data_impute_XYCH_WX[:,i:i+1])
    sum_list.append(sum)

sum_list = np.array(sum_list)
top_k_index = sum_list.argsort()[::-1][0:top_k]
print(top_k_index)

X_XYCH_WX = np.array(normalized_data_impute_XYCH_WX)
X_GYCH_WX = np.array(normalized_data_impute_GYCH_WX)
X = np.vstack((X_XYCH_WX,X_GYCH_WX))
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
    width=2000,
)

clustergram.show()
dcc.Graph(figure=clustergram)
# df.to_excel('data_new.xlsx')