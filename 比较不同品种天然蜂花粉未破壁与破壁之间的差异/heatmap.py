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

# data = pd.read_excel('../files/pollen files/results/process_output_quantid_pos_camera_noid/peaktablePOSout_POS_noid_replace.xlsx')
data = pd.read_excel('../files/pollen files/results/process_output_quantid_neg_camera_noid/peaktableNEGout_NEG_noid_replace.xlsx')
print(data)

sample_labels = []


color_exist = []
targets = data.columns.values[1:]


for i in range(len(targets)):
    if 'WX' not in targets[i]:
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

# 分别比较样本1和6、
keywords1 = ['XYCH_WX_','XYCH_WXPB_']
# 样本2和7、
keywords2 = ['GYCH_WX_','GYCH_WXPB_']
# 样本3和8、
keywords3 = ['GWBZ_WX_','GWBZ_WXPB_']
# 样本4和9、
keywords4 = ['GHH_WX_','GHH_WXPB_']
# 样本5和10 研究单个样本破壁与未破壁的变化差异
keywords5 = ['GCH_WX_','GCH_WXPB_']
# 把样本1、2、3、4、5作为一组，
# 把样本6、7、8、9、10作为一组，进行比较，研究整体未破壁样本与破壁样本的变化
keywords6 = ['WX_','WXPB_']


x_index=[]
y_index=[]
print(targets)
keywords = keywords6
for i in range(len(targets)):
    if keywords[0] in targets[i]:
        x_index.append(i)
    elif keywords[1] in targets[i]:
        y_index.append(i)

print(x_index)
print(y_index)
targets = np.hstack((targets[x_index],targets[y_index]))
print(targets)




normalized_data_impute_x = []
for index in x_index:
    normalized_data_impute_x.append(normalized_data_impute[:,index].T)
normalized_data_impute_x = np.array(normalized_data_impute_x)

normalized_data_impute_y =[]
for index in y_index:
    normalized_data_impute_y.append(normalized_data_impute[:,index].T)
normalized_data_impute_y = np.array(normalized_data_impute_y)

data_impute_XYCH_WX = normalized_data_impute_x
data_impute_GYCH_WX = normalized_data_impute_y



top_k = 20
sum_list =[]
for i in range(normalized_data_impute_x.shape[1]):
    sum = np.sum(data_impute_XYCH_WX[:,i:i+1])
    sum_list.append(sum)

sum_list = np.array(sum_list)
top_k_index = sum_list.argsort()[::-1][0:top_k]
print(top_k_index)

X_XYCH_WX = np.array(normalized_data_impute_x)
X_GYCH_WX = np.array(normalized_data_impute_y)
X = np.vstack((X_XYCH_WX,X_GYCH_WX))
X = X.transpose()
print(X.shape)

X_top = []
for k in top_k_index:
    X_top.append(X[k])
X_top = np.array(X_top)
print(X_top)


df = pd.DataFrame()
for i in range(len(targets)):
    temp = []
    for j in X_top[:,i:i+1]:
        print(j)
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

if len(keywords) >6:
    width = 1500
else:
    width = 2000
clustergram = dashbio.Clustergram(
    data=df,
    column_labels=list(df.columns.values),
    row_labels=list(df.index),
    height=1000,
    width=width,
)

clustergram.show()
dcc.Graph(figure=clustergram)
# df.to_excel('data_new.xlsx')