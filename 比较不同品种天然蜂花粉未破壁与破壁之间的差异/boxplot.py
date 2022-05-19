import math
import random
import matplotlib
matplotlib.rc('font',family='Microsoft YaHei')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.preprocessing
from matplotlib.patches import Ellipse
from scipy.stats import ttest_ind, mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from skimage.measure import EllipseModel
data = pd.read_excel('../files/pollen files/results/peaktableBOTHout_BOTH_noid_replace_mean_full.xlsx')
# data = pd.read_excel('../files/pollen files/results/process_output_quantid_pos_camera_noid/peaktablePOSout_POS_noid_replace_puring.xlsx')
# data = pd.read_excel('../files/pollen files/results/process_output_quantid_neg_camera_noid/peaktableNEGout_NEG_noid_replace.xlsx')

print(data)

targets = data.columns.values[1:]
for i in range(len(targets)):
    if 'WX' not in targets[i]:
        del data[targets[i]]
targets = data.columns.values[1:]

# 分别比较样本1和6、
keywords1 = ['XYCH_WX_','XYCH_WXPB_']
# 样本2和7、
keywords2 = ['GYCH_WX_','GYCH_WXPB_']
# 样本3和8、
keywords3 = ['GWBZ_WX_','GWBZ_WXPB_']
# 样本4和9、
keywords4 = ['GHH_WX_','GHH_WXPB_']
# 样本5和10
keywords5 = ['GCH_WX_','GCH_WXPB_']
# 研究单个样本破壁与未破壁的变化差异
keywords6 = ['WX_','WXPB_']
keywords = keywords6

print(targets)

for i in range(len(targets)):
    if keywords[0] not in targets[i] and keywords[1] not in targets[i]:
        print(targets[i])
        del data[targets[i]]

print(data)
data = data.dropna().reset_index(drop=True)

saved_label = data['dataMatrix'].values
del data['dataMatrix']
targets = data.columns.values
print(targets)

print(data)

data_impute = data.values

for i in range(data_impute.shape[1]):
    data_impute[:, i] = (data_impute[:, i] / np.sum(data_impute[:, i])) * 100

normalized_data_impute = data_impute
print(normalized_data_impute)


x_index=[]
y_index=[]
print(targets)

for i in range(len(targets)):
    if keywords[0] in targets[i]:
        x_index.append(i)
    elif keywords[1] in targets[i]:
        y_index.append(i)

print(x_index)
print(y_index)
targets = np.hstack((targets[x_index],targets[y_index]))


normalized_data_impute_x = []
for index in x_index:
    normalized_data_impute_x.append(normalized_data_impute[:,index].T)
normalized_data_impute_x = np.array(normalized_data_impute_x)

normalized_data_impute_y =[]
for index in y_index:
    normalized_data_impute_y.append(normalized_data_impute[:,index].T)
normalized_data_impute_y = np.array(normalized_data_impute_y)


top_k = 20
p_list =[]
for i in range(normalized_data_impute_x.shape[1]):
    t,p = mannwhitneyu(normalized_data_impute_x[:,i:i+1],normalized_data_impute_y[:,i:i+1])
    p_list.append(p[0])
p_list = np.array(p_list)
count = 0
for p in p_list:
    if p < 0.05:
        count +=1

top_k_index = p_list.argsort()[::-1][len(p_list)-count:]
print(top_k_index)


X_XYCH_WX = np.array(normalized_data_impute_x)
X_GYCH_WX = np.array(normalized_data_impute_y)



X_diff_XYCH_WX = []
for i in top_k_index:
    X_diff_XYCH_WX.append(X_XYCH_WX[:,i:i+1])



X_diff_GYCH_WX = []
for i in top_k_index:
    X_diff_GYCH_WX.append(X_GYCH_WX[:,i:i+1])



data_XYCH_WX = []
labels = []
for i in range(len(X_diff_XYCH_WX)):
    data_XYCH_WX.append(X_diff_XYCH_WX[i])
    labels += [saved_label[top_k_index[i]], '']


data_GYCH_WX = []
for i in range(len(X_diff_GYCH_WX)):
    data_GYCH_WX.append(X_diff_GYCH_WX[i])





# Creating axes instance
data_XYCH_WXs = []
for i in data_XYCH_WX:
    data_XYCH_WXs.append(i.reshape(i.shape[0]))
data_XYCH_WX = data_XYCH_WXs

data_GYCH_WXs = []
for i in data_GYCH_WX:
    data_GYCH_WXs.append(i.reshape(i.shape[0]))
data_GYCH_WX = data_GYCH_WXs

data_XYCH_WX = np.array(data_XYCH_WX)
data_GYCH_WX = np.array(data_GYCH_WX)
print(data_XYCH_WX.shape)
print(data_GYCH_WX.shape)
data = np.hstack((data_XYCH_WX,data_GYCH_WX))

data = []
for i in range(data_XYCH_WX.shape[0]):
    data.append(data_XYCH_WX[i,:])
    data.append(data_GYCH_WX[i, :])



bp = plt.boxplot(data,labels=labels,patch_artist=True)
plt.xticks(rotation = 90)
for i in range(len(bp['boxes'])):
    if i %2 == 0:
        bp['boxes'][i].set(color='r')
    else:
        bp['boxes'][i].set(color='b')
plt.legend(handles=[bp['boxes'][0],bp['boxes'][1]],labels=['{}group'.format(keywords[0]),'{}group'.format(keywords[1])])
plt.title('不同品种天然蜂花粉未破壁与破壁之间的差异（未洗）')
plt.show()
# plt.savefig('figures/pos_plots/干燥油菜花粉.png')





