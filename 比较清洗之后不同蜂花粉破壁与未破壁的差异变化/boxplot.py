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
    if 'QX' not in targets[i]:
        del data[targets[i]]
targets = data.columns.values[1:]
for i in range(len(targets)):
    if 'QXRY' in targets[i]:
        del data[targets[i]]
targets = data.columns.values[1:]
print(targets)
print(len(targets))


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
keywords1 = ['XYCH_QX_','XYCH_QXPB_']
# 样本2和7、
keywords2 = ['GYCH_QX_','GYCH_QXPB_']
# 样本3和8、
keywords3 = ['GWBZ_QX_','GWBZ_QXPB_']
# 样本4和9、
keywords4 = ['GHH_QX_','GHH_QXPB_']
# 样本5和10
keywords5 = ['GCH_QX_','GCH_QXPB_']
# 研究单个样本破壁与未破壁的变化差异
keywords6 = ['QX_','QXPB_']
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

print(normalized_data_impute_x.shape)
print(normalized_data_impute_y.shape)



top_k = 20
p_list =[]
for i in range(normalized_data_impute_x.shape[1]):
    t,p = ttest_ind(normalized_data_impute_x[:,i:i+1],normalized_data_impute_y[:,i:i+1],equal_var=True)
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
labels_XYCH_WX = []
for i in range(len(X_diff_XYCH_WX)):
    data_XYCH_WX.append(X_diff_XYCH_WX[i])
    labels_XYCH_WX.append(saved_label[top_k_index[i]])

data_GYCH_WX = []
labels_GYCH_WX = []
for i in range(len(X_diff_GYCH_WX)):
    data_GYCH_WX.append(X_diff_GYCH_WX[i])
    labels_GYCH_WX.append(saved_label[top_k_index[i]])




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
color_list=[]
for i in range(data_XYCH_WX.shape[0]):
    color_list.append('r')
    data.append(data_XYCH_WX[i,:])
for i in range(data_GYCH_WX.shape[0]):
    color_list.append('b')
    data.append(data_GYCH_WX[i, :])



print(data)
bp = plt.boxplot(data,labels=labels_XYCH_WX+labels_GYCH_WX,patch_artist=True)
plt.xticks(rotation = 90)
for i in range(len(bp['boxes'])):
    if i < len(bp['boxes'])/2:
        bp['boxes'][i].set(color='r')
    else:
        bp['boxes'][i].set(color='b')
plt.legend(handles=[bp['boxes'][0],bp['boxes'][0+count]],labels=['{}group'.format(keywords[0]),'{}group'.format(keywords[1])])
plt.title('清洗之后不同蜂花粉破壁与未破壁的差异变化')
plt.show()




