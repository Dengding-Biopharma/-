import math
import random
import matplotlib

matplotlib.rc('font', family='Microsoft YaHei')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.preprocessing
from matplotlib.patches import Ellipse
from scipy.stats import ttest_ind, stats
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from skimage.measure import EllipseModel
from statsmodels.stats.anova import anova_lm

data = pd.read_excel(
    '../files/pollen files/results/process_output_quantid_pos_camera_noid/peaktablePOSout_POS_noid_replace.xlsx')
# data = pd.read_excel('../files/pollen files/results/process_output_quantid_neg_camera_noid/peaktableNEGout_NEG_noid_replace.xlsx')
print(data)

sample_labels = []

color_exist = []
targets = data.columns.values[1:]

print(targets)
print(len(targets))

saved_label = data['dataMatrix'].values
print(saved_label)
del data['dataMatrix']

imputer_mean_XYCH_WX = SimpleImputer(missing_values=np.nan, strategy='mean')
data_impute = imputer_mean_XYCH_WX.fit_transform(data)

sum_baseline = 10000
for i in range(data_impute.shape[1]):
    coe = sum_baseline / np.sum(data_impute[:, i])
    data_impute[:, i] = (data_impute[:, i] * coe) / sum_baseline

normalized_data_impute = data_impute
print(normalized_data_impute)

# 分别比较样本1和6、
keywords1 = ['XYCH_WX_', 'XYCH_QX_', 'XYCH_QXRY_']
# 样本2和7、
keywords2 = ['GYCH_WX_', 'GYCH_QX_', 'GYCH_QXRY_']
# 样本3和8、
keywords3 = ['GWBZ_WX_', 'GWBZ_QX_', 'GWBZ_QXRY_']
# 样本4和9、
keywords4 = ['GHH_WX_', 'GHH_QX_', 'GHH_QXRY_']
# 样本5和10
keywords5 = ['GCH_WX_', 'GCH_QX_', 'GCH_QXRY_']
# 研究单个样本破壁与未破壁的变化差异
keywords6 = ['WX_', 'QX_', 'QXRY']
x_index = []
y_index = []
z_index = []
print(targets)
keywords = keywords6
for i in range(len(targets)):
    if keywords[0] in targets[i]:
        x_index.append(i)
    elif keywords[1] in targets[i]:
        y_index.append(i)
    elif keywords[2] in targets[i]:
        z_index.append(i)

print(x_index)
print(y_index)
print(z_index)
targets = np.hstack((targets[x_index], targets[y_index], targets[z_index]))
print(targets)
print(len(targets))

normalized_data_impute_x = []
for index in x_index:
    normalized_data_impute_x.append(normalized_data_impute[:, index].T)
normalized_data_impute_x = np.array(normalized_data_impute_x)

normalized_data_impute_y = []
for index in y_index:
    normalized_data_impute_y.append(normalized_data_impute[:, index].T)
normalized_data_impute_y = np.array(normalized_data_impute_y)

normalized_data_impute_z = []
for index in z_index:
    normalized_data_impute_z.append(normalized_data_impute[:, index].T)
normalized_data_impute_z = np.array(normalized_data_impute_z)

top_k = 20
p_list = []
for i in range(normalized_data_impute_x.shape[1]):
    f, p = stats.f_oneway(normalized_data_impute_x[:, i:i + 1], normalized_data_impute_y[:, i:i + 1],
                          normalized_data_impute_z[:, i:i + 1])
    p_list.append(p[0])
p_list = np.array(p_list)
count = 0
for p in p_list:
    if p < 0.05:
        count += 1

top_k_index = p_list.argsort()[::-1][len(p_list) - top_k:]
print(top_k_index)

print(len(top_k_index))

x = np.array(normalized_data_impute_x)
y = np.array(normalized_data_impute_y)
z = np.array(normalized_data_impute_z)

x_diff = []
for i in top_k_index:
    x_diff.append(x[:, i:i + 1])

y_diff = []
for i in top_k_index:
    y_diff.append(y[:, i:i + 1])

z_diff = []
for i in top_k_index:
    z_diff.append(z[:, i:i + 1])

data_x = []
labels = []
for i in range(len(x_diff)):
    data_x.append(x_diff[i])
    labels.append(saved_label[top_k_index[i]])

data_y = []

for i in range(len(y_diff)):
    data_y.append(y_diff[i])

data_z = []
for i in range(len(z_diff)):
    data_z.append(z_diff[i])

# Creating axes instance
data_xs = []
for i in data_x:
    data_xs.append(i.reshape(i.shape[0]))
data_x = data_xs

data_ys = []
for i in data_y:
    data_ys.append(i.reshape(i.shape[0]))
data_y = data_ys

data_zs = []
for i in data_z:
    data_zs.append(i.reshape(i.shape[0]))
data_z = data_zs

data_x = np.array(data_x)
data_y = np.array(data_y)
data_z = np.array(data_z)
print(data_x.shape)
print(data_y.shape)
print(data_z.shape)

data = np.hstack((data_x, data_y, data_z))

data = []

for i in range(data_x.shape[0]):
    data.append(data_x[i, :])
for i in range(data_y.shape[0]):
    data.append(data_y[i, :])
for i in range(data_z.shape[0]):
    data.append(data_z[i, :])

print(data)

bp = plt.boxplot(data, labels=labels + labels + labels, patch_artist=True)

plt.xticks(rotation=90)
for i in range(len(bp['boxes'])):
    if i < len(bp['boxes']) / 3:
        bp['boxes'][i].set(color='r')
    elif len(bp['boxes']) / 3 <= i < 2 * (len(bp['boxes']) / 3):
        bp['boxes'][i].set(color='g')
    else:
        bp['boxes'][i].set(color='b')
plt.show()
