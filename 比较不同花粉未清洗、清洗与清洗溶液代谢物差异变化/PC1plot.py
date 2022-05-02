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

# data = pd.read_excel(
#     '../files/pollen files/results/process_output_quantid_pos_camera_noid/peaktablePOSout_POS_noid_replace.xlsx')
data = pd.read_excel('../files/pollen files/results/process_output_quantid_neg_camera_noid/peaktableNEGout_NEG_noid_replace.xlsx')
print(data)

sample_labels = []

color_exist = []
targets = data.columns.values[1:]

for i in range(len(targets)):
    if 'WX_' not in targets[i] and 'QX_' not in targets[i] and 'QXRY_' not in targets[i]:
        del data[targets[i]]
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
    normalized_data_impute_x.append(normalized_data_impute[:,index].T)
normalized_data_impute_x = np.array(normalized_data_impute_x)

normalized_data_impute_y =[]
for index in y_index:
    normalized_data_impute_y.append(normalized_data_impute[:,index].T)
normalized_data_impute_y = np.array(normalized_data_impute_y)

normalized_data_impute_z = []
for index in z_index:
    normalized_data_impute_z.append(normalized_data_impute[:, index].T)
normalized_data_impute_z = np.array(normalized_data_impute_z)


print(normalized_data_impute_x.shape)
print(normalized_data_impute_y.shape)
normalized_data_impute = np.vstack((normalized_data_impute_x,normalized_data_impute_y,normalized_data_impute_z))
print(normalized_data_impute.shape)



# PCA
pca = PCA(n_components=2)
pca.fit(normalized_data_impute)
X_new = pca.fit_transform(normalized_data_impute)
print(X_new)
print(pca.explained_variance_ratio_)

targets = pd.DataFrame(data = targets)
print(targets[0].values)
print(X_new.shape)

df = pd.DataFrame()
df['targets'] = targets[0].values
df['PC1'] = X_new[:,:1]
df = df.sort_values(by='PC1')
print(df)

fig = plt.figure()
plt.scatter(df['targets'],df['PC1'],s=20,c='black')

X_mean = np.mean(X_new[:,:1])
X_mean_list = []
for i in range(len(targets[0].values)):
    X_mean_list.append(X_mean)
plt.plot(X_mean_list,linestyle = 'solid')
plt.text(1,X_mean,'Mean',fontsize=8)

X_std = np.std(X_new[:,:1])
X_mean_list = []
for i in range(len(targets[0].values)):
    X_mean_list.append(X_mean+X_std)
plt.plot(X_mean_list,linestyle = 'dashed')
plt.text(1,X_mean+X_std,'Mean+1*std',fontsize=8)

X_mean_list = []
for i in range(len(targets[0].values)):
    X_mean_list.append(X_mean+2*X_std)
plt.plot(X_mean_list,linestyle = 'dashed')
plt.text(1,X_mean+2*X_std,'Mean+2*std',fontsize=8)

X_mean_list = []
for i in range(len(targets[0].values)):
    X_mean_list.append(X_mean-X_std)
plt.plot(X_mean_list,linestyle = 'dashed')
plt.text(1,X_mean-X_std,'Mean-1*std',fontsize=8)


plt.xlabel('samples')
plt.title('PC1 for every sample')
plt.xticks(rotation = 90)
plt.show()


#
# principalDf = pd.DataFrame(data = X_new
#              , columns = ['principal component 1', 'principal component 2'])
# finalDf = pd.concat([principalDf, targets], axis = 1)
# print(finalDf)
#
# targets = list(targets[0])
#
# colors = color_exist
# fig = plt.figure(figsize = (20,20))
# ax = fig.XYCH_WXd_subplot(1,1,1)
# ax.set_xlabel('Principal Component 1 {}%'.format(round(pca.explained_variance_ratio_[0]*100,2)), fontsize = 15)
# ax.set_ylabel('Principal Component 2 {}%'.format(round(pca.explained_variance_ratio_[1]*100,2)), fontsize = 15)
# ax.set_title('2 component PCA', fontsize = 20)
#
#
#
# for target, color in zip(targets,colors):
#     indicesToKeep = finalDf[0].values == target
#     ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
#                , finalDf.loc[indicesToKeep, 'principal component 2']
#                , c = color
#                , s = 50)
#
# # ax.legend('XYCH_WX_Disease_group','GYCH_WX_Control_group')
# ax.grid()
#
#
# plt.show()