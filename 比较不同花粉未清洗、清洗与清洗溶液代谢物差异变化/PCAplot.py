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
keywords = keywords5
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
normalized_data_impute = np.vstack((normalized_data_impute_x,normalized_data_impute_y))
print(normalized_data_impute.shape)



# PCA
pca = PCA(n_components=2)
pca.fit(normalized_data_impute)
X_new = pca.fit_transform(normalized_data_impute)
print(X_new)
print(pca.explained_variance_ratio_)

y_pred = KMeans(n_clusters=2,random_state=8).fit_predict(X_new)

print(y_pred)

group0 =[]
outlier_index = []
for i in range(len(y_pred)):
    if y_pred[i] == 2:
        group0.append(X_new[i])
        outlier_index.append(i)

group0 = np.array(group0)
# ellipse_outliers = EllipseModel()
# ellipse_outliers.estimate(group0)
# outliers_x_mean,outliers_y_mean,a,b,theta = ellipse_outliers.params




# plot
print(targets)
for i in range(len(targets)):
    if 'QX_' in targets[i]:
        targets[i] = 'QX_group'
    elif 'QXPB_' in targets[i]:
        targets[i] = 'QXPB_group'
targets = pd.DataFrame(data = targets)
print(targets)

principalDf = pd.DataFrame(data = X_new
             , columns = ['PC1', 'PC2'])
finalDf = pd.concat([principalDf, targets], axis = 1)

points_XYCH_WX = []
points_GYCH_WX = []
print(finalDf)

# for i in range(X_new.shape[0]):
#     if i not in outlier_index:
#         if 'QX_' in finalDf[0][i]:
#             points_XYCH_WX.append([finalDf['PC1'][i],finalDf['PC2'][i]])
#         else:
#             points_GYCH_WX.append([finalDf['PC1'][i], finalDf['PC2'][i]])




# points_XYCH_WX = np.array(points_XYCH_WX)
# ellipse_points_XYCH_WX = EllipseModel()
# ellipse_points_XYCH_WX.estimate(points_XYCH_WX)
# XYCH_WX_x_mean,XYCH_WX_y_mean,XYCH_WX_a,XYCH_WX_b,XYCH_WX_theta = ellipse_points_XYCH_WX.params
#
# points_GYCH_WX = np.array(points_GYCH_WX)
# ellipse_points_GYCH_WX = EllipseModel()
# ellipse_points_GYCH_WX.estimate(points_GYCH_WX)
# GYCH_WX_x_mean,GYCH_WX_y_mean,GYCH_WX_a,GYCH_WX_b,GYCH_WX_theta = ellipse_points_GYCH_WX.params
#




targets = list(targets[0])

colors = color_exist
fig = plt.figure(figsize = (20,20))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1 {}%'.format(round(pca.explained_variance_ratio_[0]*100,2)), fontsize = 15)
ax.set_ylabel('Principal Component 2 {}%'.format(round(pca.explained_variance_ratio_[1]*100,2)), fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)


# ellipse_XYCH_WX = Ellipse((XYCH_WX_x_mean, XYCH_WX_y_mean), 2*XYCH_WX_a, 2*XYCH_WX_b,XYCH_WX_theta,
#                         edgecolor='r', fc='None', lw=2)
# ax.XYCH_WXd_patch(ellipse_XYCH_WX)
# ellipse_GYCH_WX = Ellipse((GYCH_WX_x_mean, GYCH_WX_y_mean), 2*GYCH_WX_a, 2*GYCH_WX_b,GYCH_WX_theta,
#                         edgecolor='b', fc='None', lw=2)
# ax.XYCH_WXd_patch(ellipse_GYCH_WX)

groups=['QX_group','QXPB_group']

for i in range(len(groups)):
    print(groups[i])
    indicesToKeep = finalDf[0].values == groups[i]
    print(indicesToKeep)
    if groups[i] == 'QX_group':
        ax_XYCH_WX = ax.scatter(finalDf.loc[indicesToKeep ,'PC1'],
               finalDf.loc[indicesToKeep, 'PC2'],
               c = 'r'
               , s = 50)
    if groups[i] == 'QXPB_group':
        ax_GYCH_WX = ax.scatter(finalDf.loc[indicesToKeep, 'PC1'],
                                finalDf.loc[indicesToKeep, 'PC2'],
                                c='b'
                                , s=50)




plt.legend(handles=[ax_XYCH_WX,ax_GYCH_WX],labels=['{}group'.format(keywords[0]),'{}group'.format(keywords[1])],loc='upper right',labelspacing=2,prop={'size': 12})
ax.grid()

plt.show()




