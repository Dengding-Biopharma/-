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
print(XYCH_WX_index)
print(GYCH_WX_index)


normalized_data_impute_XYCH_WX = []
for index in XYCH_WX_index:
    normalized_data_impute_XYCH_WX.append(normalized_data_impute[:,index].T)
normalized_data_impute_XYCH_WX = np.array(normalized_data_impute_XYCH_WX)

normalized_data_impute_GYCH_WX =[]
for index in GYCH_WX_index:
    normalized_data_impute_GYCH_WX.append(normalized_data_impute[:,index].T)
normalized_data_impute_GYCH_WX = np.array(normalized_data_impute_GYCH_WX)


top_k = 20
p_list =[]
for i in range(normalized_data_impute_XYCH_WX.shape[1]):
    t,p = ttest_ind(normalized_data_impute_XYCH_WX[:,i:i+1],normalized_data_impute_GYCH_WX[:,i:i+1],equal_var=True)
    p_list.append(p[0])
p_list = np.array(p_list)
count = 0
for p in p_list:
    if p < 0.05:
        count +=1

top_k_index = p_list.argsort()[::-1][len(p_list)-count:]
print(top_k_index)



X_XYCH_WX = np.array(normalized_data_impute_XYCH_WX)
X_GYCH_WX = np.array(normalized_data_impute_GYCH_WX)



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
plt.legend(handles=[bp['boxes'][0],bp['boxes'][0+count]],labels=['{}group'.format('XYCH_'),'{}group'.format('GYCH_')])
plt.title('新鲜油菜花粉与干燥油菜花粉的差异（未洗）')
plt.show()



#
# ax1 = fig.XYCH_WXd_subplot(211)
# ax1.boxplot(data_XYCH_WX.T,  vert=0)
# ax1.set_yticklabels(labels_XYCH_WX)
# plt.title('boxplot for top 20 variables which have significant differences between groups\nXYCH_WX group')
#
# ax2 = fig.XYCH_WXd_subplot(212)
# ax2.boxplot(data_GYCH_WX.T,  vert=0)
# ax2.set_yticklabels(labels_GYCH_WX)
# plt.title('GYCH_WX group')
# for cap in bp['caps']:
#     cap.set(color ='#8B008B',
#             linewidth = 2)
# ax2.get_xaxis().tick_bottom()
# ax2.get_yaxis().tick_left()
# for flier in bp['fliers']:
#     flier.set(marker ='D',
#               color ='#e7298a',
#               alpha = 0.5)
# for median in bp['medians']:
#     median.set(color ='red',
#                linewidth = 3)
# for whisker in bp['whiskers']:
#     whisker.set(color ='#8B008B',
#                 linewidth = 1.5,
#                 linestyle =":")


plt.show()




