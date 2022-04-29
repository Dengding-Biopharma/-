import math
import random
import matplotlib
from scipy.stats import ttest_ind

matplotlib.rc('font',family='Microsoft YaHei')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import sklearn.preprocessing
from matplotlib.patches import Ellipse
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from skimage.measure import EllipseModel

data = pd.read_excel('../files/pollen files/results/process_output_quantid_pos_camera_noid/peaktablePOSout_POS_noid_replace.xlsx')
# data = pd.reXYCH_WX_excel('../files/pollen files/results/process_output_quantid_neg_camera_noid/peaktableNEGout_NEG_noid_replace.xlsx')
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

data_impute_XYCH_WX = normalized_data_impute_XYCH_WX
data_impute_GYCH_WX = normalized_data_impute_GYCH_WX


top_k = 20
sum_list =[]
for i in range(data_impute_XYCH_WX.shape[1]):
    sum = np.sum(data_impute_XYCH_WX[:,i:i+1])

    sum_list.append(sum)
sum_list = np.array(sum_list)
top_k_index = sum_list.argsort()[::-1][0:top_k]
print(top_k_index)


X_XYCH_WX = np.array(data_impute_XYCH_WX)
X_GYCH_WX = np.array(data_impute_GYCH_WX)
X = np.vstack((X_XYCH_WX,X_GYCH_WX))

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
print(X_top.shape)
X_top = X_top.reshape(X_top.shape[1],X_top.shape[0])

labels = []
for i in top_k_index:
    labels.append(saved_label[i])




#
# targets = list(targets)
# targets.append('others')
# targets = np.array(targets)
color_exist = []
def get_random_color(color_exist):
    r = lambda: random.randint(0, 255)
    color = '#%02X%02X%02X' % (r(), r(), r())
    while color in color_exist:
        r = lambda: random.randint(0, 255)
        color = '#%02X%02X%02X' % (r(), r(), r())
    color_exist.append(color)
    return color



fig,ax = plt.subplots()

plt.xticks(rotation = 90)
ax.bar(targets,X_top[0],0.2,label=labels[0])
for i in range(1,len(X_top)):
    ax.bar(targets,X_top[i],0.2,bottom=X_top[i-1],label=labels[i],color=get_random_color(color_exist))

plt.title('Histogram of the top 20 metabolite percentage in 新鲜油菜花')
ax.legend(color_exist,bbox_to_anchor=(1, 1))
plt.show()