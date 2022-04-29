import math
import random
import matplotlib
from sklearn.ensemble import RandomForestClassifier

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
keywords = keywords1

for i in range(len(targets)):
    if keywords[0] not in targets[i] and keywords[1] not in targets[i]:
        del data[targets[i]]
targets = data.columns.values[1:]
print(targets)



saved_label = data['dataMatrix'].values
print(saved_label)
del data['dataMatrix']

imputer_mean_data = SimpleImputer(missing_values=np.nan,strategy='mean')
data_impute = imputer_mean_data.fit_transform(data)


sum_baseline = 10000
for i in range(data_impute.shape[1]):
    coe = sum_baseline/np.sum(data_impute[:,i])
    data_impute[:, i] = (data_impute[:, i]*coe)/sum_baseline

normalized_data_impute = data_impute
normalized_data_impute = normalized_data_impute.T


forest = RandomForestClassifier(n_estimators=10000,random_state=0,n_jobs=-1)
forest.fit(normalized_data_impute,targets)
importances = forest.feature_importances_
print(importances)
sorted_imps = sorted(importances,reverse=True)
print(sorted_imps)

indices = np.argsort(importances)[::-1]
print(indices)
top_20_labels = []
for i in range(20):
    top_20_labels.append(saved_label[indices[i]])
for f in range(normalized_data_impute.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, saved_label[indices[f]], importances[indices[f]]))

plt.barh(range(20),sorted_imps[:20],color='r',align='center')
plt.yticks(range(20),top_20_labels)
plt.title('Randomforest Importance graph for top 20 variables')
plt.show()
