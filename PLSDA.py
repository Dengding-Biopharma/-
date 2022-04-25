from skimage.measure import EllipseModel
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import PLSRegression
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.preprocessing
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score


data = pd.read_excel('files/peaktablePOSout_POS_noid_replace_variable.xlsx')

for column in data.columns.values:
    if '16' in column:
        del data[column]


color_exist = []
targets = data.columns.values[1:]

for i in range(len(data)):
    temp = []
    for j in targets:
        temp.append(data[j][i])
    for k in range(len(temp)):
        temp[k] = math.isnan(temp[k])
    if temp.count(True) >= len(temp) /2:
        data = data.drop(i)


for i in range(len(targets)):
    if 'AD' in targets[i]:
        targets[i] = 'AD_Disease_group'
    else:
        targets[i] = 'HC_Control_group'

for i in range(len(targets)):
    if targets[i] == 'AD_Disease_group':
        color_exist.append('r')
    else:
        color_exist.append('b')


print(targets)




saved_label = data['dataMatrix'].values
print(saved_label)
del data['dataMatrix']
# 分别插值,根据column mean（所有sample这个variable的mean）插值
imputer_mean_ad = SimpleImputer(missing_values=np.nan,strategy='mean')
data_impute = imputer_mean_ad.fit_transform(data)
# imputer_mean_hc = SimpleImputer(missing_values=np.nan,strategy='mean')
# data_impute_hc = imputer_mean_ad.fit_transform(df_hc)
print(data_impute)
sum_baseline = 13800
for i in range(data_impute.shape[1]):
    coe = sum_baseline/np.sum(data_impute[:,i])
    data_impute[:, i] = (data_impute[:, i]*coe)/sum_baseline

normalized_data_impute = data_impute
print(normalized_data_impute)

ad_index=[]
hc_index=[]
for i in range(len(targets)):
    if "AD" in targets[i]:
        ad_index.append(i)
    else:
        hc_index.append(i)
print(ad_index)
print(hc_index)

normalized_data_impute_ad = []
for index in ad_index:
    normalized_data_impute_ad.append(normalized_data_impute[:,index].T)
normalized_data_impute_ad = np.array(normalized_data_impute_ad)

normalized_data_impute_hc =[]
for index in hc_index:
    normalized_data_impute_hc.append(normalized_data_impute[:,index].T)
normalized_data_impute_hc = np.array(normalized_data_impute_hc)


print(normalized_data_impute_ad.shape)
print(normalized_data_impute_hc.shape)

X_ad = np.array(normalized_data_impute_ad)
X_hc = np.array(normalized_data_impute_hc)
X = np.vstack((X_ad,X_hc))
print(X)
int_targets = []
for i in targets:
    if 'AD' in i:
        int_targets.append(0)
    else:
        int_targets.append(1)


print(X.shape)

plsr = PLSRegression(n_components=2,scale=False)
plsr.fit(X,int_targets)

print(plsr.predict(X))
predicts = []
for predict in plsr.predict(X):
    if predict >=0.5:
        predicts.append(1)
    else:
        predicts.append(0)




colormap = {
    'AD_Disease_group': '#ff0000',  # Red
    'HC_Control_group': '#0000ff',  # Blue
}

colorlist = [colormap[c] for c in targets]
print(colorlist)

scores = pd.DataFrame(plsr.x_scores_)
scores.index = targets



ax = scores.plot(x=0, y=1, kind='scatter', s=50,
                    figsize=(6,6),c=colorlist)


print(scores)

y_pred = KMeans(n_clusters=3,random_state=8).fit_predict(plsr.x_scores_)

print(y_pred)

group0 =[]
outlier_index = []
for i in range(len(y_pred)):
    if y_pred[i] == 2:
        group0.append(plsr.x_scores_[i])
        outlier_index.append(i)

group0 = np.array(group0)

points_ad = []
points_hc = []
for i in range(len(scores)):
    if i not in outlier_index:
        if 'AD' in scores.index[i]:
            points_ad.append([scores[0][i],scores[1][i]])
        else:
            points_hc.append([scores[0][i],scores[1][i]])



points_ad = np.array(points_ad)
ellipse_points_ad = EllipseModel()
ellipse_points_ad.estimate(points_ad)
ad_x_mean,ad_y_mean,ad_a,ad_b,ad_theta = ellipse_points_ad.params

points_hc = np.array(points_hc)
ellipse_points_hc = EllipseModel()
ellipse_points_hc.estimate(points_hc)
hc_x_mean,hc_y_mean,hc_a,hc_b,hc_theta = ellipse_points_hc.params

ellipse_ad = Ellipse((ad_x_mean, ad_y_mean), 2*ad_a, 2*ad_b,ad_theta,
                        edgecolor='r', fc='None', lw=2)
ax.add_patch(ellipse_ad)
ellipse_hc = Ellipse((hc_x_mean, hc_y_mean), 2*hc_a, 2*hc_b,hc_theta,
                        edgecolor='b', fc='None', lw=2)
ax.add_patch(ellipse_hc)

print(targets)

ax.set_xlabel('PLS-DA axis 1')
ax.set_ylabel('PLS-DA axis 2')
ax.legend(handles=[ellipse_ad,ellipse_hc],labels=['AD_group','HC_group'])
plt.title('PLS-DA')
plt.show()