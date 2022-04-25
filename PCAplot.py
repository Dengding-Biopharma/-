import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.preprocessing
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from skimage.measure import EllipseModel

data = pd.read_excel('files/peaktablePOSout_POS_noid_replace_variable.xlsx')

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

imputer_mean_ad = SimpleImputer(missing_values=np.nan,strategy='mean')
data_impute = imputer_mean_ad.fit_transform(data)

print(data_impute)
sum_baseline = 30000
for i in range(data_impute.shape[1]):
    coe = sum_baseline/np.sum(data_impute[:,i])
    data_impute[:, i] = (data_impute[:, i]*coe)/sum_baseline

normalized_data_impute = data_impute
print(normalized_data_impute)

normalized_data_impute_ad = normalized_data_impute[:,:23].T
normalized_data_impute_hc = normalized_data_impute[:,23:].T


print(normalized_data_impute_ad.shape)
print(normalized_data_impute_hc.shape)




# PCA
pca = PCA(n_components=2)
pca.fit(normalized_data_impute.T)
X_new = pca.fit_transform(normalized_data_impute.T)
print(X_new)
print(pca.explained_variance_ratio_)

y_pred = KMeans(n_clusters=3,random_state=8).fit_predict(X_new)

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

targets = pd.DataFrame(data = targets)

principalDf = pd.DataFrame(data = X_new
             , columns = ['PC1', 'PC2'])
finalDf = pd.concat([principalDf, targets], axis = 1)

points_ad = []
points_hc = []
print(finalDf)

for i in range(X_new.shape[0]):
    if i not in outlier_index:
        if 'AD' in finalDf[0][i]:
            points_ad.append([finalDf['PC1'][i],finalDf['PC2'][i]])
        else:
            points_hc.append([finalDf['PC1'][i], finalDf['PC2'][i]])




points_ad = np.array(points_ad)
ellipse_points_ad = EllipseModel()
ellipse_points_ad.estimate(points_ad)
ad_x_mean,ad_y_mean,ad_a,ad_b,ad_theta = ellipse_points_ad.params

points_hc = np.array(points_hc)
ellipse_points_hc = EllipseModel()
ellipse_points_hc.estimate(points_hc)
hc_x_mean,hc_y_mean,hc_a,hc_b,hc_theta = ellipse_points_hc.params





targets = list(targets[0])

colors = color_exist
fig = plt.figure(figsize = (20,20))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1 {}%'.format(round(pca.explained_variance_ratio_[0]*100,2)), fontsize = 15)
ax.set_ylabel('Principal Component 2 {}%'.format(round(pca.explained_variance_ratio_[1]*100,2)), fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)


ellipse_ad = Ellipse((ad_x_mean, ad_y_mean), 2*ad_a, 2*ad_b,ad_theta,
                        edgecolor='r', fc='None', lw=2)
ax.add_patch(ellipse_ad)
ellipse_hc = Ellipse((hc_x_mean, hc_y_mean), 2*hc_a, 2*hc_b,hc_theta,
                        edgecolor='b', fc='None', lw=2)
ax.add_patch(ellipse_hc)



for target, color in zip(targets,colors):
    print(target)
    indicesToKeep = finalDf[0].values == target
    ax.scatter(finalDf.loc[indicesToKeep, 'PC1']
               , finalDf.loc[indicesToKeep, 'PC2']
               , c = color
               , s = 50)


ax.legend(['AD_Disease_group','HC_Control_group'],loc='lower right',borderpad=2,labelspacing=2,prop={'size': 12})
ax.grid()


plt.show()




