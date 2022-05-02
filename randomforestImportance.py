import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import ttest_ind
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
# df = pd.read_csv(url, header = None)
#
# print(df)
#
#
# from sklearn.ensemble import RandomForestClassifier
# x, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
# feat_labels = df.columns[1:]
# print(x_train.shape,y_train.shape)
# print(feat_labels[0])
# forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
# forest.fit(x_train, y_train)
# importances = forest.feature_importances_
# indices = np.argsort(importances)[::-1]
# for f in range(x_train.shape[1]):
#     print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
#
# quit()


# data = pd.read_excel('files/ad files/peaktablePOSout_POS_noid_replace.xlsx')
data = pd.read_excel('files/ad files/peaktableNEGout_NEG_noid_replace.xlsx')

for column in data.columns.values:
    if '16' in column:
        del data[column]


color_exist = []
targets = data.columns.values[1:]


print(data)
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
