import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import ttest_ind
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_excel('files/ad files/peaktablePOSout_POS_noid_more_puring_mean_full.xlsx')
# data = pd.read_excel('files/ad files/peaktableNEGout_NEG_noid_replace.xlsx')

targets = data.columns.values[1:]


print(data)
print(targets)


saved_label = data['dataMatrix'].values
print(saved_label)
del data['dataMatrix']

# imputer_mean_hc = SimpleImputer(missing_values=np.nan,strategy='mean')
# data_impute_hc = imputer_mean_ad.fit_transform(df_hc)
# print(data_impute)
# sum_baseline = 13800
# for i in range(data_impute.shape[1]):
#     coe = sum_baseline/np.sum(data_impute[:,i])
#     data_impute[:, i] = (data_impute[:, i]*coe)/sum_baseline
data_impute = data.values
# scaler = StandardScaler()
# data_impute = scaler.fit_transform(data_impute)
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
