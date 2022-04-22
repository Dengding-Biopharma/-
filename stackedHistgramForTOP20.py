import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import ttest_ind
from sklearn.impute import SimpleImputer

data = pd.read_excel('files/peaktablePOSout_POS_noid_replace_variable.xlsx')
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
sum_baseline = 30000
for i in range(data_impute.shape[1]):
    coe = sum_baseline/np.sum(data_impute[:,i])
    data_impute[:, i] = (data_impute[:, i]*coe)/sum_baseline

normalized_data_impute = data_impute
print(normalized_data_impute.shape)
data_impute_ad = normalized_data_impute.T[:23]
data_impute_hc = normalized_data_impute.T[23:]
# print(data_impute_ad[:,1778])
# quit()
top_k = 20
sum_list =[]
for i in range(data_impute_ad.shape[1]):
    sum = np.sum(data_impute_ad[:,i:i+1])

    sum_list.append(sum)
sum_list = np.array(sum_list)
top_k_index = sum_list.argsort()[::-1][0:top_k]
print(top_k_index)


X_ad = np.array(data_impute_ad)
X_hc = np.array(data_impute_hc)
X = np.vstack((X_ad,X_hc))

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
X_top = X_top.reshape(20,68)

labels = []
for i in top_k_index:
    labels.append(saved_label[i])




#
# targets = list(targets)
# targets.append('others')
# targets = np.array(targets)

fig,ax = plt.subplots()

plt.xticks(rotation = 90)
ax.bar(targets,X_top[0],0.2,label=labels[0])
for i in range(1,len(X_top)):
    ax.bar(targets,X_top[i],0.2,bottom=X_top[i-1],label=labels[i])

plt.title('Histogram of the top 20 metabolite percentage')
ax.legend(bbox_to_anchor=(1, 1))
plt.show()