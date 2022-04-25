import math
from math import nan

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

for i in range(len(data)):
    temp = []
    for j in targets:
        temp.append(data[j][i])
    for k in range(len(temp)):
        temp[k] = math.isnan(temp[k])
    if temp.count(True) >= len(temp) /2:
        data = data.drop(i)





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
print(normalized_data_impute.shape)
ad_index=[]
hc_index=[]
for i in range(len(targets)):
    if "AD" in targets[i]:
        ad_index.append(i)
    else:
        hc_index.append(i)


normalized_data_impute_ad = []
for index in ad_index:
    normalized_data_impute_ad.append(normalized_data_impute[:,index].T)
normalized_data_impute_ad = np.array(normalized_data_impute_ad)

normalized_data_impute_hc =[]
for index in hc_index:
    normalized_data_impute_hc.append(normalized_data_impute[:,index].T)
normalized_data_impute_hc = np.array(normalized_data_impute_hc)


top_k = 20
p_list =[]
for i in range(normalized_data_impute_ad.shape[1]):
    t,p = ttest_ind(normalized_data_impute_ad[:,i:i+1],normalized_data_impute_hc[:,i:i+1],equal_var=True)
    p_list.append(p[0])
p_list = np.array(p_list)
count = 0
for p in p_list:
    if p < 0.05:
        count +=1
print(count)

top_k_index = p_list.argsort()[::-1][len(p_list)-count:]
print(top_k_index)


X_ad = np.array(normalized_data_impute_ad)
X_hc = np.array(normalized_data_impute_hc)



X_diff_ad = []
for i in top_k_index:
    X_diff_ad.append(X_ad[:,i:i+1])



X_diff_hc = []
for i in top_k_index:
    X_diff_hc.append(X_hc[:,i:i+1])



data_ad = []
labels_ad = []
for i in range(len(X_diff_ad)):
    data_ad.append(X_diff_ad[i])
    labels_ad.append(saved_label[top_k_index[i]])

data_hc = []
labels_hc = []
for i in range(len(X_diff_hc)):
    data_hc.append(X_diff_hc[i])
    labels_hc.append(saved_label[top_k_index[i]])




# Creating axes instance
data_ads = []
for i in data_ad:
    data_ads.append(i.reshape(i.shape[0]))
data_ad = data_ads

data_hcs = []
for i in data_hc:
    data_hcs.append(i.reshape(i.shape[0]))
data_hc = data_hcs

data_ad = np.array(data_ad)
data_hc = np.array(data_hc)
print(data_ad.shape)
print(data_hc.shape)
data = np.hstack((data_ad,data_hc))


for i in range(data_ad.shape[0]):
    data = [data_ad[i,:],data_hc[i,:]]
    plt.boxplot(data,labels=['AD','HC'])
    plt.title(labels_ad[i])
    plt.savefig('figures/boxplots/boxplot-{}.png'.format(i))
    plt.close()



#
# ax1 = fig.add_subplot(211)
# ax1.boxplot(data_ad.T,  vert=0)
# ax1.set_yticklabels(labels_ad)
# plt.title('boxplot for top 20 variables which have significant differences between groups\nAD group')
#
# ax2 = fig.add_subplot(212)
# ax2.boxplot(data_hc.T,  vert=0)
# ax2.set_yticklabels(labels_hc)
# plt.title('HC group')
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




