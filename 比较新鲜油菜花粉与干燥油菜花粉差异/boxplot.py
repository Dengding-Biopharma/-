import math
from math import nan

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import ttest_ind
from sklearn.impute import SimpleImputer

data = pd.read_excel('../files/pollen files/results/process_output_quantid_pos_camera_noid/peaktablePOSout_POS_noid_replace.xlsx')


color_exist = []
targets = data.columns.values[1:]


print(data)
print(targets)

for i in range(len(data)):
    temp = []
    for j in targets:
        temp.append(data[j][i])
    for k in range(len(temp)):
        temp[k] = math.isnan(temp[k])
    if temp.count(True) >= len(temp) /2:
        data = data.drop(i)


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
sum_baseline = 10000
for i in range(data_impute.shape[1]):
    coe = sum_baseline/np.sum(data_impute[:,i])
    data_impute[:, i] = (data_impute[:, i]*coe)/sum_baseline

normalized_data_impute = data_impute
print(normalized_data_impute.shape)

xych_index=[]
gych_index=[]
for i in range(len(targets)):
    if "XYCH" in targets[i]:
        xych_index.append(i)
    elif 'GYCH' in targets[i]:
        gych_index.append(i)


normalized_data_impute_xych = []
for index in xych_index:
    normalized_data_impute_xych.append(normalized_data_impute[:,index].T)
normalized_data_impute_xych = np.array(normalized_data_impute_xych)

normalized_data_impute_gych =[]
for index in gych_index:
    normalized_data_impute_gych.append(normalized_data_impute[:,index].T)
normalized_data_impute_gych = np.array(normalized_data_impute_gych)


top_k = 20
p_list =[]
for i in range(normalized_data_impute_xych.shape[1]):
    t,p = ttest_ind(normalized_data_impute_xych[:,i:i+1],normalized_data_impute_xych[:,i:i+1],equal_var=True)
    p_list.append(p[0])
p_list = np.array(p_list)

count = 0
for p in p_list:
    if p < 0.05:
        count +=1
print(count)

top_k_index = p_list.argsort()[::-1][len(p_list)-top_k:]
print(top_k_index)


X_xych = np.array(normalized_data_impute_xych)
X_gych = np.array(normalized_data_impute_gych)



X_diff_xych = []
for i in top_k_index:
    X_diff_xych.append(X_xych[:,i:i+1])



X_diff_gych = []
for i in top_k_index:
    X_diff_gych.append(X_gych[:,i:i+1])



data_xych = []
labels_xych = []
for i in range(len(X_diff_xych)):
    data_xych.append(X_diff_xych[i])
    labels_xych.append(saved_label[top_k_index[i]])

data_gych = []
labels_gych = []
for i in range(len(X_diff_gych)):
    data_gych.append(X_diff_gych[i])
    labels_gych.append(saved_label[top_k_index[i]])




# Creating axes instance
data_xychs = []
for i in data_xych:
    data_xychs.append(i.reshape(i.shape[0]))
data_xych = data_xychs

data_gychs = []
for i in data_gych:
    data_gychs.append(i.reshape(i.shape[0]))
data_gych = data_gychs

data_xych = np.array(data_xych)
data_gych = np.array(data_gych)

data = np.hstack((data_xych,data_gych))


for i in range(data_xych.shape[0]):

    data = [data_xych[i,:],data_gych[i,:]]
    plt.boxplot(data,labels=['XYCH','GYCH'])
    plt.title(labels_xych[i])
    plt.savefig('figures/pos_boxplots/boxplot-{}.png'.format(i+1))
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




