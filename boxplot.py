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
data_impute_ad = normalized_data_impute.T[:45]
data_impute_hc = normalized_data_impute.T[45:]

top_k = 20
p_list =[]
for i in range(data_impute_ad.shape[1]):
    t,p = ttest_ind(data_impute_ad[:,i:i+1],data_impute_hc[:,i:i+1],equal_var=True)
    p_list.append(p[0])
p_list = np.array(p_list)

top_k_index = p_list.argsort()[::-1][len(p_list)-top_k:]
print(top_k_index)


X_ad = np.array(data_impute_ad)
X_hc = np.array(data_impute_hc)



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

fig = plt.figure(figsize=(10, 7))
ax1 = fig.add_subplot(111)

# Creating axes instance
data_ads = []
for i in data_ad:
    data_ads.append(i.reshape(45))
data_ad = data_ads

data_hcs = []
for i in data_hc:
    data_hcs.append(i.reshape(23))
data_hc = data_hcs

data_ad = np.array(data_ad)
data_hc = np.array(data_hc)
data = np.hstack((data_ad,data_hc))
print(data.shape)
print(targets)


bp = ax1.boxplot(data,  vert=1)
plt.xticks(rotation = 90)
ax1.set_xticklabels(targets)
for cap in bp['caps']:
    cap.set(color ='#8B008B',
            linewidth = 2)
ax1.get_xaxis().tick_bottom()
ax1.get_yaxis().tick_left()
for flier in bp['fliers']:
    flier.set(marker ='D',
              color ='#e7298a',
              alpha = 0.5)
for median in bp['medians']:
    median.set(color ='red',
               linewidth = 3)
for whisker in bp['whiskers']:
    whisker.set(color ='#8B008B',
                linewidth = 1.5,
                linestyle =":")

# ax2 = fig.add_subplot(211)
# # data_hcs = []
# # for i in data_hc:
# #     data_hcs.append(i.reshape(22))
# # data_hc = data_hcs
#
# bp = ax2.boxplot(data_hc,  vert=0)
# ax2.set_yticklabels(labels_hc)
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
plt.title('boxplot for top 20 variables which have significant differences between groups')

plt.show()




