import math
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import ttest_ind
from sklearn.impute import SimpleImputer

data = pd.read_excel('files/ad files/peaktablePOSout_POS_noid_more_puring_mean_full.xlsx')
# data = pd.read_excel('files/ad files/peaktableNEGout_NEG_noid_replace.xlsx')

color_exist = []
targets = data.columns.values[1:]

print(targets)

saved_label = data['dataMatrix'].values
print(saved_label)
del data['dataMatrix']
data_impute = data.values
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
data_impute_ad = normalized_data_impute_ad
data_impute_hc = normalized_data_impute_hc

top_k = 20
p_list =[]
for i in range(data_impute_ad.shape[1]):
    t,p = ttest_ind(normalized_data_impute_ad[:,i:i+1],normalized_data_impute_hc[:,i:i+1],equal_var=True)
    p_list.append(p[0])
p_list = np.array(p_list)

top_k_index = p_list.argsort()[::-1][len(p_list)-top_k:]
print(top_k_index)

X_ad = np.array(data_impute_ad)
X_hc = np.array(data_impute_hc)
X = np.vstack((X_ad,X_hc))



X_top = []
for row in range(X.shape[0]):
    sum = np.sum(X[row,:])
    temp = []
    for k in top_k_index:
        percentage = (X[row, k]/sum) * 100
        temp.append(percentage)
    X_top.append(temp)



X_top = np.array(X_top)

X_top = X_top.reshape(top_k,X_top.shape[0])

labels = []
for i in top_k_index:
    labels.append(saved_label[i])

print(X_top[0].shape)



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


plt.title('Histogram of the 20 most different metabolic distribution between groups')
ax.legend(bbox_to_anchor=(1, 1),prop={'size': 8},loc='best')
plt.show()