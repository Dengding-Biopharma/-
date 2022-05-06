import matplotlib
matplotlib.rc('font',family='Microsoft YaHei')
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.impute import SimpleImputer

data = pd.read_excel('../files/pollen files/results/process_output_quantid_pos_camera_noid/peaktablePOSout_POS_noid_replace.xlsx')

print(data)

sample_labels = []


color_exist = []
targets = data.columns.values[1:]


for i in range(len(targets)):
    if 'WX' not in targets[i]:
        del data[targets[i]]
targets = data.columns.values[1:]
print(targets)


saved_label = data['dataMatrix'].values
print(saved_label)
del data['dataMatrix']

imputer_mean_XYCH_WX = SimpleImputer(missing_values=np.nan,strategy='mean')
data_impute = imputer_mean_XYCH_WX.fit_transform(data)


sum_baseline = 10000
for i in range(data_impute.shape[1]):
    coe = sum_baseline/np.sum(data_impute[:,i])
    data_impute[:, i] = (data_impute[:, i]*coe)/sum_baseline

normalized_data_impute = data_impute
print(normalized_data_impute)

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
x_index=[]
y_index=[]
print(targets)
keywords = keywords6
for i in range(len(targets)):
    if keywords[0] in targets[i]:
        x_index.append(i)
    elif keywords[1] in targets[i]:
        y_index.append(i)

print(x_index)
print(y_index)
targets = np.hstack((targets[x_index],targets[y_index]))
print(targets)

normalized_data_impute_x = []
for index in x_index:
    normalized_data_impute_x.append(normalized_data_impute[:,index].T)
normalized_data_impute_x = np.array(normalized_data_impute_x)

normalized_data_impute_y =[]
for index in y_index:
    normalized_data_impute_y.append(normalized_data_impute[:,index].T)
normalized_data_impute_y = np.array(normalized_data_impute_y)


top_k = 20
p_list =[]
for i in range(normalized_data_impute_x.shape[1]):
    t,p = ttest_ind(normalized_data_impute_x[:,i:i+1],normalized_data_impute_y[:,i:i+1],equal_var=True)
    p_list.append(p[0])
p_list = np.array(p_list)
count = 0
for p in p_list:
    if p < 0.05:
        count +=1

top_k_index = p_list.argsort()[::-1][len(p_list)-count:]
print(top_k_index)
