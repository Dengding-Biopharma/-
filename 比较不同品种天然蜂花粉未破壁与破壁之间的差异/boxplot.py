import math
import random
import matplotlib
matplotlib.rc('font',family='Arial Unicode MS')
# matplotlib.rc('font',family='Microsoft YaHei')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.preprocessing
from matplotlib.patches import Ellipse
from scipy.stats import ttest_ind, mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from skimage.measure import EllipseModel
def boxplot(filename,mode,keywords):
    data = pd.read_excel(filename)
    targets = data.columns.values[2:]
    for i in range(len(targets)):
        if 'WX' not in targets[i]:
            del data[targets[i]]
    targets = data.columns.values[2:]

    keywords = keywords


    print(targets)

    for i in range(len(targets)):
        if keywords[0] not in targets[i] and keywords[1] not in targets[i]:
            del data[targets[i]]


    data = data.dropna().reset_index(drop=True)
    print('dataframe shape after drop rows that have NA value: ({} metabolites, {} samples)'.format(data.shape[0],
                                                                                                    data.shape[1] - 2))

    saved_label = data['dataMatrix'].values
    del data['dataMatrix']
    del data['smile']
    targets = data.columns.values
    print(targets)

    data_impute = data.values

    for i in range(data_impute.shape[1]):
        data_impute[:, i] = (data_impute[:, i] / np.sum(data_impute[:, i])) * 100

    normalized_data_impute = data_impute

    x_index=[]
    y_index=[]


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
        t,p = mannwhitneyu(normalized_data_impute_x[:,i:i+1],normalized_data_impute_y[:,i:i+1],alternative='two-sided')
        p_list.append(p[0])
    p_list = np.array(p_list)
    count = 0
    for p in p_list:
        if p < 0.05:
            count +=1

    top_k_index = p_list.argsort()[::-1][len(p_list)-count:]

    if len(top_k_index) == 0:
        print('there are no significant difference between metabolites on these two groups {} by mann whitney u test'.format(keywords))
    else:
        print(top_k_index)

        X = np.array(normalized_data_impute_x)
        Y = np.array(normalized_data_impute_y)



        X_diff = []
        for i in top_k_index:
            X_diff.append(X[:,i:i+1])



        Y_diff = []
        for i in top_k_index:
            Y_diff.append(Y[:,i:i+1])



        data_X = []
        labels = []
        for i in range(len(X_diff)):
            data_X.append(X_diff[i])
            labels += [saved_label[top_k_index[i]], '']


        data_Y = []
        for i in range(len(Y_diff)):
            data_Y.append(Y_diff[i])





        # Creating axes instance
        data_Xs = []
        for i in data_X:
            data_Xs.append(i.reshape(i.shape[0]))
        data_X = data_Xs

        data_Ys = []
        for i in data_Y:
            data_Ys.append(i.reshape(i.shape[0]))
        data_Y = data_Ys

        data_X = np.array(data_X)
        data_Y = np.array(data_Y)
        print(data_X.shape)
        print(data_Y.shape)
        data = np.hstack((data_X,data_Y))

        data = []
        for i in range(data_X.shape[0]):
            data.append(data_X[i,:])
            data.append(data_Y[i, :])



        bp = plt.boxplot(data,labels=labels,patch_artist=True)
        plt.xticks(rotation = 90)
        for i in range(len(bp['boxes'])):
            if i %2 == 0:
                bp['boxes'][i].set(color='r')
            else:
                bp['boxes'][i].set(color='b')
        plt.legend(handles=[bp['boxes'][0],bp['boxes'][1]],labels=['{}group'.format(keywords[0]),'{}group'.format(keywords[1])])
        plt.title('不同品种天然蜂花粉未破壁与破壁之间的差异（未洗）({} mode)'.format(mode))
        plt.show()
        # plt.savefig('figures/pos_plots/干燥油菜花粉.png')

if __name__ == '__main__':
    mode = 'BOTH'
    if mode == "BOTH":
        filename = '../files/pollen files/results/peaktableBOTHout_BOTH_noid_replace_mean_full.xlsx'
    elif mode == 'POS':
        filename = '../files/pollen files/results/process_output_quantid_pos_camera_noid/peaktablePOSout_POS_noid_replace.xlsx'
    elif mode == 'NEG':
        filename = '../files/pollen files/results/process_output_quantid_neg_camera_noid/peaktableNEGout_NEG_noid_replace.xlsx'

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
    keywords = keywords6

    boxplot(filename,mode,keywords)


