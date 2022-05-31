import math
import random
import matplotlib
import platform

from delete import deleteDupFromOriginalTableByDiff, Topkindex_DeleteNotInPubChem, reasonableNameForBoxplot

if platform.system() == 'Windows':
    matplotlib.rc('font', family='Microsoft YaHei')
else:
    matplotlib.rc('font',family='Arial Unicode MS')
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
        if 'QX' not in targets[i]:
            del data[targets[i]]
    targets = data.columns.values[2:]
    for i in range(len(targets)):
        if 'QXRY' in targets[i]:
            del data[targets[i]]
    targets = data.columns.values[2:]

    keywords = keywords

    print(targets)

    for i in range(len(targets)):
        if keywords[0] not in targets[i] and keywords[1] not in targets[i]:
            del data[targets[i]]

    data = data.dropna().reset_index(drop=True)
    data, diff_list = deleteDupFromOriginalTableByDiff(df=data, keywords=keywords)

    print('dataframe shape after drop rows that have NA value: ({} metabolites, {} samples)'.format(data.shape[0],
                                                                                                    data.shape[1] - 2))

    saved_label = data['dataMatrix'].values
    del data['dataMatrix']
    del data['smile']
    targets = data.columns.values


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

    print(normalized_data_impute_x.shape)
    print(normalized_data_impute_y.shape)



    top_k = 20
    top_k_index = Topkindex_DeleteNotInPubChem(saved_label,top_k)


    if len(top_k_index) == 0:
        print('there are no significant difference between metabolites on these two groups {} by mann whitney u test'.format(keywords))
    else:
        print(top_k_index)

        X_XYCH_WX = np.array(normalized_data_impute_x)
        X_GYCH_WX = np.array(normalized_data_impute_y)



        X_diff_XYCH_WX = []
        for i in top_k_index:
            X_diff_XYCH_WX.append(X_XYCH_WX[:,i:i+1])



        X_diff_GYCH_WX = []
        for i in top_k_index:
            X_diff_GYCH_WX.append(X_GYCH_WX[:,i:i+1])



        data_XYCH_WX = []
        labels= []
        for i in range(len(X_diff_XYCH_WX)):
            data_XYCH_WX.append(X_diff_XYCH_WX[i])
            temp = reasonableNameForBoxplot(saved_label[top_k_index[i]])
            print(temp)
            labels += [temp, '']

        data_GYCH_WX = []
        for i in range(len(X_diff_GYCH_WX)):
            data_GYCH_WX.append(X_diff_GYCH_WX[i])




        # Creating axes instance
        data_XYCH_WXs = []
        for i in data_XYCH_WX:
            data_XYCH_WXs.append(i.reshape(i.shape[0]))
        data_XYCH_WX = data_XYCH_WXs

        data_GYCH_WXs = []
        for i in data_GYCH_WX:
            data_GYCH_WXs.append(i.reshape(i.shape[0]))
        data_GYCH_WX = data_GYCH_WXs

        data_XYCH_WX = np.array(data_XYCH_WX)
        data_GYCH_WX = np.array(data_GYCH_WX)
        print(data_XYCH_WX.shape)
        print(data_GYCH_WX.shape)
        data = np.hstack((data_XYCH_WX,data_GYCH_WX))

        data = []
        for i in range(data_XYCH_WX.shape[0]):
            data.append(data_XYCH_WX[i,:])
            data.append(data_GYCH_WX[i, :])


        bp = plt.boxplot(data,labels=labels,patch_artist=True)
        plt.xticks(rotation = 90)
        for i in range(len(bp['boxes'])):
            if i %2 == 0:
                bp['boxes'][i].set(color='r')
            else:
                bp['boxes'][i].set(color='b')
        plt.legend(handles=[bp['boxes'][0],bp['boxes'][1]],labels=['{}group'.format(keywords[0]),'{}group'.format(keywords[1])])
        plt.title('清洗之后不同蜂花粉破壁与未破壁的差异变化 ({})'.format(mode))
        plt.show()


if __name__ == '__main__':
    mode = 'POS'
    if mode == "BOTH":
        filename = '../files/pollen files/results/peaktableBOTHout_BOTH_noid_replace_mean_full.xlsx'
    elif mode == 'POS':
        filename = '../files/pollen files/results/process_output_quantid_pos_camera_noid/peaktablePOSout_POS_noid_full_sample_replace_mean_full.xlsx'
    elif mode == 'NEG':
        filename = '../files/pollen files/results/process_output_quantid_neg_camera_noid/peaktableNEGout_NEG_noid_replace.xlsx'

    # 分别比较样本1和6、
    keywords1 = ['XYCH_QX_','XYCH_QXPB_']
    # 样本2和7、
    keywords2 = ['GYCH_QX_','GYCH_QXPB_']
    # 样本3和8、
    keywords3 = ['GWBZ_QX_','GWBZ_QXPB_']
    # 样本4和9、
    keywords4 = ['GHH_QX_','GHH_QXPB_']
    # 样本5和10
    keywords5 = ['GCH_QX_','GCH_QXPB_']
    # 研究单个样本破壁与未破壁的变化差异
    keywords6 = ['QX_','QXPB_']
    keywords = keywords5

    boxplot(filename,mode,keywords)

