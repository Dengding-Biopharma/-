import math
import random
import matplotlib
import platform
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
from delete import deleteDupFromOriginalTableByDiff, Topkindex_DeleteNotInPubChem, reasonableNameForBoxplot


def boxplot(filename,mode):
    data = pd.read_excel(filename)
    targets = data.columns.values[2:]

    keywords = ['XYCH_WX_', 'GYCH_WX_']
    for i in range(len(targets)):
        if keywords[0] not in targets[i] and keywords[1] not in targets[i]:
            del data[targets[i]]
    targets = data.columns.values[2:]
    print(targets)


    data = data.dropna().reset_index(drop=True)
    data, diff_list = deleteDupFromOriginalTableByDiff(df=data, keywords=keywords)

    print('dataframe shape after drop rows that have NA value: ({} metabolites, {} samples)'.format(data.shape[0],
                                                                                                    data.shape[1] - 2))


    saved_label = data['dataMatrix'].values
    print(saved_label)
    del data['dataMatrix']
    del data['smile']
    print(data)

    data_impute = data.values
    for i in range(data_impute.shape[1]):
        data_impute[:, i] = (data_impute[:, i] / np.sum(data_impute[:, i])) * 100

    normalized_data_impute = data_impute
    print(normalized_data_impute)

    XYCH_WX_index=[]
    GYCH_WX_index=[]
    for i in range(len(targets)):
        if "XYCH_WX_" in targets[i]:
            XYCH_WX_index.append(i)
        else:
            GYCH_WX_index.append(i)
    print(XYCH_WX_index)
    print(GYCH_WX_index)


    normalized_data_impute_XYCH_WX = []
    for index in XYCH_WX_index:
        normalized_data_impute_XYCH_WX.append(normalized_data_impute[:,index].T)
    normalized_data_impute_XYCH_WX = np.array(normalized_data_impute_XYCH_WX)

    normalized_data_impute_GYCH_WX =[]
    for index in GYCH_WX_index:
        normalized_data_impute_GYCH_WX.append(normalized_data_impute[:,index].T)
    normalized_data_impute_GYCH_WX = np.array(normalized_data_impute_GYCH_WX)


    top_k = 20
    top_k_index = Topkindex_DeleteNotInPubChem(saved_label,top_k)

    print(top_k_index)


    X_XYCH_WX = np.array(normalized_data_impute_XYCH_WX)
    X_GYCH_WX = np.array(normalized_data_impute_GYCH_WX)



    X_diff_XYCH_WX = []
    for i in top_k_index:
        X_diff_XYCH_WX.append(X_XYCH_WX[:,i:i+1])



    X_diff_GYCH_WX = []
    for i in top_k_index:
        X_diff_GYCH_WX.append(X_GYCH_WX[:,i:i+1])



    data_XYCH_WX = []
    labels = []
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




    print(data)
    bp = plt.boxplot(data,labels=labels,patch_artist=True)
    plt.xticks(rotation = 90)
    for i in range(len(bp['boxes'])):
        if i %2 == 0:
            bp['boxes'][i].set(color='r')
        else:
            bp['boxes'][i].set(color='b')
    plt.legend(handles=[bp['boxes'][0],bp['boxes'][1]],labels=['{}group'.format('XYCH_'),'{}group'.format('GYCH_')])
    plt.title('新鲜油菜花粉与干燥油菜花粉的差异（未洗）({} mode)'.format(mode))
    plt.show()




if __name__ == '__main__':
    mode = 'POS'
    if mode == "BOTH":
        filename = '../files/pollen files/results/peaktableBOTHout_BOTH_noid_replace_mean_full.xlsx'
    elif mode == 'POS':
        filename = '../files/pollen files/results/process_output_quantid_pos_camera_noid/peaktablePOSout_POS_noid_full_sample_replace_mean_full.xlsx'
    elif mode == 'NEG':
        filename = '../files/pollen files/results/process_output_quantid_neg_camera_noid/peaktableNEGout_NEG_noid_replace.xlsx'


    boxplot(filename,mode)
