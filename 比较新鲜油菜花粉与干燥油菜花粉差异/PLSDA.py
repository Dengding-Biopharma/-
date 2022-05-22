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
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from skimage.measure import EllipseModel
def plsda(filename,mode):
    data = pd.read_excel(filename)
    targets = data.columns.values[2:]

    for i in range(len(targets)):
        if 'XYCH_WX_' not in targets[i] and 'GYCH_WX_' not in targets[i]:
            del data[targets[i]]
    targets = data.columns.values[2:]
    print(targets)

    for i in range(len(targets)):
        if 'XYCH_WX' in targets[i]:
            targets[i] = 'XYCH_WX_group'
        elif 'GYCH_WX' in targets[i]:
            targets[i] = 'GYCH_WX_group'

    print(targets)

    data = data.dropna().reset_index(drop=True)
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


    print(normalized_data_impute_XYCH_WX.shape)
    print(normalized_data_impute_GYCH_WX.shape)

    X_XYCH_WX = np.array(normalized_data_impute_XYCH_WX)
    X_GYCH_WX = np.array(normalized_data_impute_GYCH_WX)
    X = np.vstack((X_XYCH_WX,X_GYCH_WX))
    print(X)
    int_targets = []
    for i in targets:
        if 'XYCH_WX' in i:
            int_targets.append(0)
        else:
            int_targets.append(1)


    print(X.shape)

    plsr = PLSRegression(n_components=2,scale=False)
    plsr.fit(X,int_targets)

    print(plsr.predict(X))
    predicts = []
    for predict in plsr.predict(X):
        if predict >=0.5:
            predicts.append(1)
        else:
            predicts.append(0)






    scores = pd.DataFrame(plsr.x_scores_)
    scores['index'] = targets

    print(scores)


    ax = scores.plot(x=0, y=1, kind='scatter', s=50,
                        figsize=(6,6),c='r')

    groups=['XYCH_WX_group','GYCH_WX_group']

    for i in range(len(groups)):
        print(groups[i])
        indicesToKeep = scores['index'].values == groups[i]
        if groups[i] == 'XYCH_WX_group':
            ax_XYCH_WX = ax.scatter(scores.loc[indicesToKeep ,0],
                   scores.loc[indicesToKeep, 1],
                   c = 'r'
                   , s = 50)
        if groups[i] == 'GYCH_WX_group':
            ax_GYCH_WX = ax.scatter(scores.loc[indicesToKeep, 0],
                                    scores.loc[indicesToKeep, 1],
                                    c='b'
                                    , s=50)




    plt.legend(handles=[ax_XYCH_WX,ax_GYCH_WX],labels=['XYCH_WX_group','GYCH_WX_group'],loc='best',labelspacing=2,prop={'size': 10})
    plt.title('PLS-DA for 新鲜油菜花粉和干燥油菜花粉 ({} mode)'.format(mode))

    plt.show()

if __name__ == '__main__':
    mode = 'POS'
    if mode == "BOTH":
        filename = '../files/pollen files/results/peaktableBOTHout_BOTH_noid_replace_mean_full.xlsx'
    elif mode == 'POS':
        filename = '../files/pollen files/results/process_output_quantid_pos_camera_noid/peaktablePOSout_POS_noid_replace.xlsx'
    elif mode == 'NEG':
        filename = '../files/pollen files/results/process_output_quantid_neg_camera_noid/peaktableNEGout_NEG_noid_replace.xlsx'


    plsda(filename,mode)