import math
import random
import matplotlib
# matplotlib.rc('font',family='Microsoft YaHei')
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

def plsda(filename,mode,keywords):
    data = pd.read_excel(filename)

    targets = data.columns.values[2:]

    for i in range(len(targets)):
        if 'WX' not in targets[i]:
            del data[targets[i]]
    targets = data.columns.values[2:]
    keywords = keywords

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
    print(normalized_data_impute)

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
    for i in range(len(targets)):
        if 'WX_' in targets[i]:
            targets[i] = 'WX_group'
        elif 'WXPB_' in targets[i]:
            targets[i] = 'WXPB_group'

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

    X_XYCH_WX = np.array(normalized_data_impute_x)
    X_GYCH_WX = np.array(normalized_data_impute_y)
    X = np.vstack((X_XYCH_WX,X_GYCH_WX))
    print(X)
    int_targets = []
    for i in targets:
        if 'WX_' in i:
            int_targets.append(0)
        elif 'WXPB_':
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

    groups=['WX_group','WXPB_group']

    for i in range(len(groups)):
        print(groups[i])
        indicesToKeep = scores['index'].values == groups[i]

        if groups[i] == 'WX_group':
            ax_XYCH_WX = ax.scatter(scores.loc[indicesToKeep ,0],
                   scores.loc[indicesToKeep, 1],
                   c = 'r'
                   , s = 50)
        if groups[i] == 'WXPB_group':
            ax_GYCH_WX = ax.scatter(scores.loc[indicesToKeep, 0],
                                    scores.loc[indicesToKeep, 1],
                                    c='b'
                                    , s=50)




    plt.legend(handles=[ax_XYCH_WX,ax_GYCH_WX],labels=['WX_group','WXPB_group'],loc='best',labelspacing=2,prop={'size': 10})
    plt.title('PLS-DA for 未洗和未洗破壁 ({} mode)'.format(mode))
    plt.show()
    # plt.savefig('figures/neg_plots/整体未破壁样本与破壁样本的变化PLS-DA.png')

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

    plsda(filename,mode,keywords)