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
    print(targets)
    for i in range(len(targets)):
        if 'WX_' not in targets[i] and 'QX_' not in targets[i] and 'QXRY_' not in targets[i]:
            del data[targets[i]]
    targets = data.columns.values[2:]
    print(targets)

    keywords = keywords

    for i in range(len(targets)):
        if keywords[0] not in targets[i] and keywords[1] not in targets[i] and keywords[2] not in targets[i]:
            del data[targets[i]]

    data = data.dropna().reset_index(drop=True)
    print('dataframe shape after drop rows that have NA value: ({} metabolites, {} samples)'.format(data.shape[0],
                                                                                                    data.shape[1] - 2))
    if data.shape[0] == 0:
        print('no metabolites exist!!! Stop!')
    else:
        saved_label = data['dataMatrix'].values
        del data['dataMatrix']
        del data['smile']
        targets = data.columns.values
        print(targets)

        data_impute = data.values

        for i in range(data_impute.shape[1]):
            data_impute[:, i] = (data_impute[:, i] / np.sum(data_impute[:, i])) * 100

        normalized_data_impute = data_impute

        x_index = []
        y_index = []
        z_index = []

        for i in range(len(targets)):
            if keywords[0] in targets[i]:
                x_index.append(i)
            elif keywords[1] in targets[i]:
                y_index.append(i)
            elif keywords[2] in targets[i]:
                z_index.append(i)

        print(x_index)
        print(y_index)
        print(z_index)
        targets = np.hstack((targets[x_index], targets[y_index], targets[z_index]))
        print(targets)
        print(len(targets))

        normalized_data_impute_x = []
        for index in x_index:
            normalized_data_impute_x.append(normalized_data_impute[:, index].T)
        normalized_data_impute_x = np.array(normalized_data_impute_x)

        normalized_data_impute_y = []
        for index in y_index:
            normalized_data_impute_y.append(normalized_data_impute[:, index].T)
        normalized_data_impute_y = np.array(normalized_data_impute_y)

        normalized_data_impute_z = []
        for index in z_index:
            normalized_data_impute_z.append(normalized_data_impute[:, index].T)
        normalized_data_impute_z = np.array(normalized_data_impute_z)

        print(normalized_data_impute_x.shape)
        print(normalized_data_impute_y.shape)


        X = np.vstack((normalized_data_impute_x,normalized_data_impute_y,normalized_data_impute_z))
        print(X.shape)


        int_targets = []
        for i in targets:
            if 'WX_' in i:
                int_targets.append(0)
            elif 'QX_' in i:
                int_targets.append(1)
            elif 'QXRY_' in i:
                int_targets.append(2)

        print(X.shape)

        plsr = PLSRegression(n_components=3,scale=False)
        plsr.fit(X,int_targets)

        print(plsr.predict(X))

        predicts = []

        for predict in plsr.predict(X):
            if predict >=0.5:
                predicts.append(1)
            else:
                predicts.append(0)



        scores = pd.DataFrame(plsr.x_scores_)

        for i in range(len(targets)):
            if 'WX_' in targets[i]:
                targets[i] = 'WX_group'
            elif 'QX_' in targets[i]:
                targets[i] = 'QX_group'
            elif 'QXRY_' in targets[i]:
                targets[i] = 'QXRY_group'

        scores['index'] = targets

        print(scores)


        fig = plt.figure()
        ax = fig.add_subplot(111)

        groups=['WX_group','QX_group','QXRY_group']

        for i in range(len(groups)):
            print(scores['index'].values)
            indicesToKeep = scores['index'].values == groups[i]
            print(indicesToKeep)
            if groups[i] == 'WX_group':
                ax_x = ax.scatter(scores.loc[indicesToKeep ,0],
                       scores.loc[indicesToKeep, 1],
                       c = 'r'
                       , s = 50)
            if groups[i] == 'QX_group':
                ax_y = ax.scatter(scores.loc[indicesToKeep, 0],
                                        scores.loc[indicesToKeep, 1],
                                        c='b'
                                        , s=50)
            if groups[i] == 'QXRY_group':
                ax_z = ax.scatter(scores.loc[indicesToKeep, 0],
                                        scores.loc[indicesToKeep, 1],
                                        c='g'
                                        , s=50)




        plt.legend(handles=[ax_x,ax_y,ax_z],labels=['{}group'.format(keywords[0]),'{}group'.format(keywords[1]),'{}group'.format(keywords[2])],loc='best',labelspacing=2,prop={'size': 10})
        plt.title('PLS-DA for 比较不同花粉未清洗、清洗与清洗溶液代谢物差异变化 ({} mode)'.format(mode))
        plt.show()

if __name__ == '__main__':
    mode = 'BOTH'
    if mode == "BOTH":
        filename = '../files/pollen files/results/peaktableBOTHout_BOTH_noid_replace_mean_full.xlsx'
    elif mode == 'POS':
        filename = '../files/pollen files/results/process_output_quantid_pos_camera_noid/peaktablePOSout_POS_noid_replace.xlsx'
    elif mode == 'NEG':
        filename = '../files/pollen files/results/process_output_quantid_neg_camera_noid/peaktableNEGout_NEG_noid_replace.xlsx'

    # 分别比较样本1和6、
    keywords1 = ['XYCH_WX_', 'XYCH_QX_', 'XYCH_QXRY_']
    # 样本2和7、
    keywords2 = ['GYCH_WX_', 'GYCH_QX_', 'GYCH_QXRY_']
    # 样本3和8、
    keywords3 = ['GWBZ_WX_', 'GWBZ_QX_', 'GWBZ_QXRY_']
    # 样本4和9、
    keywords4 = ['GHH_WX_', 'GHH_QX_', 'GHH_QXRY_']
    # 样本5和10
    keywords5 = ['GCH_WX_', 'GCH_QX_', 'GCH_QXRY_']
    # 研究单个样本破壁与未破壁的变化差异
    keywords6 = ['WX_', 'QX_', 'QXRY']
    keywords = keywords1

    plsda(filename,mode,keywords)