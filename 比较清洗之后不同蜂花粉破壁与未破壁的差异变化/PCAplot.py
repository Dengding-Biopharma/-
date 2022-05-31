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
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from skimage.measure import EllipseModel
def pca(filename,mode,keywords):
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
    print(targets)
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
    normalized_data_impute = np.vstack((normalized_data_impute_x,normalized_data_impute_y))
    print(normalized_data_impute.shape)



    # PCA
    pca = PCA(n_components=2)
    pca.fit(normalized_data_impute)
    X_new = pca.fit_transform(normalized_data_impute)
    print(X_new)
    print(pca.explained_variance_ratio_)

    y_pred = KMeans(n_clusters=2,random_state=8).fit_predict(X_new)

    print(y_pred)

    group0 =[]
    outlier_index = []
    for i in range(len(y_pred)):
        if y_pred[i] == 2:
            group0.append(X_new[i])
            outlier_index.append(i)

    group0 = np.array(group0)
    # ellipse_outliers = EllipseModel()
    # ellipse_outliers.estimate(group0)
    # outliers_x_mean,outliers_y_mean,a,b,theta = ellipse_outliers.params

    # plot
    print(targets)
    for i in range(len(targets)):
        if 'QX_' in targets[i]:
            targets[i] = 'QX_group'
        elif 'QXPB_' in targets[i]:
            targets[i] = 'QXPB_group'
    targets = pd.DataFrame(data = targets)
    print(targets)

    principalDf = pd.DataFrame(data = X_new
                 , columns = ['PC1', 'PC2'])
    finalDf = pd.concat([principalDf, targets], axis = 1)

    points_XYCH_WX = []
    points_GYCH_WX = []
    print(finalDf)

    # for i in range(X_new.shape[0]):
    #     if i not in outlier_index:
    #         if 'QX_' in finalDf[0][i]:
    #             points_XYCH_WX.append([finalDf['PC1'][i],finalDf['PC2'][i]])
    #         else:
    #             points_GYCH_WX.append([finalDf['PC1'][i], finalDf['PC2'][i]])




    # points_XYCH_WX = np.array(points_XYCH_WX)
    # ellipse_points_XYCH_WX = EllipseModel()
    # ellipse_points_XYCH_WX.estimate(points_XYCH_WX)
    # XYCH_WX_x_mean,XYCH_WX_y_mean,XYCH_WX_a,XYCH_WX_b,XYCH_WX_theta = ellipse_points_XYCH_WX.params
    #
    # points_GYCH_WX = np.array(points_GYCH_WX)
    # ellipse_points_GYCH_WX = EllipseModel()
    # ellipse_points_GYCH_WX.estimate(points_GYCH_WX)
    # GYCH_WX_x_mean,GYCH_WX_y_mean,GYCH_WX_a,GYCH_WX_b,GYCH_WX_theta = ellipse_points_GYCH_WX.params
    #




    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Principal Component 1 {}%'.format(round(pca.explained_variance_ratio_[0]*100,2)), fontsize = 15)
    ax.set_ylabel('Principal Component 2 {}%'.format(round(pca.explained_variance_ratio_[1]*100,2)), fontsize = 15)
    ax.set_title('2 component PCA ({} mode)'.format(mode), fontsize = 20)


    # ellipse_XYCH_WX = Ellipse((XYCH_WX_x_mean, XYCH_WX_y_mean), 2*XYCH_WX_a, 2*XYCH_WX_b,XYCH_WX_theta,
    #                         edgecolor='r', fc='None', lw=2)
    # ax.XYCH_WXd_patch(ellipse_XYCH_WX)
    # ellipse_GYCH_WX = Ellipse((GYCH_WX_x_mean, GYCH_WX_y_mean), 2*GYCH_WX_a, 2*GYCH_WX_b,GYCH_WX_theta,
    #                         edgecolor='b', fc='None', lw=2)
    # ax.XYCH_WXd_patch(ellipse_GYCH_WX)

    groups=['QX_group','QXPB_group']

    for i in range(len(groups)):
        print(groups[i])
        indicesToKeep = finalDf[0].values == groups[i]
        print(indicesToKeep)
        if groups[i] == 'QX_group':
            ax_XYCH_WX = ax.scatter(finalDf.loc[indicesToKeep ,'PC1'],
                   finalDf.loc[indicesToKeep, 'PC2'],
                   c = 'r'
                   , s = 50)
        if groups[i] == 'QXPB_group':
            ax_GYCH_WX = ax.scatter(finalDf.loc[indicesToKeep, 'PC1'],
                                    finalDf.loc[indicesToKeep, 'PC2'],
                                    c='b'
                                    , s=50)




    plt.legend(handles=[ax_XYCH_WX,ax_GYCH_WX],labels=['{}group'.format(keywords[0]),'{}group'.format(keywords[1])],loc='best',labelspacing=2,prop={'size': 12})
    ax.grid()
    plt.show()
    # plt.savefig('figures/neg_plots/整体/整体PCA.png')

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

    pca(filename,mode,keywords)




