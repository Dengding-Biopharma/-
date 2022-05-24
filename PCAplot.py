import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.preprocessing
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from skimage.measure import EllipseModel
from sklearn.preprocessing import StandardScaler

from matplotlib.pyplot import figure
def pca(data,mode):
    data = pd.read_excel(data)


    targets = data.columns.values[2:]  # 保存病人名称






    for i in range(len(targets)):
        if 'AD' in targets[i]:
            targets[i] = 'AD_Disease_group'
        else:
            targets[i] = 'HC_Control_group'

    saved_label = data['dataMatrix'].values  # 保存小分子名称
    saved_smile = data['smile'].values  # 小分子对应的smile
    print(saved_label)
    del data['dataMatrix']
    del data['smile']
    data_impute = data.values
    for i in range(data_impute.shape[1]):
        data_impute[:, i] = data_impute[:, i]/np.sum(data_impute[:,i])

    ad_index=[]
    hc_index=[]
    for i in range(len(targets)):
        if "AD" in targets[i]:
            ad_index.append(i)
        else:
            hc_index.append(i)
    print(ad_index)
    print(hc_index)

    # PCA
    pca = PCA(n_components=2)
    pca.fit(data_impute.T)
    X_new = pca.fit_transform(data_impute.T)

    print(pca.explained_variance_ratio_)
    y_pred = KMeans(n_clusters=3,random_state=8).fit_predict(X_new)
    print(y_pred)



    group0 =[]
    outlier_index = []
    for i in range(len(y_pred)):
        if y_pred[i] == 0:
            group0.append(X_new[i])
            outlier_index.append(i)

    group0 = np.array(group0)
    # ellipse_outliers = EllipseModel()
    # ellipse_outliers.estimate(group0)
    # outliers_x_mean,outliers_y_mean,a,b,theta = ellipse_outliers.params




    # plot

    targets = pd.DataFrame(data = targets)

    principalDf = pd.DataFrame(data = X_new
                 , columns = ['PC1', 'PC2'])
    finalDf = pd.concat([principalDf, targets], axis = 1)

    points_ad = []
    points_hc = []
    print(finalDf)

    for i in range(X_new.shape[0]):
        if i not in outlier_index:
            if 'AD' in finalDf[0][i]:
                points_ad.append([finalDf['PC1'][i],finalDf['PC2'][i]])
            else:
                points_hc.append([finalDf['PC1'][i], finalDf['PC2'][i]])




    points_ad = np.array(points_ad)
    ellipse_points_ad = EllipseModel()
    ellipse_points_ad.estimate(points_ad)
    ad_x_mean,ad_y_mean,ad_a,ad_b,ad_theta = ellipse_points_ad.params

    points_hc = np.array(points_hc)
    ellipse_points_hc = EllipseModel()
    ellipse_points_hc.estimate(points_hc)
    hc_x_mean,hc_y_mean,hc_a,hc_b,hc_theta = ellipse_points_hc.params



    fig = plt.figure(figsize = (6,6))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Principal Component 1 {}%'.format(round(pca.explained_variance_ratio_[0]*100,2)), fontsize = 15)
    ax.set_ylabel('Principal Component 2 {}%'.format(round(pca.explained_variance_ratio_[1]*100,2)), fontsize = 15)
    ax.set_title('2 component PCA ({} mode)'.format(mode), fontsize = 20)
    ax.set_aspect('equal', adjustable='box')
    plt.ylim([-0.0008,0.001])
    plt.xlim(([-0.0005,0.0015]))


    # ellipse_ad = Ellipse((ad_x_mean, ad_y_mean), 2*ad_a, 2*ad_b,ad_theta,
    #                         edgecolor='r', fc='None', lw=2)
    # ax.add_patch(ellipse_ad)
    # ellipse_hc = Ellipse((hc_x_mean, hc_y_mean), 2*hc_a, 2*hc_b,hc_theta,
    #                         edgecolor='b', fc='None', lw=2)
    # ax.add_patch(ellipse_hc)


    groups=['AD_Disease_group','HC_Control_group']

    for i in range(len(groups)):
        print(groups[i])
        indicesToKeep = finalDf[0].values == groups[i]
        print(indicesToKeep)
        if groups[i] == 'AD_Disease_group':
            ax_ad = ax.scatter(finalDf.loc[indicesToKeep ,'PC1'],
                   finalDf.loc[indicesToKeep, 'PC2'],
                   c = 'r'
                   , s = 50)
        if groups[i] == 'HC_Control_group':
            ax_hc = ax.scatter(finalDf.loc[indicesToKeep, 'PC1'],
                                    finalDf.loc[indicesToKeep, 'PC2'],
                                    c='b'
                                    , s=50)

    ax.legend(labels=['AD','HC'],handles=[ax_ad,ax_hc],loc='best',borderpad=2,labelspacing=2,prop={'size': 8})
    ax.grid()


    plt.show()



if __name__ == '__main__':
    mode = 'pos'
    if mode == 'both':
        filepath = 'files/ad files/peaktableBOTHout_BOTH_noid_replace_mean_full.xlsx'
    elif mode == 'pos':
        filepath = 'files/ad files/peaktablePOSout_POS_noid_replace_mean_full.xlsx'
    elif mode == 'neg':
        filepath = 'files/ad files/peaktableNEGout_NEG_noid_replace_mean_full.xlsx'
    pca(filepath,mode)

