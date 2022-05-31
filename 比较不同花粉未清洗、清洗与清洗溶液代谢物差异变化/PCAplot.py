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
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from skimage.measure import EllipseModel
def pca(filename,mode,keywords):
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
    print('dataframe shape after drop rows that have NA value: ({} metabolites, {} samples)'.format(data.shape[0],data.shape[1]-2))
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

        x_index=[]
        y_index=[]
        z_index=[]

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
            normalized_data_impute_x.append(normalized_data_impute[:,index].T)
        normalized_data_impute_x = np.array(normalized_data_impute_x)

        normalized_data_impute_y =[]
        for index in y_index:
            normalized_data_impute_y.append(normalized_data_impute[:,index].T)
        normalized_data_impute_y = np.array(normalized_data_impute_y)

        normalized_data_impute_z = []
        for index in z_index:
            normalized_data_impute_z.append(normalized_data_impute[:, index].T)
        normalized_data_impute_z = np.array(normalized_data_impute_z)


        print(normalized_data_impute_x.shape)
        print(normalized_data_impute_y.shape)
        normalized_data_impute = np.vstack((normalized_data_impute_x,normalized_data_impute_y,normalized_data_impute_z))
        print(normalized_data_impute.shape)



        # PCA
        pca = PCA(n_components=2)
        pca.fit(normalized_data_impute)
        X_new = pca.fit_transform(normalized_data_impute)
        print(X_new)
        print(pca.explained_variance_ratio_)

        y_pred = KMeans(n_clusters=3,random_state=8).fit_predict(X_new)

        print(y_pred)

        group0 =[]
        outlier_index = []
        # for i in range(len(y_pred)):
        #     if y_pred[i] == 2:
        #         group0.append(X_new[i])
        #         outlier_index.append(i)

        group0 = np.array(group0)
        # ellipse_outliers = EllipseModel()
        # ellipse_outliers.estimate(group0)
        # outliers_x_mean,outliers_y_mean,a,b,theta = ellipse_outliers.params




        # plot
        print(targets)
        for i in range(len(targets)):
            if 'WX_' in targets[i]:
                targets[i] = 'WX_group'
            elif 'QX_' in targets[i]:
                targets[i] = 'QX_group'
            elif 'QXRY_' in targets[i]:
                targets[i] = 'QXRY_group'
        targets = pd.DataFrame(data = targets)
        print(targets)

        principalDf = pd.DataFrame(data = X_new
                     , columns = ['PC1', 'PC2'])
        finalDf = pd.concat([principalDf, targets], axis = 1)

        points_x = []
        points_y = []
        points_z = []
        print(finalDf)

        for i in range(X_new.shape[0]):
            if i not in outlier_index:
                if 'WX_' in finalDf[0][i]:
                    points_x.append([finalDf['PC1'][i],finalDf['PC2'][i]])
                elif 'QX_' in finalDf[0][i]:
                    points_y.append([finalDf['PC1'][i], finalDf['PC2'][i]])
                elif 'QXRY_' in finalDf[0][i]:
                    points_z.append([finalDf['PC1'][i], finalDf['PC2'][i]])



        try:
            points_x = np.array(points_x)
            ellipse_points_x = EllipseModel()
            ellipse_points_x.estimate(points_x)
            x_x_mean,x_y_mean,x_a,x_b,x_theta = ellipse_points_x.params
        except:
            pass

        try:
            points_y = np.array(points_y)
            ellipse_points_y = EllipseModel()
            ellipse_points_y.estimate(points_y)
            y_x_mean,y_y_mean,y_a,y_b,y_theta = ellipse_points_y.params
        except:
            pass

        try:
            points_z = np.array(points_z)
            ellipse_points_z = EllipseModel()
            ellipse_points_z.estimate(points_z)
            z_x_mean,z_y_mean,z_a,z_b,z_theta = ellipse_points_z.params
        except:
            pass

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel('Principal Component 1 {}%'.format(round(pca.explained_variance_ratio_[0]*100,2)), fontsize = 15)
        ax.set_ylabel('Principal Component 2 {}%'.format(round(pca.explained_variance_ratio_[1]*100,2)), fontsize = 15)
        ax.set_title('2 component PCA ({} mode)'.format(mode), fontsize = 20)

        # try:
        #     ellpse_x = Ellipse((x_x_mean, x_y_mean), 2*x_a, 2*x_b,x_theta,
        #                             edgecolor='r', fc='None', lw=2)
        #     ax.add_patch(ellpse_x)
        # except:
        #     pass
        # try:
        #     ellipse_y = Ellipse((y_x_mean, y_y_mean), 2*y_a, 2*y_b,y_theta,
        #                             edgecolor='b', fc='None', lw=2)
        #     ax.add_patch(ellipse_y)
        # except:
        #     pass
        # try:
        #     ellipse_z = Ellipse((z_x_mean, z_y_mean), 2*z_a, 2*z_b,z_theta,
        #                             edgecolor='b', fc='None', lw=2)
        #     ax.add_patch(ellipse_z)
        # except:
        #     pass


        groups=['WX_group','QX_group','QXRY_group']

        for i in range(len(groups)):
            print(groups[i])
            indicesToKeep = finalDf[0].values == groups[i]
            print(indicesToKeep)
            if groups[i] == 'WX_group':
                ax_x = ax.scatter(finalDf.loc[indicesToKeep ,'PC1'],
                       finalDf.loc[indicesToKeep, 'PC2'],
                       c = 'r'
                       , s = 50)
            if groups[i] == 'QX_group':
                ax_y = ax.scatter(finalDf.loc[indicesToKeep, 'PC1'],
                                        finalDf.loc[indicesToKeep, 'PC2'],
                                        c='b'
                                        , s=50)
            if groups[i] == 'QXRY_group':
                ax_z = ax.scatter(finalDf.loc[indicesToKeep, 'PC1'],
                                        finalDf.loc[indicesToKeep, 'PC2'],
                                        c='g'
                                        , s=50)




        plt.legend(handles=[ax_x,ax_y,ax_z],labels=['{}group'.format(keywords[0]),'{}group'.format(keywords[1]),'{}group'.format(keywords[2])],loc='best',labelspacing=2,prop={'size': 12})

        ax.grid()
        plt.show()
        # plt.savefig('figures/neg_plots/整体/整体未破壁样本与破壁样本的变化PCA.png')


if __name__ == '__main__':
    mode = 'POS'
    if mode == "BOTH":
        filename = '../files/pollen files/results/peaktableBOTHout_BOTH_noid_replace_mean_full.xlsx'
    elif mode == 'POS':
        filename = '../files/pollen files/results/process_output_quantid_pos_camera_noid/peaktablePOSout_POS_noid_full_sample_replace_mean_full.xlsx'
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
    keywords6 = ['WX_', 'QX_', 'QXRY_']
    keywords = keywords5

    pca(filename,mode,keywords)



