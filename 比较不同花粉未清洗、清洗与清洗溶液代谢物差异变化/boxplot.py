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
from scipy.stats import ttest_ind, stats
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from skimage.measure import EllipseModel
from statsmodels.stats.anova import anova_lm


def boxplot(filename,mode,keywords):
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

        top_k = 20
        p_list = []
        for i in range(normalized_data_impute_x.shape[1]):
            f, p = stats.f_oneway(normalized_data_impute_x[:, i:i + 1], normalized_data_impute_y[:, i:i + 1],
                                  normalized_data_impute_z[:, i:i + 1])
            p_list.append(p[0])
        p_list = np.array(p_list)
        count = 0
        for p in p_list:
            if p < 0.05:
                count += 1

        top_k_index = p_list.argsort()[::-1][len(p_list) - top_k:]

        if len(top_k_index) == 0:
            print('there are no significant difference between metabolites on these three groups {} by ANOVA'.format(keywords))
        else:
            print(top_k_index)

            x = np.array(normalized_data_impute_x)
            y = np.array(normalized_data_impute_y)
            z = np.array(normalized_data_impute_z)

            x_diff = []
            for i in top_k_index:
                x_diff.append(x[:, i:i + 1])

            y_diff = []
            for i in top_k_index:
                y_diff.append(y[:, i:i + 1])

            z_diff = []
            for i in top_k_index:
                z_diff.append(z[:, i:i + 1])

            data_x = []
            labels = []
            for i in range(len(x_diff)):
                data_x.append(x_diff[i])
                labels += [saved_label[top_k_index[i]], '', '']

            data_y = []

            for i in range(len(y_diff)):
                data_y.append(y_diff[i])

            data_z = []
            for i in range(len(z_diff)):
                data_z.append(z_diff[i])

            # Creating axes instance
            data_xs = []
            for i in data_x:
                data_xs.append(i.reshape(i.shape[0]))
            data_x = data_xs

            data_ys = []
            for i in data_y:
                data_ys.append(i.reshape(i.shape[0]))
            data_y = data_ys

            data_zs = []
            for i in data_z:
                data_zs.append(i.reshape(i.shape[0]))
            data_z = data_zs

            data_x = np.array(data_x)
            data_y = np.array(data_y)
            data_z = np.array(data_z)
            print(data_x.shape)
            print(data_y.shape)
            print(data_z.shape)

            data = []

            for i in range(data_x.shape[0]):
                data.append(data_x[i, :])
                data.append(data_y[i, :])
                data.append(data_z[i, :])

            print(data)

            bp = plt.boxplot(data, labels=labels, patch_artist=True)

            plt.xticks(rotation=90)
            for i in range(len(bp['boxes'])):
                if i % 3 == 0:
                    bp['boxes'][i].set(color='r')
                elif i % 3 == 1:
                    bp['boxes'][i].set(color='g')
                else:
                    bp['boxes'][i].set(color='b')
            plt.legend(handles=[bp['boxes'][0], bp['boxes'][1], bp['boxes'][2]],
                       labels=['{}group'.format(keywords[0]), '{}group'.format(keywords[1]), '{}group'.format(keywords[2])])
            plt.title('比较不同花粉未清洗、清洗与清洗溶液代谢物差异变化(差异最大的20个小分子) ({} mode)'.format(mode))
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

    boxplot(filename,mode,keywords)





