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
from scipy.stats import ttest_ind, mannwhitneyu, stats
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from skimage.measure import EllipseModel
from delete import deleteDep

def find_significant(filename,mode,keywords):
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
        saved_smile = data['smile'].values
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
        if x_index and y_index and z_index:
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
                print('there are no significant difference between metabolites on these three groups {} by ANOVA (mode: {})'.format(
                    keywords,mode))
            else:
                print(top_k_index)
                df = pd.DataFrame()
                df['P'] = p_list[top_k_index]
                df['smile'] = saved_smile[top_k_index]
                df['name'] = saved_label[top_k_index]
                df = df.sort_values(by='P',ascending=True)
                df = deleteDep(df)
                df.to_excel('{}_vs_{}_vs_{}_{}_significant.xlsx'.format(keywords[0][:-1],keywords[1][:-1],keywords[2][:-1],mode),index=False,na_rep=np.nan)
                print('{}_vs_{}_vs_{}_{}_significant.xlsx generated!!!!!!!!!!!!!!'.format(keywords[0][:-1],keywords[1][:-1],keywords[2][:-1],mode))
        else:
            print('One or two or three groups has no value! keywords: {}, (mode: {})'.format(keywords,mode))
    print('*'*50)


if __name__ == '__main__':
    modes = ['BOTH','POS','NEG']
    keywords_list = [['XYCH_WX_', 'XYCH_QX_', 'XYCH_QXRY_'],['GYCH_WX_', 'GYCH_QX_', 'GYCH_QXRY_'],['GWBZ_WX_', 'GWBZ_QX_', 'GWBZ_QXRY_'],['GHH_WX_', 'GHH_QX_', 'GHH_QXRY_'],['GCH_WX_', 'GCH_QX_', 'GCH_QXRY_'],['WX_', 'QX_', 'QXRY']]
    for mode in modes:
        if mode == "BOTH":
            filename = '../files/pollen files/results/peaktableBOTHout_BOTH_noid_replace_mean_full.xlsx'
        elif mode == 'POS':
            filename = '../files/pollen files/results/process_output_quantid_pos_camera_noid/peaktablePOSout_POS_noid_replace.xlsx'
        elif mode == 'NEG':
            filename = '../files/pollen files/results/process_output_quantid_neg_camera_noid/peaktableNEGout_NEG_noid_replace.xlsx'
        for keywords in keywords_list:
            find_significant(filename,mode,keywords)