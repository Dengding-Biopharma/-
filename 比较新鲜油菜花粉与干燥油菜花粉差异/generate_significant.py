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

    for i in range(len(targets)):
        if keywords[0] not in targets[i] and keywords[1] not in targets[i]:
            del data[targets[i]]
    targets = data.columns.values[2:]

    data = data.dropna().reset_index(drop=True)
    print('dataframe shape after drop rows that have NA value: ({} metabolites, {} samples)'.format(data.shape[0],
                                                                                                    data.shape[1] - 2))

    saved_label = data['dataMatrix'].values
    saved_smile = data['smile'].values
    del data['dataMatrix']
    del data['smile']

    data_impute = data.values
    for i in range(data_impute.shape[1]):
        data_impute[:, i] = (data_impute[:, i] / np.sum(data_impute[:, i])) * 100

    normalized_data_impute = data_impute


    XYCH_WX_index = []
    GYCH_WX_index = []
    for i in range(len(targets)):
        if "XYCH_WX_" in targets[i]:
            XYCH_WX_index.append(i)
        else:
            GYCH_WX_index.append(i)
    print(XYCH_WX_index)
    print(GYCH_WX_index)
    if XYCH_WX_index and GYCH_WX_index:

        normalized_data_impute_XYCH_WX = []
        for index in XYCH_WX_index:
            normalized_data_impute_XYCH_WX.append(normalized_data_impute[:, index].T)
        normalized_data_impute_XYCH_WX = np.array(normalized_data_impute_XYCH_WX)

        normalized_data_impute_GYCH_WX = []
        for index in GYCH_WX_index:
            normalized_data_impute_GYCH_WX.append(normalized_data_impute[:, index].T)
        normalized_data_impute_GYCH_WX = np.array(normalized_data_impute_GYCH_WX)

        top_k = 20
        p_list = []
        for i in range(normalized_data_impute_XYCH_WX.shape[1]):
            t, p = mannwhitneyu(normalized_data_impute_XYCH_WX[:, i:i + 1], normalized_data_impute_GYCH_WX[:, i:i + 1],
                                alternative='two-sided')
            p_list.append(p[0])
        p_list = np.array(p_list)
        count = 0
        for p in p_list:
            if p < 0.05:
                count += 1

        top_k_index = p_list.argsort()[::-1][len(p_list) - count:]


        if len(top_k_index) == 0:
            print('there are no significant difference between metabolites on these two groups {} by mann whitney u test (mode: {})'.format(
                keywords,mode))
        else:
            print(top_k_index)
            df = pd.DataFrame()
            df['P'] = p_list[top_k_index]
            df['smile'] = saved_smile[top_k_index]
            df['name'] = saved_label[top_k_index]
            df = df.sort_values(by='P',ascending=True)
            df = deleteDep(df)
            df.to_excel('{}_vs_{}_{}_significant.xlsx'.format(keywords[0][:-1],keywords[1][:-1],mode),index=False,na_rep=np.nan)
            print('{}_vs_{}_{}_significant.xlsx generated!!!!!!!!!!!!!!'.format(keywords[0][:-1],keywords[1][:-1],mode))
    else:
        print('One or two or three groups has no value! keywords: {}, (mode: {})'.format(keywords,mode))
    print('*'*50)


if __name__ == '__main__':
    modes = ['BOTH','POS','NEG']
    keywords_list = [
        ['XYCH_WX_','GYCH_WX_']
    ]
    for mode in modes:
        if mode == "BOTH":
            filename = '../files/pollen files/results/peaktableBOTHout_BOTH_noid_replace_mean_full.xlsx'
        elif mode == 'POS':
            filename = '../files/pollen files/results/process_output_quantid_pos_camera_noid/peaktablePOSout_POS_noid_replace.xlsx'
        elif mode == 'NEG':
            filename = '../files/pollen files/results/process_output_quantid_neg_camera_noid/peaktableNEGout_NEG_noid_replace.xlsx'
        for keywords in keywords_list:
            find_significant(filename,mode,keywords)