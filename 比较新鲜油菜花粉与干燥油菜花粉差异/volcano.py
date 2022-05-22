import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu
from bioinfokit import visuz


def volcanoPlot(filename,mode):
    data = pd.read_excel(filename)
    targets = data.columns.values[2:]

    for i in range(len(targets)):
        if 'XYCH_WX_' not in targets[i] and 'GYCH_WX_' not in targets[i]:
            del data[targets[i]]
    targets = data.columns.values[2:]
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

    XYCH_WX_index = []
    GYCH_WX_index = []
    for i in range(len(targets)):
        if "XYCH_WX_" in targets[i]:
            XYCH_WX_index.append(i)
        else:
            GYCH_WX_index.append(i)
    print(XYCH_WX_index)
    print(GYCH_WX_index)

    normalized_data_impute_x = []
    for index in XYCH_WX_index:
        normalized_data_impute_x.append(normalized_data_impute[:, index].T)
    normalized_data_impute_x = np.array(normalized_data_impute_x)

    normalized_data_impute_y = []
    for index in GYCH_WX_index:
        normalized_data_impute_y.append(normalized_data_impute[:, index].T)
    normalized_data_impute_y = np.array(normalized_data_impute_y)

    top_k = 20  # top几，可调
    p_list = []
    log2fc_list = []
    for i in range(normalized_data_impute_x.shape[1]):
        t,p = mannwhitneyu(normalized_data_impute_x[:,i:i+1],normalized_data_impute_y[:,i:i+1],alternative='two-sided')
        log2fc = np.log2(np.mean(normalized_data_impute_x[:, i:i + 1])/np.mean(normalized_data_impute_y[:, i:i + 1]))
        log2fc_list.append(log2fc)
        p_list.append(p[0])
    p_list = np.array(p_list)
    log2fc_list = np.array(log2fc_list)
    count = 0
    for p in p_list:
        if p < 0.05:
            count += 1
    print(count)
    df = pd.DataFrame()
    df['name'] = saved_label
    df['log2FC'] = log2fc_list
    df['p-value'] = p_list
    print(df)
    print(np.min(df['log2FC']))
    print(np.max(df['log2FC']))
    print(np.min(df['p-value']))
    print(np.max(df['p-value']))

    # df = analys.get_data('volcano').data
    # print(type(df))

    visuz.gene_exp.volcano(df=df,lfc='log2FC',pv='p-value',show=True,lfc_thr=(0,0),ar=0,plotlegend=True)



if __name__ == '__main__':
    mode = 'POS'
    if mode == "BOTH":
        filename = '../files/pollen files/results/peaktableBOTHout_BOTH_noid_replace_mean_full.xlsx'
    elif mode == 'POS':
        filename = '../files/pollen files/results/process_output_quantid_pos_camera_noid/peaktablePOSout_POS_noid_replace.xlsx'
    elif mode == 'NEG':
        filename = '../files/pollen files/results/process_output_quantid_neg_camera_noid/peaktableNEGout_NEG_noid_replace.xlsx'


    volcanoPlot(filename,mode)