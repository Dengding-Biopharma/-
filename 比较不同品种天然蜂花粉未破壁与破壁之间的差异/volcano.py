import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu
from bioinfokit import visuz


def volcanoPlot(filename,mode,keywords):
    data = pd.read_excel(filename)
    targets = data.columns.values[2:]
    for i in range(len(targets)):
        if 'WX' not in targets[i]:
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
    print(targets)
    data_impute = data.values
    for i in range(data_impute.shape[1]):
        data_impute[:, i] = (data_impute[:, i] / np.sum(data_impute[:, i])) * 100
    print(data_impute)

    normalized_data_impute = data_impute
    print(normalized_data_impute)

    x_index = []
    y_index = []

    for i in range(len(targets)):
        if keywords[0] in targets[i]:
            x_index.append(i)
        elif keywords[1] in targets[i]:
            y_index.append(i)

    print(x_index)
    print(y_index)
    targets = np.hstack((targets[x_index], targets[y_index]))
    print(targets)

    normalized_data_impute_x = []
    for index in x_index:
        normalized_data_impute_x.append(normalized_data_impute[:, index].T)
    normalized_data_impute_x = np.array(normalized_data_impute_x)

    normalized_data_impute_y = []
    for index in y_index:
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

    volcanoPlot(filename,mode,keywords)