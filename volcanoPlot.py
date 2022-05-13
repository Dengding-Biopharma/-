import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from bioinfokit import analys,visuz


def volcanoPlot(data):
    data = pd.read_excel(data)
    print(data)
    targets = data.columns.values[1:]  # 保存病人名称
    print(targets)

    saved_label = data['dataMatrix'].values  # 保存小分子名称
    print(saved_label)

    del data['dataMatrix']
    data_impute = data.values
    for i in range(data_impute.shape[1]):
        data_impute[:, i] = data_impute[:, i] / np.sum(data_impute[:, i])
    print(data_impute)

    # 拿到组别索引
    ad_index = []
    hc_index = []
    for i in range(len(targets)):
        if "AD" in targets[i]:
            ad_index.append(i)
        else:
            hc_index.append(i)

    # 分别拿出AD和HC的数据做差异性分析
    data_impute_ad = []
    for index in ad_index:
        data_impute_ad.append(data_impute[:, index].T)
    data_impute_ad = np.array(data_impute_ad)

    data_impute_hc = []
    for index in hc_index:
        data_impute_hc.append(data_impute[:, index].T)
    data_impute_hc = np.array(data_impute_hc)

    top_k = 20  # top几，可调
    p_list = []
    log2fc_list = []
    for i in range(data_impute_ad.shape[1]):
        t, p = ttest_ind(data_impute_ad[:, i:i + 1], data_impute_hc[:, i:i + 1], equal_var=True)
        log2fc = np.log2(np.mean(data_impute_hc[:, i:i + 1])/np.mean(data_impute_ad[:, i:i + 1]))
        log2fc_list.append(log2fc)
        p_list.append(p[0])
    p_list = np.array(p_list)
    log2fc_list = np.array(log2fc_list)
    count = 0
    for p in p_list:
        if p < 0.05:
            count += 1
    # top_k_index = p_list.argsort()[::-1][len(p_list) - count:]
    # top_k_index = p_list.argsort()[::-1][:]
    # top_k_log2fc_index = log2fc_list.argsort()[::-1][:]
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
    volcanoPlot('files/ad files/peaktablePOSout_POS_noid_more_puring_mean_full.xlsx')