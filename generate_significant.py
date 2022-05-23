import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from delete import deleteDep


def find_significant(data,mode):
    data = pd.read_excel(data)  # loading data


    targets = data.columns.values[2:]  # 保存病人名称
    print(targets)

    saved_label = data['dataMatrix'].values  # 保存小分子名称
    saved_smile = data['smile'].values # 小分子对应的smile
    del data['dataMatrix']
    del data['smile']
    print(saved_label)


    data_impute = data.values
    for i in range(data_impute.shape[1]):
        data_impute[:, i] = (data_impute[:, i] / np.sum(data_impute[:, i])) * 100
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
    for i in range(data_impute_ad.shape[1]):
        t, p = ttest_ind(data_impute_ad[:, i:i + 1], data_impute_hc[:, i:i + 1], equal_var=True)
        p_list.append(p[0])
    p_list = np.array(p_list)
    count = 0
    for p in p_list:
        if p < 0.05:
            count += 1
    top_k_index = p_list.argsort()[::-1][len(p_list) - count:]
    print(top_k_index)
    # top_k_index = p_list.argsort()[::-1][:]
    print(count)
    # for special treatment only!!!!
    df = pd.DataFrame()
    df['P'] = p_list[top_k_index]
    df['name'] = saved_label[top_k_index]
    df['smile'] = saved_smile[top_k_index]
    df = deleteDep(df)
    df = df.sort_values(by='P', ascending=True)
    if mode == 'both':
        filename = 'BOTH_significant.xlsx'
    elif mode == 'pos':
        filename = 'POS_significant.xlsx'
    elif mode == 'neg':
        filename = 'NEG_significant.xlsx'
    df.to_excel(filename,index=False)

if __name__ == '__main__':
    mode = 'both'
    if mode == 'both':
        filepath = 'files/ad files/peaktableBOTHout_BOTH_noid_replace_mean_full.xlsx'
    elif mode == 'pos':
        filepath = 'files/ad files/peaktablePOSout_POS_noid_replace_mean_full.xlsx'
    elif mode == 'neg':
        filepath = 'files/ad files/peaktableNEGout_NEG_noid_replace_mean_full.xlsx'
    find_significant(filepath,mode)