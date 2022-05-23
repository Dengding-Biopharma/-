import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from delete import deleteDep


def boxplot(data,mode):
    data = pd.read_excel(data)  # loading data


    targets = data.columns.values[2:]  # 保存病人名称


    saved_label = data['dataMatrix'].values  # 保存小分子名称
    saved_smile = data['smile'].values # 小分子对应的smile
    del data['dataMatrix']
    del data['smile']



    data_impute = data.values
    for i in range(data_impute.shape[1]):
        data_impute[:, i] = (data_impute[:, i] / np.sum(data_impute[:, i])) * 100


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
    # for i in range(len(df)):
    #     if df['name'][i] == 'DG(20:4_18:0)':
    #         top_k_index.remove(df['index'][i])
    #     if df['name'][
    #         i] == "4-((11aS)-1,3-dioxo-5-(p-tolyl)-11,11a-dihydro-1H-imidazo[1',5':1,6]pyrido[3,4-b]indol-2(3H,5H,6H)-yl)-N-(4-phenylbutan-2-yl)benzamide [M+H]+":
    print('{} metabolites have significant difference after t-test!!!!!!'.format(count))
    while True:
        top_k_index = p_list.argsort()[::-1][len(p_list) - top_k:]
        print('The top 20 indexs are (原始): ', top_k_index)
        # for special treatment only!!!!
        # 要去掉什么名字的小分子
        delete_keywords = [
            'DG(20:4_18:0)',
            "4-((11aS)-1,3-dioxo-5-(p-tolyl)-11,11a-dihydro-1H-imidazo[1',5':1,6]pyrido[3,4-b]indol-2(3H,5H,6H)-yl)-N-(4-phenylbutan-2-yl)benzamide [M+H]+",
        ]
        # 要更改什么小分子的名字
        modified_keywords = []
        df = pd.DataFrame()
        df['P'] = p_list[top_k_index]
        df['name'] = saved_label[top_k_index]
        df['index'] = top_k_index
        df = deleteDep(df)
        df = df.sort_values(by='P', ascending=True)
        df = df.reset_index(drop=True)
        top_k_index = list(df['index'].values)
        for i in range(len(df)):
            if df['name'][i] in delete_keywords:
                top_k_index.remove(df['index'][i])
                print('去掉一个小分子')
        print('The top 20 indexs are (去重后并且去掉特殊小分子): ',top_k_index)
        if len(top_k_index) == 20:
            print('找到20个小分子,选取了top{}'.format(top_k))
            break
        else:
            top_k+=1
            print('还没到20个小分子被选出来！！！')
            print('*'*50)
    # 做完了数值分析，开始归一化画图
    normalized_data_impute = data_impute

    # 归一化之后还要分别取一次组别数据，用来画图
    normalized_data_impute_ad = []
    for index in ad_index:
        normalized_data_impute_ad.append(normalized_data_impute[:, index].T)
    normalized_data_impute_ad = np.array(normalized_data_impute_ad)

    normalized_data_impute_hc = []
    for index in hc_index:
        normalized_data_impute_hc.append(normalized_data_impute[:, index].T)
    normalized_data_impute_hc = np.array(normalized_data_impute_hc)

    X_ad = normalized_data_impute_ad
    X_hc = normalized_data_impute_hc

    X_diff_ad = []
    for i in top_k_index:
        X_diff_ad.append(X_ad[:, i:i + 1])

    X_diff_hc = []
    for i in top_k_index:
        X_diff_hc.append(X_hc[:, i:i + 1])

    data_ad = []
    labels = []

    for i in range(len(X_diff_ad)):
        data_ad.append(X_diff_ad[i])
        labels += [saved_label[top_k_index[i]], '']

    for i in range(len(labels)):
        labels[i] = '\n\n' + labels[i]
    print(labels)

    data_hc = []
    for i in range(len(X_diff_hc)):
        data_hc.append(X_diff_hc[i])

    data_ads = []
    for i in data_ad:
        data_ads.append(i.reshape(i.shape[0]))
    data_ad = data_ads

    data_hcs = []
    for i in data_hc:
        data_hcs.append(i.reshape(i.shape[0]))
    data_hc = data_hcs

    data_ad = np.array(data_ad)
    data_hc = np.array(data_hc)
    print(data_ad.shape)
    print(data_hc.shape)

    data = []
    color_list = []
    for i in range(data_ad.shape[0]):
        data.append(data_ad[i, :])
        data.append(data_hc[i, :])


    for i in range(len(labels)):
        if i % 2 == 0:
            print(labels[i])
            print()

    bp = plt.boxplot(data, labels=labels, patch_artist=True)
    plt.xticks(rotation=90)
    for i in range(len(bp['boxes'])):
        if i % 2 == 0:
            bp['boxes'][i].set(color='r')
        else:
            bp['boxes'][i].set(color='b')

    plt.title('boxplot for top 20 variables which have significant differences between groups ({} mode)'.format(mode))
    plt.legend(handles=[bp['boxes'][0], bp['boxes'][1]], labels=['AD', 'HC'])
    plt.show()


if __name__ == '__main__':
    mode = 'pos'
    if mode == 'both':
        filepath = 'files/ad files/peaktableBOTHout_BOTH_noid_replace_mean_full.xlsx'
    elif mode == 'pos':
        filepath = 'files/ad files/peaktablePOSout_POS_noid_replace_mean_full.xlsx'
    elif mode == 'neg':
        filepath = 'files/ad files/peaktableNEGout_NEG_noid_replace_mean_full.xlsx'
    boxplot(filepath,mode)
