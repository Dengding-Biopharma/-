import numpy as np
import pandas as pd


def heatmap(data):
    data = pd.read_excel(data)  # loading data
    # data = pd.read_excel('files/ad files/peaktableNEGout_NEG_noid_replace.xlsx')


    targets = data.columns.values[1:] # 保存病人名称

    print(targets)

    saved_label = data['dataMatrix'].values # 保存小分子名称
    print(saved_label)

    del data['dataMatrix']
    data_impute = data.values
    for i in range(data_impute.shape[1]):
        data_impute[:, i] = data_impute[:, i]/np.sum(data_impute[:,i])
    print(data_impute)

    # 拿到组别索引
    ad_index=[]
    hc_index=[]
    for i in range(len(targets)):
        if "AD" in targets[i]:
            ad_index.append(i)
        else:
            hc_index.append(i)

    # 分别拿出AD和HC的数据做差异性分析
    data_impute_ad = []
    for index in ad_index:
        data_impute_ad.append(data_impute[:,index].T)
    data_impute_ad = np.array(data_impute_ad)

    data_impute_hc =[]
    for index in hc_index:
        data_impute_hc.append(data_impute[:,index].T)
    data_impute_hc = np.array(data_impute_hc)

    top_k = 20
    sum_list =[]
    for i in range(data_impute_ad.shape[1]):
        sum = np.sum(data_impute_ad[:,i:i+1])
        sum_list.append(sum)
    sum_list = np.array(sum_list)
    top_k_index = sum_list.argsort()[::-1][0:top_k]

    # 做完了数值分析，开始归一化画图

    normalized_data_impute = data_impute

    # 归一化之后还要分别取一次组别数据，用来画图
    normalized_data_impute_ad = []
    for index in ad_index:
        normalized_data_impute_ad.append(normalized_data_impute[:,index].T)
    normalized_data_impute_ad = np.array(normalized_data_impute_ad)

    normalized_data_impute_hc =[]
    for index in hc_index:
        normalized_data_impute_hc.append(normalized_data_impute[:,index].T)
    normalized_data_impute_hc = np.array(normalized_data_impute_hc)

    X_ad = normalized_data_impute_ad
    X_hc = normalized_data_impute_hc
    X = np.vstack((X_ad,X_hc))
    X = X.transpose()
    print(X)
    print(X.shape)

    X_top = []
    for k in top_k_index:
        X_top.append(X[k])
    X_top = np.array(X_top)
    print(X_top)
    print(len(X_top))
    print(len(X_top[0]))

    df = pd.DataFrame()

    for i in range(len(targets)):
        print(X_top[:,i:i+1])
        temp = []
        for j in X_top[:,i:i+1]:
            temp.append(j[0])

        df[targets[i]] = temp
    print(saved_label)
    labels = []
    for k in top_k_index:
        labels.append(saved_label[k])
    df['label'] = labels
    print(df)
    df = df.set_index('label')
    print(df.index.values)
    print(df)
    import dash_bio as dashbio
    from dash import dcc


    clustergram = dashbio.Clustergram(
        data=df,
        column_labels=list(df.columns.values),
        row_labels=list(df.index),
        height=1000,
        width=1500,
    )

    clustergram.show()
    dcc.Graph(figure=clustergram)


if __name__ == '__main__':
    heatmap('files/ad files/peaktablePOSout_POS_noid_more_puring_mean_full.xlsx')
