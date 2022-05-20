import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def stackedHistgramTop20(data,mode):
    data = pd.read_excel(data)
    # data = pd.read_excel('files/ad files/peaktableNEGout_NEG_noid_replace.xlsx')

    targets = data.columns.values[2:]  # 保存病人名称

    saved_label = data['dataMatrix'].values  # 保存小分子名称
    saved_smile = data['smile'].values  # 小分子对应的smile
    del data['dataMatrix']
    del data['smile']



    data_impute = data.values
    for i in range(data_impute.shape[1]):
        data_impute[:, i] = data_impute[:, i]/np.sum(data_impute[:,i])
    normalized_data_impute = data_impute
    print(normalized_data_impute.shape)

    ad_index=[]
    hc_index=[]
    for i in range(len(targets)):
        if "AD" in targets[i]:
            ad_index.append(i)
        else:
            hc_index.append(i)
    print(ad_index)
    print(hc_index)

    normalized_data_impute_ad = []
    for index in ad_index:
        normalized_data_impute_ad.append(normalized_data_impute[:,index].T)
    normalized_data_impute_ad = np.array(normalized_data_impute_ad)

    normalized_data_impute_hc =[]
    for index in hc_index:
        normalized_data_impute_hc.append(normalized_data_impute[:,index].T)
    normalized_data_impute_hc = np.array(normalized_data_impute_hc)

    data_impute_ad = normalized_data_impute_ad
    data_impute_hc = normalized_data_impute_hc


    top_k = 20
    sum_list =[]
    for i in range(data_impute_ad.shape[1]):
        sum = np.sum(data_impute_ad[:,i:i+1])

        sum_list.append(sum)
    sum_list = np.array(sum_list)
    top_k_index = sum_list.argsort()[::-1][0:top_k]
    print(top_k_index)


    X_ad = np.array(data_impute_ad)
    X_hc = np.array(data_impute_hc)
    X = np.vstack((X_ad,X_hc))

    X_top = []
    sum_list = []

    for row in range(X.shape[0]):
        sum = np.sum(X[row,:])
        temp = []
        for k in top_k_index:
            percentage = (X[row, k:k + 1]/sum) * 100
            temp.append(percentage[0])

        X_top.append(temp)



    X_top = np.array(X_top)
    print(X_top.shape)
    X_top = X_top.reshape(X_top.shape[1],X_top.shape[0])

    labels = []
    for i in top_k_index:
        labels.append(saved_label[i])

    color_exist = []
    def get_random_color(color_exist):
        r = lambda: random.randint(0, 255)
        color = '#%02X%02X%02X' % (r(), r(), r())
        while color in color_exist:
            r = lambda: random.randint(0, 255)
            color = '#%02X%02X%02X' % (r(), r(), r())
        color_exist.append(color)
        return color


    fig,ax = plt.subplots()

    plt.xticks(rotation = 90)
    ax.bar(targets,X_top[0],0.2,label=labels[0])
    for i in range(1,len(X_top)):
        ax.bar(targets,X_top[i],0.2,bottom=X_top[i-1],label=labels[i],color=get_random_color(color_exist))

    plt.title('Histogram of the top 20 metabolite percentage ({} mode)'.format(mode))
    ax.legend(bbox_to_anchor=(1, 1))
    plt.show()

if __name__ == '__main__':
    mode = 'pos'
    if mode == 'both':
        filepath = 'files/ad files/peaktableBOTHout_BOTH_noid_replace_mean_full.xlsx'
    elif mode == 'pos':
        filepath = 'files/ad files/peaktablePOSout_POS_noid_replace_mean_full.xlsx'
    elif mode == 'neg':
        filepath = 'files/ad files/peaktableNEGout_NEG_noid_replace_mean_full.xlsx'
    stackedHistgramTop20(filepath,mode)

