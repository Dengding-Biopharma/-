import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def pc1(data):
    data = pd.read_excel(data)
    # data = pd.read_excel('files/ad files/peaktableNEGout_NEG_noid_replace.xlsx')
    color_exist = []
    targets = data.columns.values[1:]


    print(data)
    print(targets)


    saved_label = data['dataMatrix'].values
    print(saved_label)
    del data['dataMatrix']
    data_impute = data.values
    for i in range(data_impute.shape[1]):
        data_impute[:, i] = data_impute[:, i] / np.sum(data_impute[:, i])
    normalized_data_impute = data_impute
    print(normalized_data_impute)

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


    print(normalized_data_impute_ad.shape)
    print(normalized_data_impute_hc.shape)




    # PCA
    pca = PCA(n_components=2)
    pca.fit(normalized_data_impute.T)
    X_new = pca.fit_transform(normalized_data_impute.T)
    print(X_new)
    print(pca.explained_variance_ratio_)

    targets = pd.DataFrame(data = targets)
    print(targets[0].values)

    df = pd.DataFrame()
    df['targets'] = targets[0].values
    df['PC1'] = X_new[:,:1]
    df = df.sort_values(by='PC1')
    print(df)

    fig = plt.figure()
    plt.scatter(df['targets'],df['PC1'],s=20,c='black')

    X_mean = np.mean(X_new[:,:1])
    X_mean_list = []
    for i in range(len(targets[0].values)):
        X_mean_list.append(X_mean)
    plt.plot(X_mean_list,linestyle = 'solid')
    plt.text(2,X_mean,'Mean',fontsize=8)

    X_std = np.std(X_new[:,:1])
    X_mean_list = []
    for i in range(len(targets[0].values)):
        X_mean_list.append(X_mean+X_std)
    plt.plot(X_mean_list,linestyle = 'dashed')
    plt.text(2,X_mean+X_std,'Mean+1*std',fontsize=8)

    X_mean_list = []
    for i in range(len(targets[0].values)):
        X_mean_list.append(X_mean+2*X_std)
    plt.plot(X_mean_list,linestyle = 'dashed')
    plt.text(2,X_mean+2*X_std,'Mean+2*std',fontsize=8)

    X_mean_list = []
    for i in range(len(targets[0].values)):
        X_mean_list.append(X_mean-X_std)
    plt.plot(X_mean_list,linestyle = 'dashed')
    plt.text(2,X_mean-X_std,'Mean-1*std',fontsize=8)


    plt.xlabel('samples')
    plt.title('PC1 for every sample')
    plt.xticks(rotation = 90)
    plt.show()


if __name__ == '__main__':
    pc1('files/ad files/peaktablePOSout_POS_noid_more_puring_mean_full.xlsx')
