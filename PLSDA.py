import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from skimage.measure import EllipseModel
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import PLSRegression
from matplotlib.pyplot import figure


def plsda(data,mode):
    data = pd.read_excel(data)
    targets = data.columns.values[2:]



    for i in range(len(targets)):
        if 'AD' in targets[i]:
            targets[i] = 'AD_Disease_group'
        else:
            targets[i] = 'HC_Control_group'





    saved_label = data['dataMatrix'].values
    print(saved_label)
    del data['dataMatrix']
    del data['smile']


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

    X_ad = np.array(normalized_data_impute_ad)
    X_hc = np.array(normalized_data_impute_hc)
    X = np.vstack((X_ad,X_hc))
    print(X)
    int_targets = []
    for i in targets:
        if 'AD' in i:
            int_targets.append(0)
        else:
            int_targets.append(1)


    print(X.shape)

    plsr = PLSRegression(n_components=2,scale=False)
    plsr.fit(X,int_targets)

    print(plsr.predict(X))
    predicts = []
    for predict in plsr.predict(X):
        if predict >=0.5:
            predicts.append(1)
        else:
            predicts.append(0)




    colormap = {
        'AD_Disease_group': '#ff0000',  # Red
        'HC_Control_group': '#0000ff',  # Blue
    }

    colorlist = [colormap[c] for c in targets]
    print(colorlist)
    X_new = plsr.x_scores_

    y_pred = KMeans(n_clusters=3, random_state=8).fit_predict(X_new)
    print(y_pred)

    group0 = []
    outlier_index = []
    for i in range(len(y_pred)):
        if y_pred[i] == 2:
            group0.append(X_new[i])
            outlier_index.append(i)

    group0 = np.array(group0)
    # ellipse_outliers = EllipseModel()
    # ellipse_outliers.estimate(group0)
    # outliers_x_mean,outliers_y_mean,a,b,theta = ellipse_outliers.params

    # plot

    targets = pd.DataFrame(data=targets)

    principalDf = pd.DataFrame(data=X_new
                               , columns=['PC1', 'PC2'])
    finalDf = pd.concat([principalDf, targets], axis=1)

    points_ad = []
    points_hc = []
    print(finalDf)

    for i in range(X_new.shape[0]):
        if i not in outlier_index:
            if 'AD' in finalDf[0][i]:
                points_ad.append([finalDf['PC1'][i], finalDf['PC2'][i]])
            else:
                points_hc.append([finalDf['PC1'][i], finalDf['PC2'][i]])

    points_ad = np.array(points_ad)
    ellipse_points_ad = EllipseModel()
    ellipse_points_ad.estimate(points_ad)
    ad_x_mean, ad_y_mean, ad_a, ad_b, ad_theta = ellipse_points_ad.params

    points_hc = np.array(points_hc)
    ellipse_points_hc = EllipseModel()
    ellipse_points_hc.estimate(points_hc)
    hc_x_mean, hc_y_mean, hc_a, hc_b, hc_theta = ellipse_points_hc.params

    targets = list(targets[0])

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)

    ax.set_xlabel('PLS-DA axis 1')
    ax.set_ylabel('PLS-DA axis 2')
    ax.set_title('PLS-DA ({} mode)'.format(mode), fontsize=20)
    ax.set_aspect('equal')
    # plt.xlim([-0.00085,0.0009])

    ellipse_ad = Ellipse((ad_x_mean, ad_y_mean), 2*ad_a, 2*ad_b,ad_theta,
                            edgecolor='r', fc='None', lw=2)
    ax.add_patch(ellipse_ad)
    ellipse_hc = Ellipse((hc_x_mean, hc_y_mean), 2*hc_a, 2*hc_b,hc_theta,
                            edgecolor='b', fc='None', lw=2)
    ax.add_patch(ellipse_hc)

    groups = ['AD_Disease_group', 'HC_Control_group']

    for i in range(len(groups)):
        print(groups[i])
        indicesToKeep = finalDf[0].values == groups[i]
        print(indicesToKeep)
        if groups[i] == 'AD_Disease_group':
            ax_ad = ax.scatter(finalDf.loc[indicesToKeep, 'PC1'],
                               finalDf.loc[indicesToKeep, 'PC2'],
                               c='r'
                               , s=50)
        if groups[i] == 'HC_Control_group':
            ax_hc = ax.scatter(finalDf.loc[indicesToKeep, 'PC1'],
                               finalDf.loc[indicesToKeep, 'PC2'],
                               c='b'
                               , s=50)

    ax.legend(labels=['AD', 'HC'], handles=[ax_ad, ax_hc], loc='best', borderpad=2, labelspacing=2, prop={'size': 8})
    ax.grid()

    plt.show()


if __name__ == '__main__':
    mode = 'pos'
    if mode == 'both':
        filepath = 'files/ad files/peaktableBOTHout_BOTH_noid_replace_mean_full.xlsx'
    elif mode == 'pos':
        filepath = 'files/ad files/peaktablePOSout_POS_noid_replace_mean_full.xlsx'
    elif mode == 'neg':
        filepath = 'files/ad files/peaktableNEGout_NEG_noid_replace_mean_full.xlsx'
    plsda(filepath,mode)
