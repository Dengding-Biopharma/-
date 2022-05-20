import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def randomforestimportance(data, mode):
    data = pd.read_excel(data)
    # data = pd.read_excel('files/ad files/peaktableNEGout_NEG_noid_replace.xlsx')

    targets = data.columns.values[2:]

    saved_label = data['dataMatrix'].values

    del data['dataMatrix']
    del data['smile']

    data_impute = data.values
    for i in range(data_impute.shape[1]):
        data_impute[:, i] = data_impute[:, i] / np.sum(data_impute[:, i])

    normalized_data_impute = data_impute
    normalized_data_impute = normalized_data_impute.T

    forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
    forest.fit(normalized_data_impute, targets)
    importances = forest.feature_importances_
    print(importances)
    sorted_imps = sorted(importances, reverse=True)
    print(sorted_imps)

    indices = np.argsort(importances)[::-1]
    print(indices)
    top_20_labels = []
    for i in range(20):
        top_20_labels.append(saved_label[indices[i]])
    for f in range(normalized_data_impute.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, saved_label[indices[f]], importances[indices[f]]))

    plt.barh(range(20), sorted_imps[:20], color='r', align='center')
    plt.yticks(range(20), top_20_labels)
    plt.title('Randomforest Importance graph for top 20 variables ({} mode)'.format(mode))
    plt.show()


if __name__ == '__main__':
    mode = 'pos'
    if mode == 'both':
        filepath = 'files/ad files/peaktableBOTHout_BOTH_noid_replace_mean_full.xlsx'
    elif mode == 'pos':
        filepath = 'files/ad files/peaktablePOSout_POS_noid_replace_mean_full.xlsx'
    elif mode == 'neg':
        filepath = 'files/ad files/peaktableNEGout_NEG_noid_replace_mean_full.xlsx'
    randomforestimportance(filepath, mode)
