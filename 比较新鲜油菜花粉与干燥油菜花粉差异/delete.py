import numpy as np
import pandas as pd


def deleteDep(df):
    data = df
    print(data)

    dup_row = data.duplicated(subset=['name'], keep=False)

    dup_df = data[dup_row]
    for i in range(len(data)):
        if dup_row[i]:
            data = data.drop(i)
    dup_df = dup_df.sort_values(by='name')

    dup_df = dup_df.reset_index(drop=True)

    temp = []
    for i in range(len(dup_df)):
        if len(temp) == 0 or dup_df['name'][i] != temp[0]:
            temp = [dup_df['name'][i], dup_df['P'][i], i]
        elif dup_df['name'][i] == temp[0] and dup_df['P'][i] < temp[1]:
            dup_df = dup_df.drop(temp[2])
            temp = [dup_df['name'][i], dup_df['P'][i], i]
        elif dup_df['name'][i] == temp[0] and dup_df['P'][i] > temp[1]:
            dup_df = dup_df.drop(i)

    finalDF = pd.concat([dup_df, data])
    print(finalDF)

    return finalDF


def deleteDupFromOriginalTableByDiff(df, keywords):
    x_index = []
    y_index = []
    targets = df.columns.values
    for i in range(len(targets)):
        if keywords[0] in targets[i]:
            x_index.append(i)
        elif keywords[1] in targets[i]:
            y_index.append(i)

    diff_list = []
    for i in range(len(df)):
        diff = (np.mean(df.values[i, y_index]) - np.mean(df.values[i, x_index])) / np.mean(
            df.values[i, x_index])
        diff_list.append(diff * 100)
    df['diff'] = diff_list

    dup_row = df.duplicated(subset=['dataMatrix'], keep=False)

    dup_df = df[dup_row]

    for i in range(len(df)):
        if dup_row[i]:
            df = df.drop(i)
    dup_df = dup_df.sort_values(by='dataMatrix')
    dup_df = dup_df.reset_index(drop=True)

    current_name = None
    current_index = []
    current_diffs = []
    length = len(dup_df)
    for i in range(length):
        if current_name is None:
            assert not current_index and not current_diffs
            current_name = dup_df['dataMatrix'][i]
            current_index.append(i)
            current_diffs.append(dup_df['diff'][i])
        else:
            if i == length - 1:
                current_index.append(i)
                current_diffs.append(dup_df['diff'][i])
                max_diff_index = np.argmax(np.abs(current_diffs))
                for j in range(len(current_index)):
                    if j != max_diff_index:
                        dup_df = dup_df.drop(current_index[j])
                break
            assert current_diffs and current_diffs
            if dup_df['dataMatrix'][i] != current_name:
                max_diff_index = np.argmax(np.abs(current_diffs))
                for j in range(len(current_index)):
                    if j != max_diff_index:
                        dup_df = dup_df.drop(current_index[j])

                current_index = []
                current_diffs = []
                current_name = dup_df['dataMatrix'][i]
                current_index.append(i)
                current_diffs.append(dup_df['diff'][i])
            else:
                current_index.append(i)
                current_diffs.append(dup_df['diff'][i])

    finalDF = pd.concat([dup_df, df])
    finalDF = finalDF.sort_values(by='diff', key=abs, ascending=False)
    diff_list = finalDF['diff'].values
    del finalDF['diff']
    return finalDF, diff_list


def reasonableNameForBoxplot(label):
    xls = pd.ExcelFile('../files/operation database.xlsx')
    delete_Char_table = pd.read_excel(xls, 'Delete Char')
    renaming_table = pd.read_excel(xls, 'Renaming')
    # print(renaming_table)
    # step 1
    for char in delete_Char_table['Unnamed: 0'].values:
        if char in label:
            temp = label.split(' '+char)[0]
            label = temp

    for i in range(len(renaming_table)):
        if label == renaming_table['Original Name'][i]:
            label = renaming_table['ReName'][i]

    return label




def Topkindex_DeleteNotInPubChem(labels, top_k):
    xls = pd.ExcelFile('../files/operation database.xlsx')
    delete_PubChem_table = pd.read_excel(xls, 'Not in PubChem', header=None)
    delete_table = [i[0] for i in delete_PubChem_table.values]

    current_index = top_k - 1
    index = [i for i in range(top_k)]

    for i in range(top_k):
        if labels[i] in delete_table:
            print("delete '{}'".format(labels[i]))
            index.remove(i)

    while len(index) != top_k:
        current_index += 1
        next_name = labels[current_index]
        if next_name not in delete_table:
            index.append(current_index)
        else:
            print("delete '{}'".format(next_name))

    return index


if __name__ == '__main__':
    data_path = 'pos_significant.xlsx'
    deleteDep(data_path)