import numpy as np
import pandas as pd
def deleteDep(df):
    data = df
    print(data)

    dup_row = data.duplicated(subset=['name'],keep=False)


    dup_df = data[dup_row]
    for i in range(len(data)):
        if dup_row[i] == True:
            data = data.drop(i)
    dup_df=dup_df.sort_values(by='name')



    dup_df = dup_df.reset_index(drop=True)

    temp = []
    for i in range(len(dup_df)):
        if len(temp) == 0 or dup_df['name'][i] != temp[0]:
            temp = [dup_df['name'][i], dup_df['P'][i],i]
        elif dup_df['name'][i] == temp[0] and dup_df['P'][i] < temp[1]:
            dup_df = dup_df.drop(temp[2])
            temp =[dup_df['name'][i],dup_df['P'][i],i]
        elif dup_df['name'][i] == temp[0] and dup_df['P'][i] > temp[1]:
            dup_df = dup_df.drop(i)

    finalDF = pd.concat([dup_df,data])
    print(finalDF)

    return finalDF


if __name__ == '__main__':
    data_path = 'pos_significant.xlsx'
    deleteDep(data_path)