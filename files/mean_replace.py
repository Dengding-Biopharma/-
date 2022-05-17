import numpy as np
import pandas as pd

data = pd.read_excel('pollen files/results/process_output_quantid_pos_camera_noid/peaktablePOSout_POS_noid_replace.xlsx')

print(data)

column_labels = data.columns.values[1:]

small_group_index = [] # list of list

temp = []

for i in range(len(column_labels)):
    current = column_labels[i][:-1]
    next = column_labels[i+1][:-1]
    if current != next:
        temp.append(i)
        small_group_index.append(temp)
        temp = []
    else:
        temp.append(i)
    if i+1 == len(column_labels)-1:
        print('reach the end!')
        temp.append(i + 1)
        small_group_index.append(temp)
        break
print(small_group_index)


for i in range(len(data)):
    for small_group in small_group_index:
        temp = []
        for index in small_group:
            num = data[column_labels[index]][i]
            if np.isnan(num):
                continue
            temp.append(num)
        mean = np.mean(temp)
        if np.isnan(mean):
            data = data.drop(i)
            break
        for j in range(len(small_group)):
            num = data[column_labels[small_group[j]]][i]
            if np.isnan(num):
                data[column_labels[small_group[j]]][i] = mean

data.to_excel('pollen files/results/process_output_quantid_pos_camera_noid/peaktablePOSout_POS_noid_replace_puring.xlsx',index=False)






