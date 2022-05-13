import numpy as np
import pandas as pd

data = pd.read_excel('ad files/varsPOSout_pos_noid.xlsx')
print(data)
variable_name_list = []
variable_smile_list = []
for i in range(len(data)):
    if 'no_match' not in data['ours_name'][i] and 'no_match' not in data['gnps_name'][i]:
        if data['ours_most_common_count'][i] >= data['gnps_most_common_count'][i]:
            variable_name_list.append(data['ours_name'][i])
            variable_smile_list.append(data['ours_smile'][i])
        else:
            variable_name_list.append(data['gnps_name'][i])
            variable_smile_list.append(data['gnps_smile'][i])
        continue
    if 'no_match' not in data['ours_name'][i]:
        variable_name_list.append(data['ours_name'][i])
        variable_smile_list.append(data['ours_smile'][i])
        continue
    if 'no_match' not in data['gnps_name'][i]:
        variable_name_list.append(data['gnps_name'][i])
        variable_smile_list.append(data['gnps_smile'][i])
        continue
    if 'no_match' in data['ours_name'][i] and 'no_match' in data['gnps_name'][i]:
        variable_name_list.append(data['ours_name'][i])
        variable_smile_list.append(data['ours_smile'][i])
        continue

print(len(variable_name_list))
print(len(variable_smile_list))

data['max_name'] = variable_name_list
data['max_smile'] = variable_smile_list
count = 0
for i in range(len(data)):
    if 'no_match' not in data['max_name'][i]:
        count += 1
print(count)

data.to_excel('ad files/varsPOSout_pos_noid_more_from_gnps.xlsx',index=False,na_rep=np.nan)