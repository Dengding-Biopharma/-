import numpy as np
import pandas as pd
from collections import Counter
data_dic = pd.read_excel('varsPOSout_pos_noid.xlsx')
data = pd.read_excel('Pos-summary-0313-16.xlsx')
table = pd.read_excel('peaktablePOSout_POS_noid.xlsx')


variable_name_list = []
variable_num = len(data_dic)
for i in range(variable_num):
    mzmin = data_dic['xcmsCamera_mzmin'][i]
    mzmax = data_dic['xcmsCamera_mzmax'][i]
    rtmin = data_dic['xcmsCamera_rtmin'][i]
    rtmax = data_dic['xcmsCamera_rtmax'][i]
    bool_mz = data['Exp_pepMass'].between(mzmin,mzmax,inclusive=True)
    temp = data[bool_mz]
    if temp.empty:
        variable_name_list.append('{}_no_match'.format(data_dic['variableMetadata'][i]))
        continue
    bool_rt = temp['Exp_RT'].between(rtmin,rtmax,inclusive=True)
    temp =temp[bool_rt]
    if temp.empty:
        variable_name_list.append('variable_{}_no_match'.format(data_dic['variableMetadata'][i]))
        continue
    name_list = temp['Max_NAME'].values
    collections_names = Counter(name_list)
    most_common_name = collections_names.most_common(1)
    variable_name_list.append(most_common_name[0][0])

table = table.drop(columns='dataMatrix')
print(table)
# variable_name_list = pd.DataFrame(variable_name_list)
table.insert(0,'dataMatrix',variable_name_list)
print(table)


for i in range(len(table)):
    if 'no_match' in table['dataMatrix'][i]:
        table = table.drop(i)

print(table)

table.to_excel('peaktablePOSout_POS_noid_replace_variable.xlsx',index=False,na_rep=np.nan)