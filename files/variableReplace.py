import math

import numpy as np
import pandas as pd
from collections import Counter
data_dic = pd.read_excel('ad files/varsPOSout_pos_noid.xlsx')
data = pd.read_excel('ad files/Pos-summary-0313-16.xlsx')
table = pd.read_excel('ad files/peaktablePOSout_POS_noid.xlsx')

# data_dic = pd.read_excel('ad files/varsNEGout_neg_noid.xlsx')
# data = pd.read_excel('ad files/Neg-summary-0313-16.xlsx')
# table = pd.read_excel('ad files/peaktableNEGout_NEG_noid.xlsx')



variable_name_list = []
variable_smile_list = []
most_common_count_list=[]
candidate_count_list=[]
variable_num = len(data_dic)
for i in range(variable_num):
    mzmin = data_dic['xcmsCamera_mzmin'][i]-0.05
    mzmax = data_dic['xcmsCamera_mzmax'][i]+0.05
    rtmin = data_dic['xcmsCamera_rtmin'][i]
    rtmax = data_dic['xcmsCamera_rtmax'][i]
    bool_mz = data['Exp_pepMass'].between(mzmin,mzmax,inclusive=True)
    # bool_mz = data['SpecMZ'].between(mzmin, mzmax, inclusive=True)
    temp = data[bool_mz]
    if temp.empty:
        variable_name_list.append('{}_no_match'.format(data_dic['variableMetadata'][i]))
        variable_smile_list.append('{}_no_match'.format(data_dic['variableMetadata'][i]))
        most_common_count_list.append('{}_no_match'.format(data_dic['variableMetadata'][i]))
        candidate_count_list.append('{}_no_match'.format(data_dic['variableMetadata'][i]))
        continue
    bool_rt = temp['Exp_RT'].between(rtmin,rtmax,inclusive=True)
    # bool_rt = temp['RT_Query'].between(rtmin, rtmax, inclusive=True)
    temp =temp[bool_rt]
    if temp.empty:
        variable_name_list.append('{}_no_match'.format(data_dic['variableMetadata'][i]))
        variable_smile_list.append('{}_no_match'.format(data_dic['variableMetadata'][i]))
        most_common_count_list.append('{}_no_match'.format(data_dic['variableMetadata'][i]))
        candidate_count_list.append('{}_no_match'.format(data_dic['variableMetadata'][i]))
        continue
    name_list = temp['Max_NAME'].values
    smile_list = temp['Max_SMILES'].values
    # smile_list = temp['Smiles'].values
    # name_list = temp['Compound_Name'].values
    collections_names = Counter(name_list)
    collections_smiles = Counter(smile_list)
    most_common_smile_tuple = collections_smiles.most_common(1)
    most_common_name_tuple = collections_names.most_common(1)
    most_common_name = most_common_name_tuple[0][0]
    most_common_smile = most_common_smile_tuple[0][0]
    candidates_count = len(name_list)
    most_common_count = most_common_name_tuple[0][1]

    if type(most_common_name) == float or most_common_name == ' ':
        variable_name_list.append('{}_no_match'.format(data_dic['variableMetadata'][i]))
        variable_smile_list.append('{}_no_match'.format(data_dic['variableMetadata'][i]))
        most_common_count_list.append('{}_no_match'.format(data_dic['variableMetadata'][i]))
        candidate_count_list.append('{}_no_match'.format(data_dic['variableMetadata'][i]))
    else:
        variable_name_list.append(most_common_name)
        variable_smile_list.append(most_common_smile)
        most_common_count_list.append(most_common_count)
        candidate_count_list.append(candidates_count)


data_dic['gnps_name'] = variable_name_list
data_dic['gnps_smile'] = variable_smile_list
data_dic['gnps_most_common_count']=most_common_count_list
data_dic['gnps_candidate_count']=candidate_count_list

print(data_dic)
#
# data_dic.to_excel('ad files/varsPOSout_pos_noid.xlsx',index=False,na_rep=np.nan)

table = table.drop(columns='dataMatrix')
print(table)
# variable_name_list = pd.DataFrame(variable_name_list)
table.insert(0,'dataMatrix',variable_name_list)


print(table)


for i in range(len(table)):
    if 'no_match' in table['dataMatrix'][i]:
        table = table.drop(i)
    elif 'Massbank' in table['dataMatrix'][i]:
        temp = table['dataMatrix'][i].split(' ',1)[1].split('|')[0]
        table['dataMatrix'][i] = temp
    elif ';' in table['dataMatrix'][i]:
        temp = table['dataMatrix'][i].split(';')[0]
        table['dataMatrix'][i] = temp
    elif 'ReSpect' in table['dataMatrix'][i]:
        temp = table['dataMatrix'][i].split(' ',1)[1].split('|')[0]
        table['dataMatrix'][i] = temp
    elif 'HMDB' in table['dataMatrix'][i]:
        temp = table['dataMatrix'][i].split(' ',1)[1].split('|')[0]
        table['dataMatrix'][i] = temp

print(table)


table.to_excel('ad files/peaktablePOSout_POS_noid_replace_variable_gnps.xlsx',index=False,na_rep=np.nan)
# table.to_excel('ad files/peaktableNEGout_NEG_noid_repla7ce_variable.xlsx',index=False,na_rep=np.nan)

data = pd.read_excel('ad files/peaktablePOSout_POS_noid_replace_variable_gnps.xlsx')

targets = data.columns.values[1:]


print(data)


for i in range(len(data)):
    temp = []
    for j in targets:
        temp.append(data[j][i])
    for k in range(len(temp)):
        temp[k] = math.isnan(temp[k])
    if temp.count(True) >= len(temp) /2:
        data = data.drop(i)

print(data)

data.to_excel('ad files/peaktablePOSout_POS_noid_replace_variable_gnps.xlsx',index=False,na_rep=np.nan)