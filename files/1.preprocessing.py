import math
import random
from collections import Counter

import numpy as np
import pandas as pd


def preprocessing(peak_filename,database_name,exp_filename,hmdb_first=False,match_from_gnps=False,match_from_hmdb=False,hmdb_filename=None,gnps_file=None,mode=None):
    peak_file = pd.read_excel(peak_filename)
    database = pd.read_excel(database_name)
    exp_file = pd.read_excel(exp_filename)

    # 如果score0.7以上要用hmdb的数据
    if hmdb_first:
        print('hmdb first!!!!!!!!!!')
        try:
            database_name = database_name[:-5]+'_hmdb.xlsx'
            database = pd.read_excel(database_name)
            print('using cached hmdb-first file!')
        except:
            print('no cached hmdb database! generating!')
            for i in range(len(database)):
                if database['HMDB_Score'][i] >= 0.7:
                    database['Max_NAME'][i] = database['HMDB_Name'][i]
                    database['Max_pepmass'][i] = database['HMDB_pepmass'][i]
                    database['Max_SMILES'][i] = database['HMDB_SMILES'][i]
                    database['Max_Score'][i] = database['HMDB_Score'][i]
                    database['Max_Source'][i] = 'HMDB'
                    database['Max_INCHI'][i] = database['HMDB_INCHI'][i]
            database.to_excel(database_name,index=False,na_rep=np.nan)


    variable_name_list = []
    variable_smile_list = []
    most_common_count_list = []
    candidate_count_list = []

    for i in range(len(exp_file)):
        mzmin = exp_file['xcmsCamera_mzmin'][i] - 0.05
        mzmax = exp_file['xcmsCamera_mzmax'][i] + 0.05
        rtmin = exp_file['xcmsCamera_rtmin'][i]
        rtmax = exp_file['xcmsCamera_rtmax'][i]
        bool_mz = database['Exp_pepMass'].between(mzmin,mzmax,inclusive=True)
        temp = database[bool_mz]
        if temp.empty:
            variable_name_list.append('{}_no_match'.format(exp_file['variableMetadata'][i]))
            variable_smile_list.append('{}_no_match'.format(exp_file['variableMetadata'][i]))
            most_common_count_list.append('{}_no_match'.format(exp_file['variableMetadata'][i]))
            candidate_count_list.append('{}_no_match'.format(exp_file['variableMetadata'][i]))
            continue
        bool_rt = temp['Exp_RT'].between(rtmin, rtmax, inclusive=True)
        temp = temp[bool_rt]
        if temp.empty:
            variable_name_list.append('{}_no_match'.format(exp_file['variableMetadata'][i]))
            variable_smile_list.append('{}_no_match'.format(exp_file['variableMetadata'][i]))
            most_common_count_list.append('{}_no_match'.format(exp_file['variableMetadata'][i]))
            candidate_count_list.append('{}_no_match'.format(exp_file['variableMetadata'][i]))
            continue
        name_list = temp['Max_NAME'].values
        smile_list = temp['Max_SMILES'].values
        collections_names = Counter(name_list)
        collections_smiles = Counter(smile_list)
        most_common_smile_tuple = collections_smiles.most_common(1)
        most_common_name_tuple = collections_names.most_common(1)
        most_common_name = most_common_name_tuple[0][0]
        most_common_smile = most_common_smile_tuple[0][0]

        candidates_count = len(name_list)
        most_common_count = most_common_name_tuple[0][1]

        if type(most_common_name) == float or most_common_name == ' ':
            variable_name_list.append('{}_no_match'.format(exp_file['variableMetadata'][i]))
            variable_smile_list.append('{}_no_match'.format(exp_file['variableMetadata'][i]))
            most_common_count_list.append('{}_no_match'.format(exp_file['variableMetadata'][i]))
            candidate_count_list.append('{}_no_match'.format(exp_file['variableMetadata'][i]))
        else:
            variable_name_list.append(most_common_name)
            variable_smile_list.append(most_common_smile)
            most_common_count_list.append(most_common_count)
            candidate_count_list.append(candidates_count)

    # 如果要用gnps的数据库
    if match_from_gnps:
        variable_name_list_gnps = []
        variable_smile_list_gnps = []
        most_common_count_list_gnps = []
        candidate_count_list_gnps = []
        print('match more from gnps!!!!')
        try:
            gnps_database = pd.read_excel(gnps_file)
        except:
            print('Please check the gnps database location!')
            quit()
        for i in range(len(exp_file)):
            mzmin = exp_file['xcmsCamera_mzmin'][i] - 0.05
            mzmax = exp_file['xcmsCamera_mzmax'][i] + 0.05
            rtmin = exp_file['xcmsCamera_rtmin'][i]
            rtmax = exp_file['xcmsCamera_rtmax'][i]
            bool_mz = gnps_database['SpecMZ'].between(mzmin, mzmax, inclusive=True)
            temp = gnps_database[bool_mz]
            if temp.empty:
                variable_name_list_gnps.append('{}_no_match'.format(exp_file['variableMetadata'][i]))
                variable_smile_list_gnps.append('{}_no_match'.format(exp_file['variableMetadata'][i]))
                most_common_count_list_gnps.append('{}_no_match'.format(exp_file['variableMetadata'][i]))
                candidate_count_list_gnps.append('{}_no_match'.format(exp_file['variableMetadata'][i]))
                continue
            bool_rt = temp['RT_Query'].between(rtmin, rtmax, inclusive=True)
            temp = temp[bool_rt]
            if temp.empty:
                variable_name_list_gnps.append('{}_no_match'.format(exp_file['variableMetadata'][i]))
                variable_smile_list_gnps.append('{}_no_match'.format(exp_file['variableMetadata'][i]))
                most_common_count_list_gnps.append('{}_no_match'.format(exp_file['variableMetadata'][i]))
                candidate_count_list_gnps.append('{}_no_match'.format(exp_file['variableMetadata'][i]))
                continue
            smile_list = temp['Smiles'].values
            name_list = temp['Compound_Name'].values
            collections_names = Counter(name_list)
            collections_smiles = Counter(smile_list)
            most_common_smile_tuple = collections_smiles.most_common(1)
            most_common_name_tuple = collections_names.most_common(1)
            most_common_name = most_common_name_tuple[0][0]
            most_common_smile = most_common_smile_tuple[0][0]

            candidates_count = len(name_list)
            most_common_count = most_common_name_tuple[0][1]

            if type(most_common_name) == float or most_common_name == ' ':
                variable_name_list_gnps.append('{}_no_match'.format(exp_file['variableMetadata'][i]))
                variable_smile_list_gnps.append('{}_no_match'.format(exp_file['variableMetadata'][i]))
                most_common_count_list_gnps.append('{}_no_match'.format(exp_file['variableMetadata'][i]))
                candidate_count_list_gnps.append('{}_no_match'.format(exp_file['variableMetadata'][i]))
            else:
                variable_name_list_gnps.append(most_common_name)
                variable_smile_list_gnps.append(most_common_smile)
                most_common_count_list_gnps.append(most_common_count)
                candidate_count_list_gnps.append(candidates_count)

        exp_file['ours_name'] = variable_name_list
        exp_file['ours_smile'] = variable_smile_list
        exp_file['ours_most_common_count']=most_common_count_list
        exp_file['ours_candidate_count']=candidate_count_list

        exp_file['gnps_name'] = variable_name_list_gnps
        exp_file['gnps_smile'] = variable_smile_list_gnps
        exp_file['gnps_most_common_count']=most_common_count_list_gnps
        exp_file['gnps_candidate_count']=candidate_count_list_gnps


        variable_name_list = []
        variable_smile_list = []
        for i in range(len(exp_file)):
            if 'no_match' not in exp_file['ours_name'][i] and 'no_match' not in exp_file['gnps_name'][i]:
                if exp_file['ours_most_common_count'][i] >= exp_file['gnps_most_common_count'][i]:
                    variable_name_list.append(exp_file['ours_name'][i])
                    variable_smile_list.append(exp_file['ours_smile'][i])
                else:
                    variable_name_list.append(exp_file['gnps_name'][i])
                    variable_smile_list.append(exp_file['gnps_smile'][i])
                continue
            if 'no_match' not in exp_file['ours_name'][i]:
                variable_name_list.append(exp_file['ours_name'][i])
                variable_smile_list.append(exp_file['ours_smile'][i])
                continue
            if 'no_match' not in exp_file['gnps_name'][i]:
                variable_name_list.append(exp_file['gnps_name'][i])
                variable_smile_list.append(exp_file['gnps_smile'][i])
                continue
            if 'no_match' in exp_file['ours_name'][i] and 'no_match' in exp_file['gnps_name'][i]:
                variable_name_list.append(exp_file['ours_name'][i])
                variable_smile_list.append(exp_file['ours_smile'][i])
                continue

        exp_file['max_name'] = variable_name_list
        exp_file['max_smile'] = variable_smile_list
        count = 0
        for i in range(len(exp_file)):
            if 'no_match' not in exp_file['max_name'][i]:
                count += 1

        exp_file.to_excel(exp_filename[:-5]+'_more_from_gnps.xlsx', index=False, na_rep=np.nan)
    else:
        exp_file['ours_name'] = variable_name_list
        exp_file['ours_smile'] = variable_smile_list
        exp_file['ours_most_common_count'] = most_common_count_list
        exp_file['ours_candidate_count'] = candidate_count_list
        exp_file['max_name'] = exp_file['ours_name']
        exp_file['max_smile'] = exp_file['ours_smile']
        exp_file.to_excel(exp_filename[:-5] + '_ours.xlsx', index=False, na_rep=np.nan)


    if match_from_hmdb:
        hmdb_file = pd.read_csv(hmdb_filename,delimiter='\t')
        for i in range(len(exp_file)):
            if 'no_match' in exp_file['max_name'][i]:
                if mode == 'pos':
                    mz = exp_file['xcmsCamera_mz'][i] - 1.0032  # POS mode
                if mode == 'neg':
                    mz = exp_file['xcmsCamera_mz'][i] + 1.0032  # NEG mode
                mzmin = mz - ((15 / 1000000) * mz)
                mzmax = mz + ((15 / 1000000) * mz)
                bool_mz = hmdb_file['monisotopic_molecular_weight'].between(mzmin, mzmax, inclusive=True)
                temp = hmdb_file[bool_mz]
                if temp.empty:
                    continue
                exp_file['max_name'][i] = str(random.choice(temp['name'].values))[2:-1]
                exp_file['max_smile'][i] = random.choice(temp['smiles'].values)
        if match_from_gnps:
            exp_file.to_excel(exp_filename[:-5] + '_more_from_gnps_and_hmdb.xlsx', index=False, na_rep=np.nan)
        else:
            exp_file.to_excel(exp_filename[:-5] + '_more_from_hmdb.xlsx', index=False, na_rep=np.nan)
    peak_file = peak_file.drop(columns='dataMatrix')
    peak_file.insert(0, 'dataMatrix', exp_file['max_name'])
    peak_file.insert(1, 'smile', exp_file['max_smile'])

    for i in range(len(peak_file)):
        if 'no_match' in peak_file['dataMatrix'][i]:
            peak_file = peak_file.drop(i)
        elif 'Massbank' in peak_file['dataMatrix'][i]:
            temp = peak_file['dataMatrix'][i].split(' ', 1)[1].split('|')[0]
            peak_file['dataMatrix'][i] = temp
        elif ';' in peak_file['dataMatrix'][i]:
            temp = peak_file['dataMatrix'][i].split(';')[0]
            peak_file['dataMatrix'][i] = temp
        elif 'ReSpect' in peak_file['dataMatrix'][i]:
            temp = peak_file['dataMatrix'][i].split(' ', 1)[1].split('|')[0]
            peak_file['dataMatrix'][i] = temp
        elif 'HMDB' in peak_file['dataMatrix'][i]:
            temp = peak_file['dataMatrix'][i].split(' ', 1)[1].split('|')[0]
            peak_file['dataMatrix'][i] = temp
        elif 'Spectral Match to' in peak_file['dataMatrix'][i]:
            temp = peak_file['dataMatrix'][i].split('Spectral Match to ', 1)[1]
            peak_file['dataMatrix'][i] = temp

    peak_file = peak_file.reset_index(drop=True)
    print(peak_file)
    targets = peak_file.columns.values[2:]

    for i in range(len(peak_file)):
        temp = []
        for j in targets:
            temp.append(peak_file[j][i])
        for k in range(len(temp)):
            temp[k] = math.isnan(temp[k])
        if temp.count(True) > len(temp) / 2:
            peak_file = peak_file.drop(i)

    for i in range(len(targets)):
        if '0316' in targets[i]:
            del peak_file[targets[i]]

    peak_file.to_excel(peak_filename[:-5]+'_replace.xlsx',index=False,na_rep=np.nan)
    print(peak_file)

    # 均值填补缺失值
    targets = peak_file.columns.values

    saved_label = peak_file['dataMatrix'].values
    saved_smile = peak_file['smile'].values
    ad_index = []
    hc_index = []
    for i in range(len(targets)):
        if "AD" in targets[i]:
            ad_index.append(i)
        elif 'HC' in targets[i]:
            hc_index.append(i)
    print(ad_index)
    print(hc_index)

    data_array = []
    for i in range(len(peak_file)):
        ad = []
        for j in ad_index:
            ad.append(peak_file.values[i][j])
        temp = []
        for num in ad:
            if np.isnan(num):
                continue
            temp.append(num)
        ad_mean = np.mean(temp)
        for m in range(len(ad)):
            if np.isnan(ad[m]):
                ad[m] = ad_mean

        hc = []
        for k in hc_index:
            hc.append(peak_file.values[i][k])
        temp = []
        for num in hc:
            if np.isnan(num):
                continue
            temp.append(num)
        hc_mean = np.mean(temp)
        for n in range(len(hc)):
            if np.isnan(hc[n]):
                hc[n] = hc_mean
        row = [saved_label[i],saved_smile[i]] + ad + hc
        data_array.append(row)

    df = pd.DataFrame(data_array, columns=targets)

    df.to_excel(peak_filename[:-5]+'_replace_mean_full.xlsx', index=False, na_rep=np.nan)



if __name__ == '__main__':
    # pos files
    peak_filename = 'ad files/peaktablePOSout_POS_noid.xlsx'
    database = 'ad files/Pos-summary-0313-16.xlsx'
    exp_filename = 'ad files/varsPOSout_pos_noid.xlsx'
    gnps_file = 'ad files/Pos_analyzed.xlsx'
    mode = 'pos'

    # neg files
    # peak_filename = 'ad files/peaktableNEGout_NEG_noid.xlsx'
    # database = 'ad files/Neg-summary-0313-16.xlsx'
    # exp_filename = 'ad files/varsNEGout_neg_noid.xlsx'
    # gnps_file = 'ad files/Neg_analyzed.xlsx'
    # mode = 'neg'

    # static setting
    hmdb_filename = 'hmdb_metabolites.csv'
    hmdb_first = True
    match_from_gnps=True
    match_from_hmdb=True
    preprocessing(peak_filename,database,exp_filename,hmdb_first=hmdb_first,match_from_gnps=match_from_gnps,match_from_hmdb=match_from_hmdb,hmdb_filename=hmdb_filename,gnps_file=gnps_file,mode=mode)