import pandas as pd

ours = pd.read_excel('ad files/peaktablePOSout_POS_noid_replace_variable_ours.xlsx')
gnps = pd.read_excel('ad files/peaktablePOSout_POS_noid_replace_variable_gnps.xlsx')


for i in range(len(gnps)):
    if ('no_match' not in ours['dataMatrix'][i]) and ('no_match' not in gnps['dataMatrix'][i]):
        if ours['dataMatrix'][i]==gnps['dataMatrix'][i]:
            print(ours['dataMatrix'][i],gnps['dataMatrix'][i],ours['dataMatrix'][i]==gnps['dataMatrix'][i])

