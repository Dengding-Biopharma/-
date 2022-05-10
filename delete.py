import pandas as pd

data = pd.read_excel('pos_significant.xlsx')
print(data)

dup_row = data.duplicated(subset=['name'],keep=False)
print(dup_row)

dup_df = data[dup_row]
for i in range(len(data)):
    if dup_row[i] == True:
        data = data.drop(i)
print(data)
dup_df=dup_df.sort_values(by='name')

print(dup_df)

dup_df.to_excel('temp.xlsx',index=False)
dup_df = pd.read_excel('temp.xlsx')
print(dup_df)
temp = []
for i in range(len(dup_df)):
    if len(temp) == 0 or dup_df['name'][i] != temp[0]:
        temp = [dup_df['name'][i], dup_df['P'][i],i]
    elif dup_df['name'][i] == temp[0] and dup_df['P'][i] < temp[1]:
        dup_df = dup_df.drop(temp[2])
        temp =[dup_df['name'][i],dup_df['P'][i],i]
    elif dup_df['name'][i] == temp[0] and dup_df['P'][i] > temp[1]:
        dup_df = dup_df.drop(i)

print(len(dup_df))
finalDF = pd.concat([dup_df,data])
print(finalDF)
finalDF.to_excel('pos_significant.xlsx',index=False)


