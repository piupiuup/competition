import numpy as np
import pandas as pd



data_path = r'C:/Users/csw/Desktop/python/liangzi/data/'


entbase = pd.read_csv(data_path + '1entbase.csv')
alter = pd.read_csv(data_path + '2alter.csv')
branch = pd.read_csv(data_path + '3branch.csv')
invest = pd.read_csv(data_path + '4invest.csv')
right = pd.read_csv(data_path + '5right.csv')
project = pd.read_csv(data_path + '6project.csv')
lawsuit = pd.read_csv(data_path + '7lawsuit.csv')
breakfaith = pd.read_csv(data_path + '8breakfaith.csv')
recruit = pd.read_csv(data_path + '9recruit.csv')
qualification = pd.read_csv(data_path + '10qualification.csv',encoding='GB2312')
test = pd.read_csv(data_path + 'evaluation_public.csv')
train = pd.read_csv(data_path + 'train.csv')

print('将feature name转换为小写')
def conver2lower(data):
    new_columns = []
    for name in data.columns:
        new_columns.append(name.lower())
    data.columns = new_columns
    data.rename(columns={'eid': 'id'}, inplace=True)
    return data

entbase = conver2lower(entbase)
alter = conver2lower(alter)
branch = conver2lower(branch)
invest = conver2lower(invest)
right = conver2lower(right)
project = conver2lower(project)
lawsuit = conver2lower(lawsuit)
breakfaith = conver2lower(breakfaith)
recruit = conver2lower(recruit)
qualification = conver2lower(qualification)
test = conver2lower(test)
train = conver2lower(train)

def replace(s):
    if s is np.nan:
        return s
    if '美元' in s:
        return float(s.replace('美元', '').replace('万元', '').replace('万', '')) * 6.5
    if '港' in s:
        return float(s.replace('港', '').replace('币', '').replace('万元', '').replace('万', '')) * 0.85

    return float(s.replace('万元','').replace('人民币','').replace('万', '').replace('(单位：)', ''))
def get_area(s):
    if '美元' in s:
        return 2
    if '港币' in s:
        return 1
    return 0

print('数据清洗...')
alter['altbe'] = alter['altbe'].apply(replace)
alter['altaf'] = alter['altaf'].apply(replace)
alter['alterno'].replace('A_015','15',inplace=True)
qualification['begindate'] = qualification['begindate'].apply(lambda x: x.replace('年','-').replace('月',''))
qualification['expirydate'] = qualification['expirydate'].apply(lambda x: x.replace('年','-').replace('月','') if type(x) is str else x)
breakfaith['fbdate'] = breakfaith['fbdate'].apply(lambda x: x.replace('年','-').replace('月',''))
breakfaith['sxenddate'] = breakfaith['sxenddate'].apply(lambda x: x.replace('年','-').replace('月','') if type(x) is str else x)
lawsuit['lawdate'] = lawsuit['lawdate'].apply(lambda x: x.replace('年','-').replace('月',''))
recruit['pnum'] = recruit['pnum'].apply(lambda x: x.replace('若干','').replace('人','') if type(x) is str else x)
train.rename(columns={'target':'label'},inplace=True)

print('覆盖原来数据')
entbase.to_csv(data_path + '1entbase.csv',index=False,encoding='utf-8')
alter.to_csv(data_path + '2alter.csv',index=False,encoding='utf-8')
branch.to_csv(data_path + '3branch.csv',index=False,encoding='utf-8')
invest.to_csv(data_path + '4invest.csv',index=False,encoding='utf-8')
right.to_csv(data_path + '5right.csv',index=False,encoding='utf-8')
project.to_csv(data_path + '6project.csv',index=False,encoding='utf-8')
lawsuit.to_csv(data_path + '7lawsuit.csv',index=False,encoding='utf-8')
breakfaith.to_csv(data_path + '8breakfaith.csv',index=False,encoding='utf-8')
recruit.to_csv(data_path + '9recruit.csv',index=False,encoding='utf-8')
qualification.to_csv(data_path + '10qualification.csv',index=False,encoding='utf-8')
test.to_csv(data_path + 'evaluation_public.csv',index=False,encoding='utf-8')
train.to_csv(data_path + 'train.csv',index=False,encoding='utf-8')
