import numpy as np
import pandas as pd



new_data_path = r'C:/Users/csw/Desktop/python/liangzi/data/new_data/'
old_data_path = r'C:/Users/csw/Desktop/python/liangzi/data/old_data/'
concat_data_path = r'C:/Users/csw/Desktop/python/liangzi/data/concat_data/'

new_entbase = pd.read_csv(new_data_path + '1entbase.csv')
new_alter = pd.read_csv(new_data_path + '2alter.csv')
new_branch = pd.read_csv(new_data_path + '3branch.csv')
new_invest = pd.read_csv(new_data_path + '4invest.csv')
new_right = pd.read_csv(new_data_path + '5right.csv')
new_project = pd.read_csv(new_data_path + '6project.csv')
new_lawsuit = pd.read_csv(new_data_path + '7lawsuit.csv')
new_breakfaith = pd.read_csv(new_data_path + '8breakfaith.csv')
new_recruit = pd.read_csv(new_data_path + '9recruit.csv')
new_qualification = pd.read_csv(new_data_path + '10qualification.csv',encoding='gb2312')
new_test = pd.read_csv(new_data_path + 'evaluation_public.csv')
new_train = pd.read_csv(new_data_path + 'train.csv')

old_entbase = pd.read_csv(old_data_path + '1entbase.csv')
old_alter = pd.read_csv(old_data_path + '2alter.csv')
old_branch = pd.read_csv(old_data_path + '3branch.csv')
old_invest = pd.read_csv(old_data_path + '4invest.csv')
old_right = pd.read_csv(old_data_path + '5right.csv')
old_project = pd.read_csv(old_data_path + '6project.csv')
old_lawsuit = pd.read_csv(old_data_path + '7lawsuit.csv')
old_breakfaith = pd.read_csv(old_data_path + '8breakfaith.csv')
old_recruit = pd.read_csv(old_data_path + '9recruit.csv')
old_train = pd.read_csv(old_data_path + 'train.csv')

old_entbase['PROV'] = 12
old_entbase['EID'] = old_entbase['EID'].apply(lambda x: 's'+str(x))
new_entbase['IENUM'] = new_entbase['INUM'] - new_entbase['ENUM']
old_entbase['ENUM'] = -1
old_entbase['IENUM'] = -1
old_alter['EID'] = old_alter['EID'].apply(lambda x: 's'+str(x))
old_branch['EID'] = old_branch['EID'].apply(lambda x: 's'+str(x))
old_branch['TYPECODE'] = old_branch['TYPECODE'].apply(lambda x: 's'+str(x))
old_invest['EID'] = old_invest['EID'].apply(lambda x: 's'+str(x))
old_invest['BTEID'] = old_invest['BTEID'].apply(lambda x: 's'+str(x))
old_right['EID'] = old_right['EID'].apply(lambda x: 's'+str(x))
old_project['EID'] = old_project['EID'].apply(lambda x: 's'+str(x))
old_lawsuit['EID'] = old_lawsuit['EID'].apply(lambda x: 's'+str(x))
old_breakfaith['EID'] = old_breakfaith['EID'].apply(lambda x: 's'+str(x))
old_breakfaith['FBDATE'] = pd.to_datetime(old_breakfaith['FBDATE']).astype(str)
old_recruit['EID'] = old_recruit['EID'].apply(lambda x: 's'+str(x))
old_recruit['POSCODE'] = np.nan
old_train['EID'] = old_train['EID'].apply(lambda x: 's'+str(x))

entbase = pd.concat([new_entbase,old_entbase])
alter = pd.concat([new_alter,old_alter])
branch = pd.concat([new_branch,old_branch])
invest = pd.concat([new_invest,old_invest])
right = pd.concat([new_right,old_right])
project = pd.concat([new_project,old_project])
lawsuit = pd.concat([new_lawsuit,old_lawsuit])
breakfaith = pd.concat([new_breakfaith,old_breakfaith])
recruit = new_recruit
qualification = new_qualification
test = new_test
train = pd.concat([new_train,old_train])

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
recruit = conver2lower(new_recruit)
qualification = conver2lower(qualification)
test = conver2lower(test)
train = conver2lower(train)

def replace(s):
    if s is np.nan:
        return s
    if s == 'null万元':
        return np.nan
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
breakfaith['fbdate'] = breakfaith['fbdate'].apply(lambda x: x.replace('年','-').replace('月',''))
breakfaith['sxenddate'] = breakfaith['sxenddate'].apply(lambda x: x.replace('年','-').replace('月','') if type(x) is str else x)
lawsuit['lawdate'] = lawsuit['lawdate'].apply(lambda x: x.replace('年','-').replace('月',''))
recruit['pnum'] = recruit['pnum'].apply(lambda x: x.replace('若干','').replace('人','') if type(x) is str else x)
train.rename(columns={'target':'label'},inplace=True)

print('覆盖原来数据')
entbase.to_csv(concat_data_path + '1entbase.csv',index=False,encoding='utf-8')
alter.to_csv(concat_data_path + '2alter.csv',index=False,encoding='utf-8')
branch.to_csv(concat_data_path + '3branch.csv',index=False,encoding='utf-8')
invest.to_csv(concat_data_path + '4invest.csv',index=False,encoding='utf-8')
right.to_csv(concat_data_path + '5right.csv',index=False,encoding='utf-8')
project.to_csv(concat_data_path + '6project.csv',index=False,encoding='utf-8')
lawsuit.to_csv(concat_data_path + '7lawsuit.csv',index=False,encoding='utf-8')
breakfaith.to_csv(concat_data_path + '8breakfaith.csv',index=False,encoding='utf-8')
recruit.to_csv(concat_data_path + '9recruit.csv',index=False,encoding='utf-8')
qualification.to_csv(concat_data_path + '10qualification.csv',index=False,encoding='utf-8')
test.to_csv(concat_data_path + 'evaluation_public.csv',index=False,encoding='utf-8')
train.to_csv(concat_data_path + 'train.csv',index=False,encoding='utf-8')












