import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,auc
from sklearn.model_selection import train_test_split


import numpy as np
data_path = r'C:/Users/csw/Desktop/python/liangzi/data/other/'
entbase_path = data_path + '1entbase.csv'
alter_path = data_path + '2alter.csv'
branch_path = data_path + '3branch.csv'
invest_path = data_path + '4invest.csv'
right_path = data_path + '5right.csv'
project_path = data_path + '6project.csv'
lawsuit_path = data_path + '7lawsuit.csv'
breakfaith_path = data_path + '8breakfaith.csv'
recruit_path = data_path + '9recruit.csv'
test_path = data_path + 'evaluation_public.csv'
train_path = data_path + 'train.csv'

# 提取数据中的数值
def get_number(x):
    try:
        return float(x.replace('万元', ''))
    except:
        return np.nan

# 根目录
dir = '../public/'

print('reading train datas')
train = pd.read_csv(train_path)
# 训练集中标签为1的EID
pos_train = list(train[train['TARGET']==1]['EID'].unique())
# print(train.head())
print('train shape',train.shape)
org_train_shape = train.shape
print('positive sample',train[train.TARGET == 1].__len__())
print('positive ration',train[train.TARGET == 1].__len__() * 1.0/ len(train))
print('reading test datas')
test = pd.read_csv(test_path)
# print(test.head())
print('test shape',test.shape)
org_test_shape = test.shape
# 全部的企业EID
all_eid_number = len(set(list(test['EID'].unique()) + list(train['EID'].unique())))
print('all EID number ',all_eid_number)

# 获取企业基本信息表
print('reading 1.entbase')
entbase = pd.read_csv(entbase_path)
# 题目要求是用0填充，因此对nan进行填充
print("entbase fill nan in 0")
entbase = entbase.fillna(0)
print('entbase shape',entbase.shape)

print('reading 2alter')
alter = pd.read_csv(alter_path)
print('aleter shape',alter.shape)
print('alter in EID number ratio',len(alter['EID'].unique())*1.0 / all_eid_number)
alter = alter.fillna(0)
print('ALTERNO to cateary')
ALTERNO_to_index = list(alter['ALTERNO'].unique())
# 1 2 有金钱变化
alter['ALTERNO'] = alter['ALTERNO'].map(ALTERNO_to_index.index)
alter['ALTAF'] = np.log1p(alter['ALTAF'].map(get_number))
alter['ALTBE'] = np.log1p(alter['ALTBE'].map(get_number))
alter['ALTAF_ALTBE'] = alter['ALTAF'] - alter['ALTBE']

alter['ALTDATE_YEAR'] = alter['ALTDATE'].map(lambda x:x.split('-')[0])
alter['ALTDATE_YEAR'] = alter['ALTDATE_YEAR'].astype(int)
alter['ALTDATE_MONTH'] = alter['ALTDATE'].map(lambda x:x.split('-')[1])
alter['ALTDATE_MONTH'] = alter['ALTDATE_MONTH'].astype(int)

alter = alter.sort_values(['ALTDATE_YEAR','ALTDATE_MONTH'],ascending=True)
# 标签化 ALTERNO
alter_ALTERNO = pd.get_dummies(alter['ALTERNO'],prefix='ALTERNO')
alter_ALTERNO_merge = pd.concat([alter['EID'],alter_ALTERNO],axis=1)
alter_ALTERNO_info_sum = alter_ALTERNO_merge.groupby(['EID'],as_index=False).sum()
alter_ALTERNO_info_ration = alter_ALTERNO_merge.groupby(['EID']).sum() / alter_ALTERNO_merge.groupby(['EID']).count()
alter_ALTERNO_info_ration = alter_ALTERNO_info_ration.reset_index()
# 变更的第一年
alter_first_year = alter[['EID','ALTDATE_YEAR']].drop_duplicates(['EID'])
# 变更的最后一年
alter_last_year = alter[['EID','ALTDATE_YEAR']].sort_values(['ALTDATE_YEAR'],ascending=False).drop_duplicates(['EID'])

alter_ALTERNO_info = pd.merge(alter_ALTERNO_info_sum,alter[['ALTAF_ALTBE','EID']],on=['EID']).drop_duplicates(['EID'])
alter_ALTERNO_info = pd.merge(alter_ALTERNO_info,alter_last_year,on=['EID'])
alter_ALTERNO_info = alter_ALTERNO_info.fillna(-1)

print('reading 3branch.csv')
branch = pd.read_csv(branch_path)
branch_copy = branch.copy()
# print branch

branch['B_ENDYEAR'] = branch['B_ENDYEAR'].fillna(branch['B_REYEAR'])
branch['sub_life'] = branch['B_ENDYEAR'].fillna(branch['B_REYEAR']) - branch['B_REYEAR']
# 筛选数据
branch = branch[branch['sub_life']>=0]
branch_count = branch.groupby(['EID'],as_index=False)['TYPECODE'].count()
branch_count.rename(columns = {'TYPECODE':'branch_count'},inplace=True)
branch = pd.merge(branch,branch_count,on=['EID'],how='left')
# branch['branch_count'] = np.log1p(branch['branch_count'])
# branch['branch_count'] = branch['branch_count'].astype(int)
# branch['sub_life'] = branch['sub_life'].replace({0.0:-1})

home_prob = branch.groupby(by=['EID'])['IFHOME'].sum()/ branch.groupby(by=['EID'])['IFHOME'].count()
home_prob = home_prob.reset_index()
branch = pd.DataFrame(branch[['EID','sub_life']]).drop_duplicates('EID')
branch = pd.merge(branch,home_prob,on=['EID'],how='left')

print('reading 4invest.csv')
invest = pd.read_csv(invest_path)
invest['BTENDYEAR'] = invest['BTENDYEAR'].fillna(invest['BTYEAR'])
invest['invest_life'] = invest['BTENDYEAR'] - invest['BTYEAR']
invest_BTBL_sum = invest.groupby(['EID'],as_index=False)['BTBL'].sum()
invest_BTBL_sum.rename(columns={'BTBL':'BTBL_SUM'},inplace=True)
invest_BTBL_count = invest.groupby(['EID'],as_index=False)['BTBL'].count()
invest_BTBL_sum.rename(columns={'BTBL':'BTBL_COUNT'},inplace=True)
BTBL_INFO = pd.merge(invest_BTBL_sum,invest_BTBL_count,on=['EID'],how='left')
BTBL_INFO['BTBL_RATIO'] = BTBL_INFO['BTBL_SUM'] / BTBL_INFO['BTBL']
invest['invest_life'] = invest['invest_life'] > 0
invest['invest_life'] = invest['invest_life'].astype(int)
invest_life_ratio = invest.groupby(['EID'])['invest_life'].sum() / invest.groupby(['EID'])['invest_life'].count()
invest_life_ratio = invest_life_ratio.reset_index()
invest_life_ratio.rename(columns={'invest_life':'invest_life_ratio'},inplace=True)
invest_last_year = invest.sort_values('BTYEAR',ascending=False).drop_duplicates('EID')[['EID','BTYEAR']]
invest_first_year = invest.sort_values('BTYEAR').drop_duplicates('EID')[['EID','BTYEAR']]


invest = pd.merge(invest[['EID']],BTBL_INFO,on=['EID'],how='left').drop_duplicates(['EID'])
invest = pd.merge(invest,invest_life_ratio,on=['EID'],how='left')
invest = pd.merge(invest,invest_last_year,on=['EID'],how='left')



print('reading 5right.csv')
right = pd.read_csv(right_path)
right_RIGHTTYPE = pd.get_dummies(right['RIGHTTYPE'],prefix='RIGHTTYPE')
right_RIGHTTYPE_info = pd.concat([right['EID'],right_RIGHTTYPE],axis=1)
right_RIGHTTYPE_info_sum = right_RIGHTTYPE_info.groupby(['EID'],as_index=False).sum().drop_duplicates(['EID'])
right['ASKDATE_Y'] = right['ASKDATE'].map(lambda x:x.split('-')[0])
right_last_year = right.sort_values('ASKDATE_Y',ascending=False).drop_duplicates('EID')[['EID','ASKDATE_Y']]
right_last_year.rename(columns={'ASKDATE_Y':'right_last_year'},inplace=True)

right_count = right.groupby(['EID'],as_index=False)['RIGHTTYPE'].count()

right_count.rename(columns={'RIGHTTYPE':'right_count'},inplace=True)
right = pd.merge(right[['EID']],right_RIGHTTYPE_info_sum,on=['EID'],how='left').drop_duplicates(['EID'])
right = pd.merge(right,right_last_year,on=['EID'],how='left')
right = pd.merge(right,right_count,on=['EID'],how='left')

# print right

print('reading 6project.csv')
project = pd.read_csv(project_path)
project['DJDATE_Y'] = project['DJDATE'].map(lambda x:x.split('-')[0])
project_DJDATE_Y = pd.get_dummies(project['DJDATE_Y'],prefix='DJDATE')
project_DJDATE_Y_info = pd.concat([project['EID'],project_DJDATE_Y],axis=1)
project_DJDATE_Y_info_sum = project_DJDATE_Y_info.groupby(['EID'],as_index=False).sum()
project_DJDATE_Y_info_sum = project_DJDATE_Y_info_sum.drop_duplicates(['EID'])
project_count = project.groupby(['EID'],as_index=False)['DJDATE'].count()
project_count.rename(columns={'DJDATE':'project_count'},inplace=True)

project_last_year = project.sort_values('DJDATE_Y',ascending=False).drop_duplicates('EID')[['EID','DJDATE_Y']]

project = pd.merge(project[['EID']],project_last_year,on=['EID'],how='left').drop_duplicates(['EID'])
project = pd.merge(project,project_DJDATE_Y_info_sum,on=['EID'],how='left')

print('reading 7lawsuit.csv')

lawsuit = pd.read_csv(lawsuit_path)
lawsuit_LAWAMOUNT_sum = lawsuit.groupby(['EID'],as_index=False)['LAWAMOUNT'].sum()
lawsuit_LAWAMOUNT_sum.rename(columns={'LAWAMOUNT':'lawsuit_LAWAMOUNT_sum'},inplace=True)
lawsuit_LAWAMOUNT_sum['lawsuit_LAWAMOUNT_sum'] = np.log1p(lawsuit_LAWAMOUNT_sum['lawsuit_LAWAMOUNT_sum'])
lawsuit_LAWAMOUNT_sum['lawsuit_LAWAMOUNT_sum'] = lawsuit_LAWAMOUNT_sum['lawsuit_LAWAMOUNT_sum'].astype(int)
lawsuit_LAWAMOUNT_count = lawsuit.groupby(['EID'],as_index=False)['LAWAMOUNT'].count()
lawsuit_LAWAMOUNT_count.rename(columns={'LAWAMOUNT':'lawsuit_LAWAMOUNT_count'},inplace=True)
lawsuit['LAWDATE_Y'] = lawsuit['LAWDATE'].map(lambda x:x.split('-')[0])
lawsuit_last_year = lawsuit.sort_values('LAWDATE_Y',ascending=False).drop_duplicates('EID')[['EID','LAWDATE_Y']]

print('reading 8breakfaith.csv')
breakfaith = pd.read_csv(breakfaith_path)

breakfaith['FBDATE_Y'] = breakfaith['FBDATE'].map(lambda x:x.split('/')[0])
breakfaith_first_year = breakfaith.sort_values('FBDATE_Y').drop_duplicates('EID')[['EID','FBDATE_Y']]

breakfaith['SXENDDATE'] = breakfaith['SXENDDATE'].fillna(0)
breakfaith['is_breakfaith'] = breakfaith['SXENDDATE']!=0
breakfaith['is_breakfaith'] = breakfaith['is_breakfaith'].astype(int)

breakfaith_is_count = breakfaith.groupby(['EID'],as_index=False)['is_breakfaith'].count()
breakfaith_is_sum = breakfaith.groupby(['EID'],as_index=False)['is_breakfaith'].sum()

breakfaith_is_count.rename(columns={'is_breakfaith':'breakfaith_is_count'},inplace=True)
breakfaith_is_sum.rename(columns={'is_breakfaith':'breakfaith_is_sum'},inplace=True)
breakfaith_is_info = pd.merge(breakfaith_is_count,breakfaith_is_sum,on=['EID'],how='left')
breakfaith_is_info['ratio'] = breakfaith_is_info['breakfaith_is_sum'] / breakfaith_is_info['breakfaith_is_count']
print('reading 9recruit.csv')
recruit = pd.read_csv(recruit_path)

recruit['RECDATE_Y'] = recruit['RECDATE'].map(lambda x:x.split('-')[0])
recruit_train_last_year = recruit.sort_values('RECDATE_Y',ascending=False).drop_duplicates('EID')[['EID','RECDATE_Y']]
recruit_WZCODE = pd.get_dummies(recruit['WZCODE'],prefix='WZCODE')
recruit_WZCODE_merge = pd.concat([recruit['EID'],recruit_WZCODE],axis=1)
# 1
recruit_WZCODE_info_sum = recruit_WZCODE_merge.groupby(['EID'],as_index=False).sum().drop_duplicates(['EID'])
# 2
recruit['RECRNUM'] = recruit['RECRNUM'].fillna(0)
recruit_RECRNUM_count = recruit.groupby(['EID'],as_index=False)['RECRNUM'].count()
recruit_RECRNUM_count.rename(columns={'RECRNUM':'recruit_RECRNUM_count'},inplace=True)
# 3
recruit_RECRNUM_sum = recruit.groupby(['EID'],as_index=False)['RECRNUM'].sum()
recruit_RECRNUM_sum.rename(columns={'RECRNUM':'recruit_RECRNUM_sum'},inplace=True)
recruit_RECRNUM_sum['recruit_RECRNUM_sum'] = recruit_RECRNUM_sum['recruit_RECRNUM_sum']
# 4
recruit_RECRNUM_info = pd.merge(recruit[['EID']],recruit_RECRNUM_sum,on=['EID']).drop_duplicates(['EID'])
recruit_RECRNUM_info = pd.merge(recruit_RECRNUM_info,recruit_RECRNUM_count,on=['EID'])
recruit_RECRNUM_info['recurt_info_ration'] = recruit_RECRNUM_info['recruit_RECRNUM_sum'] / recruit_RECRNUM_info['recruit_RECRNUM_count']

print(recruit.head())

print('merge train/test')
train = pd.merge(train,entbase,on=['EID'],how='left')
# 根据注册资本简单筛选样本
print(train.shape)
print('select sample to train set...')
print(train.shape)
train = pd.merge(train,alter_ALTERNO_info,on=['EID'],how='left')
# train = pd.merge(train,branch,on=['EID'],how='left')
# train = pd.merge(train,right,on=['EID'],how='left')
# train = pd.merge(train,invest,on=['EID'],how='left')
# train = pd.merge(train,project,on=['EID'],how='left')

# train = pd.merge(train,lawsuit_LAWAMOUNT_count,on=['EID'],how='left')
# train = pd.merge(train,lawsuit_last_year,on=['EID'],how='left')
# train = pd.merge(train,breakfaith_is_info,on=['EID'],how='left')
# train = pd.merge(train,recruit_WZCODE_info_sum,on=['EID'],how='left')
# train = pd.merge(train,recruit_RECRNUM_info,on=['EID'],how='left')


test = pd.merge(test,entbase,on=['EID'],how='left')
test = pd.merge(test,alter_ALTERNO_info,on=['EID'],how='left')
# test = pd.merge(test,branch,on=['EID'],how='left')
# test = pd.merge(test,right,on=['EID'],how='left')
# test = pd.merge(test,invest,on=['EID'],how='left')
# test = pd.merge(test,project,on=['EID'],how='left')
#
# test = pd.merge(test,lawsuit_LAWAMOUNT_count,on=['EID'],how='left')
# test = pd.merge(test,lawsuit_last_year,on=['EID'],how='left')
# test = pd.merge(test,breakfaith_is_info,on=['EID'],how='left')
# test = pd.merge(test,recruit_WZCODE_info_sum,on=['EID'],how='left')
# test = pd.merge(test,recruit_RECRNUM_info,on=['EID'],how='left')

test = test.fillna(-999)
train = train.fillna(-999)

# del train['EID']
test_index = test.pop('EID')
print(test.shape,org_test_shape)

import lightgbm as lgb
# 抽样选择数据
tmp1 = train[train.TARGET==1]
tmp0 = train[train.TARGET==0]
x_valid_1 = tmp1.sample(frac=0.3, random_state=70, axis=0)
x_train_1 = tmp1.drop(x_valid_1.index.tolist())
x_valid_2 = tmp0.sample(frac=0.1, random_state=70, axis=0)
x_train_2 = tmp0.drop(x_valid_2.index.tolist())
X_train = pd.concat([x_train_1,x_train_2],axis=0)

y_train = X_train.pop('TARGET')
X_test = pd.concat([x_valid_1,x_valid_2],axis=0)
y_test = X_test.pop('TARGET')

feature_len = X_train.shape[1]
print(feature_len)
print(train.columns)

predictors = X_train.columns
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'auc'},
    'num_leaves': 128,
    'learning_rate': 0.08,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 10,
    'verbose': 0
}

evals_result = {}
print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round = 1000,
                valid_sets=lgb_eval,
                feature_name=['f' + str(i + 1) for i in range(feature_len)],
                early_stopping_rounds= 15 ,
                evals_result=evals_result)
feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
print(feat_imp)

print('Plot metrics during training...')
ax = lgb.plot_metric(evals_result, metric='auc')
plt.show()

print('Plot feature importances...')
lgb.plot_importance(gbm,max_num_features=feature_len)
plt.show()

print('Start predicting...')

y_pred = gbm.predict(test.values, num_iteration=gbm.best_iteration)
y_pred = np.round(y_pred,8)
result = pd.DataFrame({'PROB':list(y_pred),
                       })
result['FORTARGET'] = result['PROB'] > 0.22
result['PROB'] = result['PROB'].astype('str')
result['FORTARGET'] = result['FORTARGET'].astype('int')
result = pd.concat([test_index,result],axis=1)

print('predict pos tation',sum(result['FORTARGET']))

result = pd.DataFrame(result).drop_duplicates(['EID'])
result[['EID','FORTARGET','PROB']].to_csv('./evaluation_public.csv',index=None)

print(len(result.EID.unique()))
