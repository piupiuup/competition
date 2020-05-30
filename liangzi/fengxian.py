import gc
import re
import sys
import time
import jieba
import string
import codecs
import pickle
import hashlib
import os.path
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from collections import Counter
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score





######################################## 清洗数据 ########################################
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

#################################### 构造特征 #######################################

global data_path
cache_path = 'F:/liangzi_cache/'
new = False

# 获取阈值
def get_threshold(preds):
    preds_temp = sorted(preds,reverse=True)
    n = sum(preds) # 实际正例个数
    m = 0   # 提交的正例个数
    e = 0   # 正确个数的期望值
    f1 = 0  # f1的期望得分
    for threshold in preds_temp:
        e += threshold
        m += 1
        f1_temp = e/(m+n)
        if f1>f1_temp:
            break
        else:
            f1 = f1_temp
    print('阈值为：{}'.format(threshold))
    print('提交正例个数为：{}'.format(m-1))
    print('期望得分为：{}'.format(f1*2))
    return [(1  if (pred>threshold) else 0) for pred in preds]

# 合并节约内存
def concat(L):
    result = None
    for l in L:
        if result is None:
            result = l
        else:
            result[l.columns.tolist()] = l
    return result

# 分组标准化
def grp_standard(data,key,names):
    for name in names:
        mean_std = data.groupby(key, as_index=False)[name].agg({'mean': 'mean',
                                                               'std': 'std'})
        data = data.merge(mean_std, on=key, how='left')
        data[name] = ((data[name]-data['mean'])/data['std']).fillna(0)
        data[name] = data[name].replace(-np.inf, 0)
        data.drop(['mean','std'],axis=1,inplace=True)
    return data

# 分组归一化
def grp_normalize(data,key,names,start=0):
    for name in names:
        max_min = data.groupby(key,as_index=False)[name].agg({'max':'max',
                                                'min':'min'})
        data = data.merge(max_min,on=key,how='left')
        data[name] = (data[name]-data['min'])/(data['max']-data['min'])
        data[name] = data[name].replace(-np.inf, start)
        data.drop(['max','min'],axis=1,inplace=True)
    return data

# 分组排序
def grp_rank(data,key,names,ascending=True):
    for name in names:
        data.sort_values([key, name], inplace=True, ascending=ascending)
        data['rank'] = range(data.shape[0])
        min_rank = data.groupby(key, as_index=False)['rank'].agg({'min_rank': 'min'})
        data = pd.merge(data, min_rank, on=key, how='left')
        data['rank'] = data['rank'] - data['min_rank']
        data[names] = data['rank']
    data.drop(['rank'],axis=1,inplace=True)
    return data


# 基础特征
def get_base_feat(stat,data,data_key):
    def id_convert(x):
        if 'p' in x:
            return -1
        if 's' in x:
            return 0 if int(x[1:])<500000 else 1
        else:
            return 0 if int(x) < 500000 else 1
    entbase = pd.read_csv(data_path + '1entbase.csv').fillna(-1)
    # stat_temp = stat.merge(entbase,on='id',how='left')
    feat = data.merge(entbase,on='id',how='left')
    feat['hy_count'] = feat['hy'].map(stat['hy'].value_counts())
    feat['etype_count'] = feat['etype'].map(stat['etype'].value_counts())
    feat['ienum'] = feat['inum'] - feat['enum']
    feat['rgyear'] = 2020 - feat['rgyear']
    # feat['zczb2'] = feat['zczb'] * (1.14 ** feat['rgyear'])
    feat['finzb2'] = feat['finzb'] / (feat['zczb'] + 0.1)
    feat['mpnum2'] = feat['mpnum'] / (feat['zczb'] + 0.1)
    feat['inum2'] = feat['inum'] / (feat['zczb'] + 0.1)
    feat['fstinum2'] = feat['fstinum'] / (feat['zczb'] + 0.1)
    feat['tzinum2'] = feat['tzinum'] / (feat['zczb'] + 0.1)
    feat['sumnum'] = feat[['mpnum','inum','fstinum','tzinum']].sum(axis=1)
    hy = pd.get_dummies(feat['hy'], prefix='hy')
    feat = pd.concat([feat,hy],axis=1)
    etype = pd.get_dummies(feat['etype'], prefix='etype')
    feat = pd.concat([feat, etype], axis=1)
    feat['id_feat'] = feat['id'].apply(id_convert)
    feat.fillna(0,inplace=True)
    feat.drop(['id','label'],axis=1,inplace=True)
    return feat

# alter特征
def get_alter_feat(data,data_key):
    alter = pd.read_csv(data_path + '2alter.csv')
    alter['altdate'] = alter['altdate'].apply(lambda x:(2016-int(x[:4]))*12 - int(x[-2:]))
    alter.sort_values('altdate', ascending=True, inplace=True)
    n_alter = alter.groupby('id',as_index=False)['alterno'].agg({'n_alter':'size'})
    alterno = pd.get_dummies(alter['alterno'], prefix='alterno')
    alterno = pd.concat([alter[['id']], alterno], axis=1)
    alterno = alterno.groupby(['id'], as_index=False).sum()
    alter_first = alter.drop_duplicates('id',keep='first').rename(columns={'altdate':'altdate_first'})
    alter_last = alter.drop_duplicates('id', keep='last').rename(columns={'altdate':'altdate_last'})
    # alterno_time = alter.drop_duplicates(['id','alterno'], keep='last')[['id','alterno','altdate']]
    # alterno_time = alterno_time.set_index(['id','alterno']).unstack()
    # alterno_time.columns = alterno_time.columns.droplevel(0)
    # alterno_time = alterno_time.add_prefix('alterdate_').reset_index()
    # alter_money = alter[~alter['altbe'].isnull()].drop_duplicates('id', keep='first')
    # alter_money['alter_money'] = alter_money['altaf'] - alter_money['altbe']
    # alter_money['alter_rate'] = alter_money['alter_money'] / (alter_money['altbe']+0.1)
    feat = data.merge(n_alter, on='id', how='left').fillna(0)
    feat = feat.merge(alterno, on='id', how='left').fillna(0)
    # feat = feat.merge(alterno_time, on='id', how='left').fillna(0)
    feat = feat.merge(alter_first[['id', 'alterno', 'altdate_first']], on='id', how='left').fillna(-1)
    feat = feat.merge(alter_last[['id', 'altdate_last']], on='id', how='left').fillna(-1)
    # feat = feat.merge(alter_money[['id', 'alter_money', 'alter_rate','altbe']],on='id', how='left').fillna(-100000)
    feat.drop(['id', 'label'], axis=1, inplace=True)
    return feat

# branch特征
def get_branch_feat(data,data_key):
    branch = pd.read_csv(data_path + '3branch.csv')
    branch['branch_active_year'] = branch['b_endyear'] - branch['b_reyear']
    feat = branch.groupby('id').agg({'b_endyear': {'n_branch': 'size',
                                                   'n_end_branch': 'count',
                                                    'last_end_branch': 'max',
                                                   'median_end_branch': 'median'},
                                      'ifhome': {'n_home_branch': 'sum'},
                                      'b_reyear': {'last_start_branch': 'max',
                                                   'first_start_branch': 'min'},
                                     'branch_active_year': {'branch_active_year':'mean'}})
    feat.columns = feat.columns.droplevel(0)
    feat['id'] = feat.index
    feat['n_active_branch'] = feat['n_branch'] - feat['n_end_branch']
    feat['n_outer_branch'] = feat['n_branch'] - feat['n_home_branch']
    feat['active_branch_rate'] = feat['n_active_branch'] / (feat['n_branch'] + 0.1)
    feat['home_brach_rate'] = feat['n_home_branch'] / (feat['n_branch'] + 0.1)
    feat = data.merge(feat,on='id',how='left')
    feat.drop(['id', 'label'], axis=1, inplace=True)
    return feat

# invest特征
def get_invest_feat(data,data_key):
    invest = pd.read_csv(data_path + '4invest.csv')
    # invest = invest[invest['id'] != invest['bteid']]
    train = pd.read_csv(data_path + 'train.csv')
    id_label_dict = dict(zip(train['id'].values,train['label'].values))
    invest['btlabel'] = invest['bteid'].map(id_label_dict)
    invest['idlabel'] = invest['id'].map(id_label_dict)
    n_invest = invest.groupby('id',as_index=False)['id'].agg({'n_invest':'count'})
    mean_btbl = invest.groupby('id', as_index=False)['btbl'].agg({'mean_btbl': 'mean'})
    sum_btbl = invest.groupby('id', as_index=False)['btbl'].agg({'sum_btbl': 'sum'})
    n_home_invest = invest.groupby('id', as_index=False)['ifhome'].agg({'n_home_invest': 'sum'})
    n_negitive_invest = invest.groupby('id', as_index=False)['btlabel'].agg({'n_negitive_invest': 'sum'})
    n_negitive_invest2 = invest.groupby('id', as_index=False)['btendyear'].agg({'n_negitive_invest2': 'count'})
    n_negitive_invested = invest.groupby('bteid', as_index=False)['idlabel'].agg({'n_negitive_invested': 'sum'})
    n_negitive_invested.rename(columns={'bteid':'id'},inplace=True)
    last_invest = invest.groupby('id',as_index=False)['btyear'].agg({'last_invest':'max'})
    last_negitive_invest = invest.groupby('id', as_index=False)['btendyear'].agg({'last_negitive_invest': 'max'})
    bt_invest = invest[['bteid','btyear','btendyear','btbl']].rename(columns={'bteid':'id'})
    bt_invest = bt_invest.groupby('id',as_index=False).max()
    feat = data.merge(n_invest, on='id', how='left')
    feat = feat.merge(n_home_invest, on='id', how='left')
    feat = feat.merge(mean_btbl, on='id', how='left')
    feat = feat.merge(sum_btbl, on='id', how='left')
    feat = feat.merge(n_negitive_invest, on='id', how='left')
    feat = feat.merge(n_negitive_invest2, on='id', how='left')
    feat = feat.merge(n_negitive_invested, on='id', how='left')
    feat = feat.merge(last_invest, on='id', how='left')
    feat = feat.merge(last_negitive_invest, on='id', how='left')
    feat = feat.merge(bt_invest, on='id', how='left')
    feat['home_invest_rate'] = feat['n_home_invest'] / (feat['n_invest']+0.1)
    feat.drop(['id', 'label'], axis=1, inplace=True)
    return feat

# right特征
def get_right_feat(data, data_key):
    right = pd.read_csv(data_path + '5right.csv')
    nunique_right = right.groupby('id', as_index=False)['righttype'].agg({'nunique_right': 'nunique'})
    n_right1 = right[(right['askdate'] > '2012')].groupby('id', as_index=False)['righttype'].agg({'n_right1': 'count'})
    right['fbdate'] = right['fbdate'].apply(lambda x: x if x is np.nan else (2020 - int(x[:4])) * 12 - int(x[-2:]))
    right['weight'] = right['askdate'].apply(lambda x:0.5**(2015-int(x[:4])))
    right['askdate'] = right['askdate'].apply(lambda x: x if x is np.nan else (2020 - int(x[:4])) * 12 - int(x[-2:]))
    n_right = right.groupby('id', as_index=False)['id'].agg({'n_right': 'count'})
    n_right2 = right.groupby('id',as_index=False)['weight'].agg({'n_right2':'sum'})
    n_right3 = right.groupby('id', as_index=False)['fbdate'].agg({'n_right3': 'count'})
    righttype = pd.get_dummies(right['righttype'], prefix='righttype')
    righttype = pd.concat([right['id'], righttype], axis=1)
    righttype = righttype.groupby(['id'], as_index=False).sum()
    last_fbdate = right.groupby('id', as_index=False)['fbdate'].agg({'last_fbdate': 'min'})
    last_askdate = right.groupby('id', as_index=False)['askdate'].agg({'last_askdate': 'min'})
    feat = data.merge(n_right, on='id', how='left')
    feat = feat.merge(n_right2, on='id', how='left')
    feat = feat.merge(n_right3, on='id', how='left')
    feat = feat.merge(nunique_right, on='id', how='left')
    feat = feat.merge(n_right1, on='id', how='left')
    feat = feat.merge(righttype, on='id', how='left')
    feat = feat.merge(last_fbdate, on='id', how='left')
    feat = feat.merge(last_askdate, on='id', how='left')
    feat['n_right4'] = feat['n_right'] - feat['n_right3']
    feat.drop(['id', 'label'], axis=1, inplace=True)
    return feat

# project特征
def get_project_feat(data, data_key):
    project = pd.read_csv(data_path + '6project.csv')
    project['djdate'] = project['djdate'].apply(lambda x: x if x is np.nan else (2020 - int(x[:4])) * 12 - int(x[5:7]))
    feat = project.groupby('id',as_index=False)['djdate'].agg({'n_project':'count',
                                                                'max_dfdate':'min',
                                                               'min_dfdate':'max',
                                                               'mean_dfdate': 'mean'})
    feat = data.merge(feat,on='id',how='left')
    feat.drop(['id', 'label'], axis=1, inplace=True)
    return feat


# lawsuit特征
def get_lawsuit_feat(data, data_key):
    lawsuit = pd.read_csv(data_path + '7lawsuit.csv')
    lawsuit.drop_duplicates(['id','lawdate','lawamount'],inplace=True)
    n_lawsuit = lawsuit.groupby('id', as_index=False)['id'].agg({'n_lawsuit': 'size'})
    sum_lawsuit_money = lawsuit.groupby('id', as_index=False)['lawamount'].agg({'lawamount': 'sum'})
    lawsuit['lawdate'] = lawsuit['lawdate'].apply(lambda x:(2016-int(x[:4]))*12 - int(x[-2:]))
    last_lawsuit_date = lawsuit.groupby('id', as_index=False)['lawdate'].agg({'last_lawsuit_date': 'min'})
    feat = data.merge(n_lawsuit, on='id', how='left')
    feat = feat.merge(sum_lawsuit_money, on='id', how='left')
    feat = feat.merge(last_lawsuit_date, on='id', how='left')
    feat.drop(['id', 'label'], axis=1, inplace=True)
    return feat

# breakfaith特征
def get_breakfaith_feat(data, data_key):
    breakfaith = pd.read_csv(data_path + '8breakfaith.csv')
    breakfaith['fbdate'] = pd.to_datetime(breakfaith['fbdate']).apply(lambda x: (2016-x.year)*12 + x.month)
    breakfaith.drop_duplicates(['id', 'fbdate'], inplace=True)
    n_breakfaith = breakfaith.groupby('id', as_index=False)['id'].agg({'n_breakfaith': 'size'})
    last_fbdate = breakfaith.groupby('id', as_index=False)['fbdate'].agg({'last_fbdate': 'min'})
    first_fbdate = breakfaith.groupby('id', as_index=False)['fbdate'].agg({'last_fbdate': 'max'})
    feat = data.merge(n_breakfaith, on='id', how='left')
    feat = feat.merge(last_fbdate, on='id', how='left')
    feat = feat.merge(first_fbdate, on='id', how='left')
    feat.drop(['id', 'label'], axis=1, inplace=True)
    return feat

# recruit特征
def get_recruit_feat(data, data_key):
    recruit = pd.read_csv(data_path + '9recruit.csv')
    recruit['recdate'] = recruit['recdate'].apply(lambda x: (2016 - int(x[:4])) * 12 - int(x[-2:]))
    # breakfaith.drop_duplicates(['id', 'fbdate'], inplace=True)
    n_recruit = recruit.groupby('id', as_index=False)['id'].agg({'n_recruit': 'size'})
    nunique_recruit = recruit.groupby('id', as_index=False)['poscode'].agg({'nunique_recruit': 'nunique'})
    sum_recruit_people = recruit.groupby('id', as_index=False)['pnum'].agg({'sum_recruit_people': 'sum',
                                                                            'max_pnum':'max',
                                                                            'mean_pnum':'mean'})
    last_lawsuit_date = recruit.groupby('id', as_index=False)['recdate'].agg({'last_lawsuit_date': 'min'})
    wzcode = recruit.groupby(['id','wzcode'])['pnum'].sum().unstack().reset_index()
    feat = data.merge(n_recruit, on='id', how='left')
    feat = feat.merge(nunique_recruit, on='id', how='left')
    feat = feat.merge(sum_recruit_people, on='id', how='left')
    feat = feat.merge(last_lawsuit_date, on='id', how='left')
    feat = feat.merge(wzcode, on='id', how='left')
    feat.drop(['id', 'label'], axis=1, inplace=True)
    return feat

def get_qualification_feat(data, data_key):
    qualification = pd.read_csv(data_path + '10qualification.csv', encoding='gb2312')
    n_qualification = qualification.groupby('id',as_index=False)['addtype'].agg({'n_qua':'count'})
    feat = data.merge(n_qualification,on='id',how='left')
    return feat



# 二次处理特征
def second_feat(result):
    return result

# 获取样本标签
def get_labels(data):
    train = pd.read_csv(r'C:/Users/csw/Desktop/python/liangzi/data/concat_data/train.csv')
    label_dict = dict(zip(train['id'].values,train['label'].values))
    data['label'] = data['id'].map(label_dict)
    return data

# 构造训练集
def make_set(stat,data,path):
    global data_path
    data_path = path
    t0 = time.time()
    data_key = hashlib.md5(data.to_string().encode()).hexdigest()
    print('数据key为：{}'.format(data_key))
    result_path = cache_path + 'feat_set_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & 0:
        result = pd.read_hdf(result_path, 'w')
    else:
        data.index = list(range(len(data.index)))
        entbase = pd.read_csv(data_path + '1entbase.csv').fillna(0)
        stat = stat.merge(entbase,on='id',how='left')
        print('开始构造特征...')
        base_feat = get_base_feat(stat, data,data_key)          # 添加基础特征
        alter_feat = get_alter_feat(data,data_key)              # alter特征
        branch_feat = get_branch_feat(data,data_key)            # branch特征
        invest_feat = get_invest_feat(data,data_key)            # invest特征
        right_feat = get_right_feat(data, data_key)             # right特征
        project_feat = get_project_feat(data, data_key)         # project特征
        lawsuit_feat = get_lawsuit_feat(data, data_key)         # lawsuit特征
        breakfaith_feat = get_breakfaith_feat(data, data_key)   # breakfaith特征
        recruit_feat = get_recruit_feat(data, data_key)         # recruit特征
        # qualification_feat = get_qualification_feat(data, data_key)# qualification特征

        result = concat([data[['id']],base_feat,alter_feat ,branch_feat,invest_feat,right_feat,
                         project_feat,lawsuit_feat,breakfaith_feat,recruit_feat
                         ])
        result = get_labels(result)
        result = second_feat(result)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    print('特征矩阵大小：{}'.format(result.shape))
    print('生成特征一共用时{}秒'.format(time.time() - t0))
    return result


##################################### lgb重采样预测 ############################
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import KFold

print('读取train数据...')
data_path = 'C:/Users/csw/Desktop/python/liangzi/data/'
cache_path = 'F:/liangzi_cache/'

train = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'evaluation_public.csv')
test['label'] = np.nan

print('构造特征...')
train_feat_temp = make_set(train,train,data_path)
test_feat = make_set(train,test,data_path)
sumbmission = test_feat[['id']].copy()

predictors = [f for f in train_feat_temp.columns if f not in ['id','label','enddate']]

train_feat = train_feat_temp.append(train_feat_temp[train_feat_temp['prov']==11])
train_feat = train_feat.append(train_feat_temp[train_feat_temp['prov']==11])
print('开始CV 5折训练...')
scores = []
t0 = time.time()
mean_score = []
train_preds = np.zeros(len(train_feat))
test_preds11 = np.zeros(len(test_feat))
kf = KFold(len(train_feat), n_folds = 5, shuffle=True, random_state=520)
for i, (train_index, test_index) in enumerate(kf):
    lgb_train = lgb.Dataset(train_feat[predictors].iloc[train_index], train_feat['label'].iloc[train_index])
    lgb_test = lgb.Dataset(train_feat[predictors].iloc[test_index], train_feat['label'].iloc[test_index])

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 150,
        'learning_rate': 0.01,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.95,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 100,
    }
    gbm = lgb.train(params, lgb_train, 900)
    train_preds_sub = gbm.predict(train_feat[predictors].iloc[test_index])
    train_preds[test_index] += train_preds_sub
    test_preds_sub = gbm.predict(test_feat[predictors])
    test_preds11 += test_preds_sub

    score = roc_auc_score(train_feat['label'].iloc[test_index],train_preds_sub)
    scores.append(score)
    print('第{0}轮mae的得分: {1}'.format(i + 1, score))
test_preds11 = test_preds11/5
print('auc平均得分: {}'.format(np.mean(scores)))
print('CV训练用时{}秒'.format(time.time() - t0))

print('开始CV 5折训练...')
train_feat = train_feat_temp.append(train_feat_temp[(train_feat_temp['prov']==12)])
train_feat = train_feat.append(train_feat_temp[(train_feat_temp['prov']==12)])
scores = []
t0 = time.time()
mean_score = []
train_preds = np.zeros(len(train_feat))
test_preds12 = np.zeros(len(test_feat))
kf = KFold(len(train_feat), n_folds = 5, shuffle=True, random_state=520)
for i, (train_index, test_index) in enumerate(kf):
    lgb_train = lgb.Dataset(train_feat[predictors].iloc[train_index], train_feat['label'].iloc[train_index])
    lgb_test = lgb.Dataset(train_feat[predictors].iloc[test_index], train_feat['label'].iloc[test_index])

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 150,
        'learning_rate': 0.01,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.95,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 100,
    }
    gbm = lgb.train(params, lgb_train, 900)
    train_preds_sub = gbm.predict(train_feat[predictors].iloc[test_index])
    train_preds[test_index] += train_preds_sub
    test_preds_sub = gbm.predict(test_feat[predictors])
    test_preds12 += test_preds_sub

    score = roc_auc_score(train_feat['label'].iloc[test_index],train_preds_sub)
    scores.append(score)
    print('第{0}轮mae的得分: {1}'.format(i + 1, score))
test_preds12 = test_preds12/5
print('auc平均得分: {}'.format(np.mean(scores)))
print('CV训练用时{}秒'.format(time.time() - t0))

test_feat['pred11'] = test_preds11
test_feat['pred12'] = test_preds12
test_feat['pred'] = test_feat.apply(lambda x: x.pred11 if x.prov==11 else x.pred12, axis=1)
preds_scatter = get_threshold(test_feat['pred'].values)
submission = pd.DataFrame({'EID':sumbmission['id'],'FORTARGET':preds_scatter,'PROB':1-test_feat['pred'].values})
submission.to_csv(r'C:\Users\csw\Desktop\python\liangzi\submission\sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), index=False, float_format='%.4f')










################################## xgb重采样 ################################
import xgboost

print('读取train数据...')
data_path = 'C:/Users/csw/Desktop/python/liangzi/data/'
cache_path = 'F:/liangzi_cache/'

train = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'evaluation_public.csv')
test['label'] = np.nan

print('构造特征...')
train_feat_temp = make_set(train,train,data_path)
test_feat = make_set(train,test,data_path)
sumbmission = test_feat[['id']].copy()

train_feat = train_feat_temp.append(train_feat_temp[train_feat_temp['prov']==11])
train_feat = train_feat.append(train_feat_temp[train_feat_temp['prov']==11])
predictors = train_feat.columns.drop(['id','label','enddate','hy_16.0', 'hy_91.0', 'hy_94.0'])

print('开始CV 5折训练...')
scores = []
t0 = time.time()
test_preds11 = np.zeros(len(test_feat))
kf = KFold(len(train_feat), n_folds = 5, shuffle=True, random_state=520)
for i, (train_index, test_index) in enumerate(kf):
    xgb_train = xgboost.DMatrix(train_feat[predictors].iloc[train_index], train_feat['label'].iloc[train_index])
    xgb_eval = xgboost.DMatrix(test_feat[predictors])

    xgb_params = {
        "objective": "reg:logistic"
        , "eval_metric": "auc"
        , "eta": 0.01
        , "max_depth": 12
        , "min_child_weight": 10
        , "gamma": 0.70
        , "subsample": 0.76
        , "colsample_bytree": 0.95
        , "alpha": 2e-05
        , "lambda": 10
    }
    bst = xgboost.train(params=xgb_params,dtrain=xgb_train,num_boost_round=1200)
    test_preds_sub = bst.predict(xgb_eval)
    test_preds11 += test_preds_sub

test_preds11 = test_preds11/5
print('CV训练用时{}秒'.format(time.time() - t0))

print('开始CV 5折训练...')
train_feat = train_feat_temp.append(train_feat_temp[(train_feat_temp['prov']==12)])
train_feat = train_feat.append(train_feat_temp[(train_feat_temp['prov']==12)])
t0 = time.time()
test_preds12 = np.zeros(len(test_feat))
kf = KFold(len(train_feat), n_folds = 5, shuffle=True, random_state=520)
for i, (train_index, test_index) in enumerate(kf):
    xgb_train = xgboost.DMatrix(train_feat[predictors].iloc[train_index], train_feat['label'].iloc[train_index])
    xgb_eval = xgboost.DMatrix(test_feat[predictors])

    xgb_params = {
        "objective": "reg:logistic"
        , "eval_metric": "auc"
        , "eta": 0.01
        , "max_depth": 12
        , "min_child_weight": 10
        , "gamma": 0.70
        , "subsample": 0.76
        , "colsample_bytree": 0.95
        , "alpha": 2e-05
        , "lambda": 10
    }
    bst = xgboost.train(params=xgb_params, dtrain=xgb_train, num_boost_round=1200)
    test_preds_sub = bst.predict(xgb_eval)
    test_preds12 += test_preds_sub

test_preds12 = test_preds12/5
print('CV训练用时{}秒'.format(time.time() - t0))

test_feat['pred11'] = test_preds11
test_feat['pred12'] = test_preds12
test_feat['pred'] = test_feat.apply(lambda x: x.pred11 if x.prov==11 else x.pred12, axis=1)
preds_scatter = get_threshold(test_feat['pred'].values)
submission = pd.DataFrame({'EID':sumbmission['id'],'FORTARGET':preds_scatter,'PROB':test_feat['pred'].values})
submission.to_csv(r'C:\Users\csw\Desktop\python\liangzi\submission\sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)










