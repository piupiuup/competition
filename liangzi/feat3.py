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
from sklearn.metrics import roc_auc_score

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
def get_base_feat(data,data_key):
    result_path = cache_path + 'base_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & new:
        feat = pd.read_hdf(result_path, 'w')
    else:
        entbase = pd.read_csv(data_path + '1entbase.csv').fillna(0)
        # stat_temp = stat.merge(entbase,on='id',how='left')
        feat = data.merge(entbase,on='id',how='left')
        feat['rgyear'] = 2020 - feat['rgyear']
        feat['zczb2'] = feat['zczb'] * (1.14 ** feat['rgyear'])
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
        feat.fillna(0,inplace=True)
        feat.drop(['id','label'],axis=1,inplace=True)
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat


# alter特征
def get_alter_feat(data,data_key):
    result_path = cache_path + 'alter_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & new:
        feat = pd.read_hdf(result_path, 'w')
    else:
        alter = pd.read_csv(data_path + '2alter.csv')
        alter['altdate'] = alter['altdate'].apply(lambda x:(2016-int(x[:4]))*12 - int(x[-2:]))
        alter.sort_values('altdate', ascending=True, inplace=True)
        n_alter = alter.groupby('id',as_index=False)['id'].agg({'n_alter':'size'})
        alterno = pd.get_dummies(alter['alterno'], prefix='alterno')
        alterno = pd.concat([alter[['id']], alterno], axis=1)
        alterno = alterno.groupby(['id'], as_index=False).sum()
        alter_first = alter.drop_duplicates('id',keep='first').rename(columns={'altdate':'altdate_first'})
        alter_last = alter.drop_duplicates('id', keep='last').rename(columns={'altdate':'altdate_last'})
        alter_money = alter[~alter['altbe'].isnull()].drop_duplicates('id', keep='first')
        alter_money['alter_money'] = alter_money['altaf'] - alter_money['altbe']
        alter_money['alter_rate'] = alter_money['alter_money'] / (alter_money['altbe']+0.1)
        feat = data.merge(n_alter, on='id', how='left').fillna(0)
        feat = feat.merge(alterno, on='id', how='left').fillna(0)
        # feat = feat.merge(alterno_time, on='id', how='left').fillna(0)
        feat = feat.merge(alter_first[['id', 'alterno', 'altdate_first']], on='id', how='left').fillna(-1)
        feat = feat.merge(alter_last[['id', 'altdate_last']], on='id', how='left').fillna(-1)
        feat = feat.merge(alter_money[['id', 'alter_money', 'alter_rate','altbe']],on='id', how='left').fillna(-100000)
        feat.drop(['id', 'label'], axis=1, inplace=True)
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat

# branch特征
def get_branch_feat(data,data_key):
    result_path = cache_path + 'branch_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & new:
        feat = pd.read_hdf(result_path, 'w')
    else:
        branch = pd.read_csv(data_path + '3branch.csv')
        branch['branch_active_year'] = branch['b_endyear'] - branch['b_reyear']
        feat = branch.groupby('id').agg({'b_endyear': {'n_branch': 'size',
                                                        'n_end_branch': 'count',
                                                        'last_end_branch': 'max',
                                                       'first_end_branch': 'max',
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
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat

# invest特征
def get_invest_feat(data,data_key):
    result_path = cache_path + 'invest_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & new:
        feat = pd.read_hdf(result_path, 'w')
    else:
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
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat

# right特征
def get_right_feat(data, data_key):
    result_path = cache_path + 'right_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & new:
        feat = pd.read_hdf(result_path, 'w')
    else:
        right = pd.read_csv(data_path + '5right.csv')
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
        feat = feat.merge(righttype, on='id', how='left')
        feat = feat.merge(last_fbdate, on='id', how='left')
        feat = feat.merge(last_askdate, on='id', how='left')
        feat['n_right4'] = feat['n_right'] - feat['n_right3']
        feat.drop(['id', 'label'], axis=1, inplace=True)
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat

# lawsuit特征
def get_lawsuit_feat(data, data_key):
    result_path = cache_path + 'lawsuit_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & new:
        feat = pd.read_hdf(result_path, 'w')
    else:
        lawsuit = pd.read_csv(data_path + '7lawsuit.csv')
        lawsuit.drop_duplicates(['id','lawdate','lawamount'],inplace=True)
        n_lawsuit = lawsuit.groupby('id', as_index=False)['id'].agg({'n_lawsuit': 'size'})
        sum_lawsuit_money = lawsuit.groupby('id', as_index=False)['id'].agg({'lawamount': 'sum'})
        lawsuit['lawdate'] = lawsuit['lawdate'].apply(lambda x:(2016-int(x[:4]))*12 - int(x[-2:]))
        last_lawsuit_date = lawsuit.groupby('id', as_index=False)['lawdate'].agg({'last_lawsuit_date': 'min'})
        feat = data.merge(n_lawsuit, on='id', how='left')
        feat = feat.merge(sum_lawsuit_money, on='id', how='left')
        feat = feat.merge(last_lawsuit_date, on='id', how='left')
        feat.drop(['id', 'label'], axis=1, inplace=True)
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat

# breakfaith特征
def get_breakfaith_feat(data, data_key):
    result_path = cache_path + 'breakfaith_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & new:
        feat = pd.read_hdf(result_path, 'w')
    else:
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
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat

# recruit特征
def get_recruit_feat(data, data_key):
    result_path = cache_path + 'recruit_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & new:
        feat = pd.read_hdf(result_path, 'w')
    else:
        recruit = pd.read_csv(data_path + '9recruit.csv')
        recruit['recdate'] = recruit['recdate'].apply(lambda x: (2016 - int(x[:4])) * 12 - int(x[-2:]))
        # breakfaith.drop_duplicates(['id', 'fbdate'], inplace=True)
        n_recruit = recruit.groupby('id', as_index=False)['id'].agg({'n_recruit': 'size'})
        sum_recruit_people = recruit.groupby('id', as_index=False)['id'].agg({'sum_recruit_people': 'sum'})
        last_lawsuit_date = recruit.groupby('id', as_index=False)['recdate'].agg({'last_lawsuit_date': 'min'})
        wzcode = recruit[['id','wzcode','recrnum']].set_index(['id','wzcode'])
        feat = data.merge(n_recruit, on='id', how='left')
        feat = feat.merge(sum_recruit_people, on='id', how='left')
        feat = feat.merge(last_lawsuit_date, on='id', how='left')
        feat.drop(['id', 'label'], axis=1, inplace=True)
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat

# 二次处理特征
def second_feat(result):
    return result

# 获取样本标签
def get_labels(data):
    train = pd.read_csv(r'C:/Users/csw/Desktop/python/liangzi/data/train.csv')
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
    if os.path.exists(result_path) & new:
        result = pd.read_hdf(result_path, 'w')
    else:
        data.index = list(range(len(data.index)))

        print('开始构造特征...')
        base_feat = get_base_feat(data,data_key)                # 添加基础特征
        alter_feat = get_alter_feat(data,data_key)              # alter特征
        branch_feat = get_branch_feat(data,data_key)            # branch特征
        invest_feat = get_invest_feat(data,data_key)            # invest特征
        right_feat = get_right_feat(data, data_key)             # right特征
        lawsuit_feat = get_lawsuit_feat(data, data_key)         # lawsuit特征
        breakfaith_feat = get_breakfaith_feat(data, data_key)   # breakfaith特征
        recruit_feat = get_recruit_feat(data, data_key)         # recruit特征

        result = concat([data[['id']],base_feat,alter_feat
                            ,branch_feat,invest_feat,right_feat,
                         lawsuit_feat,breakfaith_feat,recruit_feat
                         ])
        result = get_labels(result)
        result = second_feat(result)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    print('特征矩阵大小：{}'.format(result.shape))
    print('生成特征一共用时{}秒'.format(time.time() - t0))
    return result







