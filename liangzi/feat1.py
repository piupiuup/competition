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

# 对特征进行分组变化
def get_flex(stat,data,group,feat):
    print(stat.head())
    data_temp = data.copy()
    std = stat.groupby(group)[feat].std()
    mean = stat.groupby(group)[feat].mean()
    data_temp['std'] = data_temp[group].map(std)
    data_temp['mean'] = data_temp[group].map(mean)
    data_temp['flex_value'] = (data_temp[feat]-data_temp['mean']) / (data_temp['mean']+0.001)
    return data_temp['flex_value']


# 基础特征
def get_base_feat(stat,data,data_key):
    result_path = cache_path + 'base_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & new:
        feat = pd.read_hdf(result_path, 'w')
    else:
        entbase = pd.read_csv(data_path + '1entbase.csv')
        # stat_temp = stat.merge(entbase,on='id',how='left')
        feat = data.merge(entbase,on='id',how='left')
        feat['rgyear'] = 2015 - feat['rgyear']
        feat['zczb'] = feat['zczb'] * (1.14 ** feat['rgyear'])
        feat['finzb_rate'] = feat['finzb'] / (feat['zczb']+0.1)
        # etype_count = stat_temp.groupby('etype').size().to_dict()
        # feat['etype_count'] = feat['etype'].map(etype_count)
        # print(feat.head())
        # feat['flex_year'] = get_flex(stat_temp, feat, 'etype', 'rgyear')
        # feat['flex_zczb'] = get_flex(stat_temp, feat, 'etype', 'zczb')
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
        n_alter = alter.groupby('id',as_index=False)['id'].agg({'n_alter':'size'})
        alter.sort_values('altdate',ascending=True,inplace=True)
        alter = alter.drop_duplicates(['id','alterno'],keep='last')
        alter['alter_money'] = alter['altaf'] - alter['altbe']
        alter['alter_rate'] = alter['alter_money'] / (alter['altbe']+0.1)
        feat = data.merge(n_alter, on='id', how='left').fillna(0)
        feat = feat.merge(alter[['id','alterno','altdate','alter_money','alter_rate','altbe','altaf']], on='id', how='left').fillna(-1)
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
        feat = branch.groupby('id').agg({'b_endyear': {'n_branch': 'size',
                                                        'n_end_branch': 'count',
                                                        'last_end_branch': 'max'},
                                          'ifhome': {'n_home_branch': 'sum'},
                                          'b_reyear': {'last_start_branch': 'max'}})
        feat.columns = feat.columns.droplevel(0)
        feat['id'] = feat.index
        feat['n_active_branch'] = feat['n_branch'] - feat['n_end_branch']
        feat['n_outer_branch'] = feat['n_branch'] - feat['n_home_branch']
        feat['active_branch_rate'] = feat['n_active_branch'] / (feat['n_branch']+0.1)
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
        invest = invest[invest['id'] != invest['bteid']]
        train = pd.read_csv(data_path + 'train.csv')
        id_label_dict = dict(zip(train['id'].values,train['label'].values))
        invest['btlabel'] = invest['bteid'].map(id_label_dict)
        invest['idlabel'] = invest['id'].map(id_label_dict)
        n_invest = invest.groupby('id',as_index=False)['id'].agg({'n_invest':'count'})
        n_negitive_invest = invest.groupby('id', as_index=False)['btlabel'].agg({'n_negitive_invest': 'sum'})
        n_negitive_invest2 = invest.groupby('id', as_index=False)['btendyear'].agg({'n_negitive_invest2': 'count'})
        n_negitive_invested = invest.groupby('bteid', as_index=False)['idlabel'].agg({'n_negitive_invested': 'sum'})
        n_negitive_invested.rename(columns={'bteid':'id'},inplace=True)
        last_invest = invest.groupby('id',as_index=False)['btyear'].agg({'last_invest':'max'})
        last_negitive_invest = invest.groupby('id', as_index=False)['btendyear'].agg({'last_negitive_invest': 'max'})
        feat = data.merge(n_invest, on='id', how='left')
        feat = feat.merge(n_negitive_invest, on='id', how='left')
        feat = feat.merge(n_negitive_invest2, on='id', how='left')
        feat = feat.merge(n_negitive_invested, on='id', how='left')
        feat = feat.merge(last_invest, on='id', how='left')
        feat = feat.merge(last_negitive_invest, on='id', how='left')
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
        right['weight'] = right['askdate'].apply(lambda x:0.3**(2015-int(x[:4])))
        n_right = right.groupby('id',as_index=False)['weight'].agg({'n_right':'sum'})
        feat = data.merge(n_right, on='id', how='left')
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
        # breakfaith.drop_duplicates(['id', 'fbdate'], inplace=True)
        n_breakfaith = breakfaith.groupby('id', as_index=False)['id'].agg({'n_breakfaith': 'size'})
        feat = data.merge(n_breakfaith, on='id', how='left')
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
        base_feat = get_base_feat(stat,data,data_key)                # 添加基础特征
        alter_feat = get_alter_feat(data,data_key)              # alter特征
        branch_feat = get_branch_feat(data,data_key)            # branch特征
        invest_feat = get_invest_feat(data,data_key)            # invest特征
        right_feat = get_right_feat(data, data_key)             # right特征
        lawsuit_feat = get_lawsuit_feat(data, data_key)         # lawsuit特征
        breakfaith_feat = get_breakfaith_feat(data, data_key)   # breakfaith特征
        recruit_feat = get_recruit_feat(data, data_key)         # recruit特征

        result = concat([data[['id']],base_feat,alter_feat,branch_feat,invest_feat,right_feat,
                         lawsuit_feat,breakfaith_feat,recruit_feat])
        result = get_labels(result)
        result = second_feat(result)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    print('特征矩阵大小：{}'.format(result.shape))
    print('生成特征一共用时{}秒'.format(time.time() - t0))
    return result







