import os
import time
import pickle
import hashlib
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from collections import Counter
from collections import defaultdict
from joblib import Parallel, delayed

data_path = 'C:/Users/csw/Desktop/python/JD/moshi/data/'
cache_path = 'F:/moshi_cache/'
flag = 0

# 获取阈值
def get_threshold(preds,count=None):
    preds_temp = sorted(preds, reverse=True)
    if count is not None:
        threshold = preds_temp[count+1]
        print('阈值为：{}'.format(threshold))
    else:
        m = sum(preds) # 实际正例个数
        n = 0   # 提交的正例个数
        e = 0   # 正确个数的期望值
        f1 = 0  # f1的期望得分
        for threshold in preds_temp:
            e += threshold
            n += 1
            f1_temp = e/(0.01*m+n)
            if f1>f1_temp:
                break
            else:
                f1 = f1_temp
        print('阈值为：{}'.format(threshold))
        print('提交正例个数为：{}'.format(m-1))
        print('期望f1得分为：{}'.format(f1*1.01))
    return [(1  if (pred>threshold) else 0) for pred in preds]

# 线下测评
def f1(true,pred):
    m = np.sum(true)
    n = np.sum(pred)
    r = sum(np.array(true.astype(int)) & np.array(pred))
    return 1.01*r/(0.01*m+n)

# 线下测评
def get_label(data):
    true_path = data_path + 'true.pkl'
    try:
        true = pickle.load(open(true_path,'+rb'))
    except:
        print('没有发现真实数据，无法测评')
    data['label'] = data['rowkey'].map(true)
    return data

# 合并节约内存
def concat(L):
    result = None
    for l in L:
        if result is None:
            result = l
        else:
            result[l.columns.tolist()] = l
    return result

# 分组排序
def group_rank(data, feat1, feat2, ascending):
    data.sort_values([feat1, feat2], inplace=True, ascending=ascending)
    data['rank'] = range(data.shape[0])
    min_rank = data.groupby(feat1, as_index=False)['rank'].agg({'min_rank': 'min'})
    data = pd.merge(data, min_rank, on=feat1, how='left')
    data['rank'] = data['rank'] - data['min_rank']
    del data['min_rank']
    return data
###################################################
#..................... 构造特征 ....................
###################################################
# 连接wifi特征
def get_sample_feat(data_label,data_key):
    result_path = cache_path + 'sample_feat_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        data_label_temp = data_label.copy()
        data_label_temp['time'] = pd.to_datetime(data_label_temp['time'])
        data_label_temp['log_time'] = pd.to_datetime(data_label_temp['log_time'])
        data_label_temp['week'] = data_label_temp['time'].dt.dayofweek
        data_label_temp['hour'] = data_label_temp['time'].dt.hour
        data_label_temp['sep_time'] = (data_label_temp['time']-data_label_temp['log_time']).dt.total_seconds()
        result = data_label_temp[['hour','week','sep_time']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 获取城市购买数量
def get_city_count_feat(data, data_feat, data_key):
    result_path = cache_path + 'city_count_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        city_count = data.groupby('city',as_index=False)['id'].agg({'city_n_people':'nunique',
                                                                       'city_count':'count'})
        city_count['city_count'] = city_count['city_count'] / city_count['city_count'].mean()  # 消除时间窗口不一致带来的误差
        city_count['city_n_people'] = city_count['city_n_people'] / city_count['city_n_people'].mean()
        city_count['city_frequency'] = city_count['city_count']/city_count['city_n_people']
        result = data_feat.merge(city_count,on='city',how='left')
        result = result[['city_count', 'city_n_people', 'city_frequency']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 用户上一次购买距离本次购买的时间
def get_around_time_feat(label, data_key):
    result_path = cache_path + 'around_time_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        label_temp = label.copy()
        label_temp['time'] = pd.to_datetime(label_temp['time'])
        label_temp = group_rank(label_temp, 'id', 'time', ascending=True)
        label_temp2 = label_temp.copy().rename(columns={'time':'time2'})
        label_temp2['rank'] = label_temp2['rank'] + 1
        label_temp3 = label_temp.copy().rename(columns={'time': 'time3'})
        label_temp3['rank'] = label_temp3['rank'] - 1
        label_temp = label_temp.merge(label_temp2[['id', 'rank', 'time2']],
                                                on=['id', 'rank'], how='left')
        label_temp['pre_time'] = (label_temp['time'] - label_temp['time2']).dt.total_seconds()
        label_temp = label_temp.merge(label_temp3[['id', 'rank', 'time3']],
                                                on=['id', 'rank'], how='left')
        label_temp['next_time'] = (label_temp['time3'] - label_temp['time']).dt.total_seconds()
        result = label_temp[['pre_time', 'next_time']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 分月份统计用户之前的购买情况
def get_id_history_feat(label, train_label, data_key):
    result_path = cache_path + 'id_history_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        label_temp = label.copy()
        label_temp['month'] = label_temp['time'].str[:7]
        month_list = sorted(label_temp['month'].unique().tolist())
        train_label_temp = train_label.copy()
        train_label_temp['month'] = train_label_temp['time'].str[:7]
        result = pd.DataFrame()
        for month in month_list:
            data_temp = train_label[train_label_temp['month']<month]
            id_his_count = data_temp.groupby('id',as_index=False)['label'].agg({'id_his_count':'count',
                                                                                'id_his_sum':'sum'})
            id_his_count['id_his_rate'] = id_his_count['id_his_sum'] / id_his_count['id_his_count']
            id_his_count['month'] = month
            result = pd.concat([result,id_his_count])
        result = label_temp.merge(result,on=['id','month'],how='left')
        result = result[['id_his_sum','id_his_count','id_his_rate']].copy()
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 分月份统计本次购买连续购买了多少个
def get_countunuous_count(label,data_key):
    result_path = cache_path + 'countunuous_count_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & 0:
        result = pd.read_hdf(result_path, 'w')
    else:
        print(label.shape)
        label_temp = label.copy()
        label_temp['time'] = pd.to_datetime(label_temp['time'])
        label_temp = group_rank(label_temp, 'id', 'time', ascending=True)
        label_temp2 = label_temp.copy().rename(columns={'time': 'time2'})
        label_temp2['rank'] = label_temp2['rank'] - 1
        label_temp = label_temp.merge(label_temp2[['id', 'rank', 'time2']],
                                      on=['id', 'rank'], how='left')
        print(label_temp.shape)
        label_temp['next_time'] = (label_temp['time'] - label_temp['time2']).dt.total_seconds().fillna(-1)
        result = []; n = 0
        for next_time in label_temp['next_time'].values:
            n += 1
            if (next_time == -1) | (next_time>900):
                result += [n]*n
                n = 0
        result += [n]*n
        label_temp['contunuous_count'] = result
        result = label.merge(label_temp,on=['rowkey'],how='left')[['contunuous_count']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 二次处理特征
def second_feat(result):

    return result


# 制作训练集
def make_feats(data, label, train_label):
    t0 = time.time()
    data_key = hashlib.md5(data['id'].to_string().encode()).hexdigest()+\
               hashlib.md5(label['id'].to_string().encode()).hexdigest()
    print('数据key为：{}'.format(data_key))
    result_path = cache_path + 'train_set_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & 0:
        result = pd.read_hdf(result_path, 'w')
    else:
        print('找到trade对应的log行为...')
        data_label = label.merge(data, on='id', how='left')
        data_label = data_label[data_label['time'] >= data_label['log_time']]
        data_label = data_label[data_label['result']!=31]
        data_label.sort_values('log_time', inplace=True)
        data_label.drop_duplicates('rowkey', keep='last', inplace=True)
        data_label = label[['rowkey','time','id']].merge(data_label.drop('time',axis=1),on=['rowkey'],how='left')
        data_label.reset_index(inplace=True,drop=True)

        print('开始构造特征...')
        sample = get_sample_feat(data_label, data_key)                  # 构造简单特征
        city_count = get_city_count_feat(data, data_label, data_key)    # 获取城市购买数量
        around_time = get_around_time_feat(label, data_key)             # 用户上一次和下一次购买距离本次购买的时间
        id_history = get_id_history_feat(label, train_label, data_key)  # 分月份统计用户之前的购买情况
        # 分月份统计ip之前的购买情况
        # 分月份统计device之前的购买情况
        continuous_count = get_countunuous_count(label,data_key)        # 分月份统计本次购买连续购买了多少个
        # 本次登录距离下一次购买的时间

        print('开始合并特征...')
        result = concat([data_label,sample,city_count,around_time,id_history,
                         continuous_count])
        result = second_feat(result)

        print('添加label')
        result = get_label(result)

        print('存储数据...')
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    print('特征矩阵大小：{}'.format(result.shape))
    print('生成特征一共用时{}秒'.format(time.time() - t0))
    return result






















