# -*-coding:utf-8 -*-
import os
import gc
import time
import random
import pickle
import Geohash
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm
import lightgbm as lgb
import multiprocessing
from sklearn.cross_validation import KFold


cache_path = '/home1/csw/cache/'
train_path = '/home1/csw/data/train.csv'
test_path = '/home1/csw/data/test.csv'

flag = True
# 相差的分钟数
def diff_of_minutes(time1, time2):
    d = {'5': 0, '6': 31, }
    try:
        days = (d[time1[6]] + int(time1[8:10])) - (d[time2[6]] + int(time2[8:10]))
        try:
            minutes1 = int(time1[11:13]) * 60 + int(time1[14:16])
        except:
            minutes1 = 0
        try:
            minutes2 = int(time2[11:13]) * 60 + int(time2[14:16])
        except:
            minutes2 = 0
        return (days * 1440 - minutes2 + minutes1)
    except:
        return np.nan

# 判断是否为节假日:
def if_holiday(date):
    holiday_list = [13,14,20,21,28,29,30]
    return 1 if int(date) in holiday_list else 0

# 计算夹角余弦值
def cal_cos(start_lat,start_lon,end_lat,end_lon,act_start_lat,act_start_lon,act_end_lat,act_end_lon):
    x1 = end_lon - start_lon
    y1 = end_lat - start_lat
    x2 = act_end_lon - act_start_lon
    y2 = act_end_lat - act_start_lat
    result = (x1*x2 + y1*y2) / (x1**2 + y1**2 + x2**2 + y2**2)**0.5
    return result

# 组内标准化
def group_normalize(data,key,feat):
    grp = data.groupby(key,as_index=False)[feat].agg({'std':'std','avg':'mean'})
    result = pd.merge(data,grp,on=key,how='left')
    result[feat] = ((result[feat]-result['std']) / result['avg']).fillna(1)
    return result[feat]

# 计算两点之间距离
def cal_distance(lat1,lon1,lat2,lon2):
    dx = np.abs(lon1 - lon2)  # 经度差
    dy = np.abs(lat1 - lat2)  # 维度差
    b = (lat1 + lat2) / 2.0
    Lx = 6371004.0 * (dx / 57.2958) * np.cos(b / 57.2958)
    Ly = 6371004.0 * (dy / 57.2958)
    L = (Lx**2 + Ly**2) ** 0.5
    return L

# 计算两点之间距离
def cal_mht_distance(lat1,lon1,lat2,lon2):
    dx = np.abs(lon1 - lon2)  # 经度差
    dy = np.abs(lat1 - lat2)  # 维度差
    b = (lat1 + lat2) / 2.0
    Lx = 6371004.0 * (dx / 57.2958) * np.cos(b / 57.2958)
    Ly = 6371004.0 * (dy / 57.2958)
    L = Lx + Ly
    return L

# 相差的分钟数
def diff_of_minutes(time1, time2):
    try:
        minutes1 = int(time1[11:13]) * 60 + int(time1[14:16])
        minutes2 = int(time2[11:13]) * 60 + int(time2[14:16])
        return min((np.abs(minutes1-minutes2)),(1440-np.abs(minutes1-minutes2)))
    except:
        return np.nan

# 分组排序
def rank(data, feat1, feat2, ascending):
    data.sort_values([feat1,feat2],inplace=True,ascending=ascending)
    data['rank'] = range(data.shape[0])
    min_rank = data.groupby(feat1,as_index=False)['rank'].agg({'min_rank':'min'})
    data = pd.merge(data,min_rank,on=feat1,how='left')
    data['rank'] = data['rank'] - data['min_rank']
    del data['min_rank']
    return data

# 解码字典
def get_loc_dict():
    dump_path = cache_path + 'loc_dict.pkl'
    if os.path.exists(dump_path):
        loc_dict = pickle.load(open(dump_path, 'rb+'))
    else:
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        locs = list(set(train['geohashed_start_loc']) | set(train['geohashed_end_loc']) | set(test['geohashed_start_loc']))
        deloc = []
        for loc in locs:
            deloc.append(Geohash.decode_exactly(loc)[:2])
        loc_dict = dict(zip(locs, deloc))
        pickle.dump(loc_dict, open(dump_path, 'wb+'))
    return loc_dict

# 计算两点之间的欧氏距离和曼哈顿距离
def get_distance(sample):
    loc_dict = get_loc_dict()
    geohashed_loc = sample[['geohashed_start_loc','geohashed_end_loc']].values
    distance = []
    for i in geohashed_loc:
        loc1, loc2 = i
        if (loc1 is np.nan) | (loc2 is np.nan):
            distance.append(np.nan)
            continue
        lat1, lon1 = loc_dict[loc1]
        lat2, lon2 = loc_dict[loc2]
        distance.append(cal_distance(lat1,lon1,lat2,lon2))
    sample.loc[:,'distance'] = distance
    return sample

# 对结果进行整理
def reshape(pred):
    result = pred.copy()
    result = rank(result,'orderid','pred',ascending=False)
    result = result[result['rank']<3][['orderid','geohashed_end_loc','rank']]
    result = result.set_index(['orderid','rank']).unstack()
    result.reset_index(inplace=True)
    result['orderid'] = result['orderid'].astype('int')
    result.columns = ['orderid', 0, 1, 2]
    return result

# 测评函数
def map(result):
    result_path = cache_path + 'true.pkl'
    if os.path.exists(result_path):
        true = pickle.load(open(result_path, 'rb+'))
    else:
        train = pd.read_csv(train_path)
        true = dict(zip(train['orderid'].values,train['geohashed_end_loc']))
        pickle.dump(true,open(result_path, 'wb+'))
    data = result.copy()
    data['true'] = data['orderid'].map(true)
    score = (sum(data['true']==data[0])
             +sum(data['true']==data[1])/2
             +sum(data['true']==data[2])/3)/data.shape[0]
    return score

# 对结果添加噪音
def get_noise(data,rate=0.1,seed=66):
    random.seed(seed)
    data_temp = data.copy()
    index = list(range(data.shape[0]))
    data.index = index
    random.shuffle(index)
    n = int(rate*data.shape[0])
    sub_index = index[:n]
    data_temp.loc[sub_index,[0,1,2]] = 'a'
    return data_temp

# 获取争取标签
def get_label(data):
    result_path = cache_path + 'true.pkl'
    if os.path.exists(result_path):
        true = pickle.load(open(result_path, 'rb+'))
    else:
        train = pd.read_csv(train_path)
        true = dict(zip(train['orderid'].values, train['geohashed_end_loc']))
        pickle.dump(true, open(result_path, 'wb+'))
    data.loc[:,'label'] = data['orderid'].map(true)
    data['label'] = (data['label'] == data['geohashed_end_loc']).astype('int')
    data['label'] = data['label'].fillna(0)
    return data

def get_near_loc():
    result_path = cache_path + 'near_loc.hdf'
    if os.path.exists(result_path):
        result = pd.read_hdf(result_path, 'w')
    else:
        import geohash
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        loc_list = train['geohashed_start_loc'].tolist() \
                   + train['geohashed_end_loc'].tolist() \
                   + test['geohashed_start_loc'].tolist()
        loc_list = np.unique(loc_list)
        result = []
        for loc in loc_list:
            nlocs = geohash.neighbors(loc)
            nlocs.append(loc)
            for nloc in nlocs:
                result.append([loc, nloc])
        result = pd.DataFrame(result, columns=['loc', 'near_loc'])
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 获取邻近小时
def get_near_hour():
    hour1 = [23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    hour2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    hour3 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0]
    result = []
    for i in hour2:
        result.append([i, hour1[i]])
        result.append([i, hour2[i]])
        result.append([i, hour3[i]])
    result = pd.DataFrame(result,columns=['hour','near_hour'])
    return result

def get_near_loc_df_9_7():
    result_path = cache_path + 'near_loc_9_7_df.hdf'
    if os.path.exists(result_path) & flag:
        result = pickle.load(open(result_path, 'rb+'))
    else:
        import geohash
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        loc_set = train['geohashed_start_loc'].tolist() \
                   + train['geohashed_end_loc'].tolist() \
                   + test['geohashed_start_loc'].tolist()
        loc_set = set(loc_set)
        result = []
        for loc in loc_set:
            loc1 = geohash.neighbors(loc)
            loc2 = geohash.neighbors(loc1[0])
            loc3 = geohash.neighbors(loc2[3])
            loc4 = geohash.neighbors(loc3[3])
            loc5 = geohash.neighbors(loc3[6])
            loc6 = geohash.neighbors(loc2[6])
            loc7 = geohash.neighbors(loc6[6])
            loc8 = geohash.neighbors(loc1[1])
            loc9 = geohash.neighbors(loc8[4])
            loc10 = geohash.neighbors(loc9[4])
            loc11 = geohash.neighbors(loc9[7])
            loc12 = geohash.neighbors(loc8[7])
            loc13 = geohash.neighbors(loc12[7])
            loc14 = geohash.neighbors(loc1[2])
            loc15 = geohash.neighbors(loc14[2])
            loc16 = geohash.neighbors(loc1[5])
            loc17 = geohash.neighbors(loc16[5])
            result_sub = [loc] + loc1 + loc2 + loc3 + loc4 + loc5 + loc6 + loc7 + loc8 + \
                         loc9 + loc10 + loc11 + loc12 + loc13 + loc14 + loc15 + loc16 + loc17
            result_sub = list(set(result_sub) & loc_set)
            for l in result_sub:
                result.append([loc,l])
        result = pd.DataFrame(result,columns=['loc','near_loc'])
        pickle.dump(result, open(result_path, 'wb+'))
    return result

####################构造负样本##################

# 将用户骑行过目的的地点加入成样本
def get_user_end_loc(data, candidate, data_key):
    result_path = cache_path + 'user_end_loc_%d.hdf' % (data_key)
    if os.path.exists(result_path):
        result = pd.read_hdf(result_path, 'w')
    else:
        user_n_end_loc = data.groupby(['userid', 'geohashed_end_loc'], as_index=False)['userid'].agg(
            {'user_eloc_count': 'count'})
        user_n_end_loc = user_n_end_loc[~user_n_end_loc['geohashed_end_loc'].isnull()]
        result = pd.merge(candidate[['orderid', 'userid']], user_n_end_loc, on=['userid'], how='inner')
        result = result[['orderid', 'geohashed_end_loc']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    print('user_end_loc样本个数为：{}'.format(result.shape))
    return result

# 将用户骑行过出发的地点加入成样本
def get_user_start_loc(data, candidate, data_key):
    result_path = cache_path + 'user_start_loc_%d.hdf' % (data_key)
    if os.path.exists(result_path):
        result = pd.read_hdf(result_path, 'w')
    else:
        user_n_start_loc = data.groupby(['userid', 'geohashed_start_loc'], as_index=False)['userid'].agg(
            {'user_sloc_count': 'count'})
        result = pd.merge(candidate[['orderid', 'userid']], user_n_start_loc, on=['userid'], how='inner')
        result.rename(columns={'geohashed_start_loc': 'geohashed_end_loc'}, inplace=True)
        result = result[['orderid', 'geohashed_end_loc']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    print('user_start_loc样本个数为：{}'.format(result.shape))
    return result

# 筛选起始地点去向最多的3个地点
def get_loc_to_loc(data, candidate, data_key):
    result_path = cache_path + 'loc_to_loc_%d.hdf' % (data_key)
    if os.path.exists(result_path):
        result = pd.read_hdf(result_path, 'w')
    else:
        sloc_eloc_count = data.groupby(['geohashed_start_loc', 'geohashed_end_loc'], as_index=False)['orderid'].agg(
            {'sloc_eloc_count': 'count'})
        sloc_eloc_count = sloc_eloc_count[~sloc_eloc_count['geohashed_end_loc'].isnull()]
        sloc_eloc_count.sort_values('sloc_eloc_count', inplace=True)
        sloc_eloc_count = sloc_eloc_count.groupby('geohashed_start_loc').tail(6)
        result = pd.merge(candidate[['orderid', 'geohashed_start_loc']], sloc_eloc_count, on='geohashed_start_loc', how='inner')
        result = result[['orderid', 'geohashed_end_loc']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    print('loc_to_loc样本个数为：{}'.format(result.shape))
    return result

# 与其交互最多的三个地点
def get_loc_with_loc(data, candidate, data_key):
    result_path = cache_path + 'loc_with_loc_%d.hdf' % (data_key)
    if os.path.exists(result_path):
        result = pd.read_hdf(result_path, 'w')
    else:
        sloc_eloc_count = data.groupby(['geohashed_start_loc', 'geohashed_end_loc'], as_index=False)['orderid'].agg(
            {'sloc_eloc_count': 'count'})
        eloc_sloc_count = sloc_eloc_count.rename(columns={'geohashed_start_loc': 'geohashed_end_loc',
                                                          'geohashed_end_loc': 'geohashed_start_loc',
                                                          'sloc_eloc_count': 'eloc_sloc_count'}).copy()
        sloc_eloc_2count = pd.merge(sloc_eloc_count, eloc_sloc_count,
                                    on=['geohashed_start_loc', 'geohashed_end_loc'], how='outer').fillna(0)
        sloc_eloc_2count['sloc_eloc_2count'] = sloc_eloc_2count['sloc_eloc_count'] + sloc_eloc_2count['eloc_sloc_count']
        sloc_eloc_2count.sort_values('sloc_eloc_2count', inplace=True)
        sloc_eloc_2count = sloc_eloc_2count.groupby('geohashed_start_loc').tail(6)
        result = pd.merge(candidate[['orderid', 'geohashed_start_loc']], sloc_eloc_2count, on='geohashed_start_loc', how='inner')
        result = result[['orderid', 'geohashed_end_loc']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    print('loc_with_loc样本个数为：{}'.format(result.shape))
    return result

# 扩大范围筛选起始地点去向最多的3个地点
def get_ex_loc_to_loc(data, candidate, data_key):
    result_path = cache_path + 'ex_loc_to_loc_%d.hdf' % (data_key)
    if os.path.exists(result_path):
        result = pd.read_hdf(result_path, 'w')
    else:
        sloc_eloc_count = data.groupby(['geohashed_start_loc', 'geohashed_end_loc'], as_index=False)['orderid'].agg(
            {'sloc_eloc_count': 'count'})
        near_loc = get_near_loc()
        sloc_eloc_count = sloc_eloc_count.merge(near_loc,left_on='geohashed_start_loc',right_on='loc',how='left')
        sloc_eloc_count['geohashed_start_loc'] = sloc_eloc_count['near_loc']
        sloc_eloc_count = sloc_eloc_count.groupby(['geohashed_start_loc', 'geohashed_end_loc'],as_index=False).sum()
        sloc_eloc_count = sloc_eloc_count[~sloc_eloc_count['geohashed_end_loc'].isnull()]
        sloc_eloc_count.sort_values('sloc_eloc_count', inplace=True)
        sloc_eloc_count = sloc_eloc_count.groupby('geohashed_start_loc').tail(6)
        result = pd.merge(candidate[['orderid', 'geohashed_start_loc']], sloc_eloc_count,on='geohashed_start_loc', how='inner')
        result = result[['orderid', 'geohashed_end_loc']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    print('ex_loc_to_loc样本个数为：{}'.format(result.shape))
    return result

# 是否工作日去向最多的三个地点
def get_holiday_loc_to_loc(data, candidate, data_key):
    result_path = cache_path + 'holiday_loc_to_loc_%d.hdf' % (data_key)
    if os.path.exists(result_path):
        result = pd.read_hdf(result_path, 'w')
    else:
        result = data.groupby(['holiday', 'geohashed_start_loc', 'geohashed_end_loc'], as_index=False)['orderid'].agg(
            {'holiday_sloc_eloc_count': 'count'})
        result.sort_values('holiday_sloc_eloc_count', inplace=True)
        result = result.groupby(['holiday', 'geohashed_start_loc']).tail(3)
        result = pd.merge(candidate[['holiday', 'orderid', 'geohashed_start_loc']], result,
                          on=['holiday', 'geohashed_start_loc'],how='inner')
        result = result[['orderid', 'geohashed_end_loc']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    print('holiday_loc_to_loc样本个数为：{}'.format(result.shape))
    return result

# 三个小时内去向最多的三个地点
def get_exhour_loc_to_loc(data, candidate, data_key):
    result_path = cache_path + 'exhour_loc_to_loc_%d.hdf' % (data_key)
    if os.path.exists(result_path):
        result = pd.read_hdf(result_path, 'w')
    else:
        result = data.groupby(['hour', 'geohashed_start_loc', 'geohashed_end_loc'], as_index=False)[
            'orderid'].agg({'exhour_sloc_eloc_count': 'count'})
        near_hour = get_near_hour()
        result = result.merge(near_hour,on='hour',how='left')
        result['hour'] = result['near_hour']
        result = result.groupby(['hour', 'geohashed_start_loc', 'geohashed_end_loc'], as_index=False)[
            'exhour_sloc_eloc_count'].sum()
        result.sort_values('exhour_sloc_eloc_count', inplace=True)
        result = result.groupby(['hour', 'geohashed_start_loc']).tail(3)
        result = pd.merge(candidate[['hour', 'orderid', 'geohashed_start_loc']], result,on=['hour','geohashed_start_loc'],how='inner')
        result = result[['orderid', 'geohashed_end_loc']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    print('exhour_loc_to_loc样本个数为：{}'.format(result.shape))
    return result

# 扩大范围三个小时内去向最多的三个地点
def get_exhour_ex_loc_to_loc(data, candidate, data_key):
    result_path = cache_path + 'exhour_ex_loc_to_loc_%d.hdf' % (data_key)
    if os.path.exists(result_path):
        result = pd.read_hdf(result_path, 'w')
    else:
        result = data.groupby(['hour', 'geohashed_start_loc', 'geohashed_end_loc'], as_index=False)[
            'orderid'].agg({'exhour_ex_sloc_eloc_count': 'count'})
        near_hour = get_near_hour()
        result = result.merge(near_hour, on='hour', how='left')
        result['hour'] = result['near_hour']
        result = result.groupby(['hour', 'geohashed_start_loc', 'geohashed_end_loc'], as_index=False)[
            'exhour_ex_sloc_eloc_count'].sum()
        near_loc = get_near_loc()
        result = result.merge(near_loc, left_on='geohashed_start_loc', right_on='loc', how='left').fillna(0)
        result['geohashed_start_loc'] = result['near_loc']
        result = result.groupby(['hour', 'geohashed_start_loc', 'geohashed_end_loc'], as_index=False).sum()
        result.sort_values('exhour_ex_sloc_eloc_count', inplace=True)
        result = result.groupby(['hour', 'geohashed_start_loc']).tail(3)
        result = pd.merge(candidate[['hour', 'orderid', 'geohashed_start_loc']], result,
                          on=['hour', 'geohashed_start_loc'], how='inner')
        result = result[['orderid', 'geohashed_end_loc']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    print('exhour_ex_loc_to_loc样本个数为：{}'.format(result.shape))
    return result

# 交易日 三个小时去向最多的地点
def get_holiday_exhour_loc_to_loc(data, candidate, data_key):
    result_path = cache_path + 'holiday_exhour_loc_to_loc_%d.hdf' % (data_key)
    if os.path.exists(result_path):
        result = pd.read_hdf(result_path, 'w')
    else:
        result = data.groupby(['hour', 'holiday','geohashed_start_loc', 'geohashed_end_loc'], as_index=False)[
            'orderid'].agg({'holiday_exhour_sloc_eloc_count': 'count'})
        near_hour = get_near_hour()
        result = result.merge(near_hour, on='hour', how='left')
        result['hour'] = result['near_hour']
        result = result.groupby(['hour','holiday', 'geohashed_start_loc', 'geohashed_end_loc'], as_index=False)[
            'holiday_exhour_sloc_eloc_count'].sum()
        result.sort_values('holiday_exhour_sloc_eloc_count', inplace=True)
        result = result.groupby(['hour','holiday', 'geohashed_start_loc']).tail(3)
        result = pd.merge(candidate[['hour','holiday', 'orderid', 'geohashed_start_loc']], result,
                          on=['hour','holiday', 'geohashed_start_loc'], how='inner')
        result = result[['orderid', 'geohashed_end_loc']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    print('holiday_exhour_loc_to_loc样本个数为：{}'.format(result.shape))
    return result

# 交易日 扩大范围最多的三个地点
def get_holiday_ex_loc_to_loc(data, candidate, data_key):
    result_path = cache_path + 'holiday_ex_loc_to_loc_%d.hdf' % (data_key)
    if os.path.exists(result_path):
        result = pd.read_hdf(result_path, 'w')
    else:
        result = data.groupby(['holiday', 'geohashed_start_loc', 'geohashed_end_loc'], as_index=False)['orderid'].agg(
            {'holiday_ex_sloc_eloc_count': 'count'})
        near_loc = get_near_loc()
        result = result.merge(near_loc, left_on='geohashed_start_loc', right_on='loc', how='left')
        result['geohashed_start_loc'] = result['near_loc']
        result = result.groupby(['holiday', 'geohashed_start_loc', 'geohashed_end_loc'], as_index=False).sum()
        result.sort_values('holiday_ex_sloc_eloc_count', inplace=True)
        result = result.groupby(['holiday','geohashed_start_loc']).tail(4)
        result = pd.merge(candidate[['orderid','holiday','geohashed_start_loc']], result,
                          on=['holiday','geohashed_start_loc'],how='inner')
        result = result[['orderid', 'geohashed_end_loc']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    print('holiday_ex_loc_to_loc样本个数为：{}'.format(result.shape))
    return result

# 交易日 三个小时 扩大范围去向最多的三个地点
def get_holiday_exhour_ex_loc_to_loc(data, candidate, data_key):
    result_path = cache_path + 'holiday_exhour_ex_loc_to_loc_%d.hdf' % (data_key)
    if os.path.exists(result_path):
        result = pd.read_hdf(result_path, 'w')
    else:
        result = data.groupby(['hour', 'holiday', 'geohashed_start_loc', 'geohashed_end_loc'], as_index=False)[
            'orderid'].agg({'holiday_exhour_ex_sloc_eloc_count': 'count'})
        near_hour = get_near_hour()
        result = result.merge(near_hour, on='hour', how='left')
        result['hour'] = result['near_hour']
        result = result.groupby(['hour', 'holiday','geohashed_start_loc', 'geohashed_end_loc'], as_index=False).sum()
        near_loc = get_near_loc()
        result = result.merge(near_loc, left_on='geohashed_start_loc', right_on='loc', how='left').fillna(0)
        result['geohashed_start_loc'] = result['near_loc']
        result = result.groupby(['hour', 'holiday','geohashed_start_loc', 'geohashed_end_loc'], as_index=False).sum()
        result.sort_values('holiday_exhour_ex_sloc_eloc_count', inplace=True)
        result = result.groupby(['hour','holiday', 'geohashed_start_loc']).tail(3)
        result = pd.merge(candidate[['hour','holiday', 'orderid', 'geohashed_start_loc']], result,
                          on=['hour', 'holiday', 'geohashed_start_loc'], how='inner')
        result = result[['orderid', 'geohashed_end_loc']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    print('holiday_exhour_ex_loc_to_loc样本个数为：{}'.format(result.shape))
    return result

# 周围9×7中的前3
def get_near_loc_to_loc(data, candidate, data_key):
    result_path = cache_path + 'near_loc_to_loc_%d.hdf' % (data_key)
    if os.path.exists(result_path):
        result = pd.read_hdf(result_path, 'w')
    else:
        eloc_count_dict = data.groupby(['geohashed_end_loc'], as_index=False).size().to_dict()
        near_loc_df = get_near_loc_df_9_7()
        near_loc_df.loc[:,'eloc_count'] = near_loc_df['near_loc'].map(eloc_count_dict)
        near_loc_df.sort_values('eloc_count',inplace=True)
        near_loc_df = near_loc_df.groupby('loc').tail(3)
        result = pd.merge(candidate[['orderid', 'geohashed_start_loc']], near_loc_df,
                          left_on='geohashed_start_loc',right_on='loc', how='inner')
        result.rename(columns={'near_loc':'geohashed_end_loc'},inplace=True)
        result = result[['orderid', 'geohashed_end_loc']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    print('near_loc_to_loc样本个数为：{}'.format(result.shape))
    return result

# 扩大范围筛选起始地点交互最多的3个地点
def get_ex_loc_with_loc(data, candidate, data_key):
    result_path = cache_path + 'ex_loc_with_loc_%d.hdf' % (data_key)
    if os.path.exists(result_path):
        result = pd.read_hdf(result_path, 'w')
    else:
        sloc_eloc_count = data.groupby(['geohashed_start_loc', 'geohashed_end_loc'], as_index=False)['orderid'].agg(
            {'sloc_eloc_count': 'count'})
        eloc_sloc_count = sloc_eloc_count.rename(columns={'geohashed_start_loc':'geohashed_end_loc',
                                                          'geohashed_end_loc':'geohashed_start_loc',
                                                          'sloc_eloc_count':'eloc_sloc_count'})
        result = pd.merge(sloc_eloc_count,eloc_sloc_count,on=['geohashed_start_loc', 'geohashed_end_loc'],how='outer')
        near_loc = get_near_loc()
        result = result.merge(near_loc,left_on='geohashed_start_loc',right_on='loc',how='left').fillna(0)
        result['geohashed_start_loc'] = result['near_loc']
        result = result.groupby(['geohashed_start_loc', 'geohashed_end_loc'],as_index=False).sum()
        result['sloc_eloc_2count'] = result['sloc_eloc_count'] + result['eloc_sloc_count']
        result = result[~result['geohashed_end_loc'].isnull()]
        result.sort_values('sloc_eloc_2count', inplace=True)
        result = result.groupby('geohashed_start_loc').tail(3)
        result = pd.merge(candidate[['orderid', 'geohashed_start_loc']], result,on='geohashed_start_loc', how='inner')
        result = result[['orderid', 'geohashed_end_loc']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    print('ex_loc_with_loc样本个数为：{}'.format(result.shape))
    return result


# 用户后续的起始地点
def get_user_next_loc(data, sample, data_key):
    result_path = cache_path + 'user_next_loc_%d.hdf' % (data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        data_temp = data.copy()
        data_temp = data_temp[data_temp['userid'].isin(sample['userid'].tolist())]
        data_temp = rank(data_temp, 'userid', 'starttime', ascending=True)
        result = pd.merge(sample,data_temp[['orderid', 'rank']], on='orderid',how='left')
        data_temp['rank'] = data_temp['rank'] - 1
        data_temp.rename(columns={'geohashed_start_loc': 'geohashed_start_loc2',
                                  'geohashed_end_loc': 'geohashed_end_loc2',
                                  'starttime': 'starttime2'}, inplace=True)
        result = pd.merge(result,data_temp[['userid', 'rank', 'geohashed_start_loc2', 'starttime2']], on=['userid', 'rank'], how='left')
        loc_dict = get_loc_dict()
        user_eloc_nloc_sep_time = []
        user_eloc_nloc_distance = []
        user_eloc_nloc_mdistance = []
        for tuple in result.itertuples():
            if tuple.geohashed_start_loc2 is np.nan:
                user_eloc_nloc_sep_time.append(-1)
                user_eloc_nloc_distance.append(-1)
                user_eloc_nloc_mdistance.append(-1)
            else:
                user_eloc_nloc_sep_time.append(diff_of_minutes(tuple.starttime, tuple.starttime2))
                lat1, lon1 = loc_dict[tuple.geohashed_start_loc2]
                lat2, lon2 = loc_dict[tuple.geohashed_end_loc]
                user_eloc_nloc_distance.append(cal_distance(lat1, lon1, lat2, lon2))
                user_eloc_nloc_mdistance.append(cal_mht_distance(lat1, lon1, lat2, lon2))
        result['user_eloc_nloc_sep_time'] = user_eloc_nloc_sep_time
        result['user_eloc_nloc_distance'] = user_eloc_nloc_distance
        result['user_eloc_nloc_mdistance'] = user_eloc_nloc_mdistance
        result = result[['user_eloc_nloc_sep_time', 'user_eloc_nloc_distance', 'user_eloc_nloc_mdistance']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


# bike后续的起始地点
def get_bike_next_loc(data, candidate, data_key):
    result_path = cache_path + 'bike_next_loc_%d.hdf' % (data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        data_temp = data.copy()
        data_temp = data_temp[data_temp['bikeid'].isin(candidate['bikeid'].values)]
        if data_temp.shape[0] == 0:
            return pd.DataFrame(columns=['orderid', 'geohashed_end_loc', 'bike_eloc_sep_time'])
        data_temp = rank(data_temp, 'bikeid', 'starttime', ascending=True)
        data_temp_temp = data_temp.copy()
        data_temp_temp['rank'] = data_temp_temp['rank'] - 1
        data_temp = pd.merge(candidate[['orderid']], data_temp, on='orderid', how='left')
        result = pd.merge(data_temp[['orderid', 'bikeid', 'rank', 'starttime']],
                          data_temp_temp[['bikeid', 'rank', 'geohashed_start_loc', 'starttime']], on=['bikeid', 'rank'],
                          how='inner')
        result['bike_eloc_sep_time'] = result.apply(lambda x: diff_of_minutes(x['starttime_y'], x['starttime_x']),
                                                    axis=1)
        result.rename(columns={'geohashed_start_loc': 'geohashed_end_loc'}, inplace=True)
        result = result[['orderid', 'geohashed_end_loc', 'bike_eloc_sep_time']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    print('bike_next_loc样本个数为：{}'.format(result.shape))
    return result

# 构造样本
def get_sample(data, candidate, data_key):
    result_path = cache_path + 'sample_%d.hdf' % (data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        user_end_loc = get_user_end_loc(data,candidate, data_key)  # 根据用户历史目的地点添加样本 ['orderid', 'geohashed_end_loc', 'user_n_end_loc']
        user_start_loc = get_user_start_loc(data,candidate, data_key)  # 根据用户历史起始地点添加样本 ['orderid', 'geohashed_end_loc', 'user_n_start_loc']
        loc_to_loc = get_loc_to_loc(data, candidate, data_key)  # 筛选起始地点去向最多的6个地点
        loc_with_loc = get_loc_with_loc(data, candidate, data_key)  # 与起始地点交互最多的6个地点
        ex_loc_with_loc = get_ex_loc_with_loc(data, candidate, data_key)  # 扩大与起始地点交互最多的三个地点
        ex_loc_to_loc = get_ex_loc_to_loc(data, candidate, data_key)  # 扩大筛选起始地点去向最多的3个地点
        holiday_loc_to_loc = get_holiday_loc_to_loc(data, candidate, data_key)  # 是否工作日去向最多的三个地点
        exhour_loc_to_loc = get_exhour_loc_to_loc(data, candidate, data_key)  # 三个小时内去向最多的三个地点
        exhour_ex_loc_to_loc = get_exhour_ex_loc_to_loc(data, candidate, data_key)  # 扩大范围三个小时内去向最多的三个地点
        holiday_exhour_loc_to_loc = get_holiday_exhour_loc_to_loc(data, candidate, data_key)  # 交易日 三个小时去向最多的地点
        holiday_ex_loc_to_loc = get_holiday_ex_loc_to_loc(data, candidate, data_key)  # 交易日 扩大范围最多的三个地点
        holiday_exhour_ex_loc_to_loc = get_holiday_exhour_ex_loc_to_loc(data, candidate,data_key)  # 交易日 三个小时 扩大范围去向最多的三个地点
        near_loc_to_loc = get_near_loc_to_loc(data, candidate, data_key)  # 周围9×7中的前3
        #bike_next_loc = get_bike_next_loc(data, candidate, data_key)  # 自行车后续起始地点
        # 汇总样本id
        result = pd.concat([user_end_loc[['orderid', 'geohashed_end_loc']],
                            user_start_loc[['orderid', 'geohashed_end_loc']],
                            loc_to_loc[['orderid', 'geohashed_end_loc']],
                            loc_with_loc[['orderid', 'geohashed_end_loc']],
                            ex_loc_to_loc[['orderid', 'geohashed_end_loc']],
                            ex_loc_with_loc[['orderid', 'geohashed_end_loc']],
                            holiday_loc_to_loc[['orderid', 'geohashed_end_loc']],
                            exhour_loc_to_loc[['orderid', 'geohashed_end_loc']],
                            exhour_ex_loc_to_loc[['orderid', 'geohashed_end_loc']],
                            holiday_exhour_loc_to_loc[['orderid', 'geohashed_end_loc']],
                            holiday_ex_loc_to_loc[['orderid', 'geohashed_end_loc']],
                            holiday_exhour_ex_loc_to_loc[['orderid', 'geohashed_end_loc']],
                            near_loc_to_loc[['orderid', 'geohashed_end_loc']],
                            #bike_next_loc[['orderid', 'geohashed_end_loc']]
                            ]).drop_duplicates()
        candidate_temp = candidate[['orderid', 'userid', 'bikeid', 'biketype', 'starttime',
                                    'geohashed_start_loc']].copy()
        result = pd.merge(result, candidate_temp, on='orderid', how='left')
        # 删除起始地点和目的地点相同的样本  和 异常值
        result = result[result['geohashed_end_loc'] != result['geohashed_start_loc']]
        result = result[(~result['geohashed_end_loc'].isnull()) & (~result['geohashed_start_loc'].isnull())]
        result.index = list(range(result.shape[0]))
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    print('样本个数为：{}'.format(result.shape))
    return result

# 构造特征
def get_single_feat(data,candidate):
    print('样本个数：{}'.format(candidate.shape[0]))
    loc_dict = get_loc_dict()
    data_temp = data[['userid', 'starttime','geohashed_start_loc', 'geohashed_end_loc','distance']]
    data_temp = data_temp[~data_temp['geohashed_end_loc'].isnull()]
    user_actions = dict()
    for tuple in data_temp.itertuples():
        user = int(tuple[1])
        if user_actions.__contains__(user):
            user_actions[user].append(tuple[2:])
        else:
            user_actions[user] = [tuple[2:]]

    result = []
    candidate_temp = candidate[['orderid','userid', 'starttime','geohashed_start_loc','geohashed_end_loc','label']]
    for tuple in tqdm(candidate_temp.itertuples()):
        orderid,userid,time,start_loc,end_loc,label = tuple[1:]
        start_lat, start_lon = loc_dict[start_loc]
        end_lat, end_lon = loc_dict[end_loc]
        self_dis = int(cal_distance(end_lat, end_lon, start_lat, start_lon))
        if user_actions.__contains__(userid):
            tup = user_actions[userid]
            for action in tup:
                act_time, act_start_loc, act_end_loc, distance = action
                if time != act_time:
                    act_start_lat,  act_start_lon   = loc_dict[act_start_loc]
                    act_end_lat,    act_end_lon     = loc_dict[act_end_loc]
                    start_dis   = int(cal_distance(start_lat, start_lon, act_start_lat, act_start_lon))
                    end_dis     = int(cal_distance(end_lat, end_lon, act_end_lat, act_end_lon))
                    se_dis      = int(cal_distance(end_lat, end_lon, act_start_lat, act_start_lon))
                    dis_rate    = self_dis/distance
                    diff_time = diff_of_minutes(time,act_time)
                    holiday = sum((if_holiday(time[11:13]),if_holiday(act_time[11:13])))
                    cos = cal_cos(start_lat,start_lon,end_lat,end_lon,act_start_lat,act_start_lon,act_end_lat,act_end_lon)
                    result.append([orderid,end_loc,start_dis,end_dis,diff_time,holiday,cos,self_dis,se_dis,dis_rate,label])
    del loc_dict,data_temp,user_actions
    gc.collect()
    result = pd.DataFrame(result,columns=['orderid','geohashed_end_loc','start_dis','end_dis','diff_time','holiday','cos','self_dis','se_dis','dis_rate','label'])
    return result


# 制作训练集
def make_train_set(data,candidate):
    t0 = time.time()
    result_path = cache_path + 'single_train_set_%d.hdf' % (data['orderid'].sum() * candidate['orderid'].sum())
    if os.path.exists(result_path) & flag:
        feat = pd.read_hdf(result_path, 'w')
    else:
        # 汇总样本id
        print('开始构造样本...')
        sample = get_sample(data,candidate)
        sample = get_label(sample)
        gc.collect()

        print('开始构造特征...')
        data = get_distance(data)
        feat = get_single_feat(data,sample)
        gc.collect()

    print('生成特征一共用时{}秒'.format(time.time()-t0))
    return feat

print('制作测试集...')
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
test.loc[:, 'geohashed_end_loc'] = np.nan
data = pd.concat([train,test])
data_list1 = pd.date_range('2017-05-10 00:00:00','2017-06-02 00:00:00')
for i in range(len(data_list1)-1):
    data_path = cache_path + str(data_list1[i])[:10] + '.hdf'
    if (os.path.exists(data_path) | (i == 7)) & flag:
        continue
    else:
        gc.collect()
        print('构造{}号训练集...'.format(str(data_list1[i])[8:10]))
        train1_eval = data[data['starttime'] < str(data_list1[i])].copy()
        train2_eval = data[data['starttime'] >= str(data_list1[i+1])].copy()
        test_eval = data[(data['starttime'] < str(data_list1[i+1])) &
                          (data['starttime'] >= str(data_list1[i]))].copy()
        test_eval.loc[:,'geohashed_end_loc'] = np.nan
        train_eval = pd.concat([train1_eval,train2_eval,test_eval])
        test_eval_feat = make_train_set(train_eval,test_eval)
        del train1_eval,train2_eval,test_eval
        gc.collect()
        test_eval_feat.to_hdf(data_path, 'w', complib='blosc', complevel=5)
