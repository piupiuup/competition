import os
import gc
import time
import random
import pickle
import Geohash
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime
from datetime import timedelta

cache_path = 'F:/mobike_cache1/'
train_path = r'C:\Users\csw\Desktop\python\mobike\data\train.csv'
test_path = r'C:\Users\csw\Desktop\python\mobike\data\test.csv'

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


# 合并节约内存
def concat(L):
    result = None
    for l in L:
        if result is None:
            result = l
        else:
            result[l.columns.tolist()] = l
    return result


# 添加all_pred特征
def get_all_order_pred(result):
    all_order_pred_path = cache_path + 'all_order_pred.hdf'
    if os.path.exists(all_order_pred_path):
        all_order_pred = pd.read_hdf(all_order_pred_path)
        all_order_pred.rename(columns={'pred': 'all_order_pred'}, inplace=True)
        result = result.merge(all_order_pred, on=['orderid', 'geohashed_end_loc'], how='left')
        return result
    return result


# 周围的8个地点加自己DataFrame
def get_near_loc():
    result_path = cache_path + 'near_loc.hdf'
    if os.path.exists(result_path) & flag:
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


# 周围的8个地点加自己dict
def get_near_loc_dict():
    result_path = cache_path + 'near_loc_dict.pkl'
    if os.path.exists(result_path) & flag:
        result = pickle.load(open(result_path, 'rb+'))
    else:
        import geohash
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        loc_list = train['geohashed_start_loc'].tolist() \
                   + train['geohashed_end_loc'].tolist() \
                   + test['geohashed_start_loc'].tolist()
        loc_list = np.unique(loc_list)
        result = dict()
        for loc in loc_list:
            result[loc] = geohash.neighbors(loc) + [loc]
        pickle.dump(result, open(result_path, 'wb+'))
    return result


# 是否被自己统计 且在其附近
def if_around(feat, data, loc='geohashed_end_loc'):
    data_temp = data[~data['geohashed_end_loc'].isnull()]
    order_loc_dict = dict(zip(data_temp['orderid'].values, data_temp[loc].values))
    near_loc_dict = get_near_loc_dict()
    result = []
    for loc, orderid in zip(feat[loc], feat['orderid']):
        if order_loc_dict.__contains__(orderid):
            if loc in near_loc_dict[order_loc_dict[orderid]]:
                result.append(1)
            else:
                result.append(0)
        else:
            result.append(0)
    return result


# 组内标准化
def group_normalize(data, key, feat):
    grp = data.groupby(key, as_index=False)[feat].agg({'std': 'std', 'avg': 'mean'})
    result = pd.merge(data, grp, on=key, how='left')
    result[feat] = ((result[feat] - result['std']) / result['avg']).fillna(1)
    return result[feat]


# 对时间进行分区
def split_time(time):
    t = time[:11]
    if t < '05:00:00':
        return 0
    elif t < '09:30:00':
        return 1
    elif t < '15:30:00':
        return 2
    elif t < '21:00:00':
        return 3
    else:
        return 0


# 是否为节假日
def if_holiday(date):
    holiday_dict = {10: 0, 11: 0, 12: 0, 13: 1, 14: 1, 15: 0, 16: 0, 18: 0, 19: 0, 20: 1, 21: 1, \
                    22: 0, 23: 0, 24: 0, 25: 0, 26: 0, 27: 0, 28: 1, 29: 1, 30: 1, 31: 0, 1: 0}
    return holiday_dict[int(date[8:10])]


# 计算两点之间距离
def cal_distance(lat1, lon1, lat2, lon2):
    dx = np.abs(lon1 - lon2)  # 经度差
    dy = np.abs(lat1 - lat2)  # 维度差
    b = (lat1 + lat2) / 2.0
    Lx = 6371004.0 * (dx / 57.2958) * np.cos(b / 57.2958)
    Ly = 6371004.0 * (dy / 57.2958)
    L = (Lx ** 2 + Ly ** 2) ** 0.5
    return L


# 计算两点之间距离
def cal_mht_distance(lat1, lon1, lat2, lon2):
    dx = np.abs(lon1 - lon2)  # 经度差
    dy = np.abs(lat1 - lat2)  # 维度差
    b = (lat1 + lat2) / 2.0
    Lx = 6371004.0 * (dx / 57.2958) * np.cos(b / 57.2958)
    Ly = 6371004.0 * (dy / 57.2958)
    L = Lx + Ly
    return L


# 分组排序
def rank(data, feat1, feat2, ascending):
    data.sort_values([feat1, feat2], inplace=True, ascending=ascending)
    data['rank'] = range(data.shape[0])
    min_rank = data.groupby(feat1, as_index=False)['rank'].agg({'min_rank': 'min'})
    data = pd.merge(data, min_rank, on=feat1, how='left')
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
        locs = list(
            set(train['geohashed_start_loc']) | set(train['geohashed_end_loc']) | set(test['geohashed_start_loc']))
        deloc = []
        for loc in locs:
            deloc.append(Geohash.decode_exactly(loc)[:2])
        loc_dict = dict(zip(locs, deloc))
        pickle.dump(loc_dict, open(dump_path, 'wb+'))
    return loc_dict


# 计算两点之间的欧氏距离和曼哈顿距离
def get_distance(sample, sloc='geohashed_start_loc', eloc='geohashed_end_loc'):
    loc_dict = get_loc_dict()
    geohashed_loc = sample[[sloc, eloc]].values
    distance = []
    mht_distance = []
    for i in geohashed_loc:
        loc1, loc2 = i
        if (loc1 is np.nan) | (loc2 is np.nan):
            distance.append(np.nan)
            mht_distance.append(np.nan)
        else:
            lat1, lon1 = loc_dict[loc1]
            lat2, lon2 = loc_dict[loc2]
            distance.append(cal_distance(lat1, lon1, lat2, lon2))
            mht_distance.append(cal_mht_distance(lat1, lon1, lat2, lon2))
    result = sample.copy()
    result['distance'] = distance
    result['mht_distance'] = mht_distance
    result = result[['distance', 'mht_distance']]
    return result


# 对结果进行整理
def reshape(pred):
    result = pred.copy()
    result = rank(result, 'orderid', 'pred', ascending=False)
    result = result[result['rank'] < 3][['orderid', 'geohashed_end_loc', 'rank']]
    result = result.set_index(['orderid', 'rank']).unstack()
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
        true = dict(zip(train['orderid'].values, train['geohashed_end_loc']))
        pickle.dump(true, open(result_path, 'wb+'))
    data = result.copy()
    data['true'] = data['orderid'].map(true)
    score = (sum(data['true'] == data[0]) / 1.0
             + sum(data['true'] == data[1]) / 2.0
             + sum(data['true'] == data[2]) / 3.0) / data.shape[0]
    return score


# 对结果添加噪音
def get_noise(data, rate=0.1, seed=66):
    random.seed(seed)
    data_temp = data.copy()
    index = list(range(data.shape[0]))
    data.index = index
    random.shuffle(index)
    n = int(rate * data.shape[0])
    sub_index = index[:n]
    data_temp.loc[sub_index, [0, 1, 2]] = 'a'
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
    data.loc[:, 'label'] = data['orderid'].map(true)
    data['label'] = (data['label'] == data['geohashed_end_loc']).astype('int')
    data['label'] = data['label'].fillna(0)
    return data


# 热点目的地
def get_loc_matrix():
    result_path = cache_path + 'loc_matrix.hdf'
    if os.path.exists(result_path):
        result = pd.read_hdf(result_path, 'w')
    else:
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        end_loc = pd.DataFrame({'geohashed_end_loc': list(train['geohashed_end_loc'].unique())})
        end_loc['end_loc_lat'] = end_loc['geohashed_end_loc'].apply(lambda x: Geohash.decode_exactly(x)[0])
        end_loc['end_loc_lon'] = end_loc['geohashed_end_loc'].apply(lambda x: Geohash.decode_exactly(x)[1])
        end_loc['end_loc_lat_box'] = end_loc['end_loc_lat'].apply(lambda x: x // 0.003)
        end_loc['end_loc_lon_box'] = end_loc['end_loc_lon'].apply(lambda x: x // 0.00375)
        count_of_loc = train.groupby('geohashed_end_loc', as_index=False)['geohashed_end_loc'].agg(
            {'count_of_loc': 'count'})
        end_loc = pd.merge(end_loc, count_of_loc, on='geohashed_end_loc', how='left')
        max_index = end_loc.groupby(['end_loc_lat_box', 'end_loc_lon_box']).apply(lambda x: x['count_of_loc'].argmax())
        end_loc = end_loc.loc[max_index.tolist(), ['geohashed_end_loc', 'end_loc_lat', 'end_loc_lon']]
        end_loc.sort_values('end_loc_lat', inplace=True)
        end_loc = end_loc.values
        start_loc = pd.DataFrame(
            {'geohashed_start_loc': list(pd.concat([train, test])['geohashed_start_loc'].unique())})
        start_loc['start_loc_lat'] = start_loc['geohashed_start_loc'].apply(lambda x: Geohash.decode_exactly(x)[0])
        start_loc['start_loc_lon'] = start_loc['geohashed_start_loc'].apply(lambda x: Geohash.decode_exactly(x)[1])
        start_loc = start_loc.values
        start_end_loc_arr = []
        for i in start_loc:
            for j in end_loc:
                if (np.abs(i[1] - j[1]) < 0.012) & (np.abs(i[2] - j[2]) < 0.015):
                    start_end_loc_arr.append([i[0], j[0]])
        result = pd.DataFrame(start_end_loc_arr, columns=['geohashed_start_loc', 'geohashed_end_loc'])
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


####################构造负样本##################

# 将用户骑行过目的的地点加入成样本
def get_user_end_loc(data, candidate, data_key):
    result_path = cache_path + 'user_end_loc_%d.hdf' % (data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        user_n_end_loc = data.groupby(['userid', 'geohashed_end_loc'], as_index=False)['userid'].agg(
            {'user_eloc_count': 'count'})
        user_n_end_loc = user_n_end_loc[~user_n_end_loc['geohashed_end_loc'].isnull()]
        result = pd.merge(candidate[['orderid', 'userid']], user_n_end_loc, on=['userid'], how='left')
        order_eloc_dict = dict(zip(data['orderid'].values, data['geohashed_end_loc'].values))
        result['user_eloc_count'] = result['user_eloc_count'] - (
        result['geohashed_end_loc'] == result['orderid'].map(order_eloc_dict))
        result = result[result['user_eloc_count'] > 0]
        result = result[['orderid', 'geohashed_end_loc']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


# 将用户骑行过出发的地点加入成样本
def get_user_start_loc(data, candidate, data_key):
    result_path = cache_path + 'user_start_loc_%d.hdf' % (data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        user_n_start_loc = data.groupby(['userid', 'geohashed_start_loc'], as_index=False)['userid'].agg(
            {'user_sloc_count': 'count'})
        result = pd.merge(candidate[['orderid', 'userid']], user_n_start_loc, on=['userid'], how='left')
        order_sloc_dict = dict(zip(data['orderid'].values, data['geohashed_start_loc'].values))
        result['user_sloc_count'] = result['user_sloc_count'] - (
        result['geohashed_start_loc'] == result['orderid'].map(order_sloc_dict))
        result = result[result['user_sloc_count'] > 0]
        result.rename(columns={'geohashed_start_loc': 'geohashed_end_loc'}, inplace=True)
        result = result[['orderid', 'geohashed_end_loc']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


# 筛选起始地点去向最多的3个地点
def get_loc_to_loc(data, candidate, data_key):
    result_path = cache_path + 'loc_to_loc_%d.hdf' % (data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        candidate_temp = candidate.copy()
        candidate_temp.rename(columns={'geohashed_end_loc': 'true_loc'}, inplace=True)
        sloc_eloc_count = data.groupby(['geohashed_start_loc', 'geohashed_end_loc'], as_index=False)['orderid'].agg(
            {'sloc_eloc_count': 'count'})
        sloc_eloc_count = sloc_eloc_count[~sloc_eloc_count['geohashed_end_loc'].isnull()]
        sloc_eloc_count.sort_values('sloc_eloc_count', inplace=True)
        sloc_eloc_count = sloc_eloc_count.groupby('geohashed_start_loc').tail(4)
        result = pd.merge(candidate_temp[['orderid', 'geohashed_start_loc', 'true_loc']], sloc_eloc_count,
                          on='geohashed_start_loc', how='left')
        order_eloc_dict = dict(zip(data['orderid'].values, data['geohashed_end_loc'].values))
        result['sloc_eloc_count'] = result['sloc_eloc_count'] - (
        result['geohashed_end_loc'] == result['orderid'].map(order_eloc_dict))
        result.sort_values('sloc_eloc_count', inplace=True)
        result = result.groupby('orderid').tail(3)
        result = result[result['sloc_eloc_count'] > 0]
        result = result[['orderid', 'geohashed_end_loc']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


# 与其交互最多的三个地点
def get_loc_with_loc(data, candidate, data_key):
    result_path = cache_path + 'loc_with_loc_%d.hdf' % (data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        candidate_temp = candidate.copy()
        candidate_temp.rename(columns={'geohashed_end_loc': 'true_loc'}, inplace=True)
        sloc_eloc_count = data.groupby(['geohashed_start_loc', 'geohashed_end_loc'], as_index=False)['orderid'].agg(
            {'sloc_eloc_count': 'count'})
        sloc_eloc_count = sloc_eloc_count[~sloc_eloc_count['geohashed_end_loc'].isnull()]
        eloc_sloc_count = sloc_eloc_count.rename(columns={'geohashed_start_loc': 'geohashed_end_loc',
                                                          'geohashed_end_loc': 'geohashed_start_loc',
                                                          'sloc_eloc_count': 'eloc_sloc_count'}).copy()
        sloc_eloc_2count = pd.merge(sloc_eloc_count, eloc_sloc_count,
                                    on=['geohashed_start_loc', 'geohashed_end_loc'], how='outer').fillna(0)
        sloc_eloc_2count['sloc_eloc_2count'] = sloc_eloc_2count['sloc_eloc_count'] + sloc_eloc_2count['eloc_sloc_count']
        sloc_eloc_2count.sort_values('sloc_eloc_2count', inplace=True)
        sloc_eloc_2count = sloc_eloc_2count.groupby('geohashed_start_loc').tail(4)
        result = pd.merge(candidate_temp[['orderid', 'geohashed_start_loc', 'true_loc']], sloc_eloc_2count,
                          on='geohashed_start_loc', how='left')
        order_eloc_dict = dict(zip(data['orderid'].values, data['geohashed_end_loc'].values))
        result['sloc_eloc_2count'] = result['sloc_eloc_2count'] - (
        result['geohashed_end_loc'] == result['orderid'].map(order_eloc_dict))
        result.sort_values('sloc_eloc_2count', inplace=True)
        result = result.groupby('orderid').tail(3)
        result = result[result['sloc_eloc_2count'] > 0]
        result = result[['orderid', 'geohashed_end_loc']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


# 筛选起始地点周围流入量最大的3个地点
def get_loc_near_loc(data, candidate, data_key):
    result_path = cache_path + 'loc_near_loc_%d.hdf' % (data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        candidate_temp = candidate.copy()
        candidate_temp.rename(columns={'geohashed_end_loc': 'true_loc'}, inplace=True)
        loc_matrix = get_loc_matrix()
        loc_input = data.groupby('geohashed_end_loc', as_index=False)['userid'].agg({'eloc_n_input': 'count'})
        loc_matrix = pd.merge(loc_matrix, loc_input, on='geohashed_end_loc', how='left')
        loc_matrix = loc_matrix[~loc_matrix['eloc_n_input'].isnull()]
        loc_matrix.sort_values('eloc_n_input', inplace=True)
        loc_matrix = loc_matrix.groupby('geohashed_start_loc').tail(4)
        result = pd.merge(candidate_temp[['orderid', 'geohashed_start_loc', 'true_loc']], loc_matrix,
                          on='geohashed_start_loc', how='left')
        order_eloc_dict = dict(zip(data['orderid'].values, data['geohashed_end_loc'].values))
        result['eloc_n_input'] = result['eloc_n_input'] - (
        result['geohashed_end_loc'] == result['orderid'].map(order_eloc_dict))
        result.sort_values('eloc_n_input', inplace=True)
        result = result.groupby('orderid').tail(3)
        result = result[result['eloc_n_input'] > 0]
        result = result[['orderid', 'geohashed_end_loc']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


# 用户后续的起始地点
def get_user_next_loc(data, candidate, data_key):
    result_path = cache_path + 'user_next_loc_%d.hdf' % (data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        data_temp = data.copy()
        data_temp = data_temp[data_temp['userid'].isin(candidate['userid'].unique().tolist())]
        data_temp = rank(data_temp, 'userid', 'starttime', ascending=True)
        data_temp_temp = data_temp.copy()
        data_temp_temp['rank'] = data_temp_temp['rank'] - 1
        result = pd.merge(data_temp[['orderid', 'userid', 'rank', 'starttime']],
                          data_temp_temp[['userid', 'rank', 'geohashed_start_loc', 'starttime']], on=['userid', 'rank'],
                          how='inner')
        result['user_eloc_nloc_sep_time'] = result.apply(lambda x: diff_of_minutes(x['starttime_y'], x['starttime_x']),
                                                         axis=1)
        result.rename(columns={'geohashed_start_loc': 'geohashed_end_loc'}, inplace=True)
        result = result[['orderid', 'geohashed_end_loc', 'user_eloc_nloc_sep_time']]
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
        result = pd.merge(data_temp[['orderid', 'bikeid', 'rank', 'starttime']],
                          data_temp_temp[['bikeid', 'rank', 'geohashed_start_loc', 'starttime']], on=['bikeid', 'rank'],
                          how='inner')
        result['bike_eloc_sep_time'] = result.apply(lambda x: diff_of_minutes(x['starttime_y'], x['starttime_x']),
                                                    axis=1)
        result.rename(columns={'geohashed_start_loc': 'geohashed_end_loc'}, inplace=True)
        result = result[['orderid', 'geohashed_end_loc', 'bike_eloc_sep_time']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


# 用户前一次的起始地点
def get_user_forward_loc(data, candidate, data_key):
    result_path = cache_path + 'user_forward_loc_%d.hdf' % (data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        data_temp = data.copy()
        data_temp = data_temp[data_temp['userid'].isin(candidate['userid'].values)]
        if data_temp.shape[0] == 0:
            return pd.DataFrame(columns=['orderid', 'geohashed_end_loc', 'user_eloc_nloc_sep_time'])
        data_temp = rank(data_temp, 'userid', 'starttime', ascending=True)
        data_temp_temp = data_temp.copy()
        data_temp_temp['rank'] = data_temp_temp['rank'] + 1
        result = pd.merge(data_temp[['orderid', 'userid', 'rank', 'starttime']],
                          data_temp_temp[['userid', 'rank', 'geohashed_start_loc', 'starttime']], on=['userid', 'rank'],
                          how='inner')
        result['user_eloc_forward_sep_time'] = result.apply(
            lambda x: diff_of_minutes(x['starttime_y'], x['starttime_x']), axis=1)
        result.rename(columns={'geohashed_start_loc': 'geohashed_end_loc'}, inplace=True)
        result = result[['orderid', 'geohashed_end_loc', 'user_eloc_forward_sep_time']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


#####################构造特征####################
# 获取用户历史行为次数
def get_user_count(data, sample, data_key):
    feat_path = cache_path + 'user_count_feat_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        data_temp = data.copy()
        loc_dict = get_loc_dict()
        data_temp['lat1'] = data_temp['geohashed_start_loc'].apply(lambda x: np.nan if x is np.nan else loc_dict[x][0])
        data_temp['lon1'] = data_temp['geohashed_start_loc'].apply(lambda x: np.nan if x is np.nan else loc_dict[x][1])
        data_temp['lat2'] = data_temp['geohashed_end_loc'].apply(lambda x: np.nan if x is np.nan else loc_dict[x][0])
        data_temp['lon2'] = data_temp['geohashed_end_loc'].apply(lambda x: np.nan if x is np.nan else loc_dict[x][1])
        feat = data_temp.groupby('userid').agg({'userid': {'user_count': 'count'},
                                                'geohashed_end_loc': {'eloc_no_nan': 'count'},
                                                'lat1': {'sum_lat1': 'sum'},
                                                'lon1': {'sum_lon1': 'sum'},
                                                'lat2': {'sum_lat2': 'sum'},
                                                'lon2': {'sum_lon2': 'sum'}})
        feat.columns = feat.columns.droplevel(0)
        feat.reset_index(inplace=True)
        feat['sum_lat'] = feat['sum_lat1'] + feat['sum_lat2']
        feat['sum_lon'] = feat['sum_lon1'] + feat['sum_lon2']
        feat = pd.merge(sample, feat, on=['userid'], how='left')
        feat['user_count'].fillna(0, inplace=True)
        order_eloc_dict = dict(zip(data['orderid'].values, data['geohashed_end_loc'].values))
        feat['sum_lat'] = feat['sum_lat'] - (feat['geohashed_end_loc'] == feat['orderid'].map(order_eloc_dict)) \
                                            * feat['geohashed_end_loc'].apply(
            lambda x: np.nan if x is np.nan else loc_dict[x][0])
        feat['sum_lon'] = feat['sum_lon'] - (feat['geohashed_end_loc'] == feat['orderid'].map(order_eloc_dict)) \
                                            * feat['geohashed_end_loc'].apply(
            lambda x: np.nan if x is np.nan else loc_dict[x][1])
        feat['mean_lat'] = feat['sum_lat'] / (feat['user_count'] + feat['eloc_no_nan'] -
                                              (feat['geohashed_end_loc'] == feat['orderid'].map(order_eloc_dict)))
        feat['mean_lon'] = feat['sum_lon'] / (feat['user_count'] + feat['eloc_no_nan'] -
                                              (feat['geohashed_end_loc'] == feat['orderid'].map(order_eloc_dict)))
        feat['user_count'] = feat['user_count'] - (feat['orderid'].isin(data['orderid'].values))
        feat['lat2'] = feat['geohashed_end_loc'].apply(lambda x: np.nan if x is np.nan else loc_dict[x][0])
        feat['lon2'] = feat['geohashed_end_loc'].apply(lambda x: np.nan if x is np.nan else loc_dict[x][1])
        feat['center_dis'] = ((feat['lat2'] - feat['mean_lat']) ** 2 + (
        np.cos(feat['mean_lon'] / 57.2958) * (feat['lon2'] - feat['mean_lon'])) ** 2) ** 0.5
        feat = feat[['user_count', 'center_dis']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat


# 获取出发地热度（作为出发地的次数、人数）
def get_start_loc(data, sample, data_key):
    feat_path = cache_path + 'start_loc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        start_loc = data.groupby('geohashed_start_loc', as_index=False)['userid'].agg({'sloc_count': 'count',
                                                                                       'sloc_n_user': 'nunique'})
        feat = pd.merge(sample, start_loc, on='geohashed_start_loc', how='left').fillna(0)
        feat['sloc_count'] = feat['sloc_count'] - feat['orderid'].isin(data['orderid'].values)
        feat = feat[['sloc_count', 'sloc_n_user']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat


# 获取目标地点的热度(目的地)
def get_end_loc(data, sample, data_key):
    feat_path = cache_path + 'end_loc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        eloc_as_sloc = data.groupby('geohashed_start_loc', as_index=False)['userid'].agg(
            {'eloc_as_sloc_count': 'count',
             'eloc_as_sloc_n_uesr': 'nunique'})
        eloc = data.groupby('geohashed_end_loc', as_index=False)['userid'].agg(
            {'eloc_count': 'count',
             'eloc_n_uesr': 'nunique'})
        eloc_as_sloc.rename(columns={'geohashed_start_loc': 'geohashed_end_loc'}, inplace=True)
        feat = pd.merge(sample, eloc_as_sloc, on='geohashed_end_loc', how='left')
        feat = pd.merge(feat, eloc, on='geohashed_end_loc', how='left').fillna(0)
        order_eloc_dict = dict(zip(data['orderid'].values, data['geohashed_end_loc'].values))
        feat['eloc_count'] = feat['eloc_count'] - (feat['geohashed_end_loc'] == feat['orderid'].map(order_eloc_dict))
        feat['eloc_2count'] = feat['eloc_count'] + feat['eloc_as_sloc_count']
        feat['eloc_in_out_rate'] = feat['eloc_count'] / (feat['eloc_as_sloc_count'] + 0.001)
        feat = feat[['eloc_as_sloc_count', 'eloc_as_sloc_n_uesr', 'eloc_2count',
                     'eloc_count', 'eloc_n_uesr', 'eloc_in_out_rate']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat


# 用户×出发地（次数 目的地个数）
def get_user_sloc(data, sample, data_key):
    feat_path = cache_path + 'user_sloc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        user_sloc = data.groupby(['userid', 'geohashed_start_loc'], as_index=False)['userid'].agg(
            {'user_sloc_count': 'count',
             'user_sloc_n_eloc': 'nunique'})
        feat = pd.merge(sample, user_sloc, on=['userid', 'geohashed_start_loc'], how='left').fillna(0)
        order_sloc_dict = dict(zip(data['orderid'].values, data['geohashed_start_loc'].values))
        feat['user_sloc_count'] = feat['user_sloc_count'] - (
        feat['geohashed_start_loc'] == feat['orderid'].map(order_sloc_dict))
        feat = feat[['user_sloc_count', 'user_sloc_n_eloc']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat


# 用户×目的地
def get_user_eloc(data, sample, candidate, data_key):
    feat_path = cache_path + 'user_eloc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        user_eloc_as_sloc = data.groupby(['userid', 'geohashed_start_loc'], as_index=False)['userid'].agg(
            {'user_eloc_as_sloc_count': 'count',
             'user_eloc_as_sloc_n_sloc': 'nunique'})
        user_eloc = data.groupby(['userid', 'geohashed_end_loc'], as_index=False)['geohashed_start_loc'].agg(
            {'user_eloc_count': 'count',
             'user_eloc_n_sloc': 'nunique'})
        user_eloc_as_sloc.rename(columns={'geohashed_start_loc': 'geohashed_end_loc'}, inplace=True)
        feat = pd.merge(sample, user_eloc_as_sloc, on=['userid', 'geohashed_end_loc'], how='left')
        feat = pd.merge(feat, user_eloc, on=['userid', 'geohashed_end_loc'], how='left').fillna(0)
        order_eloc_dict = dict(zip(data['orderid'].values, data['geohashed_end_loc'].values))
        feat['user_eloc_count'] = feat['user_eloc_count'] - (
        feat['geohashed_end_loc'] == feat['orderid'].map(order_eloc_dict))
        feat['user_eloc_2count'] = feat['user_eloc_count'] + feat['user_eloc_as_sloc_count']
        feat['user_sloc_goback_rate'] = feat['user_eloc_count'] / (feat['user_eloc_as_sloc_count'] + 0.001)
        user_next_loc = get_user_next_loc(data, candidate,data_key)
        feat = feat.merge(user_next_loc, on=['orderid', 'geohashed_end_loc'], how='left').fillna(10000)
        user_next_loc.rename(columns={'geohashed_end_loc': 'geohashed_end_loc2'}, inplace=True)
        feat = feat.merge(user_next_loc[['orderid', 'geohashed_end_loc2']], on='orderid', how='left')
        feat[['user_eloc_next_loc_dis', 'user_eloc_next_loc_mdis']] = get_distance(feat, sloc='geohashed_end_loc',
                                                                                   eloc='geohashed_end_loc2')
        user_forward_loc = get_user_forward_loc(data, candidate, data_key)
        feat = feat.merge(user_forward_loc, on=['orderid', 'geohashed_end_loc'], how='left').fillna(10000)
        feat = feat[['user_eloc_as_sloc_count', 'user_eloc_as_sloc_n_sloc', 'user_eloc_count',
                     'user_eloc_n_sloc', 'user_eloc_2count', 'user_eloc_next_loc_mdis',
                     'user_sloc_goback_rate', 'user_eloc_nloc_sep_time',
                     'user_eloc_next_loc_dis', 'user_eloc_forward_sep_time']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat


# 出发地×目的地
def get_sloc_eloc(data, sample, data_key):
    feat_path = cache_path + 'sloc_eloc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        sloc_eloc = data.groupby(['geohashed_start_loc', 'geohashed_end_loc'], as_index=False)['userid'].agg(
            {'sloc_eloc_count': 'count',
             'sloc_eloc_n_user': 'nunique'})
        eloc_sloc = sloc_eloc.rename(columns={'geohashed_start_loc': 'geohashed_end_loc',
                                              'geohashed_end_loc': 'geohashed_start_loc',
                                              'sloc_eloc_count': 'eloc_sloc_count',
                                              'sloc_eloc_n_user': 'eloc_sloc_n_user'})
        feat = pd.merge(sample, sloc_eloc, on=['geohashed_start_loc', 'geohashed_end_loc'], how='left')
        feat = pd.merge(feat, eloc_sloc, on=['geohashed_start_loc', 'geohashed_end_loc'], how='left').fillna(0)
        order_eloc_dict = dict(zip(data['orderid'].values, data['geohashed_end_loc'].values))
        feat['sloc_eloc_count'] = feat['sloc_eloc_count'] - (
        feat['geohashed_end_loc'] == feat['orderid'].map(order_eloc_dict))
        feat['sloc_eloc_2count'] = feat['sloc_eloc_count'] + feat['eloc_sloc_count']
        feat['sloc_eloc_goback_rate'] = feat['sloc_eloc_count'] / (feat['eloc_sloc_count'] + 0.001)
        distance = get_distance(feat)
        feat = pd.concat([feat, distance], axis=1)
        feat = feat[['sloc_eloc_count', 'sloc_eloc_n_user', 'eloc_sloc_count',
                     'eloc_sloc_n_user', 'sloc_eloc_2count',
                     'sloc_eloc_goback_rate', 'distance', 'mht_distance']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat


# 自行车×目的地
def get_bike_eloc(data, sample, candidate, data_key):
    feat_path = cache_path + 'bike_eloc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        bike_next_loc = get_bike_next_loc(data, candidate, data_key)
        feat = sample.merge(bike_next_loc, on=['orderid', 'geohashed_end_loc'], how='left').fillna(10000)
        bike_next_loc.rename(columns={'geohashed_end_loc': 'geohashed_end_loc2'}, inplace=True)
        feat = feat.merge(bike_next_loc[['orderid', 'geohashed_end_loc2']], on='orderid', how='left')
        feat[['biek_eloc_nloc_dis', 'biek_eloc_nloc_mdis']] = get_distance(feat, sloc='geohashed_end_loc',
                                                                           eloc='geohashed_end_loc2')
        feat = feat[['bike_eloc_sep_time', 'biek_eloc_nloc_dis', 'biek_eloc_nloc_mdis']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat


# 用户×出发地×目的地
def get_user_sloc_eloc(data, sample, data_key):
    feat_path = cache_path + 'user_sloc_eloc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        user_sloc_eloc_count = data.groupby(['userid', 'geohashed_start_loc', 'geohashed_end_loc'], as_index=False)[
            'userid'].agg({'user_sloc_eloc_count': 'count'})
        user_eloc_sloc_count = user_sloc_eloc_count.rename(columns={'geohashed_start_loc': 'geohashed_end_loc',
                                                                    'geohashed_end_loc': 'geohashed_start_loc',
                                                                    'user_sloc_eloc_count': 'user_eloc_sloc_count'})
        feat = pd.merge(sample, user_sloc_eloc_count, on=['userid', 'geohashed_start_loc', 'geohashed_end_loc'],
                        how='left')
        feat = pd.merge(feat, user_eloc_sloc_count, on=['userid', 'geohashed_start_loc', 'geohashed_end_loc'],
                        how='left').fillna(0)
        order_eloc_dict = dict(zip(data['orderid'].values, data['geohashed_end_loc'].values))
        feat['user_sloc_eloc_count'] = feat['user_sloc_eloc_count'] - (
        feat['geohashed_end_loc'] == feat['orderid'].map(order_eloc_dict))
        feat['user_sloc_eloc_2count'] = feat['user_sloc_eloc_count'] + feat['user_eloc_sloc_count']
        feat = feat[['user_sloc_eloc_count', 'user_eloc_sloc_count', 'user_sloc_eloc_2count']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat


# 时间×出发地
def get_time_sloc(data, sample, data_key):
    feat_path = cache_path + 'time_sloc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        time_sloc_count = data.groupby(['time', 'geohashed_start_loc'], as_index=False)[
            'userid'].agg({'time_sloc_count': 'count'})
        feat = pd.merge(sample, time_sloc_count, on=['time', 'geohashed_start_loc'], how='left').fillna(0)
        feat = feat['time_sloc_count']
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat


# 时间×目的地
def get_time_eloc(data, sample, data_key):
    feat_path = cache_path + 'time_eloc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        time_eloc_count = data.groupby(['time', 'geohashed_end_loc'], as_index=False)[
            'userid'].agg({'time_eloc_count': 'count'})
        feat = pd.merge(sample, time_eloc_count, on=['time', 'geohashed_end_loc'], how='left').fillna(0)
        feat = feat['time_eloc_count']
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat


# 时间×出发地×目的地
def get_time_sloc_eloc(data, sample, data_key):
    feat_path = cache_path + 'time_sloc_eloc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        time_sloc_eloc_count = data.groupby(['time', 'geohashed_start_loc', 'geohashed_end_loc'], as_index=False)[
            'userid'].agg({'time_sloc_eloc_count': 'count'})
        feat = pd.merge(sample, time_sloc_eloc_count, on=['time', 'geohashed_start_loc', 'geohashed_end_loc'],
                        how='left').fillna(0)
        feat = feat['time_sloc_eloc_count']
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat


# 用户×目的地×时间
def get_time_user_eloc(data, sample, data_key):
    feat_path = cache_path + 'time_user_eloc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        time_user_eloc_count = data.groupby(['userid', 'time', 'geohashed_end_loc'], as_index=False)[
            'userid'].agg({'time_user_eloc_count': 'count'})
        feat = pd.merge(sample, time_user_eloc_count, on=['userid', 'time', 'geohashed_end_loc'], how='left').fillna(0)
        feat = feat['time_user_eloc_count']
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat


###############扩大统计范围##################
# 获取出发地热度（作为出发地的次数、人数）
def get_ex_sloc(data, sample, data_key):
    feat_path = cache_path + 'ex_sloc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        start_loc = data.groupby('geohashed_start_loc', as_index=False)['userid'].agg({'ex_sloc_count': 'count'})
        near_loc = get_near_loc()
        start_loc = start_loc.merge(near_loc, left_on='geohashed_start_loc', right_on='loc', how='left')
        start_loc['geohashed_start_loc'] = start_loc['near_loc']
        start_loc = start_loc.groupby('geohashed_start_loc', as_index=False)['ex_sloc_count'].sum()
        feat = pd.merge(sample, start_loc, on='geohashed_start_loc', how='left').fillna(0)
        feat['ex_sloc_count'] = feat['ex_sloc_count'] - feat['orderid'].isin(data['orderid'].values)
        feat = feat[['ex_sloc_count']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat


# 获取目标地点的热度(目的地)
def get_ex_eloc(data, sample, data_key):
    feat_path = cache_path + 'ex_eloc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        eloc_as_sloc = data.groupby('geohashed_start_loc', as_index=False)['userid'].agg(
            {'ex_eloc_as_sloc_count': 'count'})
        eloc = data.groupby('geohashed_end_loc', as_index=False)['userid'].agg(
            {'ex_eloc_count': 'count'})
        eloc_as_sloc.rename(columns={'geohashed_start_loc': 'geohashed_end_loc'}, inplace=True)
        feat = pd.merge(eloc, eloc_as_sloc, on='geohashed_end_loc', how='outer').fillna(0)
        near_loc = get_near_loc()
        feat = feat.merge(near_loc, left_on='geohashed_end_loc', right_on='loc', how='left').fillna(0)
        feat = feat.groupby('near_loc', as_index=False).sum()
        feat = pd.merge(sample, feat, left_on='geohashed_end_loc', right_on='near_loc', how='left').fillna(0)
        feat['ex_eloc_count'] = feat['ex_eloc_count'] - if_around(feat, data, 'geohashed_end_loc')
        feat['ex_eloc_2count'] = feat['ex_eloc_count'] + feat['ex_eloc_as_sloc_count']
        feat['ex_eloc_in_out_rate'] = feat['ex_eloc_count'] / (feat['ex_eloc_as_sloc_count'] + 0.001)
        feat = feat[['ex_eloc_as_sloc_count', 'ex_eloc_count',
                     'ex_eloc_2count', 'ex_eloc_in_out_rate']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat


# 用户×出发地（次数 目的地个数）
def get_ex_user_sloc(data, sample, data_key):
    feat_path = cache_path + 'ex_user_sloc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        user_sloc = data.groupby(['userid', 'geohashed_start_loc'], as_index=False)['geohashed_start_loc'].agg(
            {'ex_user_sloc_count': 'count'})
        near_loc = get_near_loc()
        feat = user_sloc.merge(near_loc, left_on='geohashed_start_loc', right_on='loc', how='left').fillna(0)
        feat['geohashed_start_loc'] = feat['near_loc']
        feat = feat.groupby(['userid', 'geohashed_start_loc'], as_index=False).sum()
        feat = pd.merge(sample, feat, on=['userid', 'geohashed_start_loc'], how='left').fillna(0)
        feat['ex_user_sloc_count'] = feat['ex_user_sloc_count'] - feat['orderid'].isin(data['orderid'].values)
        feat = feat[['ex_user_sloc_count']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat


# 用户×目的地
def get_ex_user_eloc(data, sample, data_key):
    feat_path = cache_path + 'ex_user_eloc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        user_eloc_as_sloc = data.groupby(['userid', 'geohashed_start_loc'], as_index=False)['userid'].agg(
            {'ex_user_eloc_as_sloc_count': 'count'})
        user_eloc = data.groupby(['userid', 'geohashed_end_loc'], as_index=False)['userid'].agg(
            {'ex_user_eloc_count': 'count'})
        user_eloc_as_sloc.rename(columns={'geohashed_start_loc': 'geohashed_end_loc'}, inplace=True)
        feat = pd.merge(user_eloc, user_eloc_as_sloc, on=['userid', 'geohashed_end_loc'], how='outer').fillna(0)
        near_loc = get_near_loc()
        feat = feat.merge(near_loc, left_on='geohashed_end_loc', right_on='loc', how='left')
        feat['geohashed_end_loc'] = feat['near_loc']
        feat = feat.groupby(['userid', 'geohashed_end_loc'], as_index=False).sum()
        feat = pd.merge(sample, feat, on=['userid', 'geohashed_end_loc'], how='left').fillna(0)
        feat['ex_user_eloc_count'] = feat['ex_user_eloc_count'] - if_around(feat, data, 'geohashed_end_loc')
        feat['ex_user_eloc_2count'] = feat['ex_user_eloc_count'] + feat['ex_user_eloc_as_sloc_count']
        feat['ex_user_sloc_goback_rate'] = feat['ex_user_eloc_count'] / (feat['ex_user_eloc_as_sloc_count'] + 0.001)
        feat = feat[['ex_user_eloc_as_sloc_count', 'ex_user_eloc_count',
                     'ex_user_eloc_2count', 'ex_user_sloc_goback_rate']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat


# 出发地×目的地
def get_ex_sloc_eloc(data, sample, data_key):
    feat_path = cache_path + 'ex_sloc_eloc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        sloc_eloc = data.groupby(['geohashed_start_loc', 'geohashed_end_loc'], as_index=False)['userid'].agg(
            {'ex_sloc_eloc_count': 'count'})
        eloc_sloc = sloc_eloc.rename(columns={'geohashed_start_loc': 'geohashed_end_loc',
                                              'geohashed_end_loc': 'geohashed_start_loc',
                                              'ex_sloc_eloc_count': 'ex_eloc_sloc_count'})
        feat = pd.merge(sloc_eloc, eloc_sloc, on=['geohashed_start_loc', 'geohashed_end_loc'], how='left')
        near_loc = get_near_loc()
        feat = feat.merge(near_loc, left_on='geohashed_start_loc', right_on='loc', how='left').fillna(0)
        feat['geohashed_start_loc'] = feat['near_loc']
        feat = feat.groupby(['geohashed_start_loc', 'geohashed_end_loc'], as_index=False).sum()
        feat = feat.merge(near_loc, left_on='geohashed_end_loc', right_on='loc', how='left').fillna(0)
        feat['geohashed_end_loc'] = feat['near_loc']
        feat = feat.groupby(['geohashed_start_loc', 'geohashed_end_loc'], as_index=False).sum()
        feat = pd.merge(sample, feat, on=['geohashed_end_loc', 'geohashed_start_loc'], how='left').fillna(0)
        feat['ex_sloc_eloc_count'] = feat['ex_sloc_eloc_count'] - if_around(feat, data, 'geohashed_end_loc')
        feat['ex_sloc_eloc_2count'] = feat['ex_sloc_eloc_count'] + feat['ex_eloc_sloc_count']
        feat['ex_sloc_eloc_goback_rate'] = feat['ex_sloc_eloc_count'] / (feat['ex_eloc_sloc_count'] + 0.001)
        feat = feat[['ex_sloc_eloc_count', 'ex_eloc_sloc_count',
                     'ex_sloc_eloc_2count', 'ex_sloc_eloc_goback_rate']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat


# 用户×出发地×目的地
def get_ex_user_sloc_eloc(data, sample, candidate, data_key):
    feat_path = cache_path + 'ex_user_sloc_eloc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        user_sloc_eloc_count = data.groupby(['userid', 'geohashed_start_loc', 'geohashed_end_loc'], as_index=False)[
            'userid'].agg({'ex_user_sloc_eloc_count': 'count'})
        user_eloc_sloc_count = user_sloc_eloc_count.rename(columns={'geohashed_start_loc': 'geohashed_end_loc',
                                                                    'geohashed_end_loc': 'geohashed_start_loc',
                                                                    'ex_user_sloc_eloc_count': 'ex_user_eloc_sloc_count'})
        feat = pd.merge(user_sloc_eloc_count, user_eloc_sloc_count,
                        on=['userid', 'geohashed_start_loc', 'geohashed_end_loc'], how='outer').fillna(0)
        near_loc = get_near_loc()
        feat = feat.merge(near_loc, left_on='geohashed_start_loc', right_on='loc', how='left').fillna(0)
        feat['geohashed_start_loc'] = feat['near_loc']
        candidate_temp = candidate[['userid', 'geohashed_start_loc']].drop_duplicates()
        feat = pd.merge(candidate_temp, feat, on=['userid', 'geohashed_start_loc'], how='inner')
        feat = feat.groupby(['userid', 'geohashed_start_loc', 'geohashed_end_loc'], as_index=False).sum()
        feat = feat.merge(near_loc, left_on='geohashed_end_loc', right_on='loc', how='left').fillna(0)
        feat['geohashed_end_loc'] = feat['near_loc']
        feat = feat.groupby(['userid', 'geohashed_start_loc', 'geohashed_end_loc'], as_index=False).sum()
        feat = pd.merge(sample, feat, on=['userid', 'geohashed_end_loc', 'geohashed_start_loc'], how='left').fillna(0)
        feat['ex_user_sloc_eloc_count'] = feat['ex_user_sloc_eloc_count'] - if_around(feat, data, 'geohashed_end_loc')
        feat['ex_user_sloc_eloc_2count'] = feat['ex_user_sloc_eloc_count'] + feat['ex_user_eloc_sloc_count']
        feat = feat[['ex_user_sloc_eloc_count', 'ex_user_eloc_sloc_count', 'ex_user_sloc_eloc_2count']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat


# 二次处理特征
def second_feat(data, result):
    order_eloc_dict = dict(zip(data['orderid'].values, data['geohashed_end_loc'].values))
    order_sloc_dict = dict(zip(data['orderid'].values, data['geohashed_start_loc'].values))
    result['eloc_n_uesr'] = result['eloc_n_uesr'] - ((result['user_eloc_count'] == 0)
                                                     & ((result['geohashed_end_loc'] == result['orderid'].map(
        order_eloc_dict))))
    result['sloc_n_user'] = result['sloc_n_user'] - ((result['user_sloc_count'] == 0)
                                                     & ((result['geohashed_start_loc'] == result['orderid'].map(
        order_sloc_dict))))
    result['sloc_eloc_n_user'] = result['sloc_eloc_n_user'] - ((result['user_sloc_eloc_count'] == 0)
                                                               & ((result['geohashed_end_loc'] == result['orderid'].map(
        order_eloc_dict))))
    result['sloc_eloc_2n_user'] = result['sloc_eloc_n_user'] + result['eloc_sloc_n_user']
    result['user_eloc_n_sloc'] = result['user_eloc_n_sloc'] - ((result['user_sloc_eloc_count'] == 0)
                                                               & ((result['geohashed_end_loc'] == result['orderid'].map(
        order_eloc_dict))))
    result['eloc_count_sloc_rate'] = result['eloc_count'] / (result['sloc_count'] + 0.001)
    result['user_eloc_count_user_rate'] = result['user_eloc_count'] / (result['user_count'] + 0.001)
    result['user_eloc_2count_user_rate'] = result['user_eloc_2count'] / (result['user_count'] + 0.001)
    result['user_sloc_eloc_count_user_rate'] = result['user_sloc_eloc_count'] / (result['user_count'] + 0.001)
    result['user_sloc_eloc_2count_user_rate'] = result['user_sloc_eloc_2count'] / (result['user_count'] + 0.001)
    result['user_sloc_eloc_count_user_sloc_rate'] = result['user_sloc_eloc_count'] / (result['user_sloc_count'] + 0.001)
    result['user_sloc_eloc_count_user_eloc_rate'] = result['user_sloc_eloc_count'] / (result['user_eloc_count'] + 0.001)
    result['sloc_eloc_count_sloc_rate'] = result['sloc_eloc_count'] / (result['sloc_count'] + 0.001)
    result['sloc_eloc_n_user_sloc_rate'] = result['sloc_eloc_n_user'] / (result['sloc_n_user'] + 0.001)
    # result['time_sloc_eloc_count_ts_rate'] = result['time_sloc_eloc_count'] / (result['time_sloc_count'] + 0.001)
    # result['time_eloc_count_sloc_rate'] = result['time_eloc_count'] / (result['time_sloc_count'] + 0.001)
    result['ex_user_eloc_count_user_rate'] = result['ex_user_eloc_count'] / (result['user_count'] + 0.001)
    result['ex_sloc_eloc_count_ex_sloc_rate'] = result['ex_sloc_eloc_count'] / (result['ex_sloc_count'] + 0.001)
    result['ex_user_sloc_eloc_count_user_sloc_rate'] = result['ex_user_sloc_eloc_count'] / (
    result['ex_user_sloc_count'] + 0.001)
    result['holiday'] = result['starttime'].apply(lambda x: if_holiday(x))
    result['hour'] = result['starttime'].str[11:13].astype(int)
    loc_dict = get_loc_dict()
    result['lat1'] = result['geohashed_start_loc'].apply(lambda x: loc_dict[x][0])
    result['lon1'] = result['geohashed_start_loc'].apply(lambda x: loc_dict[x][1])
    result['lat2'] = result['geohashed_end_loc'].apply(lambda x: loc_dict[x][0])
    result['lon2'] = result['geohashed_end_loc'].apply(lambda x: loc_dict[x][1])
    return result


# 构造样本
def get_sample(data, candidate, data_key):
    result_path = cache_path + 'sample_%d.hdf' % (data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        user_end_loc = get_user_end_loc(data,candidate,data_key)  # 根据用户历史目的地点添加样本 ['orderid', 'geohashed_end_loc', 'user_n_end_loc']
        user_start_loc = get_user_start_loc(data,candidate,data_key)  # 根据用户历史起始地点添加样本 ['orderid', 'geohashed_end_loc', 'user_n_start_loc']
        loc_to_loc = get_loc_to_loc(data, candidate,data_key)  # 筛选起始地点去向最多的3个地点
        loc_with_loc = get_loc_with_loc(data, candidate,data_key)  # 与起始地点交互最多的三个地点
        loc_near_loc = get_loc_near_loc(data, candidate,data_key)  # 筛选起始地点周围3个热门地点
        bike_next_loc = get_bike_next_loc(data, candidate,data_key)  # 自行车后续起始地点
        # 汇总样本id
        result = pd.concat([user_end_loc[['orderid', 'geohashed_end_loc']],
                            user_start_loc[['orderid', 'geohashed_end_loc']],
                            loc_to_loc[['orderid', 'geohashed_end_loc']],
                            loc_with_loc[['orderid', 'geohashed_end_loc']],
                            loc_near_loc[['orderid', 'geohashed_end_loc']],
                            bike_next_loc[['orderid', 'geohashed_end_loc']]
                            ]).drop_duplicates()
        candidate_temp = candidate[['orderid', 'userid', 'bikeid', 'biketype', 'starttime',
                                    'geohashed_start_loc', 'time']].copy()
        result = pd.merge(result, candidate_temp, on='orderid', how='left')
        # 删除起始地点和目的地点相同的样本  和 异常值
        result = result[result['geohashed_end_loc'] != result['geohashed_start_loc']]
        result = result[(~result['geohashed_end_loc'].isnull()) & (~result['geohashed_start_loc'].isnull())]
        result.index = list(range(result.shape[0]))
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    print('样本个数为：{}'.format(result.shape))
    return result


# 制作训练集
def make_train_set(data, candidate):
    t0 = time.time()
    data_key = (sum(~candidate['geohashed_end_loc'].isnull()) + sum(~data['geohashed_end_loc'].isnull())) * data[
        'orderid'].sum() * candidate['orderid'].sum()
    result_path = cache_path + 'train_set_%d.hdf' % (data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        data.loc[:, 'time'] = data['starttime'].apply(lambda x: split_time(x))
        candidate.loc[:, 'time'] = candidate['starttime'].apply(lambda x: split_time(x))
        data.loc[:, 'holiday'] = data['starttime'].apply(lambda x: if_holiday(x))
        candidate.loc[:, 'holiday'] = candidate['starttime'].apply(lambda x: if_holiday(x))
        # 汇总样本id
        print('开始构造样本...')
        sample = get_sample(data, candidate, data_key)
        gc.collect()

        print('开始构造特征...')
        user_count = get_user_count(data, sample, data_key)  # 获取用户历史行为次数
        start_loc = get_start_loc(data, sample, data_key)  # 获取出发地热度（作为出发地的次数、人数）
        end_loc = get_end_loc(data, sample, data_key)  # 获取目的地热度（作为目的地的个数、人数，作为出发地的个数、人数，折返比）
        user_sloc = get_user_sloc(data, sample, data_key)  # 用户×出发地（次数 目的地个数）
        user_eloc = get_user_eloc(data, sample, candidate, data_key)  # 用户×目的地
        sloc_eloc = get_sloc_eloc(data, sample, data_key)  # 出发地×目的地
        bike_eloc = get_bike_eloc(data, sample, candidate, data_key)  # 自行车×目的地
        user_sloc_eloc = get_user_sloc_eloc(data, sample, data_key)  # 用户×出发地×目的地
        gc.collect()

        # time_sloc = get_time_sloc(data,sample)                              # 时间×出发地
        # gc.collect()
        # time_eloc = get_time_eloc(data,sample)                              # 时间×目的地
        # gc.collect()
        # time_sloc_eloc = get_time_sloc_eloc(data, sample)                   # 时间×目的地
        # gc.collect()
        # time_user_eloc = get_time_user_eloc(data, sample)                   # 时间×用户×目的地
        # gc.collect()

        # 扩大统计范围重新统计
        ex_sloc = get_ex_sloc(data, sample, data_key)  # 扩大 出发地
        ex_eloc = get_ex_eloc(data, sample, data_key)  # 获取目的地热度（作为目的地的个数、人数，作为出发地的个数、人数，折返比）
        ex_user_sloc = get_ex_user_sloc(data, sample, data_key)  # 用户×出发地（次数 目的地个数）
        ex_user_eloc = get_ex_user_eloc(data, sample, data_key)  # 用户×目的地
        ex_sloc_eloc = get_ex_sloc_eloc(data, sample, data_key)  # 出发地×目的地
        ex_user_sloc_eloc = get_ex_user_sloc_eloc(data, sample, candidate, data_key)  # 用户×出发地×目的地
        gc.collect()

        print('开始合并特征...')
        result = concat([sample, user_count, start_loc, end_loc, user_sloc, user_eloc, sloc_eloc,
                         bike_eloc, user_sloc_eloc, ex_sloc, ex_eloc, ex_user_sloc, ex_user_eloc,
                         ex_sloc_eloc, ex_user_sloc_eloc
                         ])
        del sample, user_count, start_loc, end_loc, user_sloc, user_eloc, sloc_eloc, bike_eloc, \
            user_sloc_eloc, ex_sloc, ex_eloc, ex_user_sloc, ex_user_eloc, ex_sloc_eloc, ex_user_sloc_eloc
        gc.collect()
        result = second_feat(data, result)
        result = get_all_order_pred(result)
        gc.collect()

        print('添加label')
        # 添加标签
        result = get_label(result)

        print('存储数据...')
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    print('特征矩阵大小：{}'.format(result.shape))
    print('生成特征一共用时{}秒'.format(time.time() - t0))
    return result


# 线下测试集
train = pd.read_csv(train_path)
train1 = train[(train['starttime'] < '2017-05-23 00:00:00')].copy()
train2 = train[(train['starttime'] >= '2017-05-23 00:00:00') & (train['starttime'] < '2017-05-25 00:00:00')].copy()
train2.loc[:, 'geohashed_end_loc'] = np.nan
test = pd.read_csv(test_path)
test.loc[:, 'geohashed_end_loc'] = np.nan
data = pd.concat([train1, train2, test])
del train, test
train_data = data[(data['starttime'] >= '2017-05-21 00:00:00') & ((data['starttime'] < '2017-05-23 00:00:00'))].copy()
train_feat = make_train_set(data, train_data)
test_data = data[(data['starttime'] >= '2017-05-23 00:00:00') & ((data['starttime'] < '2017-05-25 00:00:00'))].copy()
test_feat = make_train_set(data, test_data)

predictors = train_feat.columns.drop(['orderid', 'geohashed_end_loc', 'label', 'userid',
                                      'bikeid', 'starttime', 'geohashed_start_loc', 'time'])

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'max_depth': 10,
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5,
    'verbose': 0,
    'seed': 66
}
print('Start training...')

lgb_train = lgb.Dataset(train_feat[predictors], train_feat.label)
lgb_eval = lgb.Dataset(test_feat[predictors], test_feat.label, reference=lgb_train)

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=1000,
                valid_sets=lgb_eval,
                verbose_eval = 50,
                early_stopping_rounds=100)
feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
feat_imp.to_csv(cache_path + 'feat_imp.csv')

preds = gbm.predict(test_feat[predictors])
test_feat['pred'] = preds
result = reshape(test_feat)
result = pd.merge(test_data[['orderid']], result, on='orderid', how='left')
print('map得分为:{}'.format(map(result)))

# 线上提交
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
test.loc[:, 'geohashed_end_loc'] = np.nan
data = pd.concat([train, test])
del train, test
train_data = data[(data['starttime'] >= '2017-05-23 00:00:00') & ((data['starttime'] < '2017-05-25 00:00:00'))].copy()
train_feat = make_train_set(data, train_data)

predictors = train_feat.columns.drop(['orderid', 'geohashed_end_loc', 'label', 'userid',
                                      'bikeid', 'starttime', 'geohashed_start_loc', 'time'])
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'max_depth': 10,
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5,
    'verbose': 0,
    'seed': 66
}

lgb_train = lgb.Dataset(train_feat[predictors].values, train_feat.label.values)
print('Start training...')
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=5000,
                valid_sets=lgb_eval,
                verbose_eval = 50,
                early_stopping_rounds=100)
pickle.dump(gbm,open(cache_path+'gmb.model','wb+'))
feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
feat_imp.to_csv(cache_path + 'feat_imp.csv')
del train_feat, lgb_train
gc.collect()

test_data = data[(data['starttime']>='2017-05-25 00:00:00')].copy()
test_feat = make_train_set(data,test_data)
test_pred = test_feat[['orderid','geohashed_end_loc']].copy()
test_feat = test_feat[predictors].values
del data,test_data
gc.collect()


print('Start predicting...')
preds = gbm.predict(test_feat)
del test_feat
test_pred['pred'] = preds
gc.collect()
result = reshape(test_pred)
test = pd.read_csv(test_path)
result = pd.merge(test[['orderid']], result, on='orderid', how='left')
result.fillna('wx4f8mt', inplace=True)
result = get_noise(result, 0.1)
result.to_csv(r'C:\Users\csw\Desktop\python\mobike\submission\0826(1).csv', index=False, header=False)














# 第一组
gbm = pickle.load(open(cache_path+'gmb_().model'.format(n),'wb+'))
test_data1 = data[(data['starttime'] >= '2017-05-25 00:00:00') & ((data['starttime'] < '2017-05-28 00:00:00'))].copy()
test_feat1 = make_train_set(data, test_data1)
test_pred1 = test_feat1[['orderid', 'geohashed_end_loc']].copy()
test_feat1 = test_feat1[predictors].values
preds1 = gbm.predict(test_feat1)
test_pred1.loc[:,'pred'] = preds1
del test_data1,test_feat1
gc.collect()
# 第二组
test_data2 = data[(data['starttime'] >= '2017-05-28 00:00:00')].copy()
test_feat2 = make_train_set(data, test_data2)
test_pred2 = test_feat2[['orderid', 'geohashed_end_loc']].copy()
test_feat2 = test_feat2[predictors].values
preds2 = gbm.predict(test_feat2)
test_pred2.loc[:,'pred'] = preds2
del test_data2,test_feat2,data
gc.collect()

# 合并生成结果
test_pred = pd.concat([test_pred1,test_pred2],axis=0)
result = reshape(test_pred)
test = pd.read_csv(test_path)
result = pd.merge(test[['orderid']], result, on='orderid', how='left')
result.fillna('wx4f8mt', inplace=True)
result = get_noise(result, 0.1)
result.to_csv(r'C:\Users\csw\Desktop\python\mobike\submission\0829(1).csv', index=False, header=False)
