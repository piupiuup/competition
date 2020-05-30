import os
import gc
import time
import random
import pickle
import Geohash
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime
from datetime import timedelta
from sklearn.cross_validation import KFold

cache_path = 'F:/mobike_cache11/'
data_path = 'C:/Users/csw/Desktop/python/mobike/data/'
train_path = data_path + 'train.csv'
test_path = data_path + 'test.csv'
select_sample_path = data_path + 'select_sample.csv'
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

# 将时间转换为当天相对时间
def int_time(time):
    result = int(time[11:13]) * 60 + int(time[14:16])
    return result

# 合并节约内存
def concat(L):
    result = None
    for l in L:
        if result is None:
            result = l
        else:
            result[l.columns.tolist()] = l
    return result


# 抽样函数
def make_sample(n,n_sub=2,seed=None):
    import random
    if seed is not None:
        random.seed(seed)
    if type(n) is int:
        l = list(range(n))
        s = int(n / n_sub)
    else:
        l = n
        s = int(len(n) / n_sub)
    random.shuffle(l)
    result = []
    for i in range(n_sub):
        if i == n_sub:
            result.append(l[i*s:])
        else:
            result.append(l[i*s: (i+1)*s])
    return result

# 周围的8个地点加自己DataFrame
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

# 周围的9×7个地点
def get_near_loc_dict_9_7():
    result_path = cache_path + 'near_loc_9_7.hdf'
    if os.path.exists(result_path):
        result = pickle.load(open(result_path, 'rb+'))
    else:
        import geohash
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        loc_set = train['geohashed_start_loc'].tolist() \
                   + train['geohashed_end_loc'].tolist() \
                   + test['geohashed_start_loc'].tolist()
        loc_set = set(loc_set)
        result = {}
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
            result[loc] = list(set(result_sub) & loc_set)
        pickle.dump(result, open(result_path, 'wb+'))
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
# 计算周围9×7范围内的目的地个数：
def get_count_9_7(orderids,locs,data):
    eloc_count = data.groupby('geohashed_end_loc')['userid'].count().to_dict()
    near_loc_dict_9_7 = get_near_loc_dict_9_7()
    result = []
    for orderid,locs in zip(orderids,locs):
        sum_count = 0
        near_locs = near_loc_dict_9_7[locs]
        for near_loc in near_locs:
            if near_loc in eloc_count:
                sum_count += eloc_count[near_loc]
        result.append(sum_count)
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
def if_around2(feat, data):
    data_temp = data[~data['geohashed_end_loc'].isnull()]
    order_sloc_dict = dict(zip(data_temp['orderid'].values, data_temp['geohashed_start_loc'].values))
    order_eloc_dict = dict(zip(data_temp['orderid'].values, data_temp['geohashed_end_loc'].values))
    near_loc_dict = get_near_loc_dict()
    result = []
    for eloc, orderid in zip(feat['geohashed_end_loc'], feat['orderid']):
        if order_sloc_dict.__contains__(orderid):
            if ((order_eloc_dict[orderid] in near_loc_dict[order_sloc_dict[orderid]]) &
                (eloc in near_loc_dict[order_sloc_dict[orderid]])):
                result.append(1)
            else:
                result.append(0)
        else:
            result.append(0)
    return result


def group_normalize(data, key, feat):
    data_temp = data[[key,feat]]
    grp = data_temp.groupby(key, as_index=False)[feat].agg({'std': 'std', 'avg': 'mean'})
    result = pd.merge(data_temp, grp, on=key, how='left')
    result[feat] = ((result[feat] - result['avg']) / result['std']).fillna(1)
    return result[feat]


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

# 获取角度
def get_direction(data):
    loc_dict = get_loc_dict()
    data_temp = data[['geohashed_start_loc', 'geohashed_end_loc']].copy()
    data_temp['lat1'] = data_temp['geohashed_start_loc'].apply(lambda x: loc_dict[x][0])
    data_temp['lon1'] = data_temp['geohashed_start_loc'].apply(lambda x: loc_dict[x][1])
    data_temp['lat2'] = data_temp['geohashed_end_loc'].apply(lambda x: loc_dict[x][0])
    data_temp['lon2'] = data_temp['geohashed_end_loc'].apply(lambda x: loc_dict[x][1])
    data_temp['x'] = data_temp['lat2'] - data_temp['lat1']
    data_temp['y'] = data_temp['lon2'] - data_temp['lon1']
    data_temp['z'] = (data_temp['x']**2 + data_temp['y']**2) ** 0.5
    result = np.arccos(data_temp['x']/data_temp['z']) * (data_temp['y'].apply(lambda x:1 if x>0 else -1))
    return result
# 获取夹角
def get_angle(data):
    loc_dict = get_loc_dict()
    data_temp = data[['geohashed_start_loc', 'mean_lat', 'mean_lon', 'lat', 'lon']].copy()
    data_temp['slat'] = data_temp['geohashed_start_loc'].apply(lambda x: loc_dict[x][0])
    data_temp['slon'] = data_temp['geohashed_start_loc'].apply(lambda x: loc_dict[x][1])
    data_temp['x1'] = data_temp['lat'] - data_temp['slat']
    data_temp['y1'] = data_temp['lon'] - data_temp['slon']
    data_temp['x2'] = data_temp['mean_lat'] - data_temp['slat']
    data_temp['y2'] = data_temp['mean_lon'] - data_temp['slon']
    result = np.arccos((data_temp['x1'] * data_temp['x2'] + data_temp['y1']*data_temp['y2'])/
                       (2*(data_temp['x1']**2 + data_temp['y1']**2)**0.5 * (data_temp['x2']**2 + data_temp['y2']**2)**0.5))
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
    true = get_true()
    result_temp = result.copy()
    result_temp['true'] = result_temp['orderid'].map(true)
    n = result_temp.shape[0]
    score1 = sum(result_temp['true'] == result_temp[0]) / 1.0 / n
    score2 = sum(result_temp['true'] == result_temp[1]) / 2.0 / n
    score3 = sum(result_temp['true'] == result_temp[2]) / 3.0 / n
    score = score1 + score2 + score3
    return score

def leak_map(result):
    true = get_leak_true()
    result_temp = result.copy()
    result_temp['true'] = result_temp['orderid'].map(true)
    n = result_temp.shape[0]
    score1 = sum(result_temp['true'] == result_temp[0]) / 1.0 / n
    score2 = sum(result_temp['true'] == result_temp[1]) / 2.0 / n
    score3 = sum(result_temp['true'] == result_temp[2]) / 3.0 / n
    score = score1 + score2 + score3
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
    true = get_true()
    data.loc[:, 'label'] = data['orderid'].map(true)
    data['label'] = (data['label'] == data['geohashed_end_loc']).astype('int')
    data['label'] = data['label'].fillna(0)
    return data

def get_leak_label(data):
    true = get_leak_true()
    data.loc[:, 'label'] = data['orderid'].map(true)
    data['label'] = (data['label'] == data['geohashed_end_loc']).astype('int')
    data['label'] = data['label'].fillna(0)
    return data

def get_leak_true():
    result_path = cache_path + 'leak_true.pkl'
    if os.path.exists(result_path):
        true = pickle.load(open(result_path, 'rb+'))
    else:
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        test.loc[:,'geohashed_end_loc'] = np.nan
        data = pd.concat([train,test])
        bike_next_loc = get_bike_next_loc(data, data, 'all')
        data = pd.merge(data,bike_next_loc,on=['orderid','geohashed_end_loc'],how='left')
        data = data[((data['bike_eloc_sep_time'].isnull()) & (~data['geohashed_end_loc'].isnull()))]
        true = dict(zip(data['orderid'].values,data['geohashed_end_loc'].values))
        pickle.dump(true,open(result_path, 'wb+'))
    return true

def get_true():
    result_path = cache_path + 'true.pkl'
    if os.path.exists(result_path):
        true = pickle.load(open(result_path, 'rb+'))
    else:
        train = pd.read_csv(train_path)
        true = dict(zip(train['orderid'].values,train['geohashed_end_loc'].values))
        pickle.dump(true,open(result_path, 'wb+'))
    return true


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


# 筛选起始地点去向最多的6个地点
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
        near_loc = get_near_loc()
        result = result.merge(near_loc, left_on='geohashed_start_loc', right_on='loc', how='left').fillna(0)
        result['geohashed_start_loc'] = result['near_loc']
        result = result.groupby(['hour', 'holiday', 'geohashed_start_loc', 'geohashed_end_loc'], as_index=False).sum()
        result = result[result['geohashed_start_loc'].isin(candidate['geohashed_start_loc'].values)]
        near_hour = get_near_hour()
        result = result.merge(near_hour, on='hour', how='left')
        result['hour'] = result['near_hour']
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
    result_path = cache_path + 'bike_next_loc_{}.hdf'.format(data_key)
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


# 用户前一次的起始地点
def get_user_forward_loc(data, sample, data_key):
    result_path = cache_path + 'user_forward_loc_%d.hdf' % (data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        data_temp = data.copy()
        data_temp = data_temp[data_temp['userid'].isin(sample['userid'].values)]
        if data_temp.shape[0] == 0:
            return pd.DataFrame(columns=['orderid', 'geohashed_end_loc', 'user_eloc_nloc_sep_time'])
        data_temp = rank(data_temp, 'userid', 'starttime', ascending=True)
        result = pd.merge(sample, data_temp[['orderid', 'rank']], on='orderid', how='left')
        data_temp['rank'] = data_temp['rank'] + 1
        data_temp.rename(columns={'geohashed_start_loc':'geohashed_start_loc2',
                                  'geohashed_end_loc':'geohashed_end_loc2',
                                  'starttime':'starttime2'},inplace=True)
        result = pd.merge(result,data_temp[['userid', 'rank', 'geohashed_start_loc2','geohashed_end_loc2','starttime2']],
                          on=['userid', 'rank'], how='left')
        loc_dict = get_loc_dict()
        user_eloc_forward_sep_time = []
        user_eloc_forward_distance = []
        user_eloc_forward_mdistance = []
        for tuple in result.itertuples():
            if tuple.geohashed_start_loc2 is np.nan:
                user_eloc_forward_sep_time.append(-1)
                user_eloc_forward_distance.append(-1)
                user_eloc_forward_mdistance.append(-1)
            else:
                user_eloc_forward_sep_time.append(diff_of_minutes(tuple.starttime, tuple.starttime2))
                lat1, lon1 = loc_dict[tuple.geohashed_start_loc2]
                lat2, lon2 = loc_dict[tuple.geohashed_end_loc]
                user_eloc_forward_distance.append(cal_distance(lat1, lon1, lat2, lon2))
                user_eloc_forward_mdistance.append(cal_mht_distance(lat1, lon1, lat2, lon2))
        result['user_eloc_forward_sep_time'] = user_eloc_forward_sep_time
        result['user_eloc_forward_distance'] = user_eloc_forward_distance
        result['user_eloc_forward_mdistance'] = user_eloc_forward_mdistance
        result = result[['user_eloc_forward_sep_time', 'user_eloc_forward_distance','user_eloc_forward_mdistance']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result


#####################构造特征####################
# 获取order时间特征
def get_order_feat(data, sample, data_key):
    feat_path = cache_path + 'order_feat_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        loc_dict = get_loc_dict()
        data_temp = data[~data['geohashed_end_loc'].isnull()].copy()
        data_temp['int_time2'] = data_temp['starttime'].apply(int_time)
        data_temp['direction2'] = get_direction(data_temp)
        data_temp.rename(columns={'orderid': 'orderid2'}, inplace=True)
        sample_temp = sample.copy()
        sample_temp['int_time'] = sample_temp['starttime'].apply(int_time)
        sample_temp['direction'] = get_direction(sample_temp)

        user_time = pd.merge(data_temp[['orderid2', 'userid', 'geohashed_end_loc', 'int_time2']],
                             sample_temp[['orderid', 'userid', 'geohashed_end_loc', 'int_time']],
                             on=['userid', 'geohashed_end_loc'], how='inner')
        user_time['diff_time'] = (user_time['int_time'] - user_time['int_time2']).apply(lambda x: min(abs(1440 - abs(x)), abs(x)))
        user_time = user_time.groupby(['orderid', 'geohashed_end_loc'], as_index=False)['diff_time'].agg(
            {'user_mean_diff_time': 'mean',
             'user_min_diff_time': 'min'})

        sloc_time = pd.merge(data_temp[['orderid2', 'geohashed_start_loc', 'geohashed_end_loc', 'int_time2']],
                             sample_temp[['orderid', 'geohashed_start_loc', 'geohashed_end_loc', 'int_time']],
                             on=['geohashed_start_loc', 'geohashed_end_loc'], how='inner')
        sloc_time['diff_time'] = (sloc_time['int_time'] - sloc_time['int_time2']).apply(lambda x: min(abs(1440 - abs(x)), abs(x)))
        sloc_time = sloc_time.groupby(['orderid', 'geohashed_end_loc'], as_index=False)['diff_time'].agg(
            {'sloc_mean_diff_time': 'mean',
             'sloc_min_diff_time': 'min'})

        user_direction = pd.merge(data_temp[['orderid2', 'userid', 'direction2']],
                             sample_temp[['orderid', 'userid', 'geohashed_end_loc', 'direction']],
                             on='userid', how='inner')
        user_direction = user_direction[user_direction['orderid'] != user_direction['orderid2']]
        user_direction['diff_direction'] = (user_direction['direction'] - user_direction['direction2']).apply(
            lambda x: min(abs(np.pi - abs(x)), abs(x)))
        user_mean_diff_direction = user_direction.groupby(['orderid','geohashed_end_loc'], as_index=False)['diff_direction'].agg(
            {'user_mean_diff_direction': 'mean'})
        user_direction['diff_direction'] = user_direction['diff_direction'].apply(lambda x: min((np.pi*0.5-x),x))
        user_mean_diff_direction2 = user_direction.groupby(['orderid','geohashed_end_loc'], as_index=False)['diff_direction'].agg(
            {'user_mean_diff_direction2': 'mean'})

        sloc_direction_grp = data_temp.groupby('geohashed_start_loc')
        sloc_direction_dict = {}
        for sloc,grp in sloc_direction_grp:
            sloc_direction_dict[sloc] = grp['direction2'].values
        sloc_mean_diff_direction = []
        sloc_mean_diff_direction2 = []
        for tuple in sample_temp.itertuples():
            sloc = tuple.geohashed_start_loc
            if sloc in sloc_direction_dict:
                direction = tuple.direction
                diff_direction = np.abs(sloc_direction_dict[sloc]-direction)
                diff_direction = np.min([diff_direction, abs(np.pi - diff_direction)], axis=0)
                sloc_mean_diff_direction.append(diff_direction.mean())
                diff_direction = np.min([diff_direction, abs(np.pi*0.5 - diff_direction)], axis=0)
                sloc_mean_diff_direction2.append(diff_direction.mean())
            else:
                sloc_mean_diff_direction.append(-1)
                sloc_mean_diff_direction2.append(-1)

        eloc_direction_grp = data_temp.groupby('geohashed_end_loc')
        eloc_direction_dict = {}
        for eloc, grp in eloc_direction_grp:
            eloc_direction_dict[eloc] = grp['direction2'].values
        eloc_mean_diff_direction = []
        eloc_mean_diff_direction2 = []
        for tuple in sample_temp.itertuples():
            eloc = tuple.geohashed_end_loc
            if eloc in eloc_direction_dict:
                direction = tuple.direction
                diff_direction = np.abs(eloc_direction_dict[eloc] - direction)
                diff_direction = np.min([diff_direction, abs(np.pi - diff_direction)], axis=0)
                eloc_mean_diff_direction.append(diff_direction.mean())
                diff_direction = np.min([diff_direction, abs(np.pi * 0.5 - diff_direction)], axis=0)
                eloc_mean_diff_direction2.append(diff_direction.mean())
            else:
                eloc_mean_diff_direction.append(-1)
                eloc_mean_diff_direction2.append(-1)


        feat = sample.merge(user_time, on=['orderid', 'geohashed_end_loc'], how='left')
        feat = feat.merge(sloc_time, on=['orderid', 'geohashed_end_loc'], how='left')
        feat = feat.merge(user_mean_diff_direction, on=['orderid', 'geohashed_end_loc'], how='left')
        feat = feat.merge(user_mean_diff_direction2, on=['orderid', 'geohashed_end_loc'], how='left')
        feat['sloc_mean_diff_direction'] = sloc_mean_diff_direction
        feat['sloc_mean_diff_direction2'] = sloc_mean_diff_direction2
        feat['eloc_mean_diff_direction'] = eloc_mean_diff_direction
        feat['eloc_mean_diff_direction2'] = eloc_mean_diff_direction2

        feat['lat1'] = feat['geohashed_start_loc'].apply(lambda x: loc_dict[x][0])
        feat['lon1'] = feat['geohashed_start_loc'].apply(lambda x: loc_dict[x][1])
        feat['lat2'] = feat['geohashed_end_loc'].apply(lambda x: loc_dict[x][0])
        feat['lon2'] = feat['geohashed_end_loc'].apply(lambda x: loc_dict[x][1])
        feat['direction'] = get_direction(feat)

        feat = feat[['user_mean_diff_time','user_min_diff_time','sloc_mean_diff_time',
                     'sloc_min_diff_time','sloc_mean_diff_direction','sloc_mean_diff_direction2',
                     'eloc_mean_diff_direction','eloc_mean_diff_direction2','lat1',
                     'lon1','lat2','lon2','direction', 'user_mean_diff_direction',
                     'user_mean_diff_direction2']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat

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
        data_temp['lat2'] = data_temp['geohashed_end_loc'].apply(lambda x: 0 if x is np.nan else loc_dict[x][0])
        data_temp['lon2'] = data_temp['geohashed_end_loc'].apply(lambda x: 0 if x is np.nan else loc_dict[x][1])
        feat = data_temp.groupby('userid').agg({'userid': {'user_count2': 'count'},
                                                'geohashed_end_loc': {'user_count': 'count'},
                                                'lat1': {'sum_lat1': 'sum'},
                                                'lon1': {'sum_lon1': 'sum'},
                                                'lat2': {'sum_lat2': 'sum'},
                                                'lon2': {'sum_lon2': 'sum'}})
        feat.columns = feat.columns.droplevel(0)
        feat.reset_index(inplace=True)
        feat['sum_lat'] = feat['sum_lat1'] + feat['sum_lat2']
        feat['sum_lon'] = feat['sum_lon1'] + feat['sum_lon2']
        feat = pd.merge(sample, feat, on=['userid'], how='left')
        feat[['user_count2','user_count']].fillna(0, inplace=True)
        feat['mean_lat'] = feat['sum_lat'] / (feat['user_count2'] + feat['user_count'])
        feat['mean_lon'] = feat['sum_lon'] / (feat['user_count2'] + feat['user_count'])
        feat['lat2'] = feat['geohashed_end_loc'].apply(lambda x: np.nan if x is np.nan else loc_dict[x][0])
        feat['lon2'] = feat['geohashed_end_loc'].apply(lambda x: np.nan if x is np.nan else loc_dict[x][1])
        feat['center_dis'] = ((feat['lat2'] - feat['mean_lat']) ** 2 + (
        np.cos(feat['mean_lon'] / 57.2958) * (feat['lon2'] - feat['mean_lon'])) ** 2) ** 0.5
        feat = feat[['user_count', 'user_count2', 'center_dis']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat


# 获取出发地热度（作为出发地的次数、人数）
def get_start_loc(data, sample, data_key):
    feat_path = cache_path + 'start_loc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        data_temp = data[~data['geohashed_end_loc'].isnull()].copy()
        loc_dict = get_loc_dict()
        data_temp['lat'] = data_temp['geohashed_end_loc'].apply(lambda x: 0 if x is np.nan else loc_dict[x][0])
        data_temp['lon'] = data_temp['geohashed_end_loc'].apply(lambda x: 0 if x is np.nan else loc_dict[x][1])
        sloc_count = data.groupby('geohashed_start_loc',as_index=False)['userid'].agg({'sloc_count2': 'count',
                                                                        'sloc_n_user2': 'nunique'})
        feat = data_temp.groupby('geohashed_start_loc').agg({'userid': {'sloc_count': 'count',
                                                                        'sloc_n_user': 'nunique'},
                                                             'geohashed_end_loc':{'eloc_no_nan':'count'},
                                                                        'lat': {'sum_lat': 'sum'},
                                                                        'lon': {'sum_lon': 'sum'}})
        feat.columns = feat.columns.droplevel(0)
        feat.reset_index(inplace=True)
        feat = feat.merge(sloc_count,on='geohashed_start_loc',how='outer')
        feat = pd.merge(sample, feat, on=['geohashed_start_loc'], how='left').fillna(0)
        feat['mean_lat'] = feat['sum_lat'] / (feat['eloc_no_nan'] + 0.001)
        feat['mean_lon'] = feat['sum_lon'] / (feat['eloc_no_nan'] + 0.001)
        feat['lat'] = feat['geohashed_end_loc'].apply(lambda x: np.nan if x is np.nan else loc_dict[x][0])
        feat['lon'] = feat['geohashed_end_loc'].apply(lambda x: np.nan if x is np.nan else loc_dict[x][1])
        feat['sloc_center_dis'] = ((feat['lat'] - feat['mean_lat']) ** 2 + (
            np.cos(feat['mean_lon'] / 57.2958) * (feat['lon'] - feat['mean_lon'])) ** 2) ** 0.5
        feat['sloc_direction'] = get_angle(feat)
        feat['seloc_count_9_7'] = get_count_9_7(feat['orderid'],feat['geohashed_start_loc'],data)
        feat = feat[['sloc_count', 'sloc_count2', 'sloc_n_user2', 'sloc_n_user',
                     'sloc_center_dis','sloc_direction','seloc_count_9_7']].fillna(100000)
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat


# 获取目标地点的热度(目的地)
def get_end_loc(data, sample, data_key):
    feat_path = cache_path + 'end_loc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        data_temp = data[~data['geohashed_end_loc'].isnull()].copy()
        loc_dict = get_loc_dict()
        data_temp['lat'] = data_temp['geohashed_start_loc'].apply(lambda x: 0 if x is np.nan else loc_dict[x][0])
        data_temp['lon'] = data_temp['geohashed_start_loc'].apply(lambda x: 0 if x is np.nan else loc_dict[x][1])
        feat = data_temp.groupby('geohashed_end_loc').agg({'userid': {'eloc_count': 'count',
                                                                      'eloc_n_user': 'nunique'},
                                                           'lat': {'sum_lat': 'sum'},
                                                           'lon': {'sum_lon': 'sum'}})
        feat.columns = feat.columns.droplevel(0)
        feat.reset_index(inplace=True)
        feat = pd.merge(sample, feat, on=['geohashed_end_loc'], how='left').fillna(0)
        feat['mean_lat'] = feat['sum_lat'] / (feat['eloc_count'] + 0.001)
        feat['mean_lon'] = feat['sum_lon'] / (feat['eloc_count'] + 0.001)
        feat['lat'] = feat['geohashed_start_loc'].apply(lambda x: np.nan if x is np.nan else loc_dict[x][0])
        feat['lon'] = feat['geohashed_start_loc'].apply(lambda x: np.nan if x is np.nan else loc_dict[x][1])
        feat['eloc_center_dis'] = ((feat['lat'] - feat['mean_lat']) ** 2 + (
            np.cos(feat['mean_lon'] / 57.2958) * (feat['lon'] - feat['mean_lon'])) ** 2) ** 0.5

        eloc_as_sloc = data.groupby('geohashed_start_loc', as_index=False)['userid'].agg(
            {'eloc_as_sloc_count': 'count',
             'eloc_as_sloc_n_user': 'nunique'})
        eloc_as_sloc.rename(columns={'geohashed_start_loc': 'geohashed_end_loc'}, inplace=True)
        feat = pd.merge(feat, eloc_as_sloc, on='geohashed_end_loc', how='left').fillna(0)
        feat['eloc_2count'] = feat['eloc_count'] + feat['eloc_as_sloc_count']
        feat['eloc_in_out_rate'] = feat['eloc_count'] / (feat['eloc_as_sloc_count'] + 0.001)
        feat['eeloc_count_9_7'] = get_count_9_7(feat['orderid'], feat['geohashed_end_loc'], data)
        feat = feat[['eloc_as_sloc_count', 'eloc_as_sloc_n_user', 'eloc_2count',
                     'eloc_count', 'eloc_n_user', 'eloc_in_out_rate','eloc_center_dis','eeloc_count_9_7']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat


# 用户×出发地（次数 目的地个数）
def get_user_sloc(data, sample, data_key):
    feat_path = cache_path + 'user_sloc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        user_sloc2 = data.groupby(['userid', 'geohashed_start_loc'], as_index=False)['userid'].agg(
            {'user_sloc_count2': 'count'})
        user_sloc = data.groupby(['userid', 'geohashed_start_loc'], as_index=False)['geohashed_end_loc'].agg(
            {'user_sloc_count': 'count'})
        feat = pd.merge(sample, user_sloc, on=['userid', 'geohashed_start_loc'], how='left').fillna(0)
        feat = pd.merge(feat, user_sloc2, on=['userid', 'geohashed_start_loc'], how='left').fillna(0)
        feat = feat[['user_sloc_count','user_sloc_count2']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat


# 用户×目的地
def get_user_eloc(data, sample, data_key):
    feat_path = cache_path + 'user_eloc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        data_temp = data[~data['geohashed_end_loc'].isnull()].copy()
        loc_dict = get_loc_dict()
        data_temp['lat'] = data_temp['geohashed_start_loc'].apply(lambda x: np.nan if x is np.nan else loc_dict[x][0])
        data_temp['lon'] = data_temp['geohashed_start_loc'].apply(lambda x: np.nan if x is np.nan else loc_dict[x][1])
        user_eloc_as_sloc2 = data.groupby(['userid', 'geohashed_start_loc'], as_index=False)['geohashed_start_loc'].agg(
            {'user_eloc_as_sloc_count2': 'count'})
        user_eloc_as_sloc = data_temp.groupby(['userid', 'geohashed_start_loc'], as_index=False)['geohashed_end_loc'].agg(
            {'user_eloc_as_sloc_count': 'count',
             'user_eloc_as_sloc_n_eloc': 'nunique'})
        user_eloc = data_temp.groupby(['userid', 'geohashed_end_loc']).agg(
            {'geohashed_start_loc':{'user_eloc_count': 'count',
                                    'user_eloc_n_sloc': 'nunique'},
             'lat':{'sum_lat':'sum'},
             'lon':{'sum_lon':'sum'},})
        user_eloc.columns = user_eloc.columns.droplevel(0)
        user_eloc.reset_index(inplace=True)
        user_eloc_as_sloc.rename(columns={'geohashed_start_loc': 'geohashed_end_loc'}, inplace=True)
        user_eloc_as_sloc2.rename(columns={'geohashed_start_loc': 'geohashed_end_loc'}, inplace=True)
        feat = pd.merge(sample, user_eloc_as_sloc, on=['userid', 'geohashed_end_loc'], how='left')
        feat = pd.merge(feat, user_eloc_as_sloc2, on=['userid', 'geohashed_end_loc'], how='left')
        feat = pd.merge(feat, user_eloc, on=['userid', 'geohashed_end_loc'], how='left').fillna(0)
        feat['mean_lat'] = feat['sum_lat'] / feat['user_eloc_count']
        feat['mean_lon'] = feat['sum_lon'] / feat['user_eloc_count']
        feat['lat'] = feat['geohashed_end_loc'].apply(lambda x: np.nan if x is np.nan else loc_dict[x][0])
        feat['lon'] = feat['geohashed_end_loc'].apply(lambda x: np.nan if x is np.nan else loc_dict[x][1])
        feat['user_eloc_center_dis'] = cal_distance(feat['lat'],feat['mean_lat'],feat['lon'],feat['mean_lon'])
        feat['user_eloc_2count'] = feat['user_eloc_count'] + feat['user_eloc_as_sloc_count']
        feat['user_sloc_goback_rate'] = feat['user_eloc_count'] / (feat['user_eloc_as_sloc_count'] + 0.001)
        feat[['user_eloc_nloc_sep_time', 'user_eloc_nloc_distance', 'user_eloc_nloc_mdistance']] = get_user_next_loc(data, sample, data_key)
        feat[['user_eloc_forward_sep_time', 'user_eloc_forward_distance','user_eloc_forward_mdistance']] = get_user_forward_loc(data, sample, data_key)
        feat = feat[['user_eloc_as_sloc_count', 'user_eloc_count','user_eloc_as_sloc_n_eloc',
                     'user_eloc_n_sloc', 'user_eloc_2count', 'user_eloc_nloc_mdistance',
                     'user_sloc_goback_rate', 'user_eloc_nloc_sep_time','user_eloc_as_sloc_count2',
                     'user_eloc_nloc_distance', 'user_eloc_forward_sep_time','user_eloc_forward_distance',
                     'user_eloc_forward_mdistance','user_eloc_center_dis']]
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
        feat['sloc_eloc_2count'] = feat['sloc_eloc_count'] + feat['eloc_sloc_count']
        feat['sloc_eloc_2n_user'] = feat['sloc_eloc_n_user'] + feat['eloc_sloc_n_user']
        feat['sloc_eloc_goback_rate'] = feat['sloc_eloc_count'] / (feat['eloc_sloc_count'] + 0.001)
        feat['sloc_eloc_goback_n_user_rate'] = feat['sloc_eloc_n_user'] / (feat['eloc_sloc_n_user'] + 0.001)
        distance = get_distance(feat)
        feat = pd.concat([feat, distance], axis=1)
        feat = feat[['sloc_eloc_count', 'sloc_eloc_n_user', 'eloc_sloc_count',
                     'eloc_sloc_n_user', 'sloc_eloc_2count','sloc_eloc_goback_n_user_rate',
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
        feat[['bike_eloc_nloc_dis', 'bike_eloc_nloc_mdis']] = get_distance(feat, sloc='geohashed_end_loc',
                                                                           eloc='geohashed_end_loc2')
        feat = feat[['bike_eloc_sep_time', 'bike_eloc_nloc_dis', 'bike_eloc_nloc_mdis']].fillna(-1)
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
        feat['user_sloc_eloc_2count'] = feat['user_sloc_eloc_count'] + feat['user_eloc_sloc_count']
        feat = feat[['user_sloc_eloc_count', 'user_eloc_sloc_count', 'user_sloc_eloc_2count']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat



#####################按照时间统计分析#########################
# 获取用户历史行为次数
def get_hour_user(data, sample, data_key):
    feat_path = cache_path + 'hour_user_feat_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        data_temp = data.copy()
        loc_dict = get_loc_dict()
        data_temp['lat1'] = data_temp['geohashed_start_loc'].apply(lambda x: np.nan if x is np.nan else loc_dict[x][0])
        data_temp['lon1'] = data_temp['geohashed_start_loc'].apply(lambda x: np.nan if x is np.nan else loc_dict[x][1])
        data_temp['lat2'] = data_temp['geohashed_end_loc'].apply(lambda x: 0 if x is np.nan else loc_dict[x][0])
        data_temp['lon2'] = data_temp['geohashed_end_loc'].apply(lambda x: 0 if x is np.nan else loc_dict[x][1])
        feat = data_temp.groupby(['hour','userid']).agg({'userid': {'hour_user_count': 'count'},
                                                'geohashed_end_loc': {'eloc_no_nan': 'count'},
                                                'lat1': {'sum_lat1': 'sum'},
                                                'lon1': {'sum_lon1': 'sum'},
                                                'lat2': {'sum_lat2': 'sum'},
                                                'lon2': {'sum_lon2': 'sum'}})
        feat.columns = feat.columns.droplevel(0)
        feat.reset_index(inplace=True)
        feat['sum_lat'] = feat['sum_lat1'] + feat['sum_lat2']
        feat['sum_lon'] = feat['sum_lon1'] + feat['sum_lon2']
        feat = pd.merge(sample, feat, on=['hour','userid'], how='left')
        feat['hour_user_count'].fillna(0, inplace=True)
        order_eloc_dict = dict(zip(data['orderid'].values, data['geohashed_end_loc'].values))
        feat['sum_lat'] = feat['sum_lat'] - feat['orderid'].map(order_eloc_dict).apply(lambda x: 0 if x is np.nan else loc_dict[x][0])
        feat['sum_lon'] = feat['sum_lon'] - feat['orderid'].map(order_eloc_dict).apply(lambda x: 0 if x is np.nan else loc_dict[x][1])
        feat['mean_lat'] = feat['sum_lat'] / (feat['hour_user_count'] + feat['eloc_no_nan'] - (~feat['orderid'].map(order_eloc_dict).isnull()))
        feat['mean_lon'] = feat['sum_lon'] / (feat['hour_user_count'] + feat['eloc_no_nan'] - (~feat['orderid'].map(order_eloc_dict).isnull()))
        feat['lat2'] = feat['geohashed_end_loc'].apply(lambda x: np.nan if x is np.nan else loc_dict[x][0])
        feat['lon2'] = feat['geohashed_end_loc'].apply(lambda x: np.nan if x is np.nan else loc_dict[x][1])
        feat['hour_center_dis'] = ((feat['lat2'] - feat['mean_lat']) ** 2 + (
        np.cos(feat['mean_lon'] / 57.2958) * (feat['lon2'] - feat['mean_lon'])) ** 2) ** 0.5
        feat = feat[['hour_user_count', 'hour_center_dis']].fillna(-1)
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat

# 时间×出发地
def get_hour_sloc(data, sample, data_key):
    feat_path = cache_path + 'hour_sloc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        data_temp = data.copy()
        loc_dict = get_loc_dict()
        data_temp['lat'] = data_temp['geohashed_end_loc'].apply(lambda x: 0 if x is np.nan else loc_dict[x][0])
        data_temp['lon'] = data_temp['geohashed_end_loc'].apply(lambda x: 0 if x is np.nan else loc_dict[x][1])
        feat = data_temp.groupby(['hour','geohashed_start_loc']).agg({'userid': {'hour_sloc_count': 'count'},
                                                                      'geohashed_end_loc': {'eloc_no_nan': 'count'},
                                                                      'lat': {'sum_lat': 'sum'},
                                                                      'lon': {'sum_lon': 'sum'}})
        feat.columns = feat.columns.droplevel(0)
        feat.reset_index(inplace=True)
        feat = pd.merge(sample, feat, on=['hour','geohashed_start_loc'], how='left').fillna(0)
        order_eloc_dict = dict(zip(data['orderid'].values, data['geohashed_end_loc'].values))
        feat['sum_lat'] = feat['sum_lat'] - feat['orderid'].map(order_eloc_dict).apply(
            lambda x: 0 if x is np.nan else loc_dict[x][0])
        feat['sum_lon'] = feat['sum_lon'] - feat['orderid'].map(order_eloc_dict).apply(
            lambda x: 0 if x is np.nan else loc_dict[x][1])
        feat['hour_sloc_count'] = feat['hour_sloc_count'] - (feat['orderid'].isin(data['orderid'].values))
        order_eloc_dict = dict(zip(data['orderid'].values, data['geohashed_end_loc'].values))
        feat['eloc_no_nan'] = feat['eloc_no_nan'] - (~feat['orderid'].map(order_eloc_dict).isnull())
        feat['mean_lat'] = feat['sum_lat'] / (feat['eloc_no_nan'] + 0.001)
        feat['mean_lon'] = feat['sum_lon'] / (feat['eloc_no_nan'] + 0.001)
        feat['lat'] = feat['geohashed_end_loc'].apply(lambda x: np.nan if x is np.nan else loc_dict[x][0])
        feat['lon'] = feat['geohashed_end_loc'].apply(lambda x: np.nan if x is np.nan else loc_dict[x][1])
        feat['hour_sloc_center_dis'] = ((feat['lat'] - feat['mean_lat']) ** 2 + (
            np.cos(feat['mean_lon'] / 57.2958) * (feat['lon'] - feat['mean_lon'])) ** 2) ** 0.5
        feat = feat[['hour_sloc_count','hour_sloc_center_dis']].fillna(100000)
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat


# 时间×目的地
def get_hour_eloc(data, sample, data_key):
    feat_path = cache_path + 'hour_eloc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        hour_eloc_count = data.groupby(['hour', 'geohashed_end_loc'], as_index=False)[
            'userid'].agg({'hour_eloc_count': 'count'})
        hour_eloc_as_sloc_count = data.groupby(['hour', 'geohashed_start_loc'], as_index=False)[
            'userid'].agg({'hour_eloc_as_sloc_count': 'count'})
        hour_eloc_as_sloc_count.rename(columns={'geohashed_start_loc':'geohashed_end_loc'},inplace=True)
        feat = pd.merge(hour_eloc_count,hour_eloc_as_sloc_count,on=['hour','geohashed_end_loc'],how='outer').fillna(0)
        feat = pd.merge(sample, feat, on=['hour', 'geohashed_end_loc'], how='left').fillna(0)
        order_eloc_dict = dict(zip(data['orderid'].values, data['geohashed_end_loc'].values))
        feat['hour_eloc_count'] = feat['hour_eloc_count'] - (feat['geohashed_end_loc'] == feat['orderid'].map(order_eloc_dict))
        feat['hour_eloc_in_out_rate'] = feat['hour_eloc_count'] / (feat['hour_eloc_as_sloc_count']+0.001)
        feat = feat[['hour_eloc_count','hour_eloc_as_sloc_count','hour_eloc_in_out_rate']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat


# 时间×出发地×目的地
def get_hour_sloc_eloc(data, sample, data_key):
    feat_path = cache_path + 'hour_sloc_eloc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        hour_sloc_eloc_count = data.groupby(['hour', 'geohashed_start_loc', 'geohashed_end_loc'], as_index=False)[
            'userid'].agg({'hour_sloc_eloc_count': 'count'})
        feat = pd.merge(sample, hour_sloc_eloc_count, on=['hour', 'geohashed_start_loc', 'geohashed_end_loc'],
                        how='left').fillna(0)
        order_eloc_dict = dict(zip(data['orderid'].values, data['geohashed_end_loc'].values))
        feat['hour_sloc_eloc_count'] = feat['hour_sloc_eloc_count'] - (feat['geohashed_end_loc'] == feat['orderid'].map(order_eloc_dict))
        feat = feat[['hour_sloc_eloc_count']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat

# 时间×用户×目的地
def get_hour_user_eloc(data, sample, data_key):
    feat_path = cache_path + 'hour_user_eloc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        hour_user_eloc_count = data.groupby(['hour','userid', 'geohashed_end_loc'], as_index=False)['geohashed_start_loc'].agg(
            {'hour_user_eloc_count': 'count'})
        feat = pd.merge(sample, hour_user_eloc_count, on=['hour', 'userid', 'geohashed_end_loc'], how='left').fillna(0)
        order_eloc_dict = dict(zip(data['orderid'].values, data['geohashed_end_loc'].values))
        feat['hour_user_eloc_count'] = feat['hour_user_eloc_count'] - (feat['geohashed_end_loc'] == feat['orderid'].map(order_eloc_dict))
        feat = feat[['hour_user_eloc_count']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat

#####################扩大时间统计分析#########################
# 获取用户历史行为次数
def get_exhour_user(data, sample, data_key):
    feat_path = cache_path + 'exhour_user_feat_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        exhour_user_count = data.groupby(['hour','userid'], as_index=False)['userid'].agg({'exhour_user_count': 'count'})
        near_hour = get_near_hour()
        exhour_user_count = exhour_user_count.merge(near_hour, on='hour',  how='left')
        exhour_user_count['hour'] = exhour_user_count['near_hour']
        exhour_user_count = exhour_user_count.groupby(['hour','userid'], as_index=False).sum()
        feat = pd.merge(sample, exhour_user_count, on=['hour','userid'], how='left').fillna(0)
        feat['exhour_user_count'] = feat['exhour_user_count'] - feat['orderid'].isin(data['orderid'].values)
        feat = feat[['exhour_user_count']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat

# 时间×出发地
def get_exhour_sloc(data, sample, data_key):
    feat_path = cache_path + 'exhour_sloc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        data_temp = data.copy()
        loc_dict = get_loc_dict()
        data_temp['lat'] = data_temp['geohashed_end_loc'].apply(lambda x: 0 if x is np.nan else loc_dict[x][0])
        data_temp['lon'] = data_temp['geohashed_end_loc'].apply(lambda x: 0 if x is np.nan else loc_dict[x][1])
        feat = data_temp.groupby(['hour','geohashed_start_loc']).agg({'userid': {'exhour_sloc_count': 'count'},
                                                                      'geohashed_end_loc':{'eloc_no_nan':'count'},
                                                                      'lat': {'sum_lat': 'sum'},
                                                                      'lon': {'sum_lon': 'sum'}})
        feat.columns = feat.columns.droplevel(0)
        feat.reset_index(inplace=True)
        near_hour = get_near_hour()
        feat = feat.merge(near_hour, on='hour', how='left')
        feat['hour'] = feat['near_hour']
        feat = feat.groupby(['hour', 'geohashed_start_loc'], as_index=False).sum()
        feat = pd.merge(sample, feat, on=['hour', 'geohashed_start_loc'], how='left').fillna(0)
        order_eloc_dict = dict(zip(data['orderid'].values, data['geohashed_end_loc'].values))
        feat['sum_lat'] = feat['sum_lat'] - feat['orderid'].map(order_eloc_dict).apply(
            lambda x: 0 if x is np.nan else loc_dict[x][0])
        feat['sum_lon'] = feat['sum_lon'] - feat['orderid'].map(order_eloc_dict).apply(
            lambda x: 0 if x is np.nan else loc_dict[x][1])
        feat['exhour_sloc_count'] = feat['exhour_sloc_count'] - (feat['orderid'].isin(data['orderid'].values))
        order_eloc_dict = dict(zip(data['orderid'].values, data['geohashed_end_loc'].values))
        feat['eloc_no_nan'] = feat['eloc_no_nan'] - (~feat['orderid'].map(order_eloc_dict).isnull())
        feat['mean_lat'] = feat['sum_lat'] / (feat['eloc_no_nan'] + 0.001)
        feat['mean_lon'] = feat['sum_lon'] / (feat['eloc_no_nan'] + 0.001)
        feat['lat'] = feat['geohashed_end_loc'].apply(lambda x: np.nan if x is np.nan else loc_dict[x][0])
        feat['lon'] = feat['geohashed_end_loc'].apply(lambda x: np.nan if x is np.nan else loc_dict[x][1])
        feat['exhour_sloc_center_dis'] = ((feat['lat'] - feat['mean_lat']) ** 2 + (
            np.cos(feat['mean_lon'] / 57.2958) * (feat['lon'] - feat['mean_lon'])) ** 2) ** 0.5
        feat = feat[['exhour_sloc_count','exhour_sloc_center_dis']].fillna(100000)
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat


# 时间×目的地
def get_exhour_eloc(data, sample, data_key):
    feat_path = cache_path + 'exhour_eloc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        exhour_eloc_count = data.groupby(['hour', 'geohashed_end_loc'], as_index=False)[
            'userid'].agg({'exhour_eloc_count': 'count'})
        near_hour = get_near_hour()
        exhour_eloc_count = exhour_eloc_count.merge(near_hour, on='hour', how='left')
        exhour_eloc_count['hour'] = exhour_eloc_count['near_hour']
        exhour_eloc_count = exhour_eloc_count.groupby(['hour', 'geohashed_end_loc'], as_index=False).sum()
        feat = pd.merge(sample, exhour_eloc_count, on=['hour', 'geohashed_end_loc'], how='left').fillna(0)
        order_eloc_dict = dict(zip(data['orderid'].values, data['geohashed_end_loc'].values))
        feat['exhour_eloc_count'] = feat['exhour_eloc_count'] - (feat['geohashed_end_loc'] == feat['orderid'].map(order_eloc_dict))
        feat = feat[['exhour_eloc_count']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat


# 时间×出发地×目的地
def get_exhour_sloc_eloc(data, sample, data_key):
    feat_path = cache_path + 'exhour_sloc_eloc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        exhour_sloc_eloc_count = data.groupby(['hour', 'geohashed_start_loc', 'geohashed_end_loc'], as_index=False)[
            'userid'].agg({'exhour_sloc_eloc_count': 'count'})
        near_hour = get_near_hour()
        exhour_sloc_eloc_count = exhour_sloc_eloc_count.merge(near_hour, on='hour', how='left')
        exhour_sloc_eloc_count['hour'] = exhour_sloc_eloc_count['near_hour']
        exhour_sloc_eloc_count = exhour_sloc_eloc_count.groupby(['hour', 'geohashed_start_loc', 'geohashed_end_loc'], as_index=False).sum()
        exhour_eloc_sloc_count = exhour_sloc_eloc_count.rename(columns={'geohashed_start_loc': 'geohashed_end_loc',
                                                                        'geohashed_end_loc': 'geohashed_start_loc',
                                                                        'exhour_sloc_eloc_count': 'exhour_eloc_sloc_count'})
        feat = pd.merge(sample, exhour_sloc_eloc_count, on=['hour', 'geohashed_start_loc', 'geohashed_end_loc'],
                        how='left').fillna(0)
        feat = pd.merge(feat, exhour_eloc_sloc_count, on=['hour', 'geohashed_start_loc', 'geohashed_end_loc'],
                        how='left').fillna(0)
        feat['exhour_sloc_eloc_goback_rate'] = feat['exhour_sloc_eloc_count'] / (feat['exhour_eloc_sloc_count'] + 0.001)
        feat = feat[['exhour_sloc_eloc_count','exhour_eloc_sloc_count','exhour_sloc_eloc_goback_rate']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat

# 时间×用户×目的地
def get_exhour_user_eloc(data, sample, data_key):
    feat_path = cache_path + 'exhour_user_eloc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        exhour_user_eloc_count = data.groupby(['hour','userid', 'geohashed_end_loc'], as_index=False)['geohashed_start_loc'].agg(
            {'exhour_user_eloc_count': 'count'})
        near_hour = get_near_hour()
        exhour_user_eloc_count = exhour_user_eloc_count.merge(near_hour, on='hour', how='left')
        exhour_user_eloc_count['hour'] = exhour_user_eloc_count['near_hour']
        exhour_user_eloc_count = exhour_user_eloc_count.groupby(['hour','userid', 'geohashed_end_loc'], as_index=False).sum()
        feat = pd.merge(sample, exhour_user_eloc_count, on=['hour', 'userid', 'geohashed_end_loc'], how='left').fillna(0)
        order_eloc_dict = dict(zip(data['orderid'].values, data['geohashed_end_loc'].values))
        feat['exhour_user_eloc_count'] = feat['exhour_user_eloc_count'] - (feat['geohashed_end_loc'] == feat['orderid'].map(order_eloc_dict))
        feat = feat[['exhour_user_eloc_count']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat

###############扩大统计范围##################
# 获取出发地热度（作为出发地的次数、人数）
def get_ex_sloc(data, sample, data_key):
    feat_path = cache_path + 'ex_sloc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        data_temp = data[~data['geohashed_end_loc'].isnull()]
        start_loc = data_temp.groupby('geohashed_start_loc', as_index=False)['userid'].agg({'ex_sloc_count': 'count',
                                                                                            'ex_sloc_n_user': 'nunique'})
        near_loc = get_near_loc()
        start_loc = start_loc.merge(near_loc, left_on='geohashed_start_loc', right_on='loc', how='left')
        start_loc['geohashed_start_loc'] = start_loc['near_loc']
        start_loc = start_loc.groupby('geohashed_start_loc', as_index=False).sum()
        feat = pd.merge(sample, start_loc, on='geohashed_start_loc', how='left').fillna(0)
        feat = feat[['ex_sloc_count','ex_sloc_n_user']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat


# 获取目标地点的热度(目的地)
def get_ex_eloc(data, sample, data_key):
    feat_path = cache_path + 'ex_eloc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        eloc_as_sloc = data.groupby('geohashed_start_loc', as_index=False)['geohashed_end_loc'].agg(
            {'ex_eloc_as_sloc_count': 'count',
             'ex_eloc_as_sloc_count2': 'size'})
        eloc = data.groupby('geohashed_end_loc', as_index=False)['geohashed_end_loc'].agg(
            {'ex_eloc_count': 'count'})
        eloc_as_sloc.rename(columns={'geohashed_start_loc': 'geohashed_end_loc'}, inplace=True)
        feat = pd.merge(eloc, eloc_as_sloc, on='geohashed_end_loc', how='outer').fillna(0)
        near_loc = get_near_loc()
        feat = feat.merge(near_loc, left_on='geohashed_end_loc', right_on='loc', how='left').fillna(0)
        feat = feat.groupby('near_loc', as_index=False).sum()
        feat = pd.merge(sample, feat, left_on='geohashed_end_loc', right_on='near_loc', how='left').fillna(0)
        feat['ex_eloc_2count'] = feat['ex_eloc_count'] + feat['ex_eloc_as_sloc_count']
        feat['ex_eloc_in_out_rate'] = feat['ex_eloc_count'] / (feat['ex_eloc_as_sloc_count'] + 0.001)
        feat = feat[['ex_eloc_as_sloc_count', 'ex_eloc_count','ex_eloc_as_sloc_count2',
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
            {'ex_sloc_eloc_count': 'count',
             'ex_sloc_eloc_n_user': 'nunique'})
        eloc_sloc = sloc_eloc.rename(columns={'geohashed_start_loc': 'geohashed_end_loc',
                                              'geohashed_end_loc': 'geohashed_start_loc',
                                              'ex_sloc_eloc_count': 'ex_eloc_sloc_count',
                                              'ex_sloc_eloc_n_user':'ex_eloc_sloc_n_user'})
        feat = pd.merge(sloc_eloc, eloc_sloc, on=['geohashed_start_loc', 'geohashed_end_loc'], how='outer').fillna(0)
        near_loc = get_near_loc()
        feat = feat.merge(near_loc, left_on='geohashed_start_loc', right_on='loc', how='left').fillna(0)
        feat['geohashed_start_loc'] = feat['near_loc']
        feat = feat.groupby(['geohashed_start_loc', 'geohashed_end_loc'], as_index=False).sum()
        feat = feat[feat['geohashed_start_loc'].isin(sample['geohashed_start_loc'].values)]
        feat = feat.merge(near_loc, left_on='geohashed_end_loc', right_on='loc', how='left').fillna(0)
        feat['geohashed_end_loc'] = feat['near_loc']
        feat = feat.groupby(['geohashed_start_loc', 'geohashed_end_loc'], as_index=False).sum()
        feat = pd.merge(sample, feat, on=['geohashed_end_loc', 'geohashed_start_loc'], how='left').fillna(0)
        feat['ex_sloc_eloc_2count'] = feat['ex_sloc_eloc_count'] + feat['ex_eloc_sloc_count']
        feat['ex_sloc_eloc_2n_user'] = feat['ex_sloc_eloc_n_user'] + feat['ex_eloc_sloc_n_user']
        feat['ex_sloc_eloc_goback_rate'] = feat['ex_sloc_eloc_count'] / (feat['ex_eloc_sloc_count'] + 0.001)
        feat = feat[['ex_sloc_eloc_count', 'ex_eloc_sloc_count','ex_sloc_eloc_2count',
                     'ex_sloc_eloc_n_user','ex_eloc_sloc_n_user','ex_sloc_eloc_2n_user',
                     'ex_sloc_eloc_goback_rate']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat

# 扩大出发地×扩大目的地×扩大时间
def get_ex_exhour_sloc_eloc(data, sample, data_key):
    feat_path = cache_path + 'ex_exhour_sloc_eloc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        ex_exhour_sloc_eloc_count = data.groupby(['hour', 'geohashed_start_loc', 'geohashed_end_loc'], as_index=False)[
            'userid'].agg({'ex_exhour_sloc_eloc_count': 'count'})
        near_loc = get_near_loc()
        ex_exhour_sloc_eloc_count = ex_exhour_sloc_eloc_count.merge(near_loc, left_on='geohashed_start_loc', right_on='loc', how='left').fillna(0)
        ex_exhour_sloc_eloc_count['geohashed_start_loc'] = ex_exhour_sloc_eloc_count['near_loc']
        ex_exhour_sloc_eloc_count = ex_exhour_sloc_eloc_count.groupby(['hour', 'geohashed_start_loc', 'geohashed_end_loc'], as_index=False).sum()
        near_hour = get_near_hour()
        ex_exhour_sloc_eloc_count = ex_exhour_sloc_eloc_count.merge(near_hour, on='hour', how='left')
        ex_exhour_sloc_eloc_count['hour'] = ex_exhour_sloc_eloc_count['near_hour']
        ex_exhour_sloc_eloc_count = ex_exhour_sloc_eloc_count.groupby(['hour', 'geohashed_start_loc', 'geohashed_end_loc'],as_index=False).sum()
        ex_exhour_eloc_sloc_count = ex_exhour_sloc_eloc_count.rename(columns={'geohashed_start_loc': 'geohashed_end_loc',
                                                                              'geohashed_end_loc': 'geohashed_start_loc',
                                                                              'ex_exhour_sloc_eloc_count': 'ex_exhour_eloc_sloc_count'})
        feat = pd.merge(sample, ex_exhour_sloc_eloc_count, on=['hour', 'geohashed_start_loc', 'geohashed_end_loc'],
                        how='left').fillna(0)
        feat = pd.merge(feat, ex_exhour_eloc_sloc_count, on=['hour', 'geohashed_start_loc', 'geohashed_end_loc'],
                        how='left').fillna(0)
        feat['exhour_ex_sloc_eloc_goback_rate'] = feat['ex_exhour_sloc_eloc_count'] / (feat['ex_exhour_eloc_sloc_count'] + 0.001)
        feat = feat[['ex_exhour_sloc_eloc_count', 'ex_exhour_eloc_sloc_count', 'exhour_ex_sloc_eloc_goback_rate']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat

# 用户×扩大目的地×扩大时间
def get_ex_exhour_user_eloc(data, sample, data_key):
    feat_path = cache_path + 'ex_exhour_user_eloc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        ex_exhour_user_eloc_count = data.groupby(['hour', 'userid', 'geohashed_end_loc'], as_index=False)[
            'userid'].agg({'ex_exhour_user_eloc_count': 'count'})
        ex_exhour_user_eloc_as_sloc_count = data.groupby(['hour', 'userid', 'geohashed_start_loc'], as_index=False)[
            'userid'].agg({'ex_exhour_user_eloc_as_sloc_count': 'count'})
        ex_exhour_user_eloc_as_sloc_count.rename(columns={'geohashed_start_loc':'geohashed_end_loc'},inplace=True)
        feat = ex_exhour_user_eloc_count.merge(ex_exhour_user_eloc_as_sloc_count,on=['hour', 'userid', 'geohashed_end_loc'],how='outer').fillna(0)
        feat = pd.merge(sample[['userid']].drop_duplicates(),feat,on=['userid'],how='inner')
        near_loc = get_near_loc()
        feat = feat.merge(near_loc, left_on='geohashed_end_loc', right_on='loc', how='left').fillna(0)
        feat['geohashed_end_loc'] = feat['near_loc']
        feat = feat.groupby(['hour', 'userid', 'geohashed_end_loc'], as_index=False).sum()
        feat = pd.merge(sample[['userid','geohashed_end_loc']].drop_duplicates(), feat, on=['userid','geohashed_end_loc'], how='inner')
        near_hour = get_near_hour()
        feat = feat.merge(near_hour, on='hour', how='left')
        feat['hour'] = feat['near_hour']
        feat = feat.groupby(['hour', 'userid', 'geohashed_end_loc'],as_index=False).sum()
        feat = pd.merge(sample, feat, on=['hour', 'userid', 'geohashed_end_loc'], how='left').fillna(0)
        feat['ex_exhour_user_eloc_goback_rate'] = feat['ex_exhour_user_eloc_count'] / (feat['ex_exhour_user_eloc_as_sloc_count'] + 0.001)
        feat = feat[['ex_exhour_user_eloc_count', 'ex_exhour_user_eloc_as_sloc_count', 'ex_exhour_user_eloc_goback_rate']]
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
        feat['ex_user_sloc_eloc_2count'] = feat['ex_user_sloc_eloc_count'] + feat['ex_user_eloc_sloc_count']
        feat = feat[['ex_user_sloc_eloc_count', 'ex_user_eloc_sloc_count', 'ex_user_sloc_eloc_2count']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat


#####################节假日#####################

# 节假日×时间
def get_holiday(data, sample, data_key):
    feat_path = cache_path + 'holiday_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        hour_count = data.groupby(['hour'], as_index=False)['geohashed_end_loc'].agg({'hour_count': 'count'})
        holiday_count = data.groupby(['holiday'], as_index=False)['geohashed_end_loc'].agg({'holiday_count': 'count'})
        holiday_hour_count = data.groupby(['hour','holiday'],as_index=False)['geohashed_end_loc'].agg({'holiday_hour_count': 'count'})
        feat = sample.merge(hour_count,on=['hour'],how='left').fillna(0)
        feat = feat.merge(holiday_count, on=['holiday'], how='left').fillna(0)
        feat = feat.merge(holiday_hour_count, on=['hour', 'holiday'], how='left').fillna(0)
        feat['hour_count'] = feat['hour_count'] - feat['orderid'].isin(data['orderid'].values)
        feat['holiday_count'] = feat['holiday_count'] - feat['orderid'].isin(data['orderid'].values)
        feat['holiday_hour_count'] = feat['holiday_hour_count'] - feat['orderid'].isin(data['orderid'].values)
        feat['all_count'] = data['geohashed_end_loc'].count()
        feat = feat[['hour_count','holiday_count','holiday_hour_count','all_count']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat

# 节假日×目的地
def get_holiday_eloc(data, sample, data_key):
    feat_path = cache_path + 'holiday_eloc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        data_temp = data[~data['geohashed_end_loc'].isnull()]
        holiday_eloc_count = data_temp.groupby(['holiday', 'geohashed_end_loc'], as_index=False)['userid'].agg(
            {'holiday_eloc_count': 'count',
             'holiday_eloc_n_user': 'nunique'})
        holiday_eloc_as_sloc_count = data_temp.groupby(['holiday', 'geohashed_start_loc'], as_index=False)['userid'].agg(
            {'holiday_eloc_as_sloc_count': 'count',
             'holiday_eloc_as_sloc_n_user': 'nunique'})
        holiday_eloc_as_sloc_count2 = data.groupby(['holiday', 'geohashed_start_loc'], as_index=False)['userid'].agg(
            {'holiday_eloc_as_sloc_count2': 'count',
             'holiday_eloc_as_sloc_n_user2': 'nunique'})
        holiday_eloc_as_sloc_count.rename(columns={'geohashed_start_loc':'geohashed_end_loc'},inplace=True)
        holiday_eloc_as_sloc_count2.rename(columns={'geohashed_start_loc': 'geohashed_end_loc'}, inplace=True)
        feat = sample.merge(holiday_eloc_count, on=['holiday','geohashed_end_loc'],how='left').fillna(0)
        feat = feat.merge(holiday_eloc_as_sloc_count, on=['holiday', 'geohashed_end_loc'], how='left').fillna(0)
        feat = feat.merge(holiday_eloc_as_sloc_count2, on=['holiday', 'geohashed_end_loc'], how='left').fillna(0)
        feat['holiday_eloc_2count'] = feat['holiday_eloc_count'] + feat['holiday_eloc_as_sloc_count']
        feat['holiday_eloc_2n_user'] = feat['holiday_eloc_n_user'] + feat['holiday_eloc_as_sloc_n_user']
        feat['holiday_eloc_in_out_rate'] = feat['holiday_eloc_count'] / (feat['holiday_eloc_as_sloc_count']+0.001)
        feat = feat[['holiday_eloc_count','holiday_eloc_as_sloc_count2','holiday_eloc_in_out_rate',
                     'holiday_eloc_n_user','holiday_eloc_as_sloc_n_user2','holiday_eloc_2count',
                     'holiday_eloc_2n_user']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat


# 节假日×扩大出发地
def get_holiday_ex_sloc(data, sample, data_key):
    feat_path = cache_path + 'holiday_ex_sloc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        data_temp = data[~data['geohashed_end_loc'].isnull()]
        holiday_start_loc = data_temp.groupby(['holiday','geohashed_start_loc'], as_index=False)['userid'].agg({
            'holiday_ex_sloc_count': 'count',
            'holiday_ex_sloc_n_user':'nunique'})
        near_loc = get_near_loc()
        start_loc = holiday_start_loc.merge(near_loc, left_on='geohashed_start_loc', right_on='loc', how='left')
        start_loc['geohashed_start_loc'] = start_loc['near_loc']
        start_loc = start_loc.groupby(['holiday','geohashed_start_loc'], as_index=False).sum()
        feat = pd.merge(sample, start_loc, on=['holiday','geohashed_start_loc'], how='left').fillna(0)
        feat = feat[['holiday_ex_sloc_count','holiday_ex_sloc_n_user']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat

# 节假日×出发地×目的地
def get_holiday_sloc_eloc(data, sample, data_key):
    feat_path = cache_path + 'holiday_sloc_eloc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        holiday_sloc_eloc = data.groupby(['holiday', 'geohashed_start_loc', 'geohashed_end_loc'], as_index=False)['userid'].agg(
            {'holiday_sloc_eloc_count': 'count',
             'holiday_sloc_eloc_n_user': 'nunique'})
        holiday_eloc_sloc = holiday_sloc_eloc.rename(columns={'geohashed_start_loc': 'geohashed_end_loc',
                                              'geohashed_end_loc': 'geohashed_start_loc',
                                              'holiday_sloc_eloc_count': 'holiday_eloc_sloc_count',
                                              'holiday_sloc_eloc_n_user': 'holiday_eloc_sloc_n_user'})
        feat = pd.merge(sample, holiday_sloc_eloc, on=['holiday', 'geohashed_start_loc', 'geohashed_end_loc'], how='left')
        feat = pd.merge(feat, holiday_eloc_sloc, on=['holiday', 'geohashed_start_loc', 'geohashed_end_loc'], how='left').fillna(0)
        feat['holiday_sloc_eloc_2count'] = feat['holiday_sloc_eloc_count'] + feat['holiday_eloc_sloc_count']
        feat['holiday_sloc_eloc_2n_user'] = feat['holiday_sloc_eloc_n_user'] + feat['holiday_eloc_sloc_n_user']
        feat['holiday_sloc_eloc_goback_rate'] = feat['holiday_sloc_eloc_count'] / (feat['holiday_eloc_sloc_count'] + 0.001)
        feat['holiday_sloc_eloc_n_user_goback_rate'] = feat['holiday_sloc_eloc_n_user'] / (feat['holiday_eloc_sloc_n_user'] + 0.001)
        feat = feat[['holiday_sloc_eloc_count', 'holiday_eloc_sloc_count', 'holiday_sloc_eloc_2count',
                     'holiday_sloc_eloc_n_user', 'holiday_eloc_sloc_n_user','holiday_sloc_eloc_2n_user',
                     'holiday_sloc_eloc_goback_rate', 'holiday_sloc_eloc_n_user_goback_rate']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat

# 节假日×扩大出发地×目的地
def get_holiday_ex_sloc_eloc(data, sample, data_key):
    feat_path = cache_path + 'holiday_ex_sloc_eloc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        holiday_ex_sloc_eloc = data.groupby(['holiday', 'geohashed_start_loc', 'geohashed_end_loc'], as_index=False)['userid'].agg(
            {'holiday_ex_sloc_eloc_count': 'count',
             'holiday_ex_sloc_eloc_n_user': 'nunique'})
        holiday_ex_eloc_sloc = holiday_ex_sloc_eloc.rename(columns={'geohashed_start_loc': 'geohashed_end_loc',
                                              'geohashed_end_loc': 'geohashed_start_loc',
                                              'holiday_ex_sloc_eloc_count': 'holiday_ex_eloc_sloc_count',
                                              'holiday_ex_sloc_eloc_n_user': 'holiday_ex_eloc_sloc_n_user'})
        feat = pd.merge(holiday_ex_eloc_sloc, holiday_ex_sloc_eloc, on=['holiday','geohashed_start_loc', 'geohashed_end_loc'], how='outer').fillna(0)
        near_loc = get_near_loc()
        feat = feat.merge(near_loc,left_on='geohashed_start_loc',right_on='loc',how='left')
        feat['geohashed_start_loc'] = feat['near_loc']
        feat = feat.groupby(['holiday', 'geohashed_start_loc', 'geohashed_end_loc'],as_index=False).sum()
        feat = pd.merge(sample[['geohashed_start_loc']].drop_duplicates(), feat, on='geohashed_start_loc', how='inner')
        feat = feat.merge(near_loc, left_on='geohashed_end_loc', right_on='loc', how='left')
        feat['geohashed_end_loc'] = feat['near_loc']
        feat = feat.groupby(['holiday', 'geohashed_start_loc', 'geohashed_end_loc'], as_index=False).sum()
        feat = pd.merge(sample,feat,on=['holiday', 'geohashed_start_loc', 'geohashed_end_loc'],how='left')
        feat['holiday_ex_sloc_eloc_2count'] = feat['holiday_ex_sloc_eloc_count'] + feat['holiday_ex_eloc_sloc_count']
        feat['holiday_ex_sloc_eloc_2n_user'] = feat['holiday_ex_sloc_eloc_n_user'] + feat['holiday_ex_eloc_sloc_n_user']
        feat['holiday_ex_sloc_eloc_goback_rate'] = feat['holiday_ex_sloc_eloc_count'] / (feat['holiday_ex_eloc_sloc_count'] + 0.001)
        feat['holiday_ex_sloc_eloc_n_user_goback_rate'] = feat['holiday_ex_sloc_eloc_n_user'] / (feat['holiday_ex_eloc_sloc_n_user'] + 0.001)
        feat = feat[['holiday_ex_sloc_eloc_count', 'holiday_ex_eloc_sloc_count', 'holiday_ex_sloc_eloc_2count',
                     'holiday_ex_sloc_eloc_n_user', 'holiday_ex_eloc_sloc_n_user','holiday_ex_sloc_eloc_2n_user',
                     'holiday_ex_sloc_eloc_goback_rate', 'holiday_ex_sloc_eloc_n_user_goback_rate']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat

# 节假日×时间×目的地
def get_holiday_hour_eloc(data, sample, data_key):
    feat_path = cache_path + 'holiday_hour_eloc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        feat = data.groupby(['hour','holiday','geohashed_end_loc'],as_index=False)['userid'].agg({'holiday_hour_eloc_count': 'count'})
        holiday_hour_sloc_count = data.groupby(['hour', 'holiday', 'geohashed_start_loc'],as_index=False)['userid'].agg({'holiday_hour_sloc_count': 'count'})
        holiday_hour_sloc_count.rename(columns={'geohashed_start_loc':'geohashed_end_loc'},inplace=True)
        feat = sample.merge(feat,on=['hour','holiday','geohashed_end_loc'],how='left')
        feat = feat.merge(holiday_hour_sloc_count, on=['hour', 'holiday', 'geohashed_end_loc'], how='left').fillna(0)
        feat['holiday_hour_sloc_in_out_rate'] = feat['holiday_hour_eloc_count'] / (feat['holiday_hour_sloc_count']+0.001)
        feat['holiday_hour_eloc_2count'] = feat['holiday_hour_eloc_count'] + feat['holiday_hour_sloc_count'] + 0.001
        feat = feat[['holiday_hour_eloc_count','holiday_hour_eloc_2count','holiday_hour_sloc_in_out_rate']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat

# 节假日×时间×出发地
def get_holiday_hour_sloc(data, sample, data_key):
    feat_path = cache_path + 'holiday_hour_sloc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        feat = data.groupby(['hour','holiday','geohashed_start_loc'],as_index=False)['userid'].agg({'holiday_hour_sloc_count': 'count'})
        feat = sample.merge(feat,on=['hour','holiday','geohashed_start_loc'],how='left').fillna(0)
        feat['holiday_hour_sloc_count'] = feat['holiday_hour_sloc_count'] - (feat['orderid'].isin(data['orderid'].values))
        feat = feat[['holiday_hour_sloc_count']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat

# 节假日×扩大时间×扩大出发地
def get_holiday_exhour_ex_sloc(data, sample, data_key):
    feat_path = cache_path + 'holiday_exhour_ex_sloc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        feat = data.groupby(['hour','holiday','geohashed_start_loc'],as_index=False)['userid'].agg({'holiday_exhour_ex_sloc_count': 'count'})
        near_hour = get_near_hour()
        feat = feat.merge(near_hour, on='hour', how='left')
        feat['hour'] = feat['near_hour']
        feat = feat.groupby(['hour','holiday','geohashed_start_loc'], as_index=False).sum()
        near_loc = get_near_loc()
        feat = feat.merge(near_loc, left_on='geohashed_start_loc', right_on='loc', how='left')
        feat['geohashed_start_loc'] = feat['near_loc']
        feat = feat.groupby(['hour','holiday','geohashed_start_loc'], as_index=False).sum()
        feat = pd.merge(sample, feat, on=['hour','holiday','geohashed_start_loc'], how='left').fillna(0)
        feat = feat[['holiday_exhour_ex_sloc_count']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat


# 节假日×时间×出发地×目的地
def get_holiday_hour_sloc_eloc(data, sample, data_key):
    feat_path = cache_path + 'holiday_hour_sloc_eloc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        feat = data.groupby(['hour','holiday','geohashed_start_loc','geohashed_end_loc'],as_index=False)['userid'].agg({'holiday_hour_sloc_eloc_count': 'count'})
        holiday_hour_eloc_sloc_count = feat.rename(columns={'geohashed_start_loc':'geohashed_end_loc',
                                                            'geohashed_end_loc':'geohashed_start_loc',
                                                            'holiday_hour_sloc_eloc_count':'holiday_hour_eloc_sloc_count'})
        feat = feat.merge(holiday_hour_eloc_sloc_count,on=['hour','holiday','geohashed_start_loc','geohashed_end_loc'],how='outer')
        feat = sample.merge(feat,on=['hour','holiday','geohashed_start_loc','geohashed_end_loc'],how='left').fillna(0)
        feat['holiday_hour_sloc_eloc_2count'] = feat['holiday_hour_sloc_eloc_count'] + feat['holiday_hour_eloc_sloc_count']
        feat = feat[['holiday_hour_sloc_eloc_count','holiday_hour_sloc_eloc_2count']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat

# 节假日×时间×扩大出发地×目的地
def get_holiday_hour_ex_sloc_eloc(data, sample, data_key):
    feat_path = cache_path + 'holiday_hour_ex_sloc_eloc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        feat = data.groupby(['hour','holiday','geohashed_start_loc','geohashed_end_loc'],as_index=False)['userid'].agg({'holiday_hour_ex_sloc_eloc_count': 'count'})
        holiday_hour_ex_eloc_sloc_count = feat.rename(columns={'geohashed_start_loc': 'geohashed_end_loc',
                                                            'geohashed_end_loc': 'geohashed_start_loc',
                                                            'holiday_hour_ex_sloc_eloc_count': 'holiday_hour_ex_eloc_sloc_count'})
        feat = feat.merge(holiday_hour_ex_eloc_sloc_count,
                          on=['hour', 'holiday', 'geohashed_start_loc', 'geohashed_end_loc'], how='outer')
        near_loc = get_near_loc()
        feat = feat.merge(near_loc, left_on='geohashed_start_loc', right_on='loc', how='left')
        feat['geohashed_start_loc'] = feat['near_loc']
        feat = feat.groupby(['hour', 'holiday', 'geohashed_start_loc','geohashed_end_loc'], as_index=False).sum()
        feat = sample.merge(feat,on=['hour','holiday','geohashed_start_loc','geohashed_end_loc'],how='left').fillna(0)
        feat['holiday_hour_ex_sloc_eloc_2count'] = feat['holiday_hour_ex_sloc_eloc_count'] + feat['holiday_hour_ex_eloc_sloc_count']
        feat = feat[['holiday_hour_ex_sloc_eloc_count','holiday_hour_ex_sloc_eloc_2count']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat

# 节假日×扩大时间×扩大出发地×目的地
def get_holiday_exhour_ex_sloc_eloc(data, sample, data_key):
    feat_path = cache_path + 'holiday_exhour_ex_sloc_eloc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        feat = data.groupby(['hour','holiday','geohashed_start_loc','geohashed_end_loc'],as_index=False)['userid'].agg({'holiday_exhour_ex_sloc_eloc_count': 'count'})
        holiday_exhour_ex_eloc_sloc_count = feat.rename(columns={'geohashed_start_loc': 'geohashed_end_loc',
                                                               'geohashed_end_loc': 'geohashed_start_loc',
                                                               'holiday_exhour_ex_sloc_eloc_count': 'holiday_exhour_ex_eloc_sloc_count'})
        feat = feat.merge(holiday_exhour_ex_eloc_sloc_count,
                          on=['hour', 'holiday', 'geohashed_start_loc', 'geohashed_end_loc'], how='outer')
        near_loc = get_near_loc()
        feat = feat.merge(near_loc, left_on='geohashed_start_loc', right_on='loc', how='left')
        feat['geohashed_start_loc'] = feat['near_loc']
        feat = feat.groupby(['hour', 'holiday', 'geohashed_start_loc','geohashed_end_loc'], as_index=False).sum()
        feat = pd.merge(sample[['geohashed_start_loc']].drop_duplicates(),feat,on='geohashed_start_loc',how='inner')
        near_hour = get_near_hour()
        feat = feat.merge(near_hour, on='hour', how='left')
        feat['hour'] = feat['near_hour']
        feat = feat.groupby(['hour', 'holiday', 'geohashed_start_loc', 'geohashed_end_loc'], as_index=False).sum()
        feat = sample.merge(feat,on=['hour','holiday','geohashed_start_loc','geohashed_end_loc'],how='left').fillna(0)
        feat['holiday_exhour_ex_sloc_eloc_2count'] = feat['holiday_exhour_ex_sloc_eloc_count'] + feat['holiday_exhour_ex_eloc_sloc_count']
        feat['holiday_exhour_ex_sloc_eloc_in_out_rate'] = feat['holiday_exhour_ex_sloc_eloc_count'] / (feat['holiday_exhour_ex_eloc_sloc_count']+0.001)
        feat = feat[['holiday_exhour_ex_sloc_eloc_count','holiday_exhour_ex_sloc_eloc_2count','holiday_exhour_ex_sloc_eloc_in_out_rate']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat

#################工作日×用户###############
# 工作日×用户
def get_holiday_user_count(data, sample, data_key):
    feat_path = cache_path + 'holiday_user_count_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        holiday_user_count = data.groupby(['holiday', 'userid'], as_index=False)['userid'].agg(
            {'holiday_user_count': 'count'})
        feat = pd.merge(sample, holiday_user_count, on=['holiday','userid'], how='left').fillna(0)
        feat = feat[['holiday_user_count']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat

# 工作日×用户×目的地
def get_holiday_ex_user_eloc(data, sample, data_key):
    feat_path = cache_path + 'holiday_ex_user_eloc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        holiday_ex_user_eloc_as_sloc = data.groupby(['holiday', 'userid', 'geohashed_start_loc'], as_index=False)['userid'].agg(
            {'holiday_ex_user_eloc_as_sloc_count': 'count'})
        holiday_ex_user_eloc = data.groupby(['holiday', 'userid', 'geohashed_end_loc'], as_index=False)['userid'].agg(
            {'holiday_ex_user_eloc_count': 'count'})
        holiday_ex_user_eloc_as_sloc.rename(columns={'geohashed_start_loc': 'geohashed_end_loc'}, inplace=True)
        feat = pd.merge(holiday_ex_user_eloc, holiday_ex_user_eloc_as_sloc, on=['holiday',  'userid', 'geohashed_end_loc'], how='outer').fillna(0)
        feat = pd.merge(sample[['userid']].drop_duplicates(),feat,on='userid',how='inner')
        near_loc = get_near_loc()
        feat = feat.merge(near_loc, left_on='geohashed_end_loc', right_on='loc', how='left')
        feat['geohashed_end_loc'] = feat['near_loc']
        feat = feat.groupby(['holiday', 'userid', 'geohashed_end_loc'], as_index=False).sum()
        feat = pd.merge(sample, feat, on=['holiday','userid', 'geohashed_end_loc'], how='left').fillna(0)
        feat['holiday_ex_user_eloc_2count'] = feat['holiday_ex_user_eloc_count'] + feat['holiday_ex_user_eloc_as_sloc_count']
        feat = feat[['holiday_ex_user_eloc_count', 'holiday_ex_user_eloc_as_sloc_count','holiday_ex_user_eloc_2count']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat



# 日期小时目的地
def get_date_hour(data, sample, data_key):
    feat_path = cache_path + 'date_hour_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        data_temp = data.copy()
        sample_temp = sample.copy()
        data_temp.loc[:, 'date'] = data_temp['starttime'].str[:5]
        sample_temp.loc[:, 'date'] = sample_temp['starttime'].str[:5]
        date_hour_count = data_temp.groupby(['date', 'hour'], as_index=False)['userid'].agg({'date_hour_count': 'count'})
        date_count = data_temp.groupby('date', as_index=False)['userid'].agg({'data_count': 'count'})
        date_hour_eloc_as_sloc = data_temp.groupby(['date', 'hour', 'geohashed_start_loc'], as_index=False)['userid'].agg({
            'date_hour_eloc_as_sloc': 'count'})
        date_eloc_as_sloc = data_temp.groupby(['date', 'geohashed_start_loc'], as_index=False)['userid'].agg({
            'date_eloc_as_sloc': 'count'})
        date_user_eloc_as_sloc = data_temp.groupby(['date','userid', 'geohashed_start_loc'], as_index=False)['userid'].agg({
            'date_user_eloc_as_sloc': 'count'})
        date_hour_eloc_as_sloc.rename(columns={'geohashed_start_loc': 'geohashed_end_loc'}, inplace=True)
        date_eloc_as_sloc.rename(columns={'geohashed_start_loc': 'geohashed_end_loc'}, inplace=True)
        date_user_eloc_as_sloc.rename(columns={'geohashed_start_loc': 'geohashed_end_loc'}, inplace=True)
        feat = sample_temp.merge(date_hour_count, on=['date', 'hour'], how='left')
        feat = feat.merge(date_count, on='date', how='left')
        feat = feat.merge(date_eloc_as_sloc, on=['date', 'geohashed_end_loc'], how='left')
        feat = feat.merge(date_hour_eloc_as_sloc, on=['date', 'hour', 'geohashed_end_loc'], how='left').fillna(0)
        feat = feat.merge(date_user_eloc_as_sloc, on=['date', 'userid', 'geohashed_end_loc'], how='left').fillna(0)
        feat['date_hour_eloc_as_sloc_rate'] = feat['date_hour_eloc_as_sloc']/(feat['date_hour_count']+0.001)
        feat['date_eloc_as_sloc_rate'] = feat['date_eloc_as_sloc'] / (feat['data_count'] + 0.001)
        feat = feat[['date_hour_eloc_as_sloc_rate','date_eloc_as_sloc_rate','date_user_eloc_as_sloc']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat

# 添加all_pred特征
def get_all_order_pred(sample, data_key):
    all_order_pred_path = cache_path + 'all_order_pred.hdf'
    result_path = cache_path + 'order_pred_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        all_order_pred = pd.read_hdf(all_order_pred_path)
        all_order_pred.rename(columns={'pred': 'all_order_pred'}, inplace=True)
        result = sample.merge(all_order_pred, on=['orderid', 'geohashed_end_loc'], how='left')
        result = result[['all_order_pred','sum_pred']].fillna(-1)
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 添加test_pred特征
def get_test_pred(data, sample, data_key):
    result_path = cache_path + 'test_pred_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        feat = pd.read_hdf(result_path, 'w')
    else:
        test_pred_path = data_path + 'test_pred_{}.csv'.format(sample['starttime'].values[0][8:10])
        test_pred = pd.read_csv(test_pred_path)
        data_temp = data[data['starttime']>='2017-05-25 00:00:00'].copy()
        # data_temp = data_temp[(data_temp['starttime']>sample['starttime'].max()) |
        #                       (data_temp['starttime']<sample['starttime'].min())].copy()
        del data_temp['geohashed_end_loc']
        test_pred = test_pred[test_pred['pred']>0.1]
        test_pred.sort_values('pred',inplace=True)
        test_pred = test_pred.groupby('orderid').tail(1)
        data_temp = data_temp.merge(test_pred,on='orderid',how='inner')
        data_temp.rename(columns={'pred':'weight'},inplace=True)
        pred_user_eloc = data_temp.groupby(['userid', 'geohashed_end_loc'], as_index=False)['weight'].agg({'pred_user_eloc':'sum'})
        pred_sloc_eloc = data_temp.groupby(['geohashed_start_loc', 'geohashed_end_loc'], as_index=False)['weight'].agg({'pred_sloc_eloc': 'sum'})
        pred_eloc = data_temp.groupby(['geohashed_end_loc'], as_index=False)['weight'].agg({'pred_eloc': 'sum'})
        order_weight_dict = dict(zip(data_temp['orderid'].values, data_temp['weight'].values))
        feat = sample.merge(pred_user_eloc, on=['userid', 'geohashed_end_loc'], how='left')
        # feat['pred_user_eloc'] = feat['pred_user_eloc'] - (feat['orderid'].map(order_weight_dict).fillna(0))
        feat = feat.merge(pred_sloc_eloc, on=['geohashed_start_loc', 'geohashed_end_loc'], how='left')
        # feat['pred_sloc_eloc'] = feat['pred_sloc_eloc'] - (feat['orderid'].map(order_weight_dict).fillna(0))
        feat = feat.merge(pred_eloc, on=['geohashed_end_loc'], how='left').fillna(0)
        # feat['pred_eloc'] = feat['pred_eloc'] - (feat['orderid'].map(order_weight_dict).fillna(0))
        feat = feat[['pred_user_eloc', 'pred_sloc_eloc', 'pred_eloc']]
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat


# 添加用户自行车特征
def get_user_bike(data, sample, data_key):
    result_path = cache_path + 'user_bike_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        feat = pd.read_hdf(result_path, 'w')
    else:
        bike_next_loc = get_bike_next_loc(data, data, 'all')
        order_user_dict = dict(zip(data['orderid'].values, data['userid'].values))
        bike_next_loc['userid'] = bike_next_loc['orderid'].map(order_user_dict)
        user_nloc_count = bike_next_loc.groupby(['userid','geohashed_end_loc'],as_index=False)['userid'].agg({'user_nloc_count':'count'})
        order_nloc = dict(zip(data['orderid'].values, data['geohashed_end_loc'].values))
        feat = sample.merge(user_nloc_count,on=['userid','geohashed_end_loc'],how='left').fillna(0)
        feat['user_nloc_count'] = feat['user_nloc_count'] - (feat['geohashed_end_loc']==feat['orderid'].map(order_nloc))
        feat = feat[['user_nloc_count']]
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat


# 聚类特征
def get_loc_cluster(data, sample, data_key):
    result_path = cache_path + 'loc_cluster_{}.hdf'.format(data_key)
    if os.path.exists(result_path) & flag:
        feat = pd.read_hdf(result_path, 'w')
    else:
        sloc1 = pd.read_csv(r'C:\Users\csw\Documents\WeChat Files\cuishiwen00\Files\bow_cluster_500_start.csv')
        sloc2 = pd.read_csv(r'C:\Users\csw\Documents\WeChat Files\cuishiwen00\Files\bow_cluster_1000_start.csv')
        sloc3 = pd.read_csv(r'C:\Users\csw\Documents\WeChat Files\cuishiwen00\Files\bow_cluster_1750_start.csv')
        sloc4 = pd.read_csv(r'C:\Users\csw\Documents\WeChat Files\cuishiwen00\Files\bow_cluster_2500_start.csv')
        eloc1 = pd.read_csv(r'C:\Users\csw\Documents\WeChat Files\cuishiwen00\Files\bow_cluster_500.csv')
        eloc2 = pd.read_csv(r'C:\Users\csw\Documents\WeChat Files\cuishiwen00\Files\bow_cluster_1000.csv')
        eloc3 = pd.read_csv(r'C:\Users\csw\Documents\WeChat Files\cuishiwen00\Files\bow_cluster_1750.csv')
        eloc4 = pd.read_csv(r'C:\Users\csw\Documents\WeChat Files\cuishiwen00\Files\bow_cluster_2500.csv')
        eloc1.columns = ['geohashed_end_loc', 'eloc_cluster1']
        eloc2.columns = ['geohashed_end_loc', 'eloc_cluster2']
        eloc3.columns = ['geohashed_end_loc', 'eloc_cluster3']
        eloc4.columns = ['geohashed_end_loc', 'eloc_cluster4']
        sloc1.columns = ['geohashed_start_loc', 'sloc_cluster1']
        sloc2.columns = ['geohashed_start_loc', 'sloc_cluster2']
        sloc3.columns = ['geohashed_start_loc', 'sloc_cluster3']
        sloc4.columns = ['geohashed_start_loc', 'sloc_cluster4']
        data_temp = data.copy()
        data_temp = data_temp.merge(eloc1, on='geohashed_end_loc', how='left')
        data_temp = data_temp.merge(eloc2, on='geohashed_end_loc', how='left')
        data_temp = data_temp.merge(eloc3, on='geohashed_end_loc', how='left')
        data_temp = data_temp.merge(eloc4, on='geohashed_end_loc', how='left')
        data_temp = data_temp.merge(sloc1, on='geohashed_start_loc', how='left')
        data_temp = data_temp.merge(sloc2, on='geohashed_start_loc', how='left')
        data_temp = data_temp.merge(sloc3, on='geohashed_start_loc', how='left')
        data_temp = data_temp.merge(sloc4, on='geohashed_start_loc', how='left')
        sample_temp = sample.copy()
        sample_temp = sample_temp.merge(eloc1, on='geohashed_end_loc', how='left')
        sample_temp = sample_temp.merge(eloc2, on='geohashed_end_loc', how='left')
        sample_temp = sample_temp.merge(eloc3, on='geohashed_end_loc', how='left')
        sample_temp = sample_temp.merge(eloc4, on='geohashed_end_loc', how='left')
        sample_temp = sample_temp.merge(sloc1, on='geohashed_start_loc', how='left')
        sample_temp = sample_temp.merge(sloc2, on='geohashed_start_loc', how='left')
        sample_temp = sample_temp.merge(sloc3, on='geohashed_start_loc', how='left')
        sample_temp = sample_temp.merge(sloc4, on='geohashed_start_loc', how='left')

        sloc_eloc_cluster1 = data_temp.groupby(['geohashed_start_loc', 'eloc_cluster1'], as_index=False)[
            'geohashed_end_loc'].agg({'sloc_eloc_cluster1': 'count'})
        sloc_eloc_cluster2 = data_temp.groupby(['geohashed_start_loc', 'eloc_cluster2'], as_index=False)[
            'geohashed_end_loc'].agg({'sloc_eloc_cluster2': 'count'})
        sloc_eloc_cluster3 = data_temp.groupby(['geohashed_start_loc', 'eloc_cluster3'], as_index=False)[
            'geohashed_end_loc'].agg({'sloc_eloc_cluster3': 'count'})
        sloc_eloc_cluster4 = data_temp.groupby(['geohashed_start_loc', 'eloc_cluster4'], as_index=False)[
            'geohashed_end_loc'].agg({'sloc_eloc_cluster4': 'count'})

        feat = sample_temp.merge(sloc_eloc_cluster1, on=['geohashed_start_loc', 'eloc_cluster1'],how='left')
        feat = feat.merge(sloc_eloc_cluster2, on=['geohashed_start_loc', 'eloc_cluster2'], how='left')
        feat = feat.merge(sloc_eloc_cluster3, on=['geohashed_start_loc', 'eloc_cluster3'], how='left')
        feat = feat.merge(sloc_eloc_cluster4, on=['geohashed_start_loc', 'eloc_cluster4'], how='left')
        feat = feat[['sloc_eloc_cluster1','sloc_eloc_cluster2','sloc_eloc_cluster3','sloc_eloc_cluster4']]
        feat.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return feat

# 二次处理特征
def second_feat(result):

    result['user_speed'] = (result['mht_distance']+result['user_eloc_nloc_mdistance']) / result['user_eloc_nloc_sep_time']
    result['bike_speed'] = (result['mht_distance'] + result['bike_eloc_nloc_mdis']) / result['bike_eloc_sep_time']
    result['eloc_n_user_sloc_rate'] = result['eloc_n_user'] / (result['sloc_n_user'] + 0.001)
    result['eloc_as_sloc_rate'] = result['eloc_as_sloc_count'] / (result['sloc_count2'] + 0.001)
    result['eloc_as_sloc_rate'] = result['eloc_as_sloc_count'] / (result['sloc_count2'] + 0.001)
    result['eloc_ex_rate'] = result['eloc_count'] / (result['ex_eloc_count'] + 0.001)
    result['eloc_sloc_rate'] = result['eloc_count'] / (result['sloc_count'] + 0.001)
    result['ex_sloc_eloc_ex_sloc_rate'] = result['ex_sloc_eloc_count'] / (result['ex_sloc_count'] + 0.001)
    result['ex_sloc_eloc2_ex_sloc_rate'] = result['ex_sloc_eloc_2count'] / (result['ex_sloc_count'] + 0.001)
    result['ex_sloc_eloc_ex_loc_rate'] = result['ex_sloc_eloc_n_user'] / (result['ex_sloc_n_user'] + 0.001)
    result['ex_sloc_eloc2_ex_loc_rate'] = result['ex_sloc_eloc_2n_user'] / (result['ex_sloc_n_user'] + 0.001)
    result['ex_eloc_ex_sloc_rate'] = result['ex_eloc_count'] / (result['ex_sloc_count'] + 0.001)
    result['ex_eloc_as_sloc_ex_sloc_rate'] = result['ex_eloc_as_sloc_count'] / (result['ex_sloc_count'] + 0.001)
    result['ex_eloc_as_sloc2_ex_sloc_rate'] = result['ex_eloc_as_sloc_count2'] / (result['ex_sloc_count'] + 0.001)
    result['ex_eloc2_ex_sloc_rate'] = result['ex_eloc_2count'] / (result['ex_sloc_count'] + 0.001)
    result['eloc_seloc_9_7_rate'] = result['eloc_count'] / (result['seloc_count_9_7'] + 0.001)
    result['eloc_eeloc_9_7_rate'] = result['eloc_count'] / (result['eeloc_count_9_7'] + 0.001)
    result['user_eloc_user_rate'] = result['user_eloc_count'] / (result['user_count'] + 0.001)
    result['user_eloc_as_sloc_user_rate'] = result['user_eloc_as_sloc_count2'] / (result['user_count2'] + 0.001)
    result['user_eloc_2count_user_rate'] = result['user_eloc_2count'] / (result['user_count2'] + 0.001)
    result['user_sloc_eloc_user_rate'] = result['user_sloc_eloc_count'] / (result['user_count'] + 0.001)
    result['user_sloc_eloc_2count_user_rate'] = result['user_sloc_eloc_2count'] / (result['user_count2'] + 0.001)
    result['user_sloc_eloc_user_sloc_rate'] = result['user_sloc_eloc_count'] / (result['user_sloc_count'] + 0.001)
    result['user_sloc_eloc_user_eloc_rate'] = result['user_sloc_eloc_count'] / (result['user_eloc_count'] + 0.001)
    result['sloc_eloc_sloc_rate'] = result['sloc_eloc_count'] / (result['sloc_count'] + 0.001)
    result['sloc_eloc_n_user_sloc_rate'] = result['sloc_eloc_n_user'] / (result['sloc_n_user'] + 0.001)
    result['hour_sloc_eloc_hs_rate'] = result['hour_sloc_eloc_count'] / (result['hour_sloc_count'] + 0.001)
    result['hour_eloc_eloc_rate'] = result['hour_eloc_count'] / (result['eloc_count'] + 0.001)
    result['hour_eloc_hour_rate'] = result['hour_eloc_count'] / (result['hour_count'] + 0.001)
    result['hour_all_rate'] = result['hour_count'] / (result['all_count'] + 0.001)
    result['hour_user_eloc_hour_user_rate'] = result['hour_user_eloc_count'] / (result['hour_user_count'] + 0.001)
    result['exhour_eloc_eloc_rate'] = result['exhour_eloc_count'] / (result['eloc_count'] + 0.001)
    result['ex_eloc_sloc_ex_eloc_rate'] = result['ex_eloc_sloc_count'] / (result['ex_sloc_count'] + 0.001)
    result['exhour_sloc_eloc_exhour_sloc_rate'] = result['exhour_sloc_eloc_count'] / (result['exhour_sloc_count'] + 0.001)
    result['exhour_user_eloc_exhour_user_rate'] = result['exhour_user_eloc_count'] / (result['exhour_user_count'] + 0.001)
    result['ex_user_eloc_user_rate'] = result['ex_user_eloc_count'] / (result['user_count'] + 0.001)
    result['ex_user_eloc2_user_rate'] = result['ex_user_eloc_2count'] / (result['user_count2'] + 0.001)
    result['ex_sloc_eloc_ex_sloc_rate'] = result['ex_sloc_eloc_count'] / (result['ex_sloc_count'] + 0.001)
    result['ex_user_sloc_eloc_user_sloc_rate'] = result['ex_user_sloc_eloc_count'] / (result['ex_user_sloc_count'] + 0.001)
    result['holiday_sloc_eloc_holiday_sloc_rate'] = result['holiday_sloc_eloc_count'] / (result['holiday_ex_sloc_count'] + 0.001)
    result['holiday_sloc_eloc2_holiday_sloc_rate'] = result['holiday_sloc_eloc_2count'] / (result['holiday_ex_sloc_count'] + 0.001)
    result['holiday_sloc_eloc_n_user_holiday_sloc_rate'] = result['holiday_sloc_eloc_n_user'] / (result['holiday_ex_sloc_n_user'] + 0.001)
    result['holiday_sloc_eloc_2n_user_holiday_sloc_rate'] = result['holiday_sloc_eloc_2n_user'] / (result['holiday_ex_sloc_n_user'] + 0.001)
    result['holiday_ex_sloc_eloc_holiday_sloc_rate'] = result['holiday_ex_sloc_eloc_count'] / (result['holiday_ex_sloc_count'] + 0.001)
    result['holiday_ex_sloc_eloc2_holiday_sloc_rate'] = result['holiday_ex_sloc_eloc_2count'] / (result['holiday_ex_sloc_count'] + 0.001)
    result['holiday_ex_eloc_sloc_holiday_sloc_rate'] = result['holiday_ex_eloc_sloc_count'] / (result['holiday_ex_sloc_count'] + 0.001)

    result['holiday_ex_sloc_eloc_n_user_holiday_sloc_rate'] = result['holiday_ex_sloc_eloc_n_user'] / ( result['holiday_ex_sloc_n_user'] + 0.001)
    result['holiday_ex_eloc_sloc_n_user_holiday_sloc_rate'] = result['holiday_ex_eloc_sloc_n_user'] / ( result['holiday_ex_sloc_n_user'] + 0.001)
    result['holiday_ex_sloc_eloc_2n_user_holiday_sloc_rate'] = result['holiday_ex_sloc_eloc_2n_user'] / (result['holiday_ex_sloc_n_user'] + 0.001)

    result['holiday_hour_eloc_holiday_hour_rate'] = result['holiday_hour_eloc_count'] / (result['holiday_hour_count'] + 0.001)
    result['holiday_hour_eloc2_holiday_hour_rate'] = result['holiday_hour_eloc_2count'] / (result['holiday_hour_count'] + 0.001)
    result['holiday_hour_holiday_rate'] = result['holiday_hour_count'] / (result['holiday_count'] + 0.001)
    result['holiday_hour_rate_rate'] = result['holiday_hour_holiday_rate'] / (result['hour_all_rate'] + 0.000001)
    result['holiday_hour_eloc_sloc_holiday_hour_rate'] = result['holiday_hour_sloc_eloc_count'] / (
    result['holiday_exhour_ex_sloc_count'] + 0.001)
    result['holiday_hour_eloc_sloc2_holiday_hour_rate'] = result['holiday_hour_sloc_eloc_2count'] / (
        result['holiday_exhour_ex_sloc_count'] + 0.001)
    result['holiday_hour_ex_eloc_sloc_holiday_hour_rate'] = result['holiday_hour_ex_sloc_eloc_count'] / (
        result['holiday_exhour_ex_sloc_count'] + 0.001)
    result['holiday_hour_ex_eloc_sloc2_holiday_hour_rate'] = result['holiday_hour_ex_sloc_eloc_2count'] / (
    result['holiday_exhour_ex_sloc_count'] + 0.001)
    result['holiday_exhour_ex_eloc_sloc_holiday_exhour_rate'] = result['holiday_exhour_ex_sloc_eloc_count'] / (
        result['holiday_exhour_ex_sloc_count'] + 0.001)
    result['holiday_exhour_ex_eloc_sloc2_holiday_exhour_rate'] = result['holiday_exhour_ex_sloc_eloc_2count'] / (
        result['holiday_exhour_ex_sloc_count'] + 0.001)
    result['holiday_eloc_holiday_rate'] = result['holiday_eloc_count'] / (result['holiday_count'] + 0.001)
    result['holiday_eloc_holiday_sloc_rate'] = result['holiday_eloc_count'] / (result['holiday_ex_sloc_count'] + 0.001)
    result['holiday_eloc2_holiday_rate'] = result['holiday_eloc_2count'] / (result['holiday_count'] + 0.001)
    result['holiday_eloc_as_sloc_n_user2_rate'] = result['holiday_eloc_as_sloc_n_user2'] / (result['holiday_ex_sloc_n_user'] + 0.001)
    result['holiday_eloc_n_user_rate'] = result['holiday_eloc_n_user'] / (result['holiday_ex_sloc_n_user'] + 0.001)
    result['holiday_eloc_as_sloc_n_user2_rate'] = result['holiday_eloc_as_sloc_n_user2'] / (result['holiday_ex_sloc_n_user'] + 0.001)
    result['holiday_eloc_2n_user_rate'] = result['holiday_eloc_2n_user'] / (result['holiday_ex_sloc_n_user'] + 0.001)

    result['holiday_eloc_as_sloc_holiday_sloc_rate'] = result['holiday_eloc_as_sloc_count2'] / (result['holiday_ex_sloc_count'] + 0.001)

    result['holiday_ex_user_eloc_as_sloc_rate'] = result['holiday_ex_user_eloc_as_sloc_count'] / (
        result['holiday_user_count'] + 0.001)
    result['holiday_ex_user_eloc_rate'] = result['holiday_ex_user_eloc_count'] / (
        result['holiday_user_count'] + 0.001)
    result['holiday_ex_user_eloc2_rate'] = result['holiday_ex_user_eloc_2count'] / (
        result['holiday_user_count'] + 0.001)
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
        # bike_next_loc = get_bike_next_loc(data, candidate, data_key)  # 自行车后续起始地点
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
                            # bike_next_loc[['orderid', 'geohashed_end_loc']]
                            ]).drop_duplicates()
        # 过滤样本
        select_sample = pd.read_csv(select_sample_path)
        result = pd.merge(select_sample[['orderid','geohashed_end_loc']],result,on=['orderid','geohashed_end_loc'],how='inner')
        candidate_temp = candidate[['orderid', 'userid', 'bikeid', 'biketype', 'starttime',
                                    'geohashed_start_loc', 'hour', 'holiday']].copy()
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
    data_key = (sum(~candidate['geohashed_end_loc'].isnull()) + sum(~data['geohashed_end_loc'].isnull())) * data['orderid'].sum() * candidate['orderid'].sum()
    print('数据key为：{}'.format(data_key))
    result_path = cache_path + 'train_set_%d.hdf' % (data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        data.loc[:, 'hour'] = data['starttime'].str[11:13].astype(int)
        candidate.loc[:, 'hour'] = candidate['starttime'].str[11:13].astype(int)
        data.loc[:, 'holiday'] = data['starttime'].apply(lambda x: if_holiday(x))
        candidate.loc[:, 'holiday'] = candidate['starttime'].apply(lambda x: if_holiday(x))
        # 汇总样本id
        print('开始构造样本...')
        sample = get_sample(data, candidate, data_key)
        gc.collect()

        print('开始构造特征...')
        order_time = get_order_feat(data, sample, data_key)                           # 获取用户时间特征
        user_count = get_user_count(data, sample, data_key)                           # 获取用户历史行为次数
        start_loc = get_start_loc(data, sample, data_key)                             # 出发地
        end_loc = get_end_loc(data, sample, data_key)                                 # 目的地
        user_sloc = get_user_sloc(data, sample, data_key)                             # 用户×出发地（次数 目的地个数）
        user_eloc = get_user_eloc(data, sample, data_key)                             # 用户×目的地
        sloc_eloc = get_sloc_eloc(data, sample, data_key)                             # 出发地×目的地
        bike_eloc = get_bike_eloc(data, sample, candidate, data_key)                  # 自行车×目的地
        user_sloc_eloc = get_user_sloc_eloc(data, sample, data_key)                   # 用户×出发地×目的地
        gc.collect()

        # 按时间统计
        hour_user = get_hour_user(data,sample, data_key)                              # 时间×用户
        hour_sloc = get_hour_sloc(data,sample, data_key)                              # 时间×出发地
        hour_eloc = get_hour_eloc(data,sample, data_key)                              # 时间×目的地
        hour_sloc_eloc = get_hour_sloc_eloc(data, sample, data_key)                   # 时间×出发地×目的地
        hour_user_eloc = get_hour_user_eloc(data, sample, data_key)                   # 时间×用户×目的地
        gc.collect()

        #扩大时间范围
        exhour_user = get_exhour_user(data, sample, data_key)                         # 时间×用户
        exhour_sloc = get_exhour_sloc(data, sample, data_key)                         # 时间×出发地
        exhour_eloc = get_exhour_eloc(data, sample, data_key)                         # 时间×目的地
        exhour_sloc_eloc = get_exhour_sloc_eloc(data, sample, data_key)               # 时间×出发地×目的地
        exhour_user_eloc = get_exhour_user_eloc(data, sample, data_key)               # 时间×用户×目的地
        gc.collect()

        # 扩大统计范围重新统计
        ex_sloc = get_ex_sloc(data, sample, data_key)                                 # 扩大 出发地
        ex_eloc = get_ex_eloc(data, sample, data_key)                                 # 获取目的地热度（作为目的地的个数、人数，作为出发地的个数、人数，折返比）
        ex_user_sloc = get_ex_user_sloc(data, sample, data_key)                       # 用户×出发地（次数 目的地个数）
        ex_user_eloc = get_ex_user_eloc(data, sample, data_key)                       # 用户×目的地
        ex_sloc_eloc = get_ex_sloc_eloc(data, sample, data_key)                       # 出发地×目的地
        ex_exhour_sloc_eloc = get_ex_exhour_sloc_eloc(data, sample, data_key)         # 扩大出发地×扩大目的地×扩大时间
        ex_exhour_user_eloc = get_ex_exhour_user_eloc(data, sample, data_key)         # 用户×扩大目的地×扩大时间
        ex_user_sloc_eloc = get_ex_user_sloc_eloc(data, sample, candidate, data_key)  # 用户×出发地×目的地
        gc.collect()

        # 工作日× 地点
        holiday = get_holiday(data, sample, data_key)                                   # 工作日×时间
        holiday_eloc = get_holiday_eloc(data, sample, data_key)                         # 工作日×目的地
        holiday_ex_sloc = get_holiday_ex_sloc(data, sample, data_key)                   # 工作日×扩大出发地
        holiday_sloc_eloc = get_holiday_sloc_eloc(data, sample, data_key)               # 工作日×出发地×目的地
        holiday_ex_sloc_eloc = get_holiday_ex_sloc_eloc(data, sample, data_key)         # 工作日×扩大×出发地×目的地
        holiday_hour_eloc = get_holiday_hour_eloc(data, sample, data_key)               # 工作日×时间×目的地
        holiday_exhour_ex_sloc = get_holiday_exhour_ex_sloc(data, sample, data_key)     # 工作日×扩大时间×扩大出发地
        holiday_hour_sloc_eloc = get_holiday_hour_sloc_eloc(data, sample, data_key)     # 工作日×时间×出发地×目的地
        holiday_hour_ex_sloc_eloc = get_holiday_hour_ex_sloc_eloc(data, sample, data_key)  # 工作日×时间×扩大出发地×目的地
        holiday_exhour_ex_sloc_eloc = get_holiday_exhour_ex_sloc_eloc(data, sample, data_key)  # 工作日×扩大时间×扩大出发地×目的地

        # 工作日× 用户
        holiday_user_count = get_holiday_user_count(data, sample, data_key)             # 工作日×扩大时间×扩大出发地
        holiday_user_eloc = get_holiday_ex_user_eloc(data, sample, data_key)            # 工作日×时间×目的地

        # 日期小时目的地特征
        date_hour = get_date_hour(data, sample, data_key)

        # 添加另一个模型
        all_order_pred = get_all_order_pred(sample, data_key)

        # 添加pred特征
        test_pred = get_test_pred(data, sample, data_key)

        # 添加用户自行车特征
        user_bike = get_user_bike(data, sample, data_key)

        # 聚类特征
        loc_cluster = get_loc_cluster(data, sample, data_key)

        print('开始合并特征...')

        result = concat([sample, user_count, start_loc, end_loc, user_sloc, user_eloc, sloc_eloc,
                         user_sloc_eloc, ex_sloc, ex_eloc, ex_user_sloc, ex_user_eloc,
                         ex_sloc_eloc, ex_user_sloc_eloc,all_order_pred,hour_sloc,hour_eloc,hour_sloc_eloc,
                         hour_user, hour_user_eloc, exhour_user, exhour_sloc, exhour_eloc,holiday_ex_sloc_eloc,
                         exhour_sloc_eloc, exhour_user_eloc, order_time,date_hour,holiday_ex_sloc,
                         holiday, holiday_eloc,ex_exhour_sloc_eloc,ex_exhour_user_eloc,
                         holiday_hour_eloc,holiday_sloc_eloc,test_pred,bike_eloc,loc_cluster,
                         holiday_exhour_ex_sloc, holiday_hour_sloc_eloc, holiday_hour_ex_sloc_eloc,
                         holiday_exhour_ex_sloc_eloc,holiday_user_count, holiday_user_eloc,user_bike
                         ])
        del sample, user_count, start_loc, end_loc, user_sloc, user_eloc, sloc_eloc, \
            user_sloc_eloc, ex_sloc, ex_eloc, ex_user_sloc, ex_user_eloc, ex_sloc_eloc,\
            ex_user_sloc_eloc, all_order_pred, hour_sloc, hour_eloc,hour_sloc_eloc,hour_user, \
            hour_user_eloc, exhour_user, exhour_sloc, exhour_eloc, exhour_sloc_eloc, exhour_user_eloc, \
            order_time,date_hour,ex_exhour_sloc_eloc,ex_exhour_user_eloc,holiday_ex_sloc, \
            holiday, holiday_eloc, holiday_hour_eloc,holiday_sloc_eloc,holiday_ex_sloc_eloc,\
            holiday_exhour_ex_sloc, holiday_hour_sloc_eloc,test_pred,bike_eloc,\
            holiday_hour_ex_sloc_eloc, holiday_exhour_ex_sloc_eloc,user_bike
        gc.collect()
        result = second_feat(result)
        result.drop(['hour_count','holiday_count','holiday_hour_count','all_count',
                     'holiday_eloc_count','holiday_eloc_as_sloc_count2','holiday_exhour_ex_sloc_eloc_2count',
                     'holiday_ex_sloc_count','holiday_ex_sloc_n_user','holiday_hour_ex_sloc_eloc_2count',
                     'holiday_sloc_eloc_count', 'holiday_eloc_sloc_count', 'holiday_sloc_eloc_2count',
                     'holiday_sloc_eloc_n_user', 'holiday_eloc_sloc_n_user', 'holiday_sloc_eloc_2n_user',
                     'holiday_ex_sloc_eloc_count', 'holiday_ex_eloc_sloc_count', 'holiday_ex_sloc_eloc_2count',
                     'holiday_ex_sloc_eloc_n_user', 'holiday_ex_eloc_sloc_n_user', 'holiday_ex_sloc_eloc_2n_user',
                     'holiday_hour_eloc_count', 'holiday_hour_eloc_2count','holiday_hour_sloc_eloc_2count',
                     'holiday_exhour_ex_sloc_count', 'holiday_hour_sloc_eloc_count','holiday_hour_ex_sloc_eloc_count',
                     'holiday_exhour_ex_sloc_eloc_count','holiday_eloc_n_user','holiday_eloc_as_sloc_n_user2',
                     'holiday_user_count','holiday_ex_user_eloc_count', 'holiday_ex_user_eloc_as_sloc_count',
                     'holiday_ex_user_eloc_2count', 'holiday_eloc_2count','holiday_eloc_2n_user'

                     ],axis=1,inplace=True)
        gc.collect()

        print('添加label')
        result = get_leak_label(result)

        print('存储数据...')
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    print('特征矩阵大小：{}'.format(result.shape))
    print('生成特征一共用时{}秒'.format(time.time() - t0))
    return result



# 线下测试集3
train = pd.read_csv(train_path)
train.loc[(train['starttime'] >= '2017-05-24 00:00:00') & (train['starttime'] < '2017-05-25 00:00:00'), 'geohashed_end_loc'] = np.nan
test = pd.read_csv(test_path)
test.loc[:, 'geohashed_end_loc'] = np.nan
data = pd.concat([train, test])
orderid_sample = make_sample(data.orderid.tolist(),n_sub=3,seed=66)[0]
del train, test

# 23号训练集
data_temp = data.copy()
data_temp.loc[(data_temp['starttime'] >= '2017-05-23 00:00:00') & (data_temp['starttime'] < '2017-05-24 00:00:00'), 'geohashed_end_loc'] = np.nan
train_data = data_temp[(data_temp['starttime'] >= '2017-05-23 00:00:00') & (data_temp['starttime'] < '2017-05-24 00:00:00')].copy()
train_data = train_data[train_data['orderid'].isin(orderid_sample)]
train_feat = make_train_set(data_temp, train_data).fillna(-1)
train_feat.sort_values('orderid',inplace=True)
# 24号测试集
data_temp = data.copy()
test_data = data_temp[(data_temp['starttime'] >= '2017-05-24 00:00:00') & (data_temp['starttime'] < '2017-05-25 00:00:00')].copy()
test_data = test_data[test_data['orderid'].isin(orderid_sample)]
test_feat = make_train_set(data_temp, test_data).fillna(-1)
test_feat.sort_values('orderid',inplace=True)

gc.collect()
predictors = train_feat.columns.drop(['orderid', 'geohashed_end_loc', 'label', 'userid',
                                      'bikeid', 'starttime', 'geohashed_start_loc','user_speed','bike_speed'])

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'lambdarank',
    'metric': 'auc',
    'max_depth': 8,
    'num_leaves': 150,
    'learning_rate': 0.05,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'colsample_bylevel': 0.7,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5,
    'verbose': 0,
    'seed': 66,
}
print('Start training...')

l1=train_feat.groupby(['orderid']).count()['userid']
l2=test_feat.groupby(['orderid']).count()['userid']

lgb_train = lgb.Dataset(train_feat[predictors],train_feat.label, group=l1,categorical_feature=['holiday','biketype'])
lgb_eval = lgb.Dataset(test_feat[predictors], test_feat.label,group=l2,categorical_feature=['holiday','biketype'], reference=lgb_train)

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=1000,
                valid_sets=lgb_eval,
                verbose_eval = 50,
                early_stopping_rounds=100)
feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
feat_imp.to_csv(cache_path + 'feat_imp.csv')

# params = {
#     'task': 'train',
#     'boosting_type': 'gbdt',
#     'objective': 'binary',
#     'metric': 'auc',
#     'max_depth': 8,
#     'num_leaves': 150,
#     'learning_rate': 0.05,
#     'subsample': 0.7,
#     'colsample_bytree': 0.7,
#     'colsample_bylevel': 0.7,
#     'feature_fraction': 0.9,
#     'bagging_fraction': 0.95,
#     'bagging_freq': 5,
#     'verbose': 0,
#     'seed': 66,
# }
# print('Start training...')
#
# lgb_train = lgb.Dataset(train_feat[predictors], train_feat.label,categorical_feature=['holiday','biketype'])
# lgb_eval = lgb.Dataset(test_feat[predictors], test_feat.label,categorical_feature=['holiday','biketype'], reference=lgb_train)
#
# gbm = lgb.train(params,
#                 lgb_train,
#                 num_boost_round=1000,
#                 valid_sets=lgb_eval,
#                 verbose_eval = 50,
#                 early_stopping_rounds=100)
# feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
# feat_imp.to_csv(cache_path + 'feat_imp.csv')


test_pred = test_feat[['orderid','geohashed_end_loc']].copy()
preds = gbm.predict(test_feat[predictors])
test_pred['pred'] = preds
test_pred = get_label(test_pred)
bike_eloc = get_bike_next_loc(data, test_data, 'all')
test_pred = test_pred.merge(bike_eloc,on=['orderid','geohashed_end_loc'],how='left')
test_pred = test_pred[~(~test_pred['bike_eloc_sep_time'].isnull())]
test_pred = test_pred[['orderid','geohashed_end_loc','pred']]
result = reshape(test_pred)
result.fillna('a',inplace=True)
result = pd.merge(test_data[['orderid']], result, on='orderid', how='left')
print('map得分为:{}'.format(leak_map(result)))






# # xgb参数测试
# train = pd.read_csv(train_path)
# train.loc[(train['starttime'] >= '2017-05-24 00:00:00') & (train['starttime'] < '2017-05-25 00:00:00'), 'geohashed_end_loc'] = np.nan
# test = pd.read_csv(test_path)
# test.loc[:, 'geohashed_end_loc'] = np.nan
# data = pd.concat([train, test])
# orderid_sample = make_sample(data.orderid.tolist(),n_sub=3,seed=66)[0]
# del train, test
#
# # 23号训练集
# data_temp = data.copy()
# data_temp.loc[(data_temp['starttime'] >= '2017-05-23 00:00:00') & (data_temp['starttime'] < '2017-05-24 00:00:00'), 'geohashed_end_loc'] = np.nan
# train_data = data_temp[(data_temp['starttime'] >= '2017-05-23 00:00:00') & (data_temp['starttime'] < '2017-05-24 00:00:00')].copy()
# train_data = train_data[train_data['orderid'].isin(orderid_sample)]
# train_feat = make_train_set(data_temp, train_data).fillna(-1)
# # 24号测试集
# data_temp = data.copy()
# test_data = data_temp[(data_temp['starttime'] >= '2017-05-24 00:00:00') & (data_temp['starttime'] < '2017-05-25 00:00:00')].copy()
# test_data = test_data[test_data['orderid'].isin(orderid_sample)]
# test_feat = make_train_set(data_temp, test_data).fillna(-1)
#
# gc.collect()
# predictors = train_feat.columns.drop(['orderid', 'geohashed_end_loc', 'label', 'userid',
#                                       'bikeid', 'starttime', 'geohashed_start_loc'])
#
# import xgboost
# xgb_train = xgboost.DMatrix(train_feat[predictors],train_feat.label)
# xgb_eval = xgboost.DMatrix(test_feat[predictors],test_feat.label)
#
# xgb_params = {
#     "objective"         : "reg:logistic"
#     ,"eval_metric"      : "logloss"
#     ,"eta"              : 0.05
#     ,"max_depth"        : 5
#     ,"min_child_weight" :4
#     ,"gamma"            :0.70
#     ,"subsample"        :0.76
#     ,"colsample_bytree" :0.95
#     ,"alpha"            :2e-05
#     ,"lambda"           :10
#     ,'silent'           :1
# }
#
# watchlist= [(xgb_eval, "test")]
# bst = xgboost.train(params=xgb_params,
#                     dtrain=xgb_train,
#                     num_boost_round=1000,
#                     evals=watchlist,
#                     verbose_eval=50,
#                     early_stopping_rounds=50)
#
# test_pred = test_feat[['orderid','geohashed_end_loc']].copy()
# preds = bst.predict(xgb_eval)
# test_pred['pred'] = preds
# test_pred = get_label(test_pred)
# bike_eloc = get_bike_next_loc(data, test_data, 'all')
# test_pred = test_pred.merge(bike_eloc,on=['orderid','geohashed_end_loc'],how='left')
# test_pred = test_pred[~(~test_pred['bike_eloc_sep_time'].isnull())]
# test_pred = test_pred[['orderid','geohashed_end_loc','pred']]
# result = reshape(test_pred)
# result.fillna('a',inplace=True)
# result = pd.merge(test_data[['orderid']], result, on='orderid', how='left')
# print('map得分为:{}'.format(leak_map(result)))


# print('Start training...')
# weight1 = train_feat['user_count'].apply(lambda x: 3 if x<4 else 1).values
# weight2 = train_feat['user_count'].apply(lambda x: 3 if x>3 else 1).values
# lgb_train = lgb.Dataset(train_feat[predictors], train_feat.label,categorical_feature=['holiday','biketype'],weight = weight1)
# lgb_eval = lgb.Dataset(test_feat[predictors], test_feat.label,categorical_feature=['holiday','biketype'], reference=lgb_train)
#
# gbm = lgb.train(params,
#                 lgb_train,
#                 num_boost_round=1000,
#                 valid_sets=lgb_eval,
#                 verbose_eval = 50,
#                 early_stopping_rounds=100)
# feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
# feat_imp.to_csv(cache_path + 'feat_imp.csv')
#
# test_pred1 = test_feat[test_feat['user_count']<4][['orderid','geohashed_end_loc']].copy()
# preds1 = gbm.predict(test_feat[test_feat['user_count']<4][predictors])
# test_pred1['pred'] = preds1
#
# lgb_train = lgb.Dataset(train_feat[predictors], train_feat.label,categorical_feature=['holiday','biketype'],weight = weight2)
# lgb_eval = lgb.Dataset(test_feat[predictors], test_feat.label,categorical_feature=['holiday','biketype'], reference=lgb_train)
#
# gbm = lgb.train(params,
#                 lgb_train,
#                 num_boost_round=1000,
#                 valid_sets=lgb_eval,
#                 verbose_eval = 50,
#                 early_stopping_rounds=100)
# feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
# feat_imp.to_csv(cache_path + 'feat_imp.csv')
#
# test_pred2 = test_feat[test_feat['user_count']>3][['orderid','geohashed_end_loc']].copy()
# preds2 = gbm.predict(test_feat[test_feat['user_count']>3][predictors])
# test_pred2['pred'] = preds2
#
#
# test_pred = pd.concat([test_pred1,test_pred2])
# test_pred = get_label(test_pred)
# bike_eloc = get_bike_next_loc(data, test_data, 'all')
# test_pred = test_pred.merge(bike_eloc,on=['orderid','geohashed_end_loc'],how='left')
# test_pred = test_pred[~(~test_pred['bike_eloc_sep_time'].isnull())]
# test_pred = test_pred[['orderid','geohashed_end_loc','pred']]
# result = reshape(test_pred)
# result.fillna('a',inplace=True)
# result = pd.merge(test_data[['orderid']], result, on='orderid', how='left')
# print('map得分为:{}'.format(leak_map(result)))


#
# # 线下测试集3
# train = pd.read_csv(train_path)
# train.loc[(train['starttime'] >= '2017-05-24 00:00:00') & (train['starttime'] < '2017-05-25 00:00:00'), 'geohashed_end_loc'] = np.nan
# test = pd.read_csv(test_path)
# test.loc[:, 'geohashed_end_loc'] = np.nan
# data = pd.concat([train, test])
# orderid_sample = make_sample(data.orderid.tolist(),n_sub=3,seed=66)[0]
# del train, test
#
# # 23号训练集
# data_temp = data.copy()
# data_temp.loc[(data_temp['starttime'] >= '2017-05-23 00:00:00') & (data_temp['starttime'] < '2017-05-24 00:00:00'), 'geohashed_end_loc'] = np.nan
# train_data = data_temp[(data_temp['starttime'] >= '2017-05-23 00:00:00') & (data_temp['starttime'] < '2017-05-24 00:00:00')].copy()
# train_data = train_data[train_data['orderid'].isin(orderid_sample)]
# train_feat = make_train_set(data_temp, train_data).fillna(-1)
# # 24号测试集
# data_temp = data.copy()
# test_data = data_temp[(data_temp['starttime'] >= '2017-05-24 00:00:00') & (data_temp['starttime'] < '2017-05-25 00:00:00')].copy()
# test_data = test_data[test_data['orderid'].isin(orderid_sample)]
# test_feat = make_train_set(data_temp, test_data).fillna(-1)
#
#
# gc.collect()
# predictors = train_feat.columns.drop(['orderid', 'geohashed_end_loc', 'label', 'userid',
#                                       'bikeid', 'starttime', 'geohashed_start_loc'])
#
# ##############################使用随机森林预测##################################
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
# for i in [2,4,8]:
#     model = RandomForestClassifier(n_estimators=250, criterion='entropy', max_depth=i, min_samples_split=2,
#                                           min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=0.6,
#                                           max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=-1,
#                                           random_state=1301, verbose=0)
#     model = model.fit(train_feat[predictors], train_feat['label'])
#     preds = model.predict_proba(test_feat[predictors])[:,1]
#     test_pred = test_feat[['orderid','geohashed_end_loc']].copy()
#     test_pred['pred'] = preds
#     bike_eloc = get_bike_next_loc(data, test_data, 'all')
#     test_pred = test_pred.merge(bike_eloc,on=['orderid','geohashed_end_loc'],how='left')
#     test_pred = test_pred[~(~test_pred['bike_eloc_sep_time'].isnull())]
#     test_pred = test_pred[['orderid','geohashed_end_loc','pred']]
#     result = reshape(test_pred)
#     result.fillna('a',inplace=True)
#     result = pd.merge(test_data[['orderid']], result, on='orderid', how='left')
#     print('map得分为:{}'.format(leak_map(result)))
#
#
#
#
# # 删除空白方向
# def del_direction(pred):
#     columns = pred.columns
#     test = pd.read_csv(test_path)
#     pred = pd.merge(pred,test[['orderid','geohashed_start_loc']],on='orderid',how='left')
#     pred['direction'] = get_direction(pred)//0.1
#     pred = pred[((pred['direction']>14) & (pred['direction']<31)) |
#                 ((pred['direction']>-33) & (pred['direction']<1))]
#     return pred[columns]


