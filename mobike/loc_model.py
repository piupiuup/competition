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

cache_path = 'F:/mobike_loc_cache/'
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
def sample(n,n_sub=2,seed=None):
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

# 周围的9×7个地点
def get_near_loc_dict_9_7():
    result_path = cache_path + 'near_loc_9_7.hdf'
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
# 周围的9×7个地点
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
    order_eloc_dict = dict(zip(data['orderid'].values, data['geohashed_end_loc'].values))
    near_loc_dict_9_7 = get_near_loc_dict_9_7()
    result = []
    for orderid,locs in zip(orderids,locs):
        sum_count = 0
        near_locs = near_loc_dict_9_7[locs]
        for near_loc in near_locs:
            if near_loc in eloc_count:
                sum_count += eloc_count[near_loc]
        end_loc = order_eloc_dict[orderid]
        if end_loc in near_locs:
            sum_count = sum_count - 1
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
    result = np.arccos(data_temp['x']/data_temp['z']) * (data_temp['y']/np.abs(data_temp['y']))
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
    result_path = cache_path + 'true.pkl'
    if os.path.exists(result_path):
        true = pickle.load(open(result_path, 'rb+'))
    else:
        train = pd.read_csv(train_path)
        true = dict(zip(train['orderid'].values, train['geohashed_end_loc'].values))
        pickle.dump(true, open(result_path, 'wb+'))
    data = result.copy()
    data['true'] = data['orderid'].map(true)
    n = data.shape[0]
    score1 = sum(data['true'] == data[0]) / 1.0 / n
    score2 = sum(data['true'] == data[1]) / 2.0 / n
    score3 = sum(data['true'] == data[2]) / 3.0 / n
    print('第一列得分为：{}'.format(score1))
    print('第二列得分为：{}'.format(score2))
    print('第三列得分为：{}'.format(score3))
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


####################构造负样本##################


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
        sloc_eloc_count = sloc_eloc_count.groupby('geohashed_start_loc').tail(7)
        result = pd.merge(candidate[['orderid', 'geohashed_start_loc']], sloc_eloc_count, on='geohashed_start_loc', how='inner')
        order_eloc_dict = dict(zip(data['orderid'].values, data['geohashed_end_loc'].values))
        result['sloc_eloc_count'] = result['sloc_eloc_count'] - (result['geohashed_end_loc'] == result['orderid'].map(order_eloc_dict))
        result.sort_values('sloc_eloc_count', inplace=True)
        result = result.groupby('orderid').tail(6)
        result = result[result['sloc_eloc_count'] > 0]
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
        sloc_eloc_count = sloc_eloc_count[~sloc_eloc_count['geohashed_end_loc'].isnull()]
        eloc_sloc_count = sloc_eloc_count.rename(columns={'geohashed_start_loc': 'geohashed_end_loc',
                                                          'geohashed_end_loc': 'geohashed_start_loc',
                                                          'sloc_eloc_count': 'eloc_sloc_count'}).copy()
        sloc_eloc_2count = pd.merge(sloc_eloc_count, eloc_sloc_count,
                                    on=['geohashed_start_loc', 'geohashed_end_loc'], how='outer').fillna(0)
        sloc_eloc_2count['sloc_eloc_2count'] = sloc_eloc_2count['sloc_eloc_count'] + sloc_eloc_2count['eloc_sloc_count']
        sloc_eloc_2count.sort_values('sloc_eloc_2count', inplace=True)
        sloc_eloc_2count = sloc_eloc_2count.groupby('geohashed_start_loc').tail(7)
        result = pd.merge(candidate[['orderid', 'geohashed_start_loc']], sloc_eloc_2count, on='geohashed_start_loc', how='inner')
        order_eloc_dict = dict(zip(data['orderid'].values, data['geohashed_end_loc'].values))
        result['sloc_eloc_2count'] = result['sloc_eloc_2count'] - (result['geohashed_end_loc'] == result['orderid'].map(order_eloc_dict))
        result.sort_values('sloc_eloc_2count', inplace=True)
        result = result.groupby('orderid').tail(6)
        result = result[result['sloc_eloc_2count'] > 0]
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
        sloc_eloc_count = sloc_eloc_count.groupby('geohashed_start_loc').tail(7)
        result = pd.merge(candidate[['orderid', 'geohashed_start_loc']], sloc_eloc_count,on='geohashed_start_loc', how='inner')
        order_eloc_dict = dict(zip(data['orderid'].values, data['geohashed_end_loc'].values))
        result['sloc_eloc_count'] = result['sloc_eloc_count'] - (result['geohashed_end_loc'] == result['orderid'].map(order_eloc_dict))
        result.sort_values('sloc_eloc_count', inplace=True)
        result = result.groupby('orderid').tail(6)
        result = result[result['sloc_eloc_count'] > 0]
        result = result[['orderid', 'geohashed_end_loc']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    print('ex_loc_to_loc样本个数为：{}'.format(result.shape))
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
        result = result.groupby('geohashed_start_loc').tail(4)
        result = pd.merge(candidate[['orderid', 'geohashed_start_loc']], result,on='geohashed_start_loc', how='inner')
        order_eloc_dict = dict(zip(data['orderid'].values, data['geohashed_end_loc'].values))
        result['sloc_eloc_2count'] = result['sloc_eloc_2count'] - (result['geohashed_end_loc'] == result['orderid'].map(order_eloc_dict))
        result.sort_values('sloc_eloc_2count', inplace=True)
        result = result.groupby('orderid').tail(3)
        result = result[result['sloc_eloc_2count'] > 0]
        result = result[['orderid', 'geohashed_end_loc']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    print('ex_loc_with_loc样本个数为：{}'.format(result.shape))
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
        result = result.groupby(['holiday', 'geohashed_start_loc']).tail(4)
        result = pd.merge(candidate[['holiday', 'orderid', 'geohashed_start_loc']], result,
                          on=['holiday', 'geohashed_start_loc'],how='inner')
        order_eloc_dict = dict(zip(data['orderid'].values, data['geohashed_end_loc'].values))
        result['holiday_sloc_eloc_count'] = result['holiday_sloc_eloc_count'] - (result['geohashed_end_loc'] == result['orderid'].map(order_eloc_dict))
        result.sort_values('holiday_sloc_eloc_count', inplace=True)
        result = result.groupby('orderid').tail(3)
        result = result[result['holiday_sloc_eloc_count'] > 0]
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
        result = result.groupby(['hour', 'geohashed_start_loc']).tail(4)
        result = pd.merge(candidate[['hour', 'orderid', 'geohashed_start_loc']], result,on=['hour','geohashed_start_loc'],how='inner')
        order_eloc_dict = dict(zip(data['orderid'].values, data['geohashed_end_loc'].values))
        result['exhour_sloc_eloc_count'] = result['exhour_sloc_eloc_count'] - (result['geohashed_end_loc'] == result['orderid'].map(order_eloc_dict))
        result.sort_values('exhour_sloc_eloc_count', inplace=True)
        result = result.groupby('orderid').tail(3)
        result = result[result['exhour_sloc_eloc_count'] > 0]
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
        result = result.groupby(['hour', 'geohashed_start_loc']).tail(4)
        result = pd.merge(candidate[['hour', 'orderid', 'geohashed_start_loc']], result,
                          on=['hour', 'geohashed_start_loc'], how='inner')
        order_eloc_dict = dict(zip(data['orderid'].values, data['geohashed_end_loc'].values))
        result['exhour_ex_sloc_eloc_count'] = result['exhour_ex_sloc_eloc_count'] - ( result['geohashed_end_loc'] == result['orderid'].map(order_eloc_dict))
        result.sort_values('exhour_ex_sloc_eloc_count', inplace=True)
        result = result.groupby('orderid').tail(3)
        result = result[result['exhour_ex_sloc_eloc_count'] > 0]
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
        result = result.groupby(['hour','holiday', 'geohashed_start_loc']).tail(4)
        result = pd.merge(candidate[['hour','holiday', 'orderid', 'geohashed_start_loc']], result,
                          on=['hour','holiday', 'geohashed_start_loc'], how='inner')
        order_eloc_dict = dict(zip(data['orderid'].values, data['geohashed_end_loc'].values))
        result['holiday_exhour_sloc_eloc_count'] = result['holiday_exhour_sloc_eloc_count'] - (
        result['geohashed_end_loc'] == result['orderid'].map(order_eloc_dict))
        result.sort_values('holiday_exhour_sloc_eloc_count', inplace=True)
        result = result.groupby('orderid').tail(3)
        result = result[result['holiday_exhour_sloc_eloc_count'] > 0]
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
        order_eloc_dict = dict(zip(data['orderid'].values, data['geohashed_end_loc'].values))
        result['holiday_ex_sloc_eloc_count'] = result['holiday_ex_sloc_eloc_count'] - (result['geohashed_end_loc'] == result['orderid'].map(order_eloc_dict))
        result.sort_values('holiday_ex_sloc_eloc_count', inplace=True)
        result = result.groupby('orderid').tail(3)
        result = result[result['holiday_ex_sloc_eloc_count'] > 0]
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
        result = result.groupby(['hour','holiday', 'geohashed_start_loc']).tail(4)
        result = pd.merge(candidate[['hour','holiday', 'orderid', 'geohashed_start_loc']], result,
                          on=['hour', 'holiday', 'geohashed_start_loc'], how='inner')
        order_eloc_dict = dict(zip(data['orderid'].values, data['geohashed_end_loc'].values))
        result['holiday_exhour_ex_sloc_eloc_count'] = result['holiday_exhour_ex_sloc_eloc_count'] - (result['geohashed_end_loc'] == result['orderid'].map(order_eloc_dict))
        result.sort_values('holiday_exhour_ex_sloc_eloc_count', inplace=True)
        result = result.groupby('orderid').tail(3)
        result = result[result['holiday_exhour_ex_sloc_eloc_count'] > 0]
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
        near_loc_df = near_loc_df.groupby('loc').tail(4)
        result = pd.merge(candidate[['orderid', 'geohashed_start_loc']], near_loc_df,
                          left_on='geohashed_start_loc',right_on='loc', how='inner')
        result.rename(columns={'near_loc':'geohashed_end_loc'},inplace=True)
        order_eloc_dict = dict(zip(data['orderid'].values, data['geohashed_end_loc'].values))
        result['eloc_count'] = result['eloc_count'] - (result['geohashed_end_loc'] == result['orderid'].map(order_eloc_dict))
        result.sort_values('eloc_count', inplace=True)
        result = result.groupby('orderid').tail(3)
        result = result[result['eloc_count'] > 0]
        result = result[['orderid', 'geohashed_end_loc']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    print('near_loc_to_loc样本个数为：{}'.format(result.shape))
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
        data_temp.rename(columns={'orderid': 'orderid2'}, inplace=True)
        sample_temp = sample.copy()
        sample_temp['int_time'] = sample_temp['starttime'].apply(int_time)

        sloc_time = pd.merge(data_temp[['orderid2', 'geohashed_start_loc', 'geohashed_end_loc', 'int_time2']],
                             sample_temp[['orderid', 'geohashed_start_loc', 'geohashed_end_loc', 'int_time']],
                             on=['geohashed_start_loc', 'geohashed_end_loc'], how='inner')
        sloc_time = sloc_time[sloc_time['orderid'] != sloc_time['orderid2']]
        sloc_time['diff_time'] = (sloc_time['int_time'] - sloc_time['int_time2']).apply(lambda x: min(abs(1440 - abs(x)), abs(x)))
        sloc_time = sloc_time.groupby(['orderid', 'geohashed_end_loc'], as_index=False)['diff_time'].agg(
            {'sloc_mean_diff_time': 'mean',
             'sloc_min_diff_time': 'min'})


        feat = sample.merge(sloc_time, on=['orderid', 'geohashed_end_loc'], how='left')

        feat['lat1'] = feat['geohashed_start_loc'].apply(lambda x: loc_dict[x][0])
        feat['lon1'] = feat['geohashed_start_loc'].apply(lambda x: loc_dict[x][1])
        feat['lat2'] = feat['geohashed_end_loc'].apply(lambda x: loc_dict[x][0])
        feat['lon2'] = feat['geohashed_end_loc'].apply(lambda x: loc_dict[x][1])
        feat['direction'] = get_direction(feat)

        feat = feat[['sloc_mean_diff_time', 'sloc_min_diff_time','lat1','lon1',
                     'lat2','lon2','direction',]]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat


# 获取出发地热度（作为出发地的次数、人数）
def get_start_loc(data, sample, data_key):
    feat_path = cache_path + 'start_loc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        data_temp = data.copy()
        loc_dict = get_loc_dict()
        data_temp['lat'] = data_temp['geohashed_end_loc'].apply(lambda x: np.nan if x is np.nan else loc_dict[x][0])
        data_temp['lon'] = data_temp['geohashed_end_loc'].apply(lambda x: np.nan if x is np.nan else loc_dict[x][1])
        feat = data_temp.groupby('geohashed_start_loc').agg({'userid': {'sloc_count': 'count'},
                                                             'geohashed_end_loc':{'eloc_no_nan':'count'},
                                                                        'lat': {'sum_lat': 'sum'},
                                                                        'lon': {'sum_lon': 'sum'}})
        feat.columns = feat.columns.droplevel(0)
        feat.reset_index(inplace=True)
        feat = pd.merge(sample, feat, on=['geohashed_start_loc'], how='left').fillna(0)
        order_eloc_dict = dict(zip(data['orderid'].values, data['geohashed_end_loc'].values))
        feat['sum_lat'] = feat['sum_lat'] - feat['orderid'].map(order_eloc_dict).apply(
            lambda x: 0 if x is np.nan else loc_dict[x][0])
        feat['sum_lon'] = feat['sum_lon'] - feat['orderid'].map(order_eloc_dict).apply(
            lambda x: 0 if x is np.nan else loc_dict[x][1])
        feat['sloc_count'] = feat['sloc_count'] - (feat['orderid'].isin(data['orderid'].values))
        order_eloc_dict = dict(zip(data['orderid'].values, data['geohashed_end_loc'].values))
        feat['eloc_no_nan'] = feat['eloc_no_nan'] - (~feat['orderid'].map(order_eloc_dict).isnull())
        feat['mean_lat'] = feat['sum_lat'] / (feat['eloc_no_nan'] + 0.001)
        feat['mean_lon'] = feat['sum_lon'] / (feat['eloc_no_nan'] + 0.001)
        feat['lat'] = feat['geohashed_end_loc'].apply(lambda x: np.nan if x is np.nan else loc_dict[x][0])
        feat['lon'] = feat['geohashed_end_loc'].apply(lambda x: np.nan if x is np.nan else loc_dict[x][1])
        feat['sloc_center_dis'] = ((feat['lat'] - feat['mean_lat']) ** 2 + (
            np.cos(feat['mean_lon'] / 57.2958) * (feat['lon'] - feat['mean_lon'])) ** 2) ** 0.5
        feat['sloc_direction'] = get_angle(feat)
        feat['seloc_count_9_7'] = get_count_9_7(feat['orderid'],feat['geohashed_start_loc'],data)
        feat = feat[['sloc_count', 'sloc_center_dis','sloc_direction','seloc_count_9_7']].fillna(100000)
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat


# 获取目标地点的热度(目的地)
def get_end_loc(data, sample, data_key):
    feat_path = cache_path + 'end_loc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        eloc_as_sloc = data.groupby('geohashed_start_loc', as_index=False)['userid'].agg(
            {'eloc_as_sloc_count': 'count'})
        eloc = data.groupby('geohashed_end_loc', as_index=False)['userid'].agg(
            {'eloc_count': 'count'})
        eloc_as_sloc.rename(columns={'geohashed_start_loc': 'geohashed_end_loc'}, inplace=True)
        feat = pd.merge(sample, eloc_as_sloc, on='geohashed_end_loc', how='left')
        feat = pd.merge(feat, eloc, on='geohashed_end_loc', how='left').fillna(0)
        order_eloc_dict = dict(zip(data['orderid'].values, data['geohashed_end_loc'].values))
        feat['eloc_count'] = feat['eloc_count'] - (feat['geohashed_end_loc'] == feat['orderid'].map(order_eloc_dict))
        feat['eloc_2count'] = feat['eloc_count'] + feat['eloc_as_sloc_count']
        feat['eloc_in_out_rate'] = feat['eloc_count'] / (feat['eloc_as_sloc_count'] + 0.001)
        feat['eeloc_count_9_7'] = get_count_9_7(feat['orderid'], feat['geohashed_end_loc'], data)
        feat = feat[['eloc_as_sloc_count', 'eloc_2count',
                     'eloc_count', 'eloc_in_out_rate','eeloc_count_9_7']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat


# 出发地×目的地
def get_sloc_eloc(data, sample, data_key):
    feat_path = cache_path + 'sloc_eloc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        sloc_eloc = data.groupby(['geohashed_start_loc', 'geohashed_end_loc'], as_index=False)['userid'].agg(
            {'sloc_eloc_count': 'count'})
        eloc_sloc = sloc_eloc.rename(columns={'geohashed_start_loc': 'geohashed_end_loc',
                                              'geohashed_end_loc': 'geohashed_start_loc',
                                              'sloc_eloc_count': 'eloc_sloc_count'})
        feat = pd.merge(sample, sloc_eloc, on=['geohashed_start_loc', 'geohashed_end_loc'], how='left')
        feat = pd.merge(feat, eloc_sloc, on=['geohashed_start_loc', 'geohashed_end_loc'], how='left').fillna(0)
        order_eloc_dict = dict(zip(data['orderid'].values, data['geohashed_end_loc'].values))
        feat['sloc_eloc_count'] = feat['sloc_eloc_count'] - (
        feat['geohashed_end_loc'] == feat['orderid'].map(order_eloc_dict))
        feat['sloc_eloc_2count'] = feat['sloc_eloc_count'] + feat['eloc_sloc_count']
        feat['sloc_eloc_goback_rate'] = feat['sloc_eloc_count'] / (feat['eloc_sloc_count'] + 0.001)
        distance = get_distance(feat)
        feat = pd.concat([feat, distance], axis=1)
        feat = feat[['sloc_eloc_count', 'eloc_sloc_count', 'sloc_eloc_2count',
                     'sloc_eloc_goback_rate', 'distance', 'mht_distance']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat


#####################按照时间统计分析#########################
# 时间×出发地
def get_hour_sloc(data, sample, data_key):
    feat_path = cache_path + 'hour_sloc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        data_temp = data.copy()
        loc_dict = get_loc_dict()
        data_temp['lat'] = data_temp['geohashed_end_loc'].apply(lambda x: np.nan if x is np.nan else loc_dict[x][0])
        data_temp['lon'] = data_temp['geohashed_end_loc'].apply(lambda x: np.nan if x is np.nan else loc_dict[x][1])
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

#####################扩大时间统计分析#########################
# 时间×出发地
def get_exhour_sloc(data, sample, data_key):
    feat_path = cache_path + 'exhour_sloc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        data_temp = data.copy()
        loc_dict = get_loc_dict()
        data_temp['lat'] = data_temp['geohashed_end_loc'].apply(lambda x: np.nan if x is np.nan else loc_dict[x][0])
        data_temp['lon'] = data_temp['geohashed_end_loc'].apply(lambda x: np.nan if x is np.nan else loc_dict[x][1])
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
        feat = pd.merge(sample, exhour_sloc_eloc_count, on=['hour', 'geohashed_start_loc', 'geohashed_end_loc'],
                        how='left').fillna(0)
        order_eloc_dict = dict(zip(data['orderid'].values, data['geohashed_end_loc'].values))
        feat['exhour_sloc_eloc_count'] = feat['exhour_sloc_eloc_count'] - (feat['geohashed_end_loc'] == feat['orderid'].map(order_eloc_dict))
        feat = feat[['exhour_sloc_eloc_count']]
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
        feat['geohashed_end_loc'] = feat['near_loc']
        feat = feat.groupby('geohashed_end_loc', as_index=False).sum()
        feat = pd.merge(sample, feat, on='geohashed_end_loc', how='left').fillna(0)
        feat['ex_eloc_count'] = feat['ex_eloc_count'] - if_around(feat, data, 'geohashed_end_loc')
        feat['ex_eloc_2count'] = feat['ex_eloc_count'] + feat['ex_eloc_as_sloc_count']
        feat['ex_eloc_in_out_rate'] = feat['ex_eloc_count'] / (feat['ex_eloc_as_sloc_count'] + 0.001)
        feat = feat[['ex_eloc_as_sloc_count', 'ex_eloc_count',
                     'ex_eloc_2count', 'ex_eloc_in_out_rate']]
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
        feat = pd.merge(sloc_eloc, eloc_sloc, on=['geohashed_start_loc', 'geohashed_end_loc'], how='outer')
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

#####################节假日#####################

# 节假日×时间
def get_holiday_hour(data, sample, data_key):
    feat_path = cache_path + 'holiday_hour_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        feat = data.groupby(['hour','holiday'],as_index=False)['userid'].agg({'holiday_hour_count': 'count'})
        feat = sample.merge(feat,on=['hour','holiday'],how='left').fillna(0)
        feat['holiday_hour_count'] = feat['holiday_hour_count'] - feat['orderid'].isin(data['orderid'].values)
        feat = feat[['holiday_hour_count']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat

# 节假日×时间×目的地
def get_holiday_eloc(data, sample, data_key):
    feat_path = cache_path + 'holiday_eloc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        feat = data.groupby(['holiday','geohashed_end_loc'],as_index=False)['userid'].agg({'holiday_eloc_count': 'count'})
        feat = sample.merge(feat,on=['holiday','geohashed_end_loc'],how='left')
        order_eloc_dict = dict(zip(data['orderid'].values, data['geohashed_end_loc'].values))
        feat['holiday_eloc_count'] = feat['holiday_eloc_count'] - (feat['geohashed_end_loc'] == feat['orderid'].map(order_eloc_dict))
        feat = feat[['holiday_eloc_count']]
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
        order_eloc_dict = dict(zip(data['orderid'].values, data['geohashed_end_loc'].values))
        feat['holiday_hour_eloc_count'] = feat['holiday_hour_eloc_count'] - (feat['geohashed_end_loc'] == feat['orderid'].map(order_eloc_dict))
        feat['holiday_hour_sloc_count'] = feat['holiday_hour_sloc_count'] - (feat['orderid'].isin(data['orderid'].values))
        feat['holiday_hour_sloc_in_out_rate'] = feat['holiday_hour_eloc_count'] / (feat['holiday_hour_sloc_count']+0.001)
        feat = feat[['holiday_hour_eloc_count','holiday_hour_sloc_in_out_rate']]
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

# 节假日×时间×扩大出发地
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
        feat = pd.merge(sample, feat, on='geohashed_start_loc', how='left').fillna(0)
        feat['holiday_exhour_ex_sloc_count'] = feat['holiday_exhour_ex_sloc_count'] - feat['orderid'].isin(data['orderid'].values)
        feat = feat[['holiday_exhour_ex_sloc_count']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat

# 节假日×扩大时间×扩大出发地
def get_holiday_hour_ex_sloc(data, sample, data_key):
    feat_path = cache_path + 'holiday_hour_ex_sloc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        feat = data.groupby(['hour','holiday','geohashed_start_loc'],as_index=False)['userid'].agg({'holiday_hour_ex_sloc_count': 'count'})

        near_loc = get_near_loc()
        feat = feat.merge(near_loc, left_on='geohashed_start_loc', right_on='loc', how='left')
        feat['geohashed_start_loc'] = feat['near_loc']
        feat = feat.groupby(['hour','holiday','geohashed_start_loc'], as_index=False).sum()
        feat = pd.merge(sample, feat, on=['hour','holiday','geohashed_start_loc'], how='left').fillna(0)
        feat['holiday_hour_ex_sloc_count'] = feat['holiday_hour_ex_sloc_count'] - feat['orderid'].isin(data['orderid'].values)
        feat = feat[['holiday_hour_ex_sloc_count']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat

# 节假日×时间×出发地×目的地
def get_holiday_hour_sloc_eloc(data, sample, data_key):
    feat_path = cache_path + 'holiday_hour_sloc_eloc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        feat = data.groupby(['hour','holiday','geohashed_start_loc','geohashed_end_loc'],as_index=False)['userid'].agg({'holiday_hour_sloc_eloc_count': 'count'})
        feat = sample.merge(feat,on=['hour','holiday','geohashed_start_loc','geohashed_end_loc'],how='left').fillna(0)
        order_eloc_dict = dict(zip(data['orderid'].values, data['geohashed_end_loc'].values))
        feat['holiday_hour_sloc_eloc_count'] = feat['holiday_hour_sloc_eloc_count'] - (feat['geohashed_end_loc'] == feat['orderid'].map(order_eloc_dict))
        feat = feat[['holiday_hour_sloc_eloc_count']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat

# 节假日×时间×扩大出发地×目的地
def get_holiday_hour_ex_sloc_eloc(data, sample, data_key):
    feat_path = cache_path + 'holiday_hour_ex_sloc_eloc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        feat = data.groupby(['hour','holiday','geohashed_start_loc','geohashed_end_loc'],as_index=False)['userid'].agg({'holiday_hour_ex_sloc_eloc_count': 'count'})
        near_loc = get_near_loc()
        feat = feat.merge(near_loc, left_on='geohashed_start_loc', right_on='loc', how='left')
        feat['geohashed_start_loc'] = feat['near_loc']
        feat = feat.groupby(['hour', 'holiday', 'geohashed_start_loc','geohashed_end_loc'], as_index=False).sum()
        feat = sample.merge(feat,on=['hour','holiday','geohashed_start_loc','geohashed_end_loc'],how='left').fillna(0)
        order_eloc_dict = dict(zip(data['orderid'].values, data['geohashed_end_loc'].values))
        feat['holiday_hour_ex_sloc_eloc_count'] = feat['holiday_hour_ex_sloc_eloc_count'] - (feat['geohashed_end_loc'] == feat['orderid'].map(order_eloc_dict))
        feat = feat[['holiday_hour_ex_sloc_eloc_count']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat

# 节假日×扩大时间×扩大出发地×目的地
def get_holiday_exhour_ex_sloc_eloc(data, sample, data_key):
    feat_path = cache_path + 'holiday_exhour_ex_sloc_eloc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        feat = data.groupby(['hour','holiday','geohashed_start_loc','geohashed_end_loc'],as_index=False)['userid'].agg({'holiday_exhour_ex_sloc_eloc_count': 'count'})
        near_hour = get_near_hour()
        feat = feat.merge(near_hour, on='hour', how='left')
        feat['hour'] = feat['near_hour']
        feat = feat.groupby(['hour', 'holiday', 'geohashed_start_loc','geohashed_end_loc'], as_index=False).sum()
        near_loc = get_near_loc()
        feat = feat.merge(near_loc, left_on='geohashed_start_loc', right_on='loc', how='left')
        feat['geohashed_start_loc'] = feat['near_loc']
        feat = feat.groupby(['hour', 'holiday', 'geohashed_start_loc','geohashed_end_loc'], as_index=False).sum()
        feat = sample.merge(feat,on=['hour','holiday','geohashed_start_loc','geohashed_end_loc'],how='left').fillna(0)
        order_eloc_dict = dict(zip(data['orderid'].values, data['geohashed_end_loc'].values))
        feat['holiday_exhour_ex_sloc_eloc_count'] = feat['holiday_exhour_ex_sloc_eloc_count'] - (feat['geohashed_end_loc'] == feat['orderid'].map(order_eloc_dict))
        feat = feat[['holiday_exhour_ex_sloc_eloc_count']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat


# 区域
def get_exloc(data, sample, data_key):
    feat_path = cache_path + 'exloc_%d.hdf' % (data_key)
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        data_temp = data.copy()
        sample_temp = sample.copy()
        data_temp.loc[:, 'exloc'] = data_temp['geohashed_start_loc'].str[:5]
        sample_temp.loc[:, 'exloc'] = sample_temp['geohashed_start_loc'].str[:5]
        exloc_count = data_temp.groupby('exloc', as_index=False)['userid'].agg(
            {'exloc_count': 'count'})
        holiday_exloc_count = data_temp.groupby(['holiday','exloc'],as_index=False)['userid'].agg({
            'holiday_exloc_count': 'count'})
        hour_exloc_count = data_temp.groupby(['hour', 'exloc'], as_index=False)['userid'].agg({
            'hour_exloc_count': 'count'})
        holiday_hour_exloc_count = data_temp.groupby(['holiday', 'hour', 'exloc'], as_index=False)['userid'].agg({
            'holiday_hour_exloc_count': 'count'})
        feat = sample_temp.merge(exloc_count, on=['exloc'], how='left').fillna(0)
        feat = feat.merge(holiday_exloc_count,on=['holiday','exloc'],how='left').fillna(0)
        feat = feat.merge(hour_exloc_count, on=['hour', 'exloc'], how='left').fillna(0)
        feat = feat.merge(holiday_hour_exloc_count, on=['holiday', 'hour', 'exloc'], how='left').fillna(0)
        feat['exloc_count'] = feat['exloc_count'] - feat['orderid'].isin(data_temp['orderid'].values)
        feat['holiday_exloc_count'] = feat['holiday_exloc_count'] - feat['orderid'].isin(data_temp['orderid'].values)
        feat['hour_exloc_count'] = feat['hour_exloc_count'] - feat['orderid'].isin(data_temp['orderid'].values)
        feat['holiday_hour_exloc_count'] = feat['holiday_hour_exloc_count'] - feat['orderid'].isin(data_temp['orderid'].values)
        feat = feat[['exloc_count','holiday_exloc_count','hour_exloc_count','holiday_hour_exloc_count']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat


######################## 二次处理特征 #####################
def second_feat(data, result):
    order_eloc_dict = dict(zip(data['orderid'].values, data['geohashed_end_loc'].values))
    order_sloc_dict = dict(zip(data['orderid'].values, data['geohashed_start_loc'].values))
    result['eloc_count_ex_rate'] =  result['eloc_count'] / result['ex_eloc_count']
    result['eloc_count_sloc_rate'] = result['eloc_count'] / (result['sloc_count'] + 0.001)
    result['eloc_count_seloc_9_7_rate'] = result['eloc_count'] / (result['seloc_count_9_7'] + 0.001)
    result['eloc_count_eeloc_9_7_rate'] = result['eloc_count'] / (result['eeloc_count_9_7'] + 0.001)
    result['sloc_eloc_count_sloc_rate'] = result['sloc_eloc_count'] / (result['sloc_count'] + 0.001)
    result['hour_sloc_eloc_count_hs_rate'] = result['hour_sloc_eloc_count'] / (result['hour_sloc_count'] + 0.001)
    result['hour_eloc_count_eloc_rate'] = result['hour_eloc_count'] / (result['eloc_count'] + 0.001)
    result['exhour_eloc_count_eloc_rate'] = result['exhour_eloc_count'] / (result['eloc_count'] + 0.001)
    result['exhour_sloc_eloc_count_exhour_sloc_rate'] = result['exhour_sloc_eloc_count'] / (result['exhour_sloc_count'] + 0.001)
    result['ex_sloc_eloc_count_ex_sloc_rate'] = result['ex_sloc_eloc_count'] / (result['ex_sloc_count'] + 0.001)
    result['holiday_hour_eloc_count_holiday_hour_rate'] = result['holiday_hour_eloc_count'] / (result['holiday_hour_count'] + 0.001)
    result['holiday_hour_eloc_sloc_count_holiday_hour_rate'] = result['holiday_hour_sloc_eloc_count'] / (result['holiday_hour_sloc_count'] + 0.001)
    result['holiday_hour_ex_eloc_sloc_count_holiday_hour_rate'] = result['holiday_hour_ex_sloc_eloc_count'] / (
    result['holiday_hour_ex_sloc_count'] + 0.001)
    result['holiday_exhour_ex_eloc_sloc_count_holiday_exhour_rate'] = result['holiday_exhour_ex_sloc_eloc_count'] / (
        result['holiday_exhour_ex_sloc_count'] + 0.001)
    result['eloc_count_exloc_rate'] = result['eloc_count'] / (result['exloc_count'] + 0.001)
    result['hour_eloc_count_exloc_rate'] = result['hour_eloc_count'] / (result['hour_exloc_count'] + 0.001)
    result['holiday_eloc_count_exloc_rate'] = result['holiday_eloc_count'] / (result['holiday_exloc_count'] + 0.001)
    result['holiday_hour_eloc_count_holiday_hour_exloc_rate'] = result['holiday_hour_eloc_count'] / (result['holiday_hour_exloc_count'] + 0.001)
    return result


# 构造样本
def get_sample(data, candidate, data_key):
    result_path = cache_path + 'sample_%d.hdf' % (data_key)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        loc_to_loc = get_loc_to_loc(data, candidate, data_key)                      # 筛选起始地点去向最多的3个地点
        loc_with_loc = get_loc_with_loc(data, candidate, data_key)                  # 与起始地点交互最多的三个地点
        ex_loc_with_loc = get_ex_loc_with_loc(data, candidate, data_key)            # 扩大与起始地点交互最多的三个地点
        ex_loc_to_loc = get_ex_loc_to_loc(data, candidate, data_key)                # 扩大筛选起始地点去向最多的3个地点
        holiday_loc_to_loc = get_holiday_loc_to_loc(data, candidate, data_key)      # 是否工作日去向最多的三个地点
        exhour_loc_to_loc = get_exhour_loc_to_loc(data, candidate, data_key)        # 三个小时内去向最多的三个地点
        exhour_ex_loc_to_loc = get_exhour_ex_loc_to_loc(data, candidate, data_key)  # 扩大范围三个小时内去向最多的三个地点
        holiday_exhour_loc_to_loc = get_holiday_exhour_loc_to_loc(data, candidate, data_key)# 交易日 三个小时去向最多的地点
        holiday_ex_loc_to_loc = get_holiday_ex_loc_to_loc(data, candidate, data_key)# 交易日 扩大范围最多的三个地点
        holiday_exhour_ex_loc_to_loc = get_holiday_exhour_ex_loc_to_loc(data, candidate, data_key)# 交易日 三个小时 扩大范围去向最多的三个地点
        near_loc_to_loc = get_near_loc_to_loc(data, candidate, data_key)           # 周围9×7中的前3

        # 汇总样本id
        result = pd.concat([loc_to_loc[['orderid', 'geohashed_end_loc']],
                            loc_with_loc[['orderid', 'geohashed_end_loc']],
                            ex_loc_to_loc[['orderid', 'geohashed_end_loc']],
                            ex_loc_with_loc[['orderid', 'geohashed_end_loc']],
                            holiday_loc_to_loc[['orderid', 'geohashed_end_loc']],
                            exhour_loc_to_loc[['orderid', 'geohashed_end_loc']],
                            exhour_ex_loc_to_loc[['orderid', 'geohashed_end_loc']],
                            holiday_exhour_loc_to_loc[['orderid', 'geohashed_end_loc']],
                            holiday_ex_loc_to_loc[['orderid', 'geohashed_end_loc']],
                            holiday_exhour_ex_loc_to_loc[['orderid', 'geohashed_end_loc']],
                            near_loc_to_loc[['orderid', 'geohashed_end_loc']]
                            ]).drop_duplicates()
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
        order_time = get_order_feat(data, sample, data_key)                           # 获取order时间特征
        start_loc = get_start_loc(data, sample, data_key)                             # 获取出发地热度（作为出发地的次数、人数）
        end_loc = get_end_loc(data, sample, data_key)                                 # 获取目的地热度（作为目的地的个数、人数，作为出发地的个数、人数，折返比）
        sloc_eloc = get_sloc_eloc(data, sample, data_key)                             # 出发地×目的地
        gc.collect()

        # 按时间统计
        hour_sloc = get_hour_sloc(data,sample, data_key)                              # 时间×出发地
        hour_eloc = get_hour_eloc(data,sample, data_key)                              # 时间×目的地
        hour_sloc_eloc = get_hour_sloc_eloc(data, sample, data_key)                   # 时间×出发地×目的地
        gc.collect()

        #扩大时间范围
        exhour_sloc = get_exhour_sloc(data, sample, data_key)                         # 时间×出发地
        exhour_eloc = get_exhour_eloc(data, sample, data_key)                         # 时间×目的地
        exhour_sloc_eloc = get_exhour_sloc_eloc(data, sample, data_key)               # 时间×出发地×目的地
        gc.collect()

        # 扩大统计范围重新统计
        ex_sloc = get_ex_sloc(data, sample, data_key)                                 # 扩大出发地
        ex_eloc = get_ex_eloc(data, sample, data_key)                                 # 扩大目的地
        ex_sloc_eloc = get_ex_sloc_eloc(data, sample, data_key)                       # 出发地×目的地
        gc.collect()

        # 工作日非工作日
        holiday_hour = get_holiday_hour(data, sample, data_key)                       # 工作日×时间
        holiday_eloc = get_holiday_eloc(data, sample, data_key)                       # 工作日×时间×目的地
        holiday_hour_eloc = get_holiday_hour_eloc(data, sample, data_key)             # 工作日×时间×目的地
        holiday_hour_sloc = get_holiday_hour_sloc(data, sample, data_key)             # 工作日×时间×出发地
        holiday_hour_ex_sloc = get_holiday_hour_ex_sloc(data, sample, data_key)       # 工作日×时间×扩大出发地
        holiday_exhour_ex_sloc = get_holiday_exhour_ex_sloc(data, sample, data_key)   # 工作日×扩大时间×扩大出发地
        holiday_hour_sloc_eloc = get_holiday_hour_sloc_eloc(data, sample, data_key)   # 工作日×时间×出发地×目的地
        holiday_hour_ex_sloc_eloc = get_holiday_hour_ex_sloc_eloc(data, sample, data_key)# 工作日×时间×扩大出发地×目的地
        holiday_exhour_ex_sloc_eloc = get_holiday_exhour_ex_sloc_eloc(data, sample, data_key)# 工作日×扩大时间×扩大出发地×目的地

        # 区域特征
        exloc = get_exloc(data, sample, data_key)  # 区域

        print('开始合并特征...')

        result = concat([sample, start_loc, end_loc, sloc_eloc, ex_sloc, ex_eloc,
                         ex_sloc_eloc,hour_sloc,hour_eloc,hour_sloc_eloc,
                          exhour_sloc, exhour_eloc,exhour_sloc_eloc, order_time,
                         holiday_hour,holiday_hour_eloc,holiday_hour_sloc,holiday_hour_ex_sloc,
                         holiday_hour_sloc_eloc,holiday_hour_ex_sloc_eloc,holiday_exhour_ex_sloc,
                         holiday_exhour_ex_sloc_eloc,holiday_eloc,exloc
                         ])
        del sample, start_loc, end_loc, sloc_eloc, ex_sloc, ex_eloc, ex_sloc_eloc,\
            hour_sloc, hour_eloc,hour_sloc_eloc,exhour_sloc, exhour_eloc, exhour_sloc_eloc, \
            order_time,holiday_hour,holiday_hour_eloc,holiday_hour_sloc,holiday_hour_ex_sloc,\
            holiday_hour_sloc_eloc,holiday_hour_ex_sloc_eloc,holiday_exhour_ex_sloc,\
            holiday_exhour_ex_sloc_eloc,holiday_eloc,exloc
        gc.collect()
        result = second_feat(data, result)
        gc.collect()

        print('添加label')
        result = get_label(result)

        print('存储数据...')
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    print('特征矩阵大小：{}'.format(result.shape))
    print('生成特征一共用时{}秒'.format(time.time() - t0))
    return result


# 线下测试集2
train = pd.read_csv(train_path)
train1 = train[(train['starttime'] < '2017-05-14 00:00:00')].copy()
train2 = train[(train['starttime'] >= '2017-05-14 00:00:00') & (train['starttime'] < '2017-05-16 00:00:00')].copy()
train2.loc[:, 'geohashed_end_loc'] = np.nan
train3 = train[(train['starttime'] >= '2017-05-16 00:00:00')].copy()
test = pd.read_csv(test_path)
test.loc[:, 'geohashed_end_loc'] = np.nan
data = pd.concat([train1, train2, train3, test])
del train, test
train_data = data[(data['starttime'] >= '2017-05-13 00:00:00') & ((data['starttime'] < '2017-05-14 00:00:00')) |
                  (data['starttime'] >= '2017-05-16 00:00:00') & ((data['starttime'] < '2017-05-17 00:00:00'))].copy()
train_orderid_sample = sample(train_data.orderid.tolist(),n_sub=6,seed=66)[0]
train_data = train_data[train_data['orderid'].isin(train_orderid_sample)]
train_feat = make_train_set(data, train_data).fillna(-1)
test_data = data[(data['starttime'] >= '2017-05-14 00:00:00') & ((data['starttime'] < '2017-05-16 00:00:00'))].copy()
test_orderid_sample = sample(test_data.orderid.tolist(),n_sub=6,seed=66)[0]
test_data = test_data[test_data['orderid'].isin(test_orderid_sample)]
test_feat = make_train_set(data, test_data).fillna(-1)
test_pred = test_feat[['orderid','geohashed_end_loc']].copy()
predictors = train_feat.columns.drop(['orderid', 'geohashed_end_loc', 'label', 'userid',
                                      'bikeid', 'starttime', 'geohashed_start_loc'])

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'max_depth': 6,
    'num_leaves': 80,
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
test_pred['pred'] = preds
result = reshape(test_pred)
result.fillna('a',inplace=True)
result = pd.merge(test_data[['orderid']], result, on='orderid', how='left')
print('map得分为:{}'.format(map(result)))




