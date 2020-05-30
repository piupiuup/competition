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

cache_path = 'F:/mobike_cache2/'
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

# 组内标准化
def group_normalize(data,key,feat):
    grp = data.groupby(key,as_index=False)[feat].agg({'std':'std','avg':'mean'})
    result = pd.merge(data,grp,on=key,how='left')
    result[feat] = ((result[feat]-result['avg']) / result['std']).fillna(1)
    return result[feat]

# 对时间进行分区
def split_time(time):
    t = time[:11]
    if t<'05:00:00':
        return 0
    elif t<'09:30:00':
        return 1
    elif t<'16:00:00':
        return 2
    elif t<'21:00:00':
        return 3
    else:
        return 0


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
            deloc.append(Geohash.decode(loc))
        loc_dict = dict(zip(locs, deloc))
        pickle.dump(loc_dict, open(dump_path, 'wb+'))
    return loc_dict

# 计算两点之间距离
def cal_distance(lat1,lon1,lat2,lon2):
    dx = np.abs(lon1 - lon2)  # 经度差
    dy = np.abs(lat1 - lat2)  # 维度差
    b = (lat1 + lat2) / 2.0
    Lx = 6371004.0 * (dx / 57.2958) * np.cos(b / 57.2958)
    Ly = 6371004.0 * (dy / 57.2958)
    L = (Lx**2 + Ly**2) ** 0.5
    return L
def cal_distance(loc1,loc2):
    if (loc1 is None) | (loc2 is None):
        return np.nan
    loc_dict = get_loc_dict()
    lat1, lon1 = loc_dict(loc1)
    lat2, lon2 = loc_dict(loc2)
    L = cal_distance(lat1,lon1,lat2,lon2)
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

# 分组排序
def rank(data, feat1, feat2, ascending):
    data.sort_values([feat1,feat2],inplace=True,ascending=ascending)
    data['rank'] = range(data.shape[0])
    min_rank = data.groupby(feat1,as_index=False)['rank'].agg({'min_rank':'min'})
    data = pd.merge(data,min_rank,on=feat1,how='left')
    data['rank'] = data['rank'] - data['min_rank']
    del data['min_rank']
    return data

# 计算两点之间的欧氏距离和曼哈顿距离
def get_distance(sample):
    result = sample.copy()
    locs = list(set(result['geohashed_start_loc']) | set(result['geohashed_end_loc']))
    if np.nan in locs:
        locs.remove(np.nan)
    deloc = []
    for loc in locs:
        deloc.append(Geohash.decode(loc))
    loc_dict = dict(zip(locs,deloc))
    geohashed_loc = result[['geohashed_start_loc','geohashed_end_loc']].values
    distance = []
    mht_distance = []
    for i in geohashed_loc:
        lat1, lon1 = loc_dict[i[0]]
        lat2, lon2 = loc_dict[i[1]]
        distance.append(cal_distance(lat1,lon1,lat2,lon2))
        mht_distance.append(cal_mht_distance(lat1,lon1,lat2,lon2))
    result['distance'] = distance
    result['mht_distance'] = mht_distance
    result = result[['distance','mht_distance']]
    return result

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

# 热点目的地
def get_loc_matrix():
    result_path = cache_path + 'loc_matrix.hdf'
    if os.path.exists(result_path):
        result = pd.read_hdf(result_path, 'w')
    else:
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        end_loc = pd.DataFrame({'geohashed_end_loc':list(train['geohashed_end_loc'].unique())})
        end_loc['end_loc_lat'] = end_loc['geohashed_end_loc'].apply(lambda x: Geohash.decode(x)[0])
        end_loc['end_loc_lon'] = end_loc['geohashed_end_loc'].apply(lambda x: Geohash.decode(x)[1])
        end_loc['end_loc_lat_box'] = end_loc['end_loc_lat'].apply(lambda x: x//0.003)
        end_loc['end_loc_lon_box'] = end_loc['end_loc_lon'].apply(lambda x: x//0.00375)
        count_of_loc = train.groupby('geohashed_end_loc',as_index=False)['geohashed_end_loc'].agg({'count_of_loc':'count'})
        end_loc = pd.merge(end_loc,count_of_loc,on='geohashed_end_loc',how='left')
        max_index = end_loc.groupby(['end_loc_lat_box','end_loc_lon_box']).apply(lambda x: x['count_of_loc'].argmax())
        end_loc = end_loc.loc[max_index.tolist(),['geohashed_end_loc', 'end_loc_lat', 'end_loc_lon']]
        end_loc.sort_values('end_loc_lat',inplace=True)
        end_loc = end_loc.values
        start_loc = pd.DataFrame({'geohashed_start_loc': list(pd.concat([train,test])['geohashed_start_loc'].unique())})
        start_loc['start_loc_lat'] = start_loc['geohashed_start_loc'].apply(lambda x: Geohash.decode(x)[0])
        start_loc['start_loc_lon'] = start_loc['geohashed_start_loc'].apply(lambda x: Geohash.decode(x)[1])
        start_loc = start_loc.values
        start_end_loc_arr = []
        for i in start_loc:
            for j in end_loc:
                if (np.abs(i[1]-j[1])<0.012) & (np.abs(i[2]-j[2])<0.015):
                    start_end_loc_arr.append([i[0],j[0]])
        result = pd.DataFrame(start_end_loc_arr,columns=['geohashed_start_loc','geohashed_end_loc'])
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result



####################构造负样本##################

# 将用户骑行过目的的地点加入成样本
def get_user_end_loc(data,candidate):
    n = sum(candidate['geohashed_end_loc'].isnull())+sum(data['geohashed_end_loc'].isnull())
    result_path = cache_path + 'user_end_loc_%d.hdf' %(data['orderid'].sum()*candidate['orderid'].sum()*n)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        candidate_temp = candidate.copy()
        candidate_temp.rename(columns={'geohashed_end_loc':'true_loc'},inplace=True)
        user_n_end_loc = data.groupby(['userid','geohashed_end_loc'],as_index=False)['userid'].agg({'user_eloc_count':'count'})
        user_n_end_loc = user_n_end_loc[~user_n_end_loc['geohashed_end_loc'].isnull()]
        result = pd.merge(candidate_temp[['orderid','userid','true_loc']],user_n_end_loc,on=['userid'],how='left')
        result['user_eloc_count'] = result['user_eloc_count'] - (result['geohashed_end_loc']==result['true_loc'])
        result = result[result['user_eloc_count'] > 0]
        result = result[['orderid', 'geohashed_end_loc']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 将用户骑行过出发的地点加入成样本
def get_user_start_loc(data,candidate):
    n = sum(candidate['geohashed_end_loc'].isnull())+sum(data['geohashed_end_loc'].isnull())
    result_path = cache_path + 'user_start_loc_%d.hdf' %(data['orderid'].sum()*candidate['orderid'].sum()*n)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        candidate_temp = candidate.copy()
        candidate_temp.rename(columns={'geohashed_start_loc': 'true_loc'}, inplace=True)
        user_n_start_loc = data.groupby(['userid', 'geohashed_start_loc'], as_index=False)['userid'].agg(
            {'user_sloc_count': 'count'})
        result = pd.merge(candidate_temp[['orderid', 'userid', 'true_loc']], user_n_start_loc, on=['userid'], how='left')
        result['user_sloc_count'] = result['user_sloc_count'] - (result['geohashed_start_loc'] == result['true_loc'])
        result = result[result['user_sloc_count'] > 0]
        result.rename(columns={'geohashed_start_loc': 'geohashed_end_loc'}, inplace=True)
        result = result[['orderid', 'geohashed_end_loc']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 筛选起始地点去向最多的3个地点
def get_loc_to_loc(data,candidate):
    n = sum(candidate['geohashed_end_loc'].isnull())+sum(data['geohashed_end_loc'].isnull())
    result_path = cache_path + 'loc_to_loc_%d.hdf' %(data['orderid'].sum()*candidate['orderid'].sum()*n)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        candidate_temp = candidate.copy()
        candidate_temp.rename(columns={'geohashed_end_loc':'true_loc'},inplace=True)
        sloc_eloc_count = data.groupby(['geohashed_start_loc','geohashed_end_loc'],as_index=False)['orderid'].agg({'sloc_eloc_count':'count'})
        sloc_eloc_count = sloc_eloc_count[~sloc_eloc_count['geohashed_end_loc'].isnull()]
        sloc_eloc_count.sort_values('sloc_eloc_count',inplace=True)
        sloc_eloc_count = sloc_eloc_count.groupby('geohashed_start_loc').tail(4)
        result = pd.merge(candidate_temp[['orderid', 'geohashed_start_loc', 'true_loc']],sloc_eloc_count,on='geohashed_start_loc',how='left')
        result['sloc_eloc_count'] = result['sloc_eloc_count'] - (result['geohashed_end_loc']==result['true_loc'])
        result.sort_values('sloc_eloc_count', inplace=True)
        result = result.groupby('orderid').tail(3)
        result = result[result['sloc_eloc_count'] > 0]
        result = result[['orderid', 'geohashed_end_loc']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 与其交互最多的三个地点
def get_loc_with_loc(data,candidate):
    n = sum(candidate['geohashed_end_loc'].isnull())+sum(data['geohashed_end_loc'].isnull())
    result_path = cache_path + 'loc_with_loc_%d.hdf' % (data['orderid'].sum() * candidate['orderid'].sum()*n)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        candidate_temp = candidate.copy()
        candidate_temp.rename(columns={'geohashed_end_loc': 'true_loc'}, inplace=True)
        sloc_eloc_count = data.groupby(['geohashed_start_loc', 'geohashed_end_loc'], as_index=False)['orderid'].agg(
            {'sloc_eloc_count': 'count'})
        sloc_eloc_count = sloc_eloc_count[~sloc_eloc_count['geohashed_end_loc'].isnull()]
        eloc_sloc_count = sloc_eloc_count.rename(columns={'geohashed_start_loc':'geohashed_end_loc',
                                                        'geohashed_end_loc':'geohashed_start_loc',
                                                        'sloc_eloc_count':'eloc_sloc_count'}).copy()
        sloc_eloc_2count = pd.merge(sloc_eloc_count,eloc_sloc_count,
                                      on=['geohashed_start_loc','geohashed_end_loc'],how='outer').fillna(0)
        sloc_eloc_2count['sloc_eloc_2count'] = sloc_eloc_2count['sloc_eloc_count'] + sloc_eloc_2count['eloc_sloc_count']
        sloc_eloc_2count.sort_values('sloc_eloc_2count', inplace=True)
        sloc_eloc_2count = sloc_eloc_2count.groupby('geohashed_start_loc').tail(4)
        result = pd.merge(candidate_temp[['orderid', 'geohashed_start_loc', 'true_loc']], sloc_eloc_2count,
                          on='geohashed_start_loc', how='left')
        result['sloc_eloc_2count'] = result['sloc_eloc_2count'] - (result['geohashed_end_loc'] == result['true_loc'])
        result.sort_values('sloc_eloc_2count', inplace=True)
        result = result.groupby('orderid').tail(3)
        result = result[result['sloc_eloc_2count']>0]
        result = result[['orderid', 'geohashed_end_loc']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 筛选起始地点周围流入量最大的3个地点
def get_loc_near_loc(data,candidate):
    n = sum(candidate['geohashed_end_loc'].isnull())+sum(data['geohashed_end_loc'].isnull())
    result_path = cache_path + 'loc_near_loc_%d.hdf' %(data['orderid'].sum()*candidate['orderid'].sum()*n)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        candidate_temp = candidate.copy()
        candidate_temp.rename(columns={'geohashed_end_loc':'true_loc'},inplace=True)
        loc_matrix = get_loc_matrix()
        loc_input = data.groupby('geohashed_end_loc',as_index=False)['geohashed_end_loc'].agg({'eloc_n_input':'count'})
        loc_matrix = pd.merge(loc_matrix,loc_input,on='geohashed_end_loc',how='left')
        loc_matrix = loc_matrix[~loc_matrix['eloc_n_input'].isnull()]
        loc_matrix.sort_values('eloc_n_input',inplace=True)
        loc_matrix = loc_matrix.groupby('geohashed_start_loc').tail(4)
        result = pd.merge(candidate_temp[['orderid','geohashed_start_loc','true_loc']],loc_matrix,on='geohashed_start_loc',how='left')
        result['eloc_n_input'] = result['eloc_n_input'] - (result['geohashed_end_loc']==result['true_loc'])
        result.sort_values('eloc_n_input', inplace=True)
        result = result.groupby('orderid').tail(3)
        result = result[result['eloc_n_input'] > 0]
        result = result[['orderid','geohashed_end_loc']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 用户后续的起始地点
def get_user_next_loc(data,candidate):
    n = sum(candidate['geohashed_end_loc'].isnull())+sum(data['geohashed_end_loc'].isnull())
    result_path = cache_path + 'user_next_loc_%d.hdf' %(data['orderid'].sum()*candidate['orderid'].sum()*n)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        data_temp = data.copy()
        data_temp = data_temp[data_temp['userid'].isin(candidate['userid'].values)]
        if data_temp.shape[0] == 0:
            return pd.DataFrame(columns=['orderid','geohashed_end_loc','user_eloc_nloc_sep_time'])
        data_temp = rank(data_temp,'userid','starttime',ascending=True)
        data_temp_temp = data_temp.copy()
        data_temp_temp['rank'] = data_temp_temp['rank']-1
        result = pd.merge(data_temp[['orderid','userid','rank','starttime']],
                          data_temp_temp[['userid','rank','geohashed_start_loc','starttime']],on=['userid','rank'],how='inner')
        result['user_eloc_nloc_sep_time'] = result.apply(lambda x: diff_of_minutes(x['starttime_y'],x['starttime_x']),axis=1)
        result.rename(columns={'geohashed_start_loc':'geohashed_end_loc'},inplace=True)
        result = result[['orderid','geohashed_end_loc','user_eloc_nloc_sep_time']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# bike后续的起始地点
def get_bike_next_loc(data,candidate):
    n = sum(candidate['geohashed_end_loc'].isnull())+sum(data['geohashed_end_loc'].isnull())
    result_path = cache_path + 'bike_next_loc_%d.hdf' %(data['orderid'].sum()*candidate['orderid'].sum()*n)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        data_temp = data.copy()
        data_temp = data_temp[data_temp['bikeid'].isin(candidate['bikeid'].values)]
        if data_temp.shape[0] == 0:
            return pd.DataFrame(columns=['orderid', 'geohashed_end_loc', 'bike_eloc_sep_time'])
        data_temp = rank(data_temp,'bikeid','starttime',ascending=True)
        data_temp_temp = data_temp.copy()
        data_temp_temp['rank'] = data_temp_temp['rank'] - 1
        result = pd.merge(data_temp[['orderid','bikeid','rank','starttime']],
                          data_temp_temp[['bikeid','rank','geohashed_start_loc','starttime']],on=['bikeid','rank'],how='inner')
        result['bike_eloc_sep_time'] = result.apply(lambda x: diff_of_minutes(x['starttime_y'],x['starttime_x']),axis=1)
        result.rename(columns={'geohashed_start_loc':'geohashed_end_loc'},inplace=True)
        result = result[['orderid','geohashed_end_loc','bike_eloc_sep_time']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 用户前一次的起始地点
def get_user_forward_loc(data,candidate):
    n = sum(candidate['geohashed_end_loc'].isnull())+sum(data['geohashed_end_loc'].isnull())
    result_path = cache_path + 'user_forward_loc_%d.hdf' %(data['orderid'].sum()*candidate['orderid'].sum()*n)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        data_temp = data.copy()
        data_temp = data_temp[data_temp['userid'].isin(candidate['userid'].values)]
        if data_temp.shape[0] == 0:
            return pd.DataFrame(columns=['orderid','geohashed_end_loc','user_eloc_nloc_sep_time'])
        data_temp = rank(data_temp,'userid','starttime',ascending=True)
        data_temp_temp = data_temp.copy()
        data_temp_temp['rank'] = data_temp_temp['rank']+1
        result = pd.merge(data_temp[['orderid','userid','rank','starttime']],
                          data_temp_temp[['userid','rank','geohashed_start_loc','starttime']],on=['userid','rank'],how='inner')
        result['user_eloc_nloc_sep_time'] = result.apply(lambda x: diff_of_minutes(x['starttime_y'],x['starttime_x']),axis=1)
        result.rename(columns={'geohashed_start_loc':'geohashed_end_loc'},inplace=True)
        result = result[['orderid','geohashed_end_loc','user_eloc_forward_sep_time']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 获取用户历史行为次数
def get_user_count(data,sample):
    feat_path = cache_path + 'user_count_feat_%d.hdf' % (data['orderid'].sum()*sample['orderid'].sum())
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        user_count = data.groupby('userid',as_index=False)['geohashed_end_loc'].agg({'user_count':'count'})
        feat = pd.merge(sample,user_count,on=['userid'],how='left')
        feat['user_count'] = feat['user_count'] - (feat['orderid'].isin(data['orderid'].values))
        feat = feat[['user_count']].fillna(0)
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat

# 获取出发地热度（作为出发地的次数、人数）
def get_start_loc(data,sample):
    feat_path = cache_path + 'start_loc_%d.hdf' % (data['orderid'].sum() * sample['orderid'].sum())
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        start_loc = data.groupby('geohashed_start_loc',as_index=False)['userid'].agg({'sloc_count':'count',
                                                                                      'sloc_n_user':'nunique'})
        feat = pd.merge(sample, start_loc, on='geohashed_start_loc', how='left').fillna(0)
        feat['sloc_count'] = feat['sloc_count'] - feat['orderid'].isin(data['orderid'].values)
        feat = feat[['sloc_count','sloc_n_user']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat

# 获取目标地点的热度(目的地)
def get_end_loc(data,sample):
    feat_path = cache_path + 'end_loc_%d.hdf' % (data['orderid'].sum() * sample['orderid'].sum())
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        eloc_as_sloc = data.groupby('geohashed_start_loc', as_index=False)['userid'].agg(
            {'eloc_as_sloc_count': 'count',
             'eloc_as_sloc_n_uesr': 'nunique'})
        eloc = data.groupby('geohashed_end_loc', as_index=False)['userid'].agg(
            {'eloc_count': 'count',
             'eloc_n_uesr': 'nunique'})
        eloc_as_sloc.rename(columns={'geohashed_start_loc':'geohashed_end_loc'},inplace=True)
        feat = pd.merge(sample, eloc_as_sloc, on='geohashed_end_loc', how='left')
        feat = pd.merge(feat,           eloc, on='geohashed_end_loc', how='left').fillna(0)
        feat['eloc_count'] = feat['eloc_count'] - (feat['orderid'].isin(data['orderid'].tolist()) & (feat['label']==1))
        feat['eloc_in_out_rate'] = feat['eloc_count'] / (feat['eloc_as_sloc_count']+0.001)
        feat = feat[['eloc_as_sloc_count', 'eloc_as_sloc_n_uesr',
                     'eloc_count', 'eloc_n_uesr', 'eloc_in_out_rate']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat

# 用户×出发地（次数 目的地个数）
def get_user_sloc(data,sample):
    feat_path = cache_path + 'user_sloc_%d.hdf' % (data['orderid'].sum() * sample['orderid'].sum())
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        user_sloc = data.groupby(['userid','geohashed_start_loc'],as_index=False)['geohashed_end_loc'].agg(
            {'user_sloc_count':'count',
             'user_sloc_n_eloc':'nunique'})
        feat = pd.merge(sample,user_sloc,on=['userid','geohashed_start_loc'],how='left').fillna(0)
        feat['user_sloc_count'] = feat['user_sloc_count'] - feat['orderid'].isin(data['orderid'].tolist())
        feat = feat[['user_sloc_count', 'user_sloc_n_eloc']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat

# 用户×目的地
def get_user_eloc(data,sample,candidate):
    feat_path = cache_path + 'user_eloc_%d.hdf' % (data['orderid'].sum() * sample['orderid'].sum())
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        user_eloc_as_sloc = data.groupby(['userid', 'geohashed_start_loc'], as_index=False)['geohashed_end_loc'].agg(
            {'user_eloc_as_sloc_count': 'count',
             'user_eloc_as_sloc_n_sloc': 'nunique'})
        user_eloc = data.groupby(['userid', 'geohashed_end_loc'], as_index=False)['geohashed_start_loc'].agg(
            {'user_eloc_count': 'count',
             'user_eloc_n_sloc': 'nunique'})
        user_eloc_as_sloc.rename(columns={'geohashed_start_loc':'geohashed_end_loc'},inplace=True)
        feat = pd.merge(sample, user_eloc_as_sloc, on=['userid', 'geohashed_end_loc'], how='left')
        feat = pd.merge(feat, user_eloc,           on=['userid', 'geohashed_end_loc'], how='left').fillna(0)
        feat['user_eloc_count'] = feat['user_eloc_count'] - (feat['orderid'].isin(data['orderid'].tolist()) & (feat['label']==1))
        feat['user_eloc_2count'] = feat['user_eloc_count'] + feat['user_eloc_as_sloc_count']
        user_count = get_user_count(data,sample)
        feat = pd.concat([feat,user_count],axis=1)
        feat['user_eloc_count_user_rate'] = feat['user_eloc_count'] / (feat['user_count']+0.001)
        feat['user_eloc_2count_user_rate'] = feat['user_eloc_2count'] / (feat['user_count'] + 0.001)
        feat['user_sloc_goback_rate'] = feat['user_eloc_count'] / (feat['user_eloc_as_sloc_count']+0.001)
        user_next_loc = get_user_next_loc(data, candidate)
        feat = feat.merge(user_next_loc,on=['orderid','geohashed_end_loc'],how='left').fillna(10000)
        user_next_loc.rename(columns={'geohashed_end_loc':'geohashed_end_loc2'},inplace=True)
        feat = feat.merge(user_next_loc[['orderid','geohashed_end_loc2']], on='orderid', how='left')
        feat['user_eloc_next_loc_dis'] = feat.apply(lambda x:cal_distance(x['geohashed_end_loc2'],x['geohashed_end_loc']),axis=1).fillna(-100)
        user_forward_loc = get_user_forward_loc(data, candidate)
        feat = feat.merge(user_forward_loc, on=['orderid', 'geohashed_end_loc'], how='left').fillna(10000)
        feat = feat[['user_eloc_as_sloc_count', 'user_eloc_as_sloc_n_sloc','user_eloc_count',
                     'user_eloc_n_sloc','user_eloc_2count','user_eloc_count_user_rate',
                     'user_eloc_2count_user_rate','user_sloc_goback_rate','user_eloc_nloc_sep_time',
                     'user_eloc_next_loc_dis','user_eloc_forward_sep_time']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat

# 出发地×目的地
def get_sloc_eloc(data,sample):
    feat_path = cache_path + 'sloc_eloc_%d.hdf' % (data['orderid'].sum() * sample['orderid'].sum())
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        sloc_eloc = data.groupby(['geohashed_start_loc','geohashed_end_loc'],as_index=False)['userid'].agg(
            {'sloc_eloc_count':'count',
             'sloc_eloc_n_user':'nunique'})
        eloc_sloc = sloc_eloc.rename(columns={'geohashed_start_loc':'geohashed_end_loc',
                                              'geohashed_end_loc':'geohashed_start_loc',
                                              'sloc_eloc_count':'eloc_sloc_count',
                                              'sloc_eloc_n_user':'eloc_sloc_n_user'})
        feat = pd.merge(sample,sloc_eloc,on=['geohashed_start_loc','geohashed_end_loc'],how='left')
        feat = pd.merge(feat, eloc_sloc, on=['geohashed_start_loc', 'geohashed_end_loc'], how='left').fillna(0)
        feat['sloc_eloc_count'] = feat['sloc_eloc_count'] - (feat['orderid'].isin(data['orderid'].tolist()) & (feat['label']==1))
        feat['sloc_eloc_2count'] = feat['sloc_eloc_count'] + feat['eloc_sloc_count']
        feat['sloc_eloc_goback_rate'] = feat['sloc_eloc_count'] / (feat['eloc_sloc_count']+0.001)
        distance = get_distance(feat)
        feat = pd.concat([feat,distance],axis=1)
        feat = feat[['sloc_eloc_count','sloc_eloc_n_user','eloc_sloc_count',
                     'eloc_sloc_n_user','sloc_eloc_2count',
                     'sloc_eloc_goback_rate','distance','mht_distance']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat

# 自行车×目的地
def get_bike_eloc(data,sample,candidate):
    feat_path = cache_path + 'bike_eloc_%d.hdf' % (data['orderid'].sum() * sample['orderid'].sum())
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        bike_next_loc = get_bike_next_loc(data, candidate)
        feat = sample.merge(bike_next_loc, on=['orderid', 'geohashed_end_loc'], how='left').fillna(10000)
        bike_next_loc.rename(columns={'geohashed_end_loc':'geohashed_end_loc2'},inplace=True)
        feat = feat.merge(bike_next_loc[['orderid', 'geohashed_end_loc2']], on='orderid', how='left')
        feat['biek_eloc_nloc_dis'] = feat.apply(lambda x:cal_distance(x['geohashed_end_loc2'],x['geohashed_end_loc']),axis=1)
        feat = feat[['bike_eloc_sep_time','biek_eloc_nloc_dis']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat

# 用户×出发地×目的地
def get_user_sloc_eloc(data,sample,candidate):
    feat_path = cache_path + 'user_sloc_eloc_%d.hdf' % (data['orderid'].sum() * sample['orderid'].sum())
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        user_sloc_eloc_count = data.groupby(['userid','geohashed_start_loc','geohashed_end_loc'],as_index=False)[
            'userid'].agg({'user_sloc_eloc_count':'count'})
        user_eloc_sloc_count = user_sloc_eloc_count.rename(columns={'geohashed_start_loc':'geohashed_end_loc',
                                                                    'geohashed_end_loc':'geohashed_start_loc',
                                                                    'user_sloc_eloc_count':'user_eloc_sloc_count'})
        feat = pd.merge(sample, user_sloc_eloc_count, on=['userid','geohashed_start_loc','geohashed_end_loc'],how='left')
        feat = pd.merge(feat, user_eloc_sloc_count, on=['userid', 'geohashed_start_loc', 'geohashed_end_loc'], how='left').fillna(0)
        feat['user_sloc_eloc_count'] = feat['user_sloc_eloc_count'] - (feat['orderid'].isin(data['orderid'].tolist()) & (feat['label']==1))
        user_count = get_user_count(data, sample)[['user_count']]
        user_sloc = get_user_sloc(data, sample)[['user_sloc_count']]
        user_eloc = get_user_eloc(data, sample, candidate)[['user_eloc_count']]
        feat = pd.concat([feat,user_count,user_sloc,user_eloc],axis=1)
        feat['user_sloc_eloc_2count'] = feat['user_sloc_eloc_count'] + feat['user_eloc_sloc_count']
        feat['user_sloc_eloc_count_user_rate'] = feat['user_sloc_eloc_count'] / feat['user_count']
        feat['user_sloc_eloc_2count_user_rate'] = feat['user_sloc_eloc_2count'] / feat['user_count']
        feat['user_sloc_eloc_count_user_sloc_rate'] = feat['user_sloc_eloc_count'] / feat['user_sloc_count']
        feat['user_sloc_eloc_count_user_eloc_rate'] = feat['user_sloc_eloc_count'] / feat['user_eloc_count']
        feat = feat[['user_sloc_eloc_count','user_eloc_sloc_count','user_sloc_eloc_2count',
                     'user_sloc_eloc_count_user_rate','user_sloc_eloc_2count_user_rate',
                     'user_sloc_eloc_count_user_sloc_rate','user_sloc_eloc_count_user_eloc_rate']]
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat

# 用户×目的地×时间
def get_time_user_eloc(data,sample,candidate):
    feat_path = cache_path + 'time_user_eloc_%d.hdf' % (data['orderid'].sum() * sample['orderid'].sum())
    if os.path.exists(feat_path) & flag:
        feat = pd.read_hdf(feat_path, 'w')
    else:
        candidate_temp = candidate.copy()
        candidate_temp['pre_time'] = candidate_temp['starttime'].str[11:].apply(
            lambda x: str(int(x[:2]) - 1) + x[2:])
        candidate_temp['beh_time'] = candidate_temp['starttime'].str[11:].apply(
            lambda x: str(int(x[:2]) + 1) + x[2:])
        result = pd.merge(data,candidate_temp[['userid','pre_time','beh_time']],on='userid',how='left')
        result = result[(result['starttime']>=result['pre_time']) & (result['starttime']<=result['beh_time'])]
        user_eloc_time_count = result.groupby(['userid','geohashed_end_loc'],as_index=False)['userid'].agg({'user_eloc_time_count':'count'})
        feat = pd.merge(sample,user_eloc_time_count,on=['userid','geohashed_end_loc'],how='left').fillna(0)
        feat = feat['user_eloc_time_count']
        feat.to_hdf(feat_path, 'w', complib='blosc', complevel=5)
    return feat

# 二次处理特征
def second_feat(data):
    data['eloc_n_uesr'] = data['eloc_n_uesr'] - (data['user_eloc_count']==0)
    data['sloc_n_user'] = data['sloc_n_user'] - (data['user_sloc_count'] == 0)
    data['sloc_eloc_n_user'] = data['sloc_eloc_n_user'] - (data['user_sloc_eloc_count'] == 0)
    data['sloc_eloc_2n_user'] = data['sloc_eloc_n_user'] + data['eloc_sloc_n_user']
    data['user_eloc_n_sloc'] = data['user_eloc_n_sloc'] - (data['user_sloc_eloc_count'] == 0)
    data['sloc_eloc_count_sloc_rate'] = data['sloc_eloc_count'] / (data['sloc_count'] + 0.001)
    data['sloc_eloc_n_user_sloc_rate'] = data['sloc_eloc_n_user'] / (data['sloc_n_user'] + 0.001)
    return data

# 构造样本
def get_sample(data,candidate):
    n = sum(candidate['geohashed_end_loc'].isnull())+sum(data['geohashed_end_loc'].isnull())
    result_path = cache_path + 'sample_%d.hdf' % (data['orderid'].sum() * candidate['orderid'].sum()*n)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        user_end_loc = get_user_end_loc(data, candidate)            # 根据用户历史目的地点添加样本 ['orderid', 'geohashed_end_loc', 'user_n_end_loc']
        user_start_loc = get_user_start_loc(data, candidate)        # 根据用户历史起始地点添加样本 ['orderid', 'geohashed_end_loc', 'user_n_start_loc']
        loc_to_loc = get_loc_to_loc(data, candidate)                # 筛选起始地点去向最多的3个地点
        loc_with_loc = get_loc_with_loc(data, candidate)            # 与起始地点交互最多的三个地点
        loc_near_loc = get_loc_near_loc(data, candidate)            # 筛选起始地点周围3个热门地点
        bike_next_loc = get_bike_next_loc(data, candidate)          # 自行车后续起始地点
        # 汇总样本id
        result = pd.concat([user_end_loc[['orderid','geohashed_end_loc']],
                            user_start_loc[['orderid', 'geohashed_end_loc']],
                            loc_to_loc[['orderid', 'geohashed_end_loc']],
                            loc_with_loc[['orderid', 'geohashed_end_loc']],
                            loc_near_loc[['orderid', 'geohashed_end_loc']],
                            bike_next_loc[['orderid', 'geohashed_end_loc']]
                            ]).drop_duplicates()
        candidate_temp = candidate.copy()
        candidate_temp.rename(columns={'geohashed_end_loc':'label'},inplace=True)
        result = pd.merge(result, candidate_temp, on='orderid', how='left')
        result['label'] = (result['label']==result['geohashed_end_loc']).astype(int)
        # 删除起始地点和目的地点相同的样本  和 异常值
        result = result[result['geohashed_end_loc'] != result['geohashed_start_loc']]
        result = result[(~result['geohashed_end_loc'].isnull()) & (~result['geohashed_start_loc'].isnull())]
        result.index = list(range(result.shape[0]))
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 制作训练集
def make_train_set(data,candidate):
    t0 = time.time()
    n = sum(candidate['geohashed_end_loc'].isnull()) + sum(data['geohashed_end_loc'].isnull())
    result_path = cache_path + 'train_set_%d.hdf' % (data['orderid'].sum() * candidate['orderid'].sum() *n)
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        data.iloc[:,'time'] = data['starttime'].apply(lambda x:split_time(x))
        candidate.iloc[:, 'time'] = candidate['starttime'].apply(lambda x: split_time(x))
        # 汇总样本id
        print('开始构造样本...')
        sample = get_sample(data,candidate)
        gc.collect()

        print('开始构造特征...')
        user_count = get_user_count(data,sample)                            # 获取用户历史行为次数
        gc.collect()
        start_loc = get_start_loc(data,sample)                              # 获取出发地热度（作为出发地的次数、人数）
        gc.collect()
        end_loc = get_end_loc(data,sample)                                  # 获取目的地热度（作为目的地的个数、人数，作为出发地的个数、人数，折返比）
        gc.collect()
        user_sloc = get_user_sloc(data,sample)                              # 用户×出发地（次数 目的地个数）
        gc.collect()
        user_eloc = get_user_eloc(data,sample,candidate)                    # 用户×目的地
        gc.collect()
        sloc_eloc = get_sloc_eloc(data,sample)                              # 出发地×目的地
        gc.collect()
        bike_eloc = get_bike_eloc(data,sample,candidate)                    # 自行车×目的地
        gc.collect()
        user_sloc_eloc = get_user_sloc_eloc(data,sample,candidate)          # 用户×出发地×目的地
        gc.collect()
        time_user_eloc = get_time_user_eloc(data,sample,candidate)          # 时间×用户×目的地
        gc.collect()



        print('开始合并特征...')
        result = pd.concat([sample,user_count,start_loc,end_loc,user_sloc,user_eloc,sloc_eloc,
                            bike_eloc,user_sloc_eloc,time_user_eloc],axis=1)
        del sample,user_count,start_loc,end_loc,user_sloc,user_eloc,sloc_eloc,bike_eloc,\
            user_sloc_eloc,time_user_eloc
        gc.collect()
        result = second_feat(result)

        print('添加label')
        # 添加标签
        result = get_label(result)

        print('存储数据...')
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    print('生成特征一共用时{}秒'.format(time.time()-t0))
    return result

# 分开制作训练集
def make_train_set_split(data,candidate):
    t0 = time.time()
    n = candidate.shape[0] // 1100000 + 1
    print('需要循环{}次'.format(n))
    result = None
    for i in range(n):
        candidate_temp = candidate[i*1100000:(i+1)*1100000]
        train = make_train_set(data, candidate_temp)
        if result is None:
            result = train
        else:
            result = pd.concat([result,train])
        print('第{}次循环结束'.format(i+1))
    print('make_train_set_split一共用时{}秒'.format(time.time()-t0))
    return result




# # 制作训练集
# train_feat = make_train_set_split(data,train)
# del train_feat
# gc.collect()
# test_feat = make_train_set_split(data,test)
# del test_feat
# gc.collect()
#
# 线下测试集
train = pd.read_csv(train_path)
train1_eval = train[train['starttime']<'2017-05-21 00:00:00'].copy()
train2_eval = train[train['starttime']>='2017-05-23 00:00:00'].copy()
test_eval = train[(train['starttime']>='2017-05-21 00:00:00') & (train['starttime']<'2017-05-23 00:00:00')].copy()
test_eval.loc[:,'geohashed_end_loc'] = np.nan
data_eval = pd.concat([train1_eval,train2_eval,test_eval])
train_eval_feat = make_train_set(data_eval,test_eval)
del train1_eval, train2_eval, test_eval
train_eval = train[train['starttime']<'2017-05-23 00:00:00'].copy()
test_eval = train[train['starttime']>='2017-05-23 00:00:00'].copy()
test_eval.loc[:,'geohashed_end_loc'] = np.nan
data_eval = pd.concat([train_eval,test_eval])
test_eval_feat = make_train_set(data_eval,test_eval)
del train_eval, data_eval

print('对数据进行组内标准化')
for name in [ 'eloc_count','eloc_n_uesr', 'eloc_in_out_rate',
       'user_eloc_count', 'user_eloc_n_sloc',
       'user_eloc_2count', 'user_eloc_count_user_rate',
       'user_eloc_2count_user_rate', 'sloc_eloc_count', 'sloc_eloc_n_user',
        'sloc_eloc_2count','sloc_eloc_2n_user', 'sloc_eloc_goback_rate',
       'user_sloc_eloc_count', 'user_sloc_eloc_2count',
       'user_sloc_eloc_count_user_rate', 'user_sloc_eloc_2count_user_rate',
       'user_sloc_eloc_count_user_sloc_rate',
       'user_sloc_eloc_count_user_eloc_rate']:
    train_eval_feat[name] = group_normalize(train_eval_feat,'orderid',name)
    test_eval_feat[name] = group_normalize(test_eval_feat, 'orderid', name)

'''
predictors = train_eval_feat.columns.drop(['orderid', 'geohashed_end_loc', 'label', 'userid',
 'bikeid','starttime', 'geohashed_start_loc')
'''
predictors = ['biketype', 'user_count', 'sloc_count', 'sloc_n_user',
       'eloc_as_sloc_count', 'eloc_as_sloc_n_uesr', 'eloc_count',
       'eloc_n_uesr', 'eloc_in_out_rate', 'user_sloc_count',
       'user_sloc_n_eloc', 'user_eloc_as_sloc_count',
       'user_eloc_as_sloc_n_sloc', 'user_eloc_count', 'user_eloc_n_sloc',
       'user_eloc_2count', 'user_eloc_count_user_rate',
       'user_eloc_2count_user_rate', 'user_sloc_goback_rate',
       'user_eloc_nloc_sep_time', 'sloc_eloc_count', 'sloc_eloc_n_user',
       'eloc_sloc_count', 'eloc_sloc_n_user', 'sloc_eloc_2count',
       'sloc_eloc_goback_rate', 'distance', 'mht_distance',
       'bike_eloc_sep_time', 'user_sloc_eloc_count', 'user_eloc_sloc_count',
       'user_sloc_eloc_2count', 'user_sloc_eloc_count_user_rate',
       'user_sloc_eloc_2count_user_rate','user_sloc_eloc_count_user_sloc_rate',
       'user_sloc_eloc_count_user_eloc_rate', 'user_eloc_time_count',
       'sloc_eloc_2n_user', 'sloc_eloc_count_sloc_rate','biek_eloc_nloc_dis',
       'sloc_eloc_n_user_sloc_rate']


params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'max_depth':10,
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5,
    'verbose': 0,
    'seed': 66
}
print('Start training...')

lgb_train = lgb.Dataset(train_eval_feat[predictors],train_eval_feat.label)
lgb_eval = lgb.Dataset(test_eval_feat[predictors],test_eval_feat.label)

gbm = lgb.train(params,lgb_train,num_boost_round=1000)

preds = gbm.predict(test_eval_feat[predictors])
test_eval_feat['pred'] = preds
result = reshape(test_eval_feat)
result = pd.merge(test_eval[['orderid']],result,on='orderid',how='left')
print('map得分为:{}'.format(map(result)))



# 线上提交

predictors = ['biketype', 'user_count', 'sloc_count', 'sloc_n_user',
       'eloc_as_sloc_count', 'eloc_as_sloc_n_uesr', 'eloc_count',
       'eloc_n_uesr', 'eloc_in_out_rate', 'user_sloc_count',
       'user_sloc_n_eloc', 'user_eloc_as_sloc_count',
       'user_eloc_as_sloc_n_sloc', 'user_eloc_count', 'user_eloc_n_sloc',
       'user_eloc_2count', 'user_eloc_count_user_rate',
       'user_eloc_2count_user_rate', 'user_sloc_goback_rate',
       'user_eloc_nloc_sep_time', 'sloc_eloc_count', 'sloc_eloc_n_user',
       'eloc_sloc_count', 'eloc_sloc_n_user', 'sloc_eloc_2count',
       'sloc_eloc_2n_user', 'sloc_eloc_goback_rate', 'distance',
       'mht_distance', 'bike_eloc_sep_time', 'user_sloc_eloc_count',
       'user_eloc_sloc_count', 'user_sloc_eloc_2count',
       'user_sloc_eloc_count_user_rate', 'user_sloc_eloc_2count_user_rate',
       'user_sloc_eloc_count_user_sloc_rate','user_eloc_time_count',
       'user_sloc_eloc_count_user_eloc_rate']


params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'max_depth':10,
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5,
    'verbose': 0,
    'seed': 66
}
# 生成训练集
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
train_eval = train[train['starttime']<'2017-05-23 00:00:00'].copy()
test_eval = train[train['starttime']>='2017-05-23 00:00:00'].copy()
test['geohashed_end_loc'] = np.nan
test_eval['geohashed_end_loc'] = np.nan
data = pd.concat([train_eval,test_eval,test])
train_feat = make_train_set(data,test_eval)
lgb_train = lgb.Dataset(train_feat[predictors].values,train_feat.label.values)

print('Start training...')
gbm = lgb.train(params,lgb_train,1000)
del train,test,data,train_eval,test_eval,lgb_train,train_feat
gc.collect()

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
test['geohashed_end_loc'] = np.nan
data = pd.concat([train,test])
test_feat = make_train_set(data,test)
test_pred = test_feat[['orderid','geohashed_end_loc']].copy()
test_feat = test_feat[predictors].values
del train,test,data
gc.collect()


print('Start predicting...')
preds = gbm.predict(test_feat)

test_pred['pred'] = preds
gc.collect()
result = reshape(test_pred)
test = pd.read_csv(test_path)
result = pd.merge(test[['orderid']],result,on='orderid',how='left')
result.fillna('wx4f8mt',inplace=True)
result = get_noise(result,0.1)
result.to_csv(r'C:\Users\csw\Desktop\python\mobike\submission\0819(1).csv',index=False,header=False)

# 训练集 特征区间 ：'2017-05-10 00:00:00'~'2017-05-21 00:00:00'
# 训练集 标签区间 ：'2017-05-21 00:00:00'~'2017-05-23 00:00:00'
# 测试集 特征区间 ：'2017-05-10 00:00:00'~'2017-05-23 00:00:00'
# 训练集 标签区间 ：'2017-05-23 00:00:00'~'2017-05-25 00:00:00'
