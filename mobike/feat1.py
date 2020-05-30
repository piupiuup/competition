import os
import gc
import time
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

# 分组排序
def rank(data, feat1, feat2, ascending):
    data.sort_values([feat1,feat2],inplace=True,ascending=ascending)
    data['rank'] = range(data.shape[0])
    min_rank = data.groupby(feat1,as_index=False)['rank'].agg({'min_rank':'min'})
    data = pd.merge(data,min_rank,on=feat1,how='left')
    data['rank'] = data['rank'] - data['min_rank']
    del data['min_rank']
    return data

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

# 获取争取标签
def get_label(data):
    result_path = cache_path + 'true.pkl'
    if os.path.exists(result_path):
        true = pickle.load(open(result_path, 'rb+'))
    else:
        train = pd.read_csv(train_path)
        true = dict(zip(train['orderid'].values, train['geohashed_end_loc']))
        pickle.dump(true, open(result_path, 'wb+'))
    data['label'] = data['orderid'].map(true)
    data['label'] = (data['label'] == data['geohashed_end_loc']).astype('int')
    return data


####################构造负样本##################

# 将用户骑行过目的的地点加入成样本
def get_user_end_loc(train,test):
    result_path = cache_path + 'user_end_loc_%d.hdf' %(train.shape[0]*test.shape[0])
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        train_temp = train.copy()
        test_temp = test.copy()
        n_user_end_loc = train_temp.groupby(['userid','geohashed_end_loc'],as_index=False)['userid'].agg({'n_user_end_loc':'count'})
        result = pd.merge(test_temp[['orderid','userid']],n_user_end_loc,on=['userid'],how='left')
        test_temp['label'] = 1 - test_temp['geohashed_end_loc'].isnull()
        result = pd.merge(result,test_temp[['orderid','geohashed_end_loc','label']],on=['orderid','geohashed_end_loc'],how='left')
        result['label'].fillna(0,inplace=True)
        result = result[['orderid', 'geohashed_end_loc','n_user_end_loc']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 将用户骑行过出发的地点加入成样本
def get_user_start_loc(train,test):
    result_path = cache_path + 'user_start_loc_%d.hdf' %(train.shape[0]*test.shape[0])
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        train_temp = train.copy()
        test_temp = test.copy()
        n_user_start_loc = train_temp.groupby(['userid','geohashed_start_loc'],as_index=False)['userid'].agg({'n_user_start_loc':'count'})
        result = pd.merge(test_temp[['orderid','userid']],n_user_start_loc,on='userid',how='left')
        result.rename(columns={'geohashed_start_loc': 'geohashed_end_loc'}, inplace=True)
        result = result[['orderid', 'geohashed_end_loc','n_user_start_loc']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 筛选起始地点去向最多的3个地点
def get_loc_to_loc(train,test):
    result_path = cache_path + 'user_loc_to_loc_%d.hdf' %(train.shape[0]*test.shape[0])
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        train_temp = train.copy()
        test_temp = test.copy()
        result = train_temp.groupby(['geohashed_start_loc','geohashed_end_loc'],as_index=False)['orderid'].agg({'n_loc_end_loc':'count'})
        result = rank(result,'geohashed_start_loc','n_loc_end_loc',ascending=False)
        result = result[result['rank'] < 4]
        result = pd.merge(test_temp[['orderid', 'geohashed_start_loc']], result, on=['geohashed_start_loc'], how='left')
        result = rank(result, 'geohashed_start_loc', 'n_loc_end_loc', ascending=False)
        result = result[result['rank']<3]
        result = result[['orderid', 'geohashed_end_loc']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 用户后续的起始地点
def get_user_next_loc(train,test):
    result_path = cache_path + 'user_next_loc_%d.hdf' %(train.shape[0]*test.shape[0])
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        train_temp = train.copy()
        test_temp = test.copy()
        all_data = pd.concat([train_temp,test_temp]).drop_duplicates()
        all_data = rank(all_data,'userid','starttime',ascending=True)
        all_data['rank'] = range(all_data.shape[0])
        min_rank = all_data.groupby('userid',as_index=False)['rank'].agg({'min_rank':'min'})
        all_data = pd.merge(all_data,min_rank,on='userid',how='left')
        all_data['rank'] = all_data['rank']-all_data['min_rank']
        all_data_temp = all_data.copy()
        all_data_temp['rank'] = all_data_temp['rank']-1
        result = pd.merge(all_data[['orderid','userid','rank','starttime']],
                          all_data_temp[['userid','rank','geohashed_start_loc','starttime']],on=['userid','rank'],how='inner')
        result = result[result['orderid'].isin(test_temp['orderid'].values)]
        result['user_sep_time'] = result.apply(lambda x: diff_of_minutes(x['starttime_y'],x['starttime_x']),axis=1)
        result.rename(columns={'geohashed_start_loc':'geohashed_end_loc'},inplace=True)
        result = result[['orderid','geohashed_end_loc','user_sep_time']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result
# 用户后续的起始地点
def get_user_next_loc_feat(train,test,result):
    result_path = cache_path + 'user_next_loc_%d.hdf' %(train.shape[0]*test.shape[0])
    if os.path.exists(result_path) & flag:
        temp = pd.read_hdf(result_path, 'w')
        result = pd.merge(result, temp, on=['orderid', 'geohashed_end_loc'], how='left')
        result['user_sep_time'].fillna(100000,inplace=True)
    else:
        temp = get_bike_next_loc(train,test)
        result = pd.merge(result, temp, on=['orderid', 'geohashed_end_loc'], how='left')
        result['user_sep_time'].fillna(100000, inplace=True)
    return result

# bike后续的起始地点
def get_bike_next_loc(train,test):
    result_path = cache_path + 'bike_next_loc_%d.hdf' %(train.shape[0]*test.shape[0])
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        train_temp = train.copy()
        test_temp = test.copy()
        all_data = pd.concat([train_temp,test_temp]).drop_duplicates()
        all_data = rank(all_data,'bikeid','starttime',ascending=True)
        all_data_temp = all_data.copy()
        all_data_temp['rank'] = all_data_temp['rank'] - 1
        result = pd.merge(all_data[['orderid','bikeid','rank','starttime']],
                          all_data_temp[['bikeid','rank','geohashed_start_loc','starttime']],on=['bikeid','rank'],how='inner')
        result = result[result['orderid'].isin(test_temp['orderid'].values)]
        result['bike_sep_time'] = result.apply(lambda x: diff_of_minutes(x['starttime_y'],x['starttime_x']),axis=1)
        result.rename(columns={'geohashed_start_loc':'geohashed_end_loc'},inplace=True)
        result = result[['orderid','geohashed_end_loc','bike_sep_time']]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result
def get_bike_next_loc_feat(train,test,result):
    result_path = cache_path + 'bike_next_loc_%d.hdf' %(train.shape[0]*test.shape[0])
    if os.path.exists(result_path) & flag:
        temp = pd.read_hdf(result_path, 'w')
        result = pd.merge(result, temp, on=['orderid', 'geohashed_end_loc'], how='left')
        result['bike_sep_time'].fillna(100000, inplace=True)
    else:
        temp = get_bike_next_loc(train,test)
        result = pd.merge(result, temp, on=['orderid', 'geohashed_end_loc'], how='left')
        result['bike_sep_time'].fillna(100000, inplace=True)
    return result


# 获取用户历史行为次数
def get_user_count(train,result):
    result_path = cache_path + 'user_count_feat_%d.hdf' % (train.shape[0]*result.shape[0])
    if os.path.exists(result_path) & flag:
        temp = pd.read_hdf(result_path, 'w')
        result = pd.merge(result,temp,on=['orderid','geohashed_end_loc'],how='left')
    else:
        user_count = train.groupby('userid',as_index=False)['geohashed_end_loc'].agg({'user_count':'count'})
        result = pd.merge(result,user_count,on=['userid'],how='left')
        result[['orderid','geohashed_end_loc','user_count']].to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 获取用户去过某个地点历史行为次数
def get_user_end_loc_count(train, result):
    result_path = cache_path + 'user_end_loc_count_feat_%d.hdf' % (train.shape[0]*result.shape[0])
    if os.path.exists(result_path) & flag:
        temp = pd.read_hdf(result_path, 'w')
        result = pd.merge(result, temp, on=['orderid', 'geohashed_end_loc'], how='left')
    else:
        user_count = train.groupby(['userid','geohashed_end_loc'],as_index=False)['userid'].agg({'user_end_loc_count':'count'})
        result = pd.merge(result,user_count,on=['userid','geohashed_end_loc'],how='left')
        result[['orderid','geohashed_end_loc','user_end_loc_count']].to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 获取用户从某个地点出发的行为次数
def get_user_start_loc_count(train,result):
    result_path = cache_path + 'user_start_loc_count_feat_%d.hdf' % (train.shape[0]*result.shape[0])
    if os.path.exists(result_path) & flag:
        temp = pd.read_hdf(result_path, 'w')
        result = pd.merge(result, temp, on=['orderid', 'geohashed_end_loc'], how='left')
    else:
        user_start_loc_count = train.groupby(['userid','geohashed_start_loc'],as_index=False)['userid'].agg({'user_start_loc_count':'count'})
        user_start_loc_count.rename(columns={'geohashed_start_loc':'geohashed_end_loc'},inplace=True)
        result = pd.merge(result, user_start_loc_count, on=['userid', 'geohashed_end_loc'], how='left')
        result[['orderid','geohashed_end_loc','user_start_loc_count']].to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 获取用户从这个路径走过几次
def get_user_loc_loc_count(train,result):
    result_path = cache_path + 'user_loc_loc_count_feat_%d.hdf' % (train.shape[0] * result.shape[0])
    if os.path.exists(result_path) & flag:
        temp = pd.read_hdf(result_path, 'w')
        result = pd.merge(result, temp, on=['orderid', 'geohashed_end_loc'], how='left')
    else:
        user_count = train.groupby(['userid','geohashed_start_loc','geohashed_end_loc'],as_index=False)['userid'].agg({'user_loc_loc_count':'count'})
        result = pd.merge(result,user_count,on=['userid','geohashed_start_loc','geohashed_end_loc'],how='left')
        result[['orderid','geohashed_end_loc','user_loc_loc_count']].to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 获取用户从这个路径折返过几次
def get_user_loc_loc_return_count(train,result):
    result_path = cache_path + 'user_loc_loc_return_count_feat_%d.hdf' % (train.shape[0] * result.shape[0])
    if os.path.exists(result_path) & flag:
        temp = pd.read_hdf(result_path, 'w')
        result = pd.merge(result, temp, on=['orderid', 'geohashed_end_loc'], how='left')
    else:
        user_count = train.groupby(['userid','geohashed_start_loc','geohashed_end_loc'],as_index=False)['userid'].agg({'user_loc_loc_return_count':'count'})
        user_count.rename(columns = {'geohashed_start_loc':'geohashed_end_loc','geohashed_end_loc':'geohashed_start_loc'},inplace=True)
        result = pd.merge(result,user_count,on=['userid','geohashed_start_loc','geohashed_end_loc'],how='left')
        result[['orderid','geohashed_end_loc','user_loc_loc_return_count']].to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 计算两点之间的欧氏距离和曼哈顿距离
def get_distance(result):
    result_path = cache_path + 'distance_feat_%d.hdf' % (result.shape[0])
    if os.path.exists(result_path) & flag:
        temp = pd.read_hdf(result_path, 'w')
        result = pd.merge(result, temp, on=['orderid', 'geohashed_end_loc'], how='left')
    else:
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
        result[['orderid','geohashed_end_loc','distance','mht_distance']].to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 获取目标地点的热度(目的地)
def get_end_loc_count(train,result):
    result_path = cache_path + 'end_loc_count_feat_%d.hdf' % (train.shape[0]*result.shape[0])
    if os.path.exists(result_path) & flag:
        temp = pd.read_hdf(result_path, 'w')
        result = pd.merge(result, temp, on=['orderid', 'geohashed_end_loc'], how='left')
    else:
        end_loc_count = train.groupby('geohashed_end_loc', as_index=False)['userid'].agg({'end_loc_count': 'count'})
        result = pd.merge(result, end_loc_count, on='geohashed_end_loc', how='left')
        result[['orderid','geohashed_end_loc','end_loc_count']].to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 获取目标地点的热度(出发地地)
def get_start_loc_count(train,result):
    result_path = cache_path + 'start_loc_count_feat_%d.hdf' % (train.shape[0]*result.shape[0])
    if os.path.exists(result_path) & flag:
        temp = pd.read_hdf(result_path, 'w')
        result = pd.merge(result, temp, on=['orderid', 'geohashed_end_loc'], how='left')
    else:
        start_loc_count = train.groupby('geohashed_start_loc', as_index=False)['userid'].agg({'start_loc_count': 'count'})
        result = pd.merge(result, start_loc_count, on='geohashed_start_loc', how='left')
        result[['orderid','geohashed_end_loc','start_loc_count']].to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 构造样本
def get_sample(train,test):
    result_path = cache_path + 'sample_%d.hdf' % (train.shape[0] * test.shape[0])
    if os.path.exists(result_path) & flag:
        result = pd.read_hdf(result_path, 'w')
    else:
        user_end_loc = get_user_end_loc(train, test)            # 根据用户历史目的地点添加样本 ['orderid', 'geohashed_end_loc', 'n_user_end_loc']
        user_start_loc = get_user_start_loc(train, test)        # 根据用户历史起始地点添加样本 ['orderid', 'geohashed_end_loc', 'n_user_start_loc']
        loc_to_loc = get_loc_to_loc(train, test)                # 筛选起始地点去向最多的3个地点
        user_next_loc = get_user_next_loc(train, test)          # 用户后续的起始地点
        bike_next_loc = get_bike_next_loc(train, test)          # 自行车后续起始地点
        # 汇总样本id
        result = pd.concat([user_end_loc[['orderid','geohashed_end_loc']],
                            user_start_loc[['orderid', 'geohashed_end_loc']],
                            loc_to_loc[['orderid', 'geohashed_end_loc']],
                            user_next_loc[['orderid', 'geohashed_end_loc']],
                            bike_next_loc[['orderid', 'geohashed_end_loc']]
                            ]).drop_duplicates()
        # 添加标签
        test['label'] = 1 - test['geohashed_end_loc'].isnull()
        result = pd.merge(result, test[['orderid', 'geohashed_end_loc', 'label']],
                          on=['orderid', 'geohashed_end_loc'],
                          how='left')
        result['label'].fillna(0,inplace=True)
        # 添加['userid', 'bikeid', 'biketype', 'starttime','geohashed_start_loc']
        result = pd.merge(result, test.drop(['geohashed_end_loc','label'], axis=1), on='orderid', how='left')
        # 删除起始地点和目的地点相同的样本  和 异常值
        result = result[result['geohashed_end_loc'] != result['geohashed_start_loc']]
        result = result[(~result['geohashed_end_loc'].isnull()) & (~result['geohashed_start_loc'].isnull())]
        result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
    return result

# 制作训练集
def make_train_set(train,test):
    #result_path = cache_path + 'train_set_%d.pkl' %(train.shape[0]*test.shape[0])
    result_path1 = cache_path + 'train_set_%d.hdf' % (train.shape[0] * test.shape[0] + 1)
    result_path2 = cache_path + 'train_set_%d.hdf' % (train.shape[0] * test.shape[0] + 2)
    result_path3 = cache_path + 'train_set_%d.hdf' % (train.shape[0] * test.shape[0] + 3)
    #result_path = cache_path + 'train_set_%d.csv' % (train.shape[0] * test.shape[0])
    if os.path.exists(result_path1) & os.path.exists(result_path2) & os.path.exists(result_path3) & flag:
        #result = pickle.load(open(result_path, 'rb+'))
        result1 = pd.read_hdf(result_path1, 'w')
        result2 = pd.read_hdf(result_path2, 'w')
        result3 = pd.read_hdf(result_path3, 'w')
        result = pd.concat([result1,result2,result3])
    else:

        # 汇总样本id
        result = get_sample(train,test)
        result = get_user_count(train,result)                                   # 获取用户历史行为次数
        result = get_user_end_loc_count(train, result)                          # 获取用户去过这个地点几次
        result = get_user_start_loc_count(train, result)                        # 获取用户从目的地点出发过几次
        result = get_user_loc_loc_count(train, result)                          # 获取用户从这个路径走过几次
        result = get_user_loc_loc_return_count(train, result)                   # 获取用户从这个路径折返过几次
        result = get_distance(result)                                           # 获取起始点和最终地点的欧式距离和曼哈顿距离
        result = get_end_loc_count(train, result)                               # 获取目的地点的热度(目的地)
        result = get_start_loc_count(train, result)                             # 获取目的地点的热度(出发地)
        result = get_user_next_loc_feat(train, test, result)                    # 用户后续的起始地点
        result = get_bike_next_loc_feat(train, test, result)                    # 自行车后续起始地点
        print('result.columns:\n{}'.format(result.columns))
        result = get_label(result)

        #pickle.dump(result, open(result_path, 'wb+'))
        sep = result.shape[0]//3
        result1 = result.loc[:sep]
        result2 = result.loc[sep:2*sep]
        result3 = result.loc[2*sep:]
        result1.to_hdf(result_path1, 'w', complib='blosc', complevel=5)
        result2.to_hdf(result_path2, 'w', complib='blosc', complevel=5)
        result3.to_hdf(result_path3, 'w', complib='blosc', complevel=5)
        #result.to_csv(result_path,index=False)
    return result




# 读取数据
#train = pd.read_csv(train_path)
#test = pd.read_csv(test_path)
#
#test['geohashed_end_loc'] = np.nan
#train_feat = make_train_set(pd.concat([train,test]),train)
#test_feat = make_train_set(pd.concat([train,test]),test)


# 线下测试集调参
#import xgboost as xgb
#predictors = [ 'biketype', 'user_end_loc_count','user_count',
#       'user_start_loc_count', 'user_loc_loc_count','start_loc_count',
#       'user_loc_loc_return_count', 'distance', 'end_loc_count',
#       'user_sep_time', 'bike_sep_time']
#train = pd.read_csv(train_path)
#test = pd.read_csv(test_path)
#train_test = make_train_set(train,train)[:1000000]
#test_test = make_train_set(train,test)
#train_X = train_test[predictors]
#train_y = train_test['label']
#test_X = test_test[predictors]
#test_y = test_test['label']
#
#params = {
#    'objective': 'binary:logistic',
#    'eta': 0.1,
#    'colsample_bytree': 0.886,
#    'min_child_weight': 2,
#    'max_depth': 10,
#    'subsample': 0.886,
#    'alpha': 10,
#    'gamma': 30,
#    'lambda':50,
#    'verbose_eval': True,
#    'nthread': 8,
#    'eval_metric': 'auc',
#    'scale_pos_weight': 10,
#    'seed': 201703,
#    'missing':-1
#}
#
#xgtrain = xgb.DMatrix(train_X, train_y)
#xgtest = xgb.DMatrix(test_X, test_y)
#watchlist = [(xgtrain,'train'), (xgtest, 'val')]
#gbdt = xgb.train(params, xgtrain, 10000, evals = watchlist, verbose_eval = 100, early_stopping_rounds = 100)
#
#pred = gbdt.predict(xgtest)
#test_test['pred'] = pred
#result = reshape(test_test)
#print('map得分为%f' %(map(result)))
#
#
# 线下测试集
train = pd.read_csv(train_path)
predictors = [ 'biketype', 'user_end_loc_count','user_count',
      'user_start_loc_count', 'user_loc_loc_count','start_loc_count',
      'user_loc_loc_return_count', 'distance', 'end_loc_count',
      'user_sep_time', 'bike_sep_time']
test1 = train[(train['starttime']>='2017-05-21 00:00:00') & (train['starttime']<'2017-05-23 00:00:00')]
test1['geohashed_end_loc'] = np.nan
data1 = train[train['starttime']<'2017-05-21 00:00:00']
data1 = pd.concat([data1,test1])
test2 = train[train['starttime']>='2017-05-23 00:00:00']
test2['geohashed_end_loc'] = np.nan
data2 = train[train['starttime']<'2017-05-23 00:00:00']
data2 = pd.concat([data2,test2])
train_test = make_train_set(data1,test1)
train_test = get_label(train_test)
test_test = make_train_set(data2,test2)
print(train_test.columns)
train_X = train_test[predictors]
train_y = train_test['label']
test_X = test_test[predictors]
test_y = test_test['label']

import xgboost as xgb
params = {
   'objective': 'binary:logistic',
   'eta': 0.1,
   'colsample_bytree': 0.886,
   'min_child_weight': 2,
   'max_depth': 10,
   'subsample': 0.886,
   'alpha': 10,
   'gamma': 30,
   'lambda':50,
   'verbose_eval': True,
   'nthread': 8,
   'eval_metric': 'auc',
   'scale_pos_weight': 10,
   'seed': 201703,
   'missing':-1
}
xgtrain = xgb.DMatrix(train_X, train_y)
xgtest = xgb.DMatrix(test_X, test_y)
print('Start training...')
watchlist = [(xgtrain,'train'), (xgtest, 'val')]
gbdt = xgb.train(params, xgtrain,10000, evals = watchlist, verbose_eval = 10, early_stopping_rounds = 30)

pred = gbdt.predict(xgtest)
test_test['pred'] = pred
result = reshape(test_test)
print('map得分为%f' %(map(result)))
#
# # 训练提交
# train = pd.read_csv(train_path)
# test = pd.read_csv(test_path)
# predictors = [ 'biketype', 'user_end_loc_count','user_count',
#        'user_start_loc_count', 'user_loc_loc_count','start_loc_count',
#        'user_loc_loc_return_count', 'distance', 'end_loc_count',
#        'user_sep_time', 'bike_sep_time']
# test1 = train[(train['starttime']>='2017-05-21 00:00:00')]
# test1['geohashed_end_loc'] = np.nan
# data1 = train[train['starttime']<'2017-05-21 00:00:00']
# test2 = test
# test2['geohashed_end_loc'] = np.nan
# data2 = train
# train_test = make_train_set(data1,test1)
# train_test = get_label(train_test)
# test_test = make_train_set(data2,test2)
# train_X = train_test[predictors]
# train_y = train_test['label']
# test_X = test_test[predictors]
# test_y = test_test['label']
#
# import xgboost as xgb
# params = {
#     'objective': 'binary:logistic',
#     'eta': 0.1,
#     'colsample_bytree': 0.886,
#     'min_child_weight': 2,
#     'max_depth': 10,
#     'subsample': 0.886,
#     'alpha': 10,
#     'gamma': 30,
#     'lambda':50,
#     'verbose_eval': True,
#     'nthread': 8,
#     'eval_metric': 'auc',
#     'scale_pos_weight': 10,
#     'seed': 201703,
#     'missing':-1
# }
# xgtrain = xgb.DMatrix(train_X, train_y)
# xgtest = xgb.DMatrix(test_X, test_y)
# gbdt = xgb.train(params, xgtrain, num_boost_round=120)
#
# pred = gbdt.predict(xgtest)
# test_test['pred'] = pred
# result = reshape(test_test)
# result = pd.merge(test[['orderid']],result,on='orderid',how='left')
# result.fillna('wx4f8mt',inplace=True)
# result.to_csv(r'0713(1).csv',index=False,header=False)
