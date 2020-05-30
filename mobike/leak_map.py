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

# 分组排序
def rank(data, feat1, feat2, ascending):
    data.sort_values([feat1, feat2], inplace=True, ascending=ascending)
    data['rank'] = range(data.shape[0])
    min_rank = data.groupby(feat1, as_index=False)['rank'].agg({'min_rank': 'min'})
    data = pd.merge(data, min_rank, on=feat1, how='left')
    data['rank'] = data['rank'] - data['min_rank']
    del data['min_rank']
    return data

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
