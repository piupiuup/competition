from mayi.feat1 import *
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from collections import Counter
from joblib import Parallel, delayed


cache_path = 'F:/mayi_cache/'
data_path = 'C:/Users/csw/Desktop/python/mayi/data/eval/'
test_path = data_path + 'evaluation_public.csv'
shop_path = data_path + 'ccf_first_round_shop_info.csv'
train_path = data_path + 'ccf_first_round_user_shop_behavior.csv'

def acc(data,name='shop_id'):
    true_path = data_path + 'true.pkl'
    try:
        true = pickle.load(open(true_path,'+rb'))
    except:
        print('没有发现真实数据，无法测评')
    return sum(data['row_id'].map(true)==data[name])/data.shape[0]

def apply_parallel(df_groups, _func):
    nthreads = multiprocessing.cpu_count() - 1
    print("nthreads: {}".format(nthreads))
    res = Parallel(n_jobs=nthreads)(delayed(_func)(grp.copy()) for _, grp in df_groups)
    return pd.concat(res)

def rank(data, feat1, feat2, ascending):
    data.sort_values([feat1,feat2],inplace=True,ascending=ascending)
    data['rank'] = range(data.shape[0])
    min_rank = data.groupby(feat1,as_index=False)['rank'].agg({'min_rank':'min'})
    data = pd.merge(data,min_rank,on=feat1,how='left')
    data['rank'] = data['rank'] - data['min_rank']
    del data['min_rank']
    return data

# 商店对应的连接wiif
def connect_wifi(wifi_infos):
    for wifi_info in wifi_infos.split(';'):
        bssid,signal,flag = wifi_info.split('|')
        if flag == 'true':
            return bssid
    return np.nan

# 日期差
def diff_of_date(day1, day2):
    d = {'08': 0, '09': 31}
    try:
        return abs((d[day1[5:7]] + int(day1[8:10])) - (d[day2[5:7]] + int(day2[8:10])))
    except:
        return np.nan


test = pd.read_csv(test_path)
shop = pd.read_csv(shop_path)
train = pd.read_csv(train_path)
train = train.merge(shop[['shop_id','mall_id']],on='shop_id',how='left')
test = test.sample(frac=0.1,random_state=66, axis=0)
# test = test[test['time_stamp']>='2017-08-29']
data_key = hashlib.md5(train['time_stamp'].to_string().encode()).hexdigest()+\
               hashlib.md5(test['time_stamp'].to_string().encode()).hexdigest()
print('过滤wifi...')
test['time_stamp'] = pd.to_datetime(test['time_stamp'])
train['time_stamp'] = pd.to_datetime(train['time_stamp'])
train['week'] = train['time_stamp'].dt.dayofweek
train['time'] = train['time_stamp'].dt.hour*6 + train['time_stamp'].dt.minute/10
test['week'] = test['time_stamp'].dt.dayofweek
test['time'] = test['time_stamp'].dt.hour*24 + train['time_stamp'].dt.minute/10

# 商店对应的时间
def get_shop_time_dict(data):
    shop_time_dict = defaultdict(lambda :[(0,0)])
    for shop_id, week, time in zip(data['shop_id'].values, data['week'].values, data['time'].values):
        shop_time_dict[shop_id].append((week,time))
    return shop_time_dict
print('生成候选集...')
sample = get_sample(train, test, data_key)
shop_time_dict = get_shop_time_dict(train)
row_shop_dict = defaultdict(lambda : [])
row_shop_dict.update(sample.groupby('row_id')['shop_id'].unique().to_dict())
# mall_shop_dict = defaultdict(lambda : [])
# mall_shop_dict.update(sample.groupby('mall_id')['shop_id'].unique().to_dict())
for i in []:
    def time_knn2_loss(week,time,dt_week,dt_time):
        return i*(time-dt_time)

    def time_knn2_pred(row):
        week = row.week
        time = row.time
        shops = row_shop_dict[row.row_id]
        shop_knn_loss = defaultdict(lambda : 0)
        for shop_id in shops:
            for dt_week, dt_time in shop_time_dict[shop_id]:
                shop_knn_loss[shop_id] += time_knn2_loss(week,time,dt_week,dt_time)
        try:
            result = sorted(shop_knn_loss,key=lambda x:shop_knn_loss[x],reverse=True)[0]
        except:
            result = np.nan
        return result
    print('开始knn预测...')
    test['time_knn_pred_shop'] = test.apply(lambda x:time_knn2_pred(x),axis=1)
    # print(acc(test,'time_knn_pred_shop'))
    print('{0}: {1}'.format(i,acc(test,'time_knn_pred_shop')))

















