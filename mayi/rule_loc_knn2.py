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

train['longitude'] = train['longitude'] * np.cos(train['latitude'] / 57.2958)
test['longitude'] = test['longitude'] * np.cos(test['latitude'] / 57.2958)
train['cwifi'] = train['wifi_infos'].apply(get_connect_wifi)
test['cwifi'] = test['wifi_infos'].apply(get_connect_wifi)

# 商店对应的地理位置
def get_shop_loc_dict(data):
    # data_temp = data[~data['cwifi'].isnull()]
    shop_loc_dict = {}
    for shop_id,grp in data.groupby('shop_id'):
        locs = []
        for row in grp.itertuples():
            locs.append((row.longitude,row.latitude))
        shop_loc_dict[shop_id] = locs
    return shop_loc_dict

sample = get_sample(train, test, data_key)
shop_loc_dict = get_shop_loc_dict(train)
row_shop_dict = defaultdict(lambda : [])
row_shop_dict.update(sample.groupby('row_id')['shop_id'].unique().to_dict())

# def knn_loss(shop_id,wifis):
#     loss = 0
#     slen = 0
#     wlen = 0
#     for bssid in  wifis:
#         a = signal_weight(wifis[bssid])
#         b = signal_weight(shop_wifi_mean_signal_dict[(shop_id,bssid)])
#         loss += a*b
#         slen += a**2
#         wlen += b**2
#     loss_cos = loss/(slen**0.5 * wlen**0.5)
#     return loss_cos

def loc_knn2_loss(shop_id, longitude, latitude):
    loss = 0
    try:
        locs = shop_loc_dict[shop_id]
        for (lon,lat) in locs:
            loss += 0.1**(((lon-longitude)**2 + (lat-latitude)**2)**0.5*100000)
    except:
        loss = np.nan
    return loss

def loc_knn2_pred(row):
    longitude = row.longitude; latitude = row.latitude
    shops = row_shop_dict[row.row_id]
    shop_loc_knn2_loss = {}
    for shop_id in shops:
        shop_loc_knn2_loss[shop_id] = loc_knn2_loss(shop_id, longitude, latitude)
    try:
        result = sorted(shop_loc_knn2_loss,key=lambda x:shop_loc_knn2_loss[x],reverse=True)[0]
    except:
        result = np.nan
    return result

test['knn_loct_pred_shop'] = test.apply(lambda x:loc_knn2_pred(x),axis=1)
print(acc(test,'knn_loct_pred_shop'))
# print('{0}: {1}'.format(i,acc(test,'knn_loct_pred_shop')))















