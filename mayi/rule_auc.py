import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from collections import Counter
from collections import defaultdict
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

def group_rank(data, feat1, feat2, ascending):
    data.sort_values([feat1,feat2],inplace=True,ascending=ascending)
    data['rank'] = range(data.shape[0])
    min_rank = data.groupby(feat1,as_index=False)['rank'].agg({'min_rank':'min'})
    data = pd.merge(data,min_rank,on=feat1,how='left')
    data['rank'] = data['rank'] - data['min_rank']
    del data['min_rank']
    return data

# 商店对应的连接wiif
def get_connect_wifi(wifi_infos):
    if wifi_infos != '':
        for wifi_info in wifi_infos.split(';'):
            bssid,signal,Flag = wifi_info.split('|')
            if Flag == 'true':
                return bssid
    return np.nan


test = pd.read_csv(test_path)
shop = pd.read_csv(shop_path)
train = pd.read_csv(train_path)
train = train.merge(shop[['shop_id','mall_id']],on='shop_id',how='left')
# test = test.sample(frac=0.1,random_state=66, axis=0)
test = test[test['time_stamp']>='2017-08-29']

def clear(data,candidate):
    train_shop_wifi_count_dict = defaultdict(lambda : defaultdict(lambda :0))
    for mall_id, wifi_infos in zip(data['mall_id'].values,data['wifi_infos'].values):
        for wifi_info in wifi_infos.split(';'):
            bssid, signal, Flag = wifi_info.split('|')
            train_shop_wifi_count_dict[mall_id][bssid] += 1
    test_shop_wifi_count_dict = defaultdict(lambda: defaultdict(lambda: 0))
    for mall_id, wifi_infos in zip(candidate['mall_id'].values, candidate['wifi_infos'].values):
        for wifi_info in wifi_infos.split(';'):
            bssid, signal, Flag = wifi_info.split('|')
            test_shop_wifi_count_dict[mall_id][bssid] += 1
    def f_1(row):
        mall_id = row.mall_id
        return ';'.join([wifi_info for wifi_info in row.wifi_infos.split(';') if test_shop_wifi_count_dict[mall_id][wifi_info.split('|')[0]] > 0])
    def f_2(row):
        mall_id = row.mall_id
        return ';'.join([wifi_info for wifi_info in row.wifi_infos.split(';') if train_shop_wifi_count_dict[mall_id][wifi_info.split('|')[0]] > 0])
    data['wifi_infos'] = data.apply(f_1,axis=1)
    candidate['wifi_infos'] = candidate.apply(f_2,axis=1)
    return data,candidate
train,test = clear(train,test)


# 统计商店-wifi次数
shop_wifi = []
for row_id,ship_id,wifi_infos in zip(train['row_id'].values,train['shop_id'].values,train['wifi_infos'].values):
    if wifi_infos != '':
        for wifi_info in wifi_infos.split(';'):
            try:
                bssid,signal,Flag = wifi_info.split('|')
            except:
                print(wifi_info)
                bssid, signal, Flag = wifi_info.split('|')
            shop_wifi.append([row_id,ship_id,bssid,float(signal)])
shop_wifi = pd.DataFrame(shop_wifi,columns=['row_id','shop_id','bssid','signal'])
shop_wifi_mean = shop_wifi.groupby(['shop_id'],as_index=False)['bssid'].agg({'shop_wifi_mean':'mean'})
shop_wifi_mean = shop_wifi_mean(shop_wifi,'row_id','shop_wifi_mean',ascending=False)
shop_wifi['rank'] = np.exp((0-shop_wifi['rank'])*0.6)
shop_wifi_count = shop_wifi.groupby(['shop_id','bssid'],as_index=False)['rank'].agg({'shop_wifi_count':'sum'})
shop_count = shop_wifi.groupby(['shop_id'],as_index=False)['rank'].agg({'shop_count':'sum'})
shop_wifi_count = shop_wifi_count.merge(shop_count,on='shop_id',how='left')
shop_wifi_count['idf'] = shop_wifi_count['shop_wifi_count']/shop_wifi_count['shop_count']

wifi_count = shop_wifi_count.groupby(['bssid'],as_index=False)['idf'].agg({'wifi_count':'sum'})
shop_wifi_count = shop_wifi_count.merge(wifi_count,on='bssid',how='left')
shop_wifi_count['idf'] = shop_wifi_count['idf']/shop_wifi_count['wifi_count']

shop_wifi_tfidf = {}
for shop_id,grp in shop_wifi_count.groupby('shop_id'):
    wiff_idf = {}
    for tuple in grp.itertuples():
        wiff_idf[tuple.bssid] = tuple.idf
    shop_wifi_tfidf[shop_id] = wiff_idf

shop = pd.read_csv(shop_path)
mall_shop_dict = shop.groupby('mall_id')['shop_id'].unique().to_dict()

def idf_pred(row):
    wifis = {}
    shop_idf = {}
    wifi_infos = row.wifi_infos
    if wifi_infos != '':
        for wifi_info in row.wifi_infos.split(';'):
            bssid, signal, Flag = wifi_info.split('|')
            wifis[bssid] = float(signal)
        shops = mall_shop_dict[row.mall_id]
        for shop_id in shops:
            for i, bssid in enumerate(sorted(wifis, key=lambda x: wifis[x], reverse=True)):
                try:
                    idf = shop_wifi_tfidf[shop_id][bssid]*np.exp((0-i)*0.6)
                    if shop_id in shop_idf:
                        shop_idf[shop_id] += idf
                    else:
                        shop_idf[shop_id] = idf
                except:
                    pass
    try:
        result = sorted(shop_idf,key=lambda x:shop_idf[x])[-1]
    except:
        result = np.nan
    return result

test['idf_pred_shop'] = test.apply(lambda x:idf_pred(x),axis=1)
print(acc(test,'idf_pred_shop'))


