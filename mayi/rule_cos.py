import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from collections import Counter
from joblib import Parallel, delayed
from mayi.feat1 import *


cache_path = 'F:/mayi_cache/'
data_path = 'C:/Users/csw/Desktop/python/mayi/data/eval/'
test_path = data_path + 'evaluation_public.csv'
shop_path = data_path + 'ccf_first_round_shop_info.csv'
train_path = data_path + 'ccf_first_round_user_shop_behavior.csv'

def acc(data,name='shop_id'):
    true_path = data_path + 'true.pkl'
    try:
        true = pickle.load(open(true_path,'+rb'))
    except:import os
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

# 信号强度转换
def signal_weight(signal):
    return 1.014**(signal)

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
        result = [wifi_info.split('|') for wifi_info in row.wifi_infos.split(';')]
        result = ';'.join(['|'.join([wifi_info[0],str(1.014**float(wifi_info[1])),wifi_info[2]]) for wifi_info in result
                  if test_shop_wifi_count_dict[mall_id][wifi_info[0]] > 0])
        return result
    def f_2(row):
        mall_id = row.mall_id
        result = [wifi_info.split('|') for wifi_info in row.wifi_infos.split(';')]
        result = ';'.join(['|'.join([wifi_info[0], str(1.014 ** float(wifi_info[1])), wifi_info[2]]) for wifi_info in result
                           if train_shop_wifi_count_dict[mall_id][wifi_info[0]] > 0])
        return result
    data['wifi_infos'] = data.apply(f_1,axis=1)
    candidate['wifi_infos'] = candidate.apply(f_2,axis=1)
    return data,candidate



test = pd.read_csv(test_path)
shop = pd.read_csv(shop_path)
train = pd.read_csv(train_path)
train = train.merge(shop[['shop_id','mall_id']],on='shop_id',how='left')
test = test.sample(frac=0.1,random_state=66, axis=0)
# test = test[test['time_stamp']>='2017-08-29']

train,test = clear(train,test)
# 商店对应wifi的平均信号强度
def get_shop_wifi_mean_signal_dict(data):
    shop_wifi = []
    for row_id, ship_id, wifi_infos in zip(data['row_id'].values, train['shop_id'].values, train['wifi_infos'].values):
        for wifi_info in wifi_infos.split(';'):
            try:
                bssid, signal, flag = wifi_info.split('|')
                shop_wifi.append([row_id, ship_id, bssid, signal])
            except:
                pass
    shop_wifi = pd.DataFrame(shop_wifi, columns=['row_id', 'shop_id', 'bssid', 'signal'])
    shop_wifi['signal'] = shop_wifi['signal'].astype('float')
    shop_wifi_mean_signal_dict = shop_wifi.groupby(['shop_id','bssid'])['signal'].mean().to_dict()
    return shop_wifi_mean_signal_dict

shop_wifi_mean_signal_dict = defaultdict(lambda : 1.014**(-104))
shop_wifi_mean_signal_dict.update(get_shop_wifi_mean_signal_dict(train))
shop = pd.read_csv(shop_path)
mall_shop_dict = shop.groupby('mall_id')['shop_id'].unique().to_dict()


def knn_loss(shop_id,wifis):
    loss = 0
    slen = 0
    wlen = 0
    for bssid in  wifis:
        a = wifis[bssid]
        b = shop_wifi_mean_signal_dict[(shop_id,bssid)]
        loss += a*b
        slen += a*a
        wlen += b*b
    loss_cos = loss/(slen**0.5 * wlen**0.5)
    return loss_cos


def knn_pred(row):
    wifis = {}
    for wifi_infos in row.wifi_infos.split(';'):
        try:
            bssid, signal, flag = wifi_infos.split('|')
        except:
            return np.nan
        wifis[bssid] = float(signal)
    shops = mall_shop_dict[row.mall_id]
    shop_knn_loss = {}
    for shop_id in shops:
        shop_knn_loss[shop_id] = knn_loss(shop_id,wifis)
    result = sorted(shop_knn_loss,key=lambda x:shop_knn_loss[x],reverse=True)[0]
    return result

test['knn_pred_shop'] = test.apply(lambda x:knn_pred(x),axis=1)
print(acc(test,'knn_pred_shop'))




















