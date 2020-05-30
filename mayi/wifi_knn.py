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



test = pd.read_csv(test_path)
shop = pd.read_csv(shop_path)
train = pd.read_csv(train_path)
train = train.merge(shop[['shop_id','mall_id']],on='shop_id',how='left')


'''
{
将train转换为dict 和 list格式，同时对信号进行强度转换 f(x) = 1.014 ** float(x),
如果shop_id没有出现过对应的wifi，则默认强度为1.014**(-104))。
shop_id1:[{bssid:signal,...},
          {bssid:signal,...},],
shop_id2:[{bssid:signal,...},
          {bssid:signal,...},],
...}}
'''
def get_shop_wifi_signal_dict(data):
    shop_wifi_signal_dict = defaultdict(lambda :[])
    for shop_id, wifi_infos in zip(data['shop_id'].values, data['wifi_infos'].values):
        if wifi_infos != '':
            wifi_signal_dict = defaultdict(lambda: 1.014**(-104))
            for wifi_info in wifi_infos.split(';'):
                bssid, signal, Flag = wifi_info.split('|')
                wifi_signal_dict[bssid] = 1.014 ** float(wifi_info[1])
            shop_wifi_signal_dict[shop_id].append(wifi_signal_dict)
    return shop_wifi_signal_dict

# 统计每个mall中对应的shop_id, 作为接下来的候选商店
shop_wifi_signal_dict = get_shop_wifi_signal_dict(train)
mall_shop_dict = defaultdict(lambda : [])
mall_shop_dict.update(shop.groupby('mall_id')['shop_id'].unique().to_dict())


# 计算当前样本wifi  与店铺历史wifi的相似度
def knn2_loss(shop_id, wifis):
    loss = 0
    for t_wifis in shop_wifi_signal_dict[shop_id]:
        single_loss = 0
        for bssid in wifis:   #计算两个wifi_info之间的欧式距离
            diff = wifis[bssid] - t_wifis[bssid]
            single_loss += diff * diff
        loss += 0.87 ** (single_loss * 1000)  # 将距离转换为概率并累加   转换公式为 f(x) = 0.87 ** (x * 1000)
    return loss


def knn_pred(row):
    wifis = {}
    for wifi_infos in row.wifi_infos.split(';'):
        try:
            bssid, signal, flag = wifi_infos.split('|')
        except:
            return np.nan
        wifis[bssid] = float(signal)
    shops = mall_shop_dict[row.mall_id]
    shop_knn_loss_dict = {}
    for shop_id in shops:
        shop_knn_loss_dict[shop_id] = knn2_loss(shop_id,wifis)
    try:
        result = sorted(shop_knn_loss_dict,key=lambda x:shop_knn_loss_dict[x],reverse=True)[0] #排序取最大值，作为当前样本的预测shop_id
    except:
        result = np.nan
    return result

print('开始knn预测...')
test['knn_pred_shop'] = test.apply(lambda x:knn_pred(x),axis=1)

















